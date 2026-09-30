"""LeRobot policy wrapper for VTLA."""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION

from .configuration_vtla import VTLAConfig
from .networks import VTLANetwork


class _TemporalEnsembler:
    """Online action-chunk ensemble matching LeRobot ACT semantics."""

    def __init__(self, coefficient: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.weights = torch.exp(-coefficient * torch.arange(chunk_size, dtype=torch.float32))
        self.cumulative_weights = torch.cumsum(self.weights, dim=0)
        self.reset()

    def reset(self) -> None:
        self.actions: Tensor | None = None
        self.counts: Tensor | None = None

    def update(self, actions: Tensor) -> Tensor:
        weights = self.weights.to(actions.device)
        cumulative = self.cumulative_weights.to(actions.device)
        if self.actions is None:
            self.actions = actions.clone()
            self.counts = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=actions.device
            )
        else:
            self.actions *= cumulative[self.counts - 1]
            self.actions += actions[:, :-1] * weights[self.counts]
            self.actions /= cumulative[self.counts]
            self.counts = torch.clamp(self.counts + 1, max=self.chunk_size)
            self.actions = torch.cat([self.actions, actions[:, -1:]], dim=1)
            self.counts = torch.cat([self.counts, torch.ones_like(self.counts[-1:])])
        action = self.actions[:, 0]
        self.actions = self.actions[:, 1:]
        self.counts = self.counts[1:]
        return action


class VTLAPolicy(PreTrainedPolicy):
    config_class = VTLAConfig
    name = "vtla"

    def __init__(
        self,
        config: VTLAConfig,
        dataset_stats: dict[str, Any] | None = None,
        dataset_meta: Any | None = None,
        **kwargs,
    ) -> None:
        del dataset_stats, dataset_meta, kwargs
        super().__init__(config)
        config.validate_features()
        self.model = VTLANetwork(config)
        self.reset()

    def get_optim_params(self):
        backbone_prefixes = ("model.vision_backbone", "model.tactile_backbone")
        backbone_params = [
            parameter
            for name, parameter in self.named_parameters()
            if name.startswith(backbone_prefixes) and parameter.requires_grad
        ]
        policy_params = [
            parameter
            for name, parameter in self.named_parameters()
            if not name.startswith(backbone_prefixes) and parameter.requires_grad
        ]
        return [
            {"params": policy_params},
            {"params": backbone_params, "lr": self.config.optimizer_lr_backbone},
        ]

    def reset(self) -> None:
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = _TemporalEnsembler(
                self.config.temporal_ensemble_coeff, self.config.chunk_size
            )
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        del kwargs
        self.eval()
        action, _, _ = self.model(batch)
        return action

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        del kwargs
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            return self.temporal_ensembler.update(self.predict_action_chunk(batch))
        if not self._action_queue:
            action_chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(action_chunk.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        predicted_actions, (mean, log_variance), components = self.model(batch)
        target_actions = batch[ACTION]
        if target_actions.shape != predicted_actions.shape:
            raise ValueError(
                f"Action target shape {tuple(target_actions.shape)} does not match VTLA output "
                f"{tuple(predicted_actions.shape)}"
            )

        absolute_error = F.l1_loss(predicted_actions, target_actions, reduction="none")
        padding_mask = batch.get("action_is_pad")
        if padding_mask is None:
            valid_mask = torch.ones_like(absolute_error[..., :1], dtype=torch.bool)
        else:
            valid_mask = ~padding_mask.unsqueeze(-1)
        valid_values = valid_mask.sum() * absolute_error.shape[-1]
        l1_loss = (absolute_error * valid_mask).sum() / valid_values.clamp_min(1)

        loss = l1_loss
        output = {
            "l1_loss": l1_loss.item(),
            "contact_probability": components["contact_probability"].mean().item(),
            "tactile_residual_l1": components["tactile_residual"].abs().mean().item(),
        }
        if self.config.use_vae and mean is not None and log_variance is not None:
            kld_loss = (-0.5 * (1 + log_variance - mean.square() - log_variance.exp())).sum(-1).mean()
            loss = loss + self.config.kl_weight * kld_loss
            output["kld_loss"] = kld_loss.item()
        return loss, output

