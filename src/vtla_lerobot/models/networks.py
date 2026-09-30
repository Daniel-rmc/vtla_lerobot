"""Neural-network building blocks used by VTLA.

These modules only depend on PyTorch/torchvision.  They intentionally do not
import UniVTAC or private LeRobot model classes, which keeps saved VTLA
checkpoints loadable with the pinned LeRobot release.
"""

from __future__ import annotations

import math

import torch
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from .configuration_vtla import VTLAConfig


def _resolve_torchvision_weights(backbone: str, value: str | None):
    if value is None:
        return None
    weights_enum = torchvision.models.get_model_weights(backbone)
    member = value.rsplit(".", maxsplit=1)[-1]
    try:
        return getattr(weights_enum, member)
    except AttributeError as exc:
        choices = [name for name in dir(weights_enum) if name.isupper()]
        raise ValueError(f"Unknown weights {value!r} for {backbone!r}; expected one of {choices}") from exc


def _make_resnet_backbone(
    name: str,
    weights: str | None,
    freeze_batch_norm: bool,
) -> tuple[nn.Module, int]:
    if not name.startswith("resnet") or not hasattr(torchvision.models, name):
        raise ValueError(f"VTLA supports torchvision ResNet backbones, got {name!r}")
    norm_layer = FrozenBatchNorm2d if freeze_batch_norm else nn.BatchNorm2d
    model = getattr(torchvision.models, name)(
        weights=_resolve_torchvision_weights(name, weights),
        replace_stride_with_dilation=[False, False, False],
        norm_layer=norm_layer,
    )
    output_channels = model.fc.in_features
    return IntermediateLayerGetter(model, return_layers={"layer4": "feature_map"}), output_channels


def sinusoidal_position_embedding_1d(length: int, dimension: int) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError("The 1D positional embedding dimension must be even")
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    scales = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10000.0) / dimension)
    )
    output = torch.zeros(length, dimension, dtype=torch.float32)
    output[:, 0::2] = torch.sin(positions * scales)
    output[:, 1::2] = torch.cos(positions * scales)
    return output


def sinusoidal_position_embedding_2d(feature_map: Tensor, dimension: int) -> Tensor:
    """Return a deterministic ``(1, D, H, W)`` positional embedding."""
    if dimension % 4 != 0:
        raise ValueError("The 2D positional embedding dimension must be divisible by four")
    _, _, height, width = feature_map.shape
    quarter = dimension // 4
    omega = torch.arange(quarter, device=feature_map.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(quarter - 1, 1)))
    y, x = torch.meshgrid(
        torch.arange(height, device=feature_map.device, dtype=torch.float32),
        torch.arange(width, device=feature_map.device, dtype=torch.float32),
        indexing="ij",
    )
    x_phase = x.reshape(-1, 1) * omega.reshape(1, -1)
    y_phase = y.reshape(-1, 1) * omega.reshape(1, -1)
    embedding = torch.cat([x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()], dim=1)
    return embedding.to(dtype=feature_map.dtype).transpose(0, 1).reshape(1, dimension, height, width)


class CrossModalAttentionLayer(nn.Module):
    """One bidirectional visual/tactile cross-attention block."""

    def __init__(self, dimension: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.vision_to_tactile = nn.MultiheadAttention(
            dimension, n_heads, dropout=dropout, batch_first=True
        )
        self.tactile_to_vision = nn.MultiheadAttention(
            dimension, n_heads, dropout=dropout, batch_first=True
        )
        self.vision_norm1 = nn.LayerNorm(dimension)
        self.vision_norm2 = nn.LayerNorm(dimension)
        self.tactile_norm1 = nn.LayerNorm(dimension)
        self.tactile_norm2 = nn.LayerNorm(dimension)
        self.vision_ffn = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension),
            nn.Dropout(dropout),
        )
        self.tactile_ffn = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, vision: Tensor, tactile: Tensor) -> tuple[Tensor, Tensor]:
        vision_delta = self.vision_to_tactile(vision, tactile, tactile, need_weights=False)[0]
        tactile_delta = self.tactile_to_vision(tactile, vision, vision, need_weights=False)[0]
        vision = self.vision_norm1(vision + vision_delta)
        tactile = self.tactile_norm1(tactile + tactile_delta)
        vision = self.vision_norm2(vision + self.vision_ffn(vision))
        tactile = self.tactile_norm2(tactile + self.tactile_ffn(tactile))
        return vision, tactile


class BidirectionalCrossModalEncoder(nn.Module):
    def __init__(self, dimension: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossModalAttentionLayer(dimension, n_heads, dropout) for _ in range(n_layers)]
        )

    def forward(self, vision: Tensor, tactile: Tensor) -> tuple[Tensor, Tensor]:
        for layer in self.layers:
            vision, tactile = layer(vision, tactile)
        return vision, tactile


class DualPathActionHead(nn.Module):
    """Predict a base action plus a contact-gated tactile residual."""

    def __init__(self, dimension: int, action_dim: int, refine_scale: float, enabled: bool) -> None:
        super().__init__()
        self.enabled = enabled
        self.refine_scale = refine_scale
        self.main_head = nn.Linear(dimension, action_dim)
        self.contact_detector = nn.Sequential(
            nn.Linear(dimension, max(dimension // 4, 16)),
            nn.GELU(),
            nn.Linear(max(dimension // 4, 16), 1),
            nn.Sigmoid(),
        )
        self.refine_head = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, action_dim),
        )
        self.scale_predictor = nn.Sequential(nn.Linear(dimension, 1), nn.Sigmoid())

    def forward(self, decoder_features: Tensor, tactile_features: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        main_action = self.main_head(decoder_features)
        contact_probability = self.contact_detector(tactile_features)
        residual = self.refine_head(tactile_features)
        scale = self.refine_scale * self.scale_predictor(tactile_features)
        if self.enabled:
            action = main_action + residual * contact_probability * scale
        else:
            action = main_action
        return action, {
            "main_action": main_action,
            "tactile_residual": residual,
            "contact_probability": contact_probability,
            "refine_scale": scale,
        }


class VTLANetwork(nn.Module):
    """Vision/tactile CVAE transformer with a tactile residual action head."""

    def __init__(self, config: VTLAConfig) -> None:
        super().__init__()
        self.config = config
        state_dim = config.robot_state_feature.shape[0]
        action_dim = config.action_feature.shape[0]
        self.vision_names = list(config.vision_features)
        self.tactile_names = list(config.tactile_features)

        self.vision_backbone, vision_channels = _make_resnet_backbone(
            config.vision_backbone,
            config.vision_backbone_weights,
            config.freeze_backbone_batch_norm,
        )
        self.tactile_backbone, tactile_channels = _make_resnet_backbone(
            config.tactile_backbone,
            config.tactile_backbone_weights,
            config.freeze_backbone_batch_norm,
        )
        self.vision_projection = nn.Conv2d(vision_channels, config.dim_model, kernel_size=1)
        self.tactile_projection = nn.Conv2d(tactile_channels, config.dim_model, kernel_size=1)
        self.vision_source_embedding = nn.Embedding(len(self.vision_names), config.dim_model)
        self.tactile_source_embedding = nn.Embedding(len(self.tactile_names), config.dim_model)

        self.cross_modal_encoder = BidirectionalCrossModalEncoder(
            config.dim_model,
            config.n_heads,
            config.cross_attention_layers,
            config.dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)

        self.state_projection = nn.Linear(state_dim, config.dim_model)
        self.latent_projection = nn.Linear(config.latent_dim, config.dim_model)
        self.encoder_token_type = nn.Embedding(2, config.dim_model)
        self.action_queries = nn.Embedding(config.chunk_size, config.dim_model)

        if config.use_vae:
            vae_layer = nn.TransformerEncoderLayer(
                d_model=config.dim_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation="relu",
                batch_first=True,
                norm_first=False,
            )
            self.vae_encoder = nn.TransformerEncoder(
                vae_layer, num_layers=config.n_vae_encoder_layers
            )
            self.vae_cls = nn.Parameter(torch.empty(1, 1, config.dim_model))
            self.vae_state_projection = nn.Linear(state_dim, config.dim_model)
            self.vae_action_projection = nn.Linear(action_dim, config.dim_model)
            self.vae_output = nn.Linear(config.dim_model, config.latent_dim * 2)
            vae_length = config.chunk_size + 2
            self.register_buffer(
                "vae_position_embedding",
                sinusoidal_position_embedding_1d(vae_length, config.dim_model).unsqueeze(0),
            )
            nn.init.normal_(self.vae_cls, std=0.02)

        self.action_head = DualPathActionHead(
            config.dim_model,
            action_dim,
            config.tactile_refine_scale,
            config.use_tactile_refine,
        )

    def _encode_images(
        self,
        batch: dict[str, Tensor],
        names: list[str],
        backbone: nn.Module,
        projection: nn.Module,
        source_embedding: nn.Embedding,
    ) -> Tensor:
        tokens = []
        for source_index, name in enumerate(names):
            feature_map = backbone(batch[name])["feature_map"]
            feature_map = projection(feature_map)
            position = sinusoidal_position_embedding_2d(feature_map, self.config.dim_model)
            source = source_embedding.weight[source_index].view(1, -1, 1, 1)
            tokens.append((feature_map + position + source).flatten(2).transpose(1, 2))
        return torch.cat(tokens, dim=1)

    def _sample_latent(
        self,
        state: Tensor,
        actions: Tensor | None,
        action_is_pad: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        batch_size = state.shape[0]
        if not self.config.use_vae or actions is None or not self.training:
            latent = torch.zeros(
                batch_size,
                self.config.latent_dim,
                device=state.device,
                dtype=state.dtype,
            )
            return latent, None, None

        cls = self.vae_cls.expand(batch_size, -1, -1)
        state_token = self.vae_state_projection(state).unsqueeze(1)
        action_tokens = self.vae_action_projection(actions)
        vae_tokens = torch.cat([cls, state_token, action_tokens], dim=1)
        vae_tokens = vae_tokens + self.vae_position_embedding.to(dtype=vae_tokens.dtype)

        if action_is_pad is None:
            action_is_pad = torch.zeros(
                batch_size, actions.shape[1], dtype=torch.bool, device=actions.device
            )
        prefix_mask = torch.zeros(batch_size, 2, dtype=torch.bool, device=actions.device)
        padding_mask = torch.cat([prefix_mask, action_is_pad], dim=1)
        cls_output = self.vae_encoder(vae_tokens, src_key_padding_mask=padding_mask)[:, 0]
        latent_parameters = self.vae_output(cls_output)
        mean, log_variance = latent_parameters.chunk(2, dim=-1)
        latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        return latent, mean, log_variance

    def forward(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, tuple[Tensor | None, Tensor | None], dict[str, Tensor]]:
        state = batch["observation.state"]
        actions = batch.get("action")
        action_is_pad = batch.get("action_is_pad")
        latent, mean, log_variance = self._sample_latent(state, actions, action_is_pad)

        vision_tokens = self._encode_images(
            batch,
            self.vision_names,
            self.vision_backbone,
            self.vision_projection,
            self.vision_source_embedding,
        )
        tactile_tokens = self._encode_images(
            batch,
            self.tactile_names,
            self.tactile_backbone,
            self.tactile_projection,
            self.tactile_source_embedding,
        )
        vision_tokens, tactile_tokens = self.cross_modal_encoder(vision_tokens, tactile_tokens)

        latent_token = self.latent_projection(latent) + self.encoder_token_type.weight[0]
        state_token = self.state_projection(state) + self.encoder_token_type.weight[1]
        memory_tokens = torch.cat(
            [latent_token.unsqueeze(1), state_token.unsqueeze(1), vision_tokens, tactile_tokens], dim=1
        )
        memory = self.encoder(memory_tokens)

        queries = self.action_queries.weight.unsqueeze(0).expand(state.shape[0], -1, -1)
        decoder_features = self.decoder(torch.zeros_like(queries) + queries, memory)
        tactile_summary = tactile_tokens.mean(dim=1, keepdim=True).expand(-1, self.config.chunk_size, -1)
        action, components = self.action_head(decoder_features, tactile_summary)
        return action, (mean, log_variance), components

