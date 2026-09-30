"""Single LeRobot dataset adapter that emits VTLA-compatible samples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .features import FeatureMap
from .sampling_manifest import DatasetSpec


class LeRobotVTLADataset(Dataset):
    """Wrap one LeRobot v3 dataset and return VTLA training tensors.

    The adapter deliberately keeps feature mapping outside of model code.  A
    training batch produced by this dataset contains the keys consumed by the
    LeRobot VTLA policy draft: image feature columns, `observation.state`,
    `action`, and `action_is_pad`.
    """

    def __init__(self, spec: DatasetSpec, *, lerobot_dataset: Dataset | None = None):
        self.spec = spec
        self.feature_map = spec.feature_map
        self.chunk_size = int(spec.chunk_size or 100)
        if lerobot_dataset is None:
            from lerobot.datasets import LeRobotDataset

            self.dataset = LeRobotDataset(
                repo_id=spec.repo_id or spec.name,
                root=spec.root,
                episodes=spec.episodes,
                revision=spec.revision,
                video_backend=spec.video_backend,
                delta_timestamps={
                    self.feature_map.action: list(range(self.chunk_size)),
                },
            )
        else:
            self.dataset = lerobot_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def _require(self, sample: Mapping[str, Any], key: str) -> Any:
        if key not in sample:
            available = ", ".join(sorted(sample.keys())[:40])
            raise KeyError(f"Dataset {self.spec.name!r} is missing feature {key!r}. Available: {available}")
        return sample[key]

    def _as_tensor(self, value: Any, *, dtype: torch.dtype | None = torch.float32) -> Tensor:
        if isinstance(value, Tensor):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _action_chunk(self, sample: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
        action = self._as_tensor(self._require(sample, self.feature_map.action))
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim != 2:
            raise ValueError(
                f"Dataset {self.spec.name!r} action feature {self.feature_map.action!r} must be 1D or 2D, "
                f"got shape {tuple(action.shape)}"
            )
        valid = min(action.shape[0], self.chunk_size)
        if action.shape[0] < self.chunk_size:
            pad = torch.zeros(self.chunk_size - action.shape[0], action.shape[1], dtype=action.dtype)
            action = torch.cat([action, pad], dim=0)
        elif action.shape[0] > self.chunk_size:
            action = action[: self.chunk_size]
        action_is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
        if valid < self.chunk_size:
            action_is_pad[valid:] = True
        return action, action_is_pad

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        output: dict[str, Any] = {}
        output["observation.state"] = self._as_tensor(self._require(sample, self.feature_map.state))
        action, action_is_pad = self._action_chunk(sample)
        output["action"] = action
        output["action_is_pad"] = action_is_pad

        for vtla_name, source_name in self.feature_map.cameras.items():
            output[source_name] = self._as_tensor(self._require(sample, source_name))
        for vtla_name, source_name in self.feature_map.tactile.items():
            output[source_name] = self._as_tensor(self._require(sample, source_name))

        output["dataset_name"] = self.spec.name
        output["source_index"] = int(index)
        if "task" in sample:
            output["task"] = sample["task"]
        if "episode_index" in sample:
            output["episode_index"] = int(torch.as_tensor(sample["episode_index"]).item())
        return output
