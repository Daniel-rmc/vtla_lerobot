"""Multi-dataset wrappers and deterministic weighted samplers."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, Sampler

from .lerobot_adapter import LeRobotVTLADataset
from .sampling_manifest import SamplingManifest


@dataclass(frozen=True)
class MixedSampleIndex:
    dataset_index: int
    sample_index: int
    logical_index: int


class MixedLeRobotVTLADataset(Dataset):
    """Combine several LeRobot VTLA adapters behind one dataset interface."""

    def __init__(self, datasets: list[LeRobotVTLADataset], manifest: SamplingManifest):
        if not datasets:
            raise ValueError("MixedLeRobotVTLADataset requires at least one dataset")
        if len(datasets) != len(manifest.datasets):
            raise ValueError("datasets length must match manifest datasets length")
        self.datasets = datasets
        self.manifest = manifest
        self._offsets = [0]
        for dataset in datasets:
            if len(dataset) <= 0:
                raise ValueError(f"Dataset {dataset.spec.name!r} is empty")
            self._offsets.append(self._offsets[-1] + len(dataset))
        self.logical_epoch_size = manifest.sampling.epoch_size or self._offsets[-1]

    @classmethod
    def from_manifest(cls, manifest: SamplingManifest) -> "MixedLeRobotVTLADataset":
        return cls([LeRobotVTLADataset(spec) for spec in manifest.datasets], manifest)

    def __len__(self) -> int:
        return self.logical_epoch_size

    @property
    def physical_size(self) -> int:
        return self._offsets[-1]

    def resolve_physical_index(self, flat_index: int) -> tuple[int, int]:
        if flat_index < 0 or flat_index >= self.physical_size:
            raise IndexError(flat_index)
        dataset_index = bisect.bisect_right(self._offsets, flat_index) - 1
        sample_index = flat_index - self._offsets[dataset_index]
        return dataset_index, sample_index

    def __getitem__(self, index: int | tuple[int, int] | MixedSampleIndex) -> dict[str, Any]:
        if isinstance(index, MixedSampleIndex):
            dataset_index, sample_index, logical_index = index.dataset_index, index.sample_index, index.logical_index
        elif isinstance(index, tuple):
            dataset_index, sample_index = int(index[0]), int(index[1])
            logical_index = -1
        else:
            dataset_index, sample_index = self.resolve_physical_index(int(index) % self.physical_size)
            logical_index = int(index)
        sample = self.datasets[dataset_index][sample_index]
        sample["dataset_index"] = dataset_index
        sample["logical_index"] = logical_index
        return sample

    def dataset_lengths(self) -> list[int]:
        return [len(dataset) for dataset in self.datasets]


class WeightedMixedSampler(Sampler[MixedSampleIndex]):
    """Deterministically sample datasets according to manifest weights.

    The sampler is DDP-aware: each rank iterates over its strided partition of a
    shared logical epoch, so every rank sees the same weighted sequence without
    needing communication.
    """

    def __init__(
        self,
        dataset: MixedLeRobotVTLADataset,
        *,
        seed: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
    ):
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        self.dataset = dataset
        self.seed = dataset.manifest.seed if seed is None else int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.weights = torch.tensor(dataset.manifest.normalized_weights(), dtype=torch.double)
        self.lengths = torch.tensor(dataset.dataset_lengths(), dtype=torch.long)
        self.num_samples = self._rank_length()
        self.total_size = self.num_samples * self.world_size if not self.drop_last else len(dataset)

    def _rank_length(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.world_size
        return int(math.ceil((len(self.dataset) - self.rank) / self.world_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[MixedSampleIndex]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        total = len(self.dataset)
        dataset_choices = torch.multinomial(
            self.weights,
            num_samples=total,
            replacement=True,
            generator=generator,
        ).tolist()
        per_dataset_offsets = [0 for _ in self.dataset.datasets]
        for logical_index in range(self.rank, total, self.world_size):
            dataset_index = int(dataset_choices[logical_index])
            length = int(self.lengths[dataset_index].item())
            if self.dataset.manifest.sampling.replacement:
                sample_index = int(torch.randint(length, (1,), generator=generator).item())
            else:
                sample_index = per_dataset_offsets[dataset_index] % length
                per_dataset_offsets[dataset_index] += 1
            yield MixedSampleIndex(dataset_index, sample_index, logical_index)
