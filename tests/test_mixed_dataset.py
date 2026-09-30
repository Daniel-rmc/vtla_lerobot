from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import Dataset

from vtla_lerobot.data import LeRobotVTLADataset, MixedLeRobotVTLADataset, SamplingManifest, WeightedMixedSampler


FEATURE_MAP = {
    "state": "observation.state",
    "action": "action",
    "cameras": {"front": "observation.images.front"},
    "tactile": {"left": "observation.images.tactile_left"},
}


class TinyLeRobotLikeDataset(Dataset):
    def __init__(self, length: int, *, action_dim: int = 2):
        self.length = length
        self.action_dim = action_dim

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return {
            "observation.state": torch.ones(3) * index,
            "action": torch.ones(self.action_dim) * index,
            "observation.images.front": torch.ones(3, 8, 8),
            "observation.images.tactile_left": torch.ones(3, 8, 8) * 2,
            "episode_index": index,
            "task": "tiny",
        }


def make_manifest() -> SamplingManifest:
    return SamplingManifest.from_dict(
        {
            "version": 1,
            "seed": 7,
            "sampling": {"strategy": "weighted", "epoch_size": 300, "replacement": True},
            "defaults": {"chunk_size": 4, "feature_map": FEATURE_MAP},
            "datasets": [
                {"name": "a", "root": "/tmp/a", "weight": 1.0},
                {"name": "b", "root": "/tmp/b", "weight": 2.0},
            ],
        }
    )


def make_mixed_dataset() -> MixedLeRobotVTLADataset:
    manifest = make_manifest()
    datasets = [
        LeRobotVTLADataset(manifest.datasets[0], lerobot_dataset=TinyLeRobotLikeDataset(5)),
        LeRobotVTLADataset(manifest.datasets[1], lerobot_dataset=TinyLeRobotLikeDataset(7)),
    ]
    return MixedLeRobotVTLADataset(datasets, manifest)


def test_single_adapter_outputs_vtla_batch() -> None:
    manifest = make_manifest()
    dataset = LeRobotVTLADataset(manifest.datasets[0], lerobot_dataset=TinyLeRobotLikeDataset(5))
    sample = dataset[2]
    assert sample["observation.state"].shape == (3,)
    assert sample["action"].shape == (4, 2)
    assert sample["action_is_pad"].tolist() == [False, True, True, True]
    assert sample["observation.images.front"].shape == (3, 8, 8)
    assert sample["dataset_name"] == "a"


def test_weighted_sampler_is_deterministic() -> None:
    dataset = make_mixed_dataset()
    first = list(WeightedMixedSampler(dataset, seed=11))
    second = list(WeightedMixedSampler(dataset, seed=11))
    assert first == second


def test_weighted_sampler_respects_approximate_ratios() -> None:
    dataset = make_mixed_dataset()
    counts = Counter(item.dataset_index for item in WeightedMixedSampler(dataset, seed=11))
    ratio = counts[1] / (counts[0] + counts[1])
    assert 0.55 < ratio < 0.78


def test_weighted_sampler_ddp_partitions_logical_indices() -> None:
    dataset = make_mixed_dataset()
    rank0 = list(WeightedMixedSampler(dataset, seed=11, rank=0, world_size=2))
    rank1 = list(WeightedMixedSampler(dataset, seed=11, rank=1, world_size=2))
    logical0 = {item.logical_index for item in rank0}
    logical1 = {item.logical_index for item in rank1}
    assert logical0.isdisjoint(logical1)
    assert logical0 | logical1 == set(range(len(dataset)))
