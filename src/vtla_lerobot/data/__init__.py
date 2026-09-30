"""Data adapters for VTLA training on LeRobot datasets."""

from .features import FeatureMap
from .sampling_manifest import (
    DatasetSpec,
    SamplingConfig,
    SamplingManifest,
    sample_weight_file_to_manifest,
    single_dataset_manifest,
)

__all__ = [
    "DatasetSpec",
    "FeatureMap",
    "LeRobotVTLADataset",
    "MixedLeRobotVTLADataset",
    "MixedSampleIndex",
    "SamplingConfig",
    "SamplingManifest",
    "WeightedMixedSampler",
    "sample_weight_file_to_manifest",
    "single_dataset_manifest",
]


def __getattr__(name: str):
    if name == "LeRobotVTLADataset":
        from .lerobot_adapter import LeRobotVTLADataset

        return LeRobotVTLADataset
    if name in {"MixedLeRobotVTLADataset", "MixedSampleIndex", "WeightedMixedSampler"}:
        from . import mixed_dataset

        return getattr(mixed_dataset, name)
    raise AttributeError(name)
