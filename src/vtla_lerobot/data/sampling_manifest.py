"""JSON parser for dataset sample-weight files and resolved training manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .features import FeatureMap


@dataclass(frozen=True)
class SamplingConfig:
    strategy: str = "weighted"
    epoch_size: int | None = None
    replacement: bool = True
    normalize_weights: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "SamplingConfig":
        raw = raw or {}
        strategy = str(raw.get("strategy", "weighted"))
        if strategy != "weighted":
            raise ValueError(f"Only weighted sampling is supported in phase 1, got {strategy!r}")
        epoch_size = raw.get("epoch_size")
        if epoch_size is not None:
            epoch_size = int(epoch_size)
            if epoch_size <= 0:
                raise ValueError("sampling.epoch_size must be positive when provided")
        return cls(
            strategy=strategy,
            epoch_size=epoch_size,
            replacement=bool(raw.get("replacement", True)),
            normalize_weights=bool(raw.get("normalize_weights", True)),
        )


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    weight: float
    root: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    split: str | None = None
    episodes: list[int] | None = None
    video_backend: str | None = None
    chunk_size: int | None = None
    feature_map: FeatureMap = field(default_factory=FeatureMap)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], defaults: Mapping[str, Any] | None = None) -> "DatasetSpec":
        defaults = defaults or {}
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError("Each dataset entry must define a non-empty name")
        root = raw.get("root", defaults.get("root"))
        repo_id = raw.get("repo_id", defaults.get("repo_id"))
        if not root and not repo_id:
            raise ValueError(f"Dataset {name!r} must define root and/or repo_id")
        weight = float(raw.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"Dataset {name!r} has non-positive weight {weight}")
        default_feature_map = FeatureMap.from_dict(defaults.get("feature_map"))
        feature_map = default_feature_map.merged(raw.get("feature_map"))
        feature_map.require_training_features()
        episodes = raw.get("episodes")
        if episodes is not None:
            episodes = [int(item) for item in episodes]
            if any(item < 0 for item in episodes):
                raise ValueError(f"Dataset {name!r} contains negative episode indices")
        chunk_size = raw.get("chunk_size", defaults.get("chunk_size"))
        if chunk_size is not None:
            chunk_size = int(chunk_size)
            if chunk_size <= 0:
                raise ValueError(f"Dataset {name!r} chunk_size must be positive")
        return cls(
            name=name,
            weight=weight,
            root=str(root) if root else None,
            repo_id=str(repo_id) if repo_id else None,
            revision=str(raw.get("revision", defaults.get("revision")))
            if raw.get("revision", defaults.get("revision"))
            else None,
            split=str(raw.get("split", defaults.get("split")))
            if raw.get("split", defaults.get("split"))
            else None,
            episodes=episodes,
            video_backend=str(raw.get("video_backend", defaults.get("video_backend")))
            if raw.get("video_backend", defaults.get("video_backend"))
            else None,
            chunk_size=chunk_size,
            feature_map=feature_map,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["feature_map"] = asdict(self.feature_map)
        return data


@dataclass(frozen=True)
class SamplingManifest:
    version: int
    seed: int
    sampling: SamplingConfig
    defaults: dict[str, Any]
    datasets: list[DatasetSpec]
    source_path: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], source_path: str | None = None) -> "SamplingManifest":
        version = int(raw.get("version", 1))
        if version != 1:
            raise ValueError(f"Unsupported sampling manifest version {version}")
        datasets_raw = raw.get("datasets")
        if not isinstance(datasets_raw, list) or not datasets_raw:
            raise ValueError("sampling manifest must contain a non-empty datasets list")
        defaults = dict(raw.get("defaults", {}) or {})
        datasets = [DatasetSpec.from_dict(item, defaults) for item in datasets_raw]
        names = [item.name for item in datasets]
        if len(set(names)) != len(names):
            raise ValueError(f"Dataset names must be unique, got duplicates in {names}")
        return cls(
            version=version,
            seed=int(raw.get("seed", 0)),
            sampling=SamplingConfig.from_dict(raw.get("sampling")),
            defaults=defaults,
            datasets=datasets,
            source_path=source_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SamplingManifest":
        path = Path(path)
        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        return cls.from_dict(raw, source_path=str(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "sampling": asdict(self.sampling),
            "defaults": self.defaults,
            "datasets": [dataset.to_dict() for dataset in self.datasets],
            "source_path": self.source_path,
        }

    def normalized_weights(self) -> list[float]:
        weights = [dataset.weight for dataset in self.datasets]
        if not self.sampling.normalize_weights:
            return weights
        total = sum(weights)
        if total <= 0:
            raise ValueError("Sum of dataset weights must be positive")
        return [weight / total for weight in weights]

    def save_resolved(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _dataset_name_from_root(root: str) -> str:
    name = Path(root).expanduser().name
    return name or "dataset"


def sample_weight_file_to_manifest(
    path: str | Path,
    *,
    feature_map: Mapping[str, Any],
    chunk_size: int = 100,
    seed: int = 0,
    epoch_size: int | None = None,
    video_backend: str | None = "pyav",
) -> SamplingManifest:
    """Load a simple dataset-root-to-weight JSON file and resolve it to a manifest.

    Preferred file format:

    ```json
    {
      "/data/task_a_lerobot_v3": 1.0,
      "/data/task_b_lerobot_v3": 2.0
    }
    ```

    An array form is also accepted for users who want explicit names:

    ```json
    [
      {"root": "/data/task_a_lerobot_v3", "weight": 1.0, "name": "task_a"},
      {"root": "/data/task_b_lerobot_v3", "weight": 2.0, "name": "task_b"}
    ]
    ```
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        datasets = [
            {"name": _dataset_name_from_root(str(root)), "root": str(root), "weight": float(weight)}
            for root, weight in raw.items()
        ]
    elif isinstance(raw, list):
        datasets = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("Each array entry in a sample weight file must be an object")
            root = item.get("root") or item.get("dataset_dir") or item.get("path")
            if not root:
                raise ValueError("Each sample weight entry must define root, dataset_dir, or path")
            datasets.append(
                {
                    "name": str(item.get("name") or _dataset_name_from_root(str(root))),
                    "root": str(root),
                    "repo_id": item.get("repo_id"),
                    "weight": float(item.get("weight", item.get("sample_weight", 1.0))),
                    "feature_map": item.get("feature_map"),
                }
            )
    else:
        raise ValueError("Sample weight file must be a JSON object or array")
    return SamplingManifest.from_dict(
        {
            "version": 1,
            "seed": seed,
            "sampling": {
                "strategy": "weighted",
                "epoch_size": epoch_size,
                "replacement": True,
                "normalize_weights": True,
            },
            "defaults": {"chunk_size": chunk_size, "video_backend": video_backend, "feature_map": feature_map},
            "datasets": datasets,
        },
        source_path=str(path),
    )


def single_dataset_manifest(
    *,
    root: str | None,
    repo_id: str | None,
    feature_map: Mapping[str, Any],
    name: str = "dataset",
    chunk_size: int = 100,
    seed: int = 0,
    video_backend: str | None = "pyav",
) -> SamplingManifest:
    return SamplingManifest.from_dict(
        {
            "version": 1,
            "seed": seed,
            "sampling": {"strategy": "weighted", "replacement": True},
            "defaults": {"chunk_size": chunk_size, "video_backend": video_backend, "feature_map": feature_map},
            "datasets": [{"name": name, "root": root, "repo_id": repo_id, "weight": 1.0}],
        }
    )
