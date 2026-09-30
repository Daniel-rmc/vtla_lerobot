from __future__ import annotations

import json

import pytest

from vtla_lerobot.data import SamplingManifest, sample_weight_file_to_manifest, single_dataset_manifest


FEATURE_MAP = {
    "state": "observation.state",
    "action": "action",
    "cameras": {"front": "observation.images.front"},
    "tactile": {"left": "observation.images.tactile_left"},
}


def test_sampling_manifest_parses_defaults_and_overrides() -> None:
    manifest = SamplingManifest.from_dict(
        {
            "version": 1,
            "seed": 123,
            "sampling": {"strategy": "weighted", "epoch_size": 1000},
            "defaults": {"chunk_size": 8, "video_backend": "pyav", "feature_map": FEATURE_MAP},
            "datasets": [
                {"name": "a", "root": "/tmp/a", "repo_id": "local/a", "weight": 1.0},
                {
                    "name": "b",
                    "root": "/tmp/b",
                    "repo_id": "local/b",
                    "weight": 3.0,
                    "feature_map": {"cameras": {"front": "observation.images.main"}},
                },
            ],
        }
    )
    assert manifest.seed == 123
    assert manifest.sampling.epoch_size == 1000
    assert manifest.normalized_weights() == [0.25, 0.75]
    assert manifest.datasets[0].chunk_size == 8
    assert manifest.datasets[1].feature_map.cameras["front"] == "observation.images.main"
    assert manifest.datasets[1].feature_map.tactile["left"] == "observation.images.tactile_left"


def test_sampling_manifest_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        SamplingManifest.from_dict(
            {
                "defaults": {"feature_map": FEATURE_MAP},
                "datasets": [
                    {"name": "same", "root": "/tmp/a"},
                    {"name": "same", "root": "/tmp/b"},
                ],
            }
        )


def test_single_dataset_manifest_builds_weighted_manifest() -> None:
    manifest = single_dataset_manifest(root="/tmp/a", repo_id="local/a", feature_map=FEATURE_MAP, chunk_size=4)
    assert manifest.datasets[0].name == "dataset"
    assert manifest.datasets[0].chunk_size == 4
    assert manifest.normalized_weights() == [1.0]


def test_sample_weight_file_object_resolves_to_manifest(tmp_path) -> None:
    path = tmp_path / "sample_weights.json"
    path.write_text(json.dumps({"/data/task_a": 1.0, "/data/task_b": 3.0}), encoding="utf-8")
    manifest = sample_weight_file_to_manifest(path, feature_map=FEATURE_MAP, chunk_size=6, epoch_size=100, seed=9)
    assert [dataset.root for dataset in manifest.datasets] == ["/data/task_a", "/data/task_b"]
    assert [dataset.name for dataset in manifest.datasets] == ["task_a", "task_b"]
    assert [dataset.weight for dataset in manifest.datasets] == [1.0, 3.0]
    assert manifest.normalized_weights() == [0.25, 0.75]
    assert manifest.sampling.epoch_size == 100
    assert manifest.datasets[0].chunk_size == 6


def test_sample_weight_file_array_accepts_explicit_names(tmp_path) -> None:
    path = tmp_path / "sample_weights.json"
    path.write_text(
        json.dumps([
            {"name": "left", "root": "/data/left", "weight": 2.0},
            {"name": "right", "dataset_dir": "/data/right", "sample_weight": 1.0},
        ]),
        encoding="utf-8",
    )
    manifest = sample_weight_file_to_manifest(path, feature_map=FEATURE_MAP)
    assert [dataset.name for dataset in manifest.datasets] == ["left", "right"]
    assert [dataset.root for dataset in manifest.datasets] == ["/data/left", "/data/right"]
    assert manifest.normalized_weights() == [2 / 3, 1 / 3]
