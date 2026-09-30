#!/usr/bin/env python3
"""Validate local manipulationNet LeRobot v3 datasets and write a reproducibility manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_values(paths: list[Path], column: str) -> np.ndarray:
    chunks = []
    for path in paths:
        values = pq.read_table(path, columns=[column])[column].combine_chunks().to_pylist()
        chunks.append(np.asarray(values))
    if not chunks:
        raise ValueError(f"No parquet values found for column {column!r}")
    return np.concatenate(chunks, axis=0)


def validate_dataset(root: Path, decode_samples: int) -> dict:
    repo_id = f"local/{root.name}"
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    errors: list[str] = []
    warnings: list[str] = []

    episode_files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    data_files = sorted((root / "data").glob("*/*.parquet"))
    episode_table = pa.concat_tables([pq.read_table(path) for path in episode_files])
    episode_indices = parquet_values(episode_files, "episode_index").reshape(-1)
    data_episode_indices = parquet_values(data_files, "episode_index").reshape(-1)
    expected = np.arange(metadata.total_episodes)
    observed = np.unique(episode_indices)
    observed_data = np.unique(data_episode_indices)
    if not np.array_equal(observed, expected):
        errors.append("episode metadata indices are not contiguous from zero")
    if not np.array_equal(observed_data, expected):
        errors.append("data episode indices are not contiguous from zero")
    if len(episode_indices) != metadata.total_episodes:
        errors.append("episode metadata row count differs from info.total_episodes")

    data_references = set(
        zip(
            episode_table["data/chunk_index"].to_pylist(),
            episode_table["data/file_index"].to_pylist(),
            strict=True,
        )
    )
    for chunk_index, file_index in sorted(data_references):
        referenced = root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
        if not referenced.is_file():
            errors.append(f"episode metadata references missing data file {referenced.relative_to(root)}")
    metadata_references = set(
        zip(
            episode_table["meta/episodes/chunk_index"].to_pylist(),
            episode_table["meta/episodes/file_index"].to_pylist(),
            strict=True,
        )
    )
    for chunk_index, file_index in sorted(metadata_references):
        referenced = (
            root
            / "meta"
            / "episodes"
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.parquet"
        )
        if not referenced.is_file():
            errors.append(
                f"episode metadata references missing episode file {referenced.relative_to(root)}"
            )
    for camera_key in metadata.camera_keys:
        video_references = set(
            zip(
                episode_table[f"videos/{camera_key}/chunk_index"].to_pylist(),
                episode_table[f"videos/{camera_key}/file_index"].to_pylist(),
                strict=True,
            )
        )
        for chunk_index, file_index in sorted(video_references):
            referenced = (
                root
                / "videos"
                / camera_key
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            if not referenced.is_file():
                errors.append(f"episode metadata references missing video file {referenced.relative_to(root)}")

    action = parquet_values(data_files, "action")
    state = parquet_values(data_files, "observation.state")
    if action.shape != (metadata.total_frames, *metadata.features["action"]["shape"]):
        errors.append(f"unexpected action array shape {action.shape}")
    if state.shape != (metadata.total_frames, *metadata.features["observation.state"]["shape"]):
        errors.append(f"unexpected state array shape {state.shape}")
    if not np.isfinite(action).all():
        errors.append("action contains NaN or infinity")
    if not np.isfinite(state).all():
        errors.append("observation.state contains NaN or infinity")

    stats_episode_max = float(np.asarray(metadata.stats["episode_index"]["max"]).reshape(-1)[0])
    if int(stats_episode_max) != metadata.total_episodes - 1:
        warnings.append(
            "stats.json episode_index summary is stale; this field is not used by VTLA normalization"
        )

    decoded = 0
    if decode_samples:
        dataset = LeRobotDataset(
            repo_id,
            root=root,
            video_backend="pyav",
            return_uint8=True,
        )
        episode_choices = np.linspace(
            0,
            metadata.total_episodes - 1,
            min(decode_samples, metadata.total_episodes),
            dtype=int,
        )
        for episode_index in episode_choices:
            index = int(metadata.episodes["dataset_from_index"][int(episode_index)])
            sample = dataset[index]
            for key in metadata.camera_keys:
                if tuple(sample[key].shape) != (3, *metadata.features[key]["shape"][:2]):
                    errors.append(f"decoded {key} has shape {tuple(sample[key].shape)}")
            decoded += 1

    return {
        "name": root.name,
        "root": str(root.resolve()),
        "codebase_version": str(metadata._version),
        "robot_type": metadata.robot_type,
        "fps": metadata.fps,
        "total_episodes": metadata.total_episodes,
        "total_frames": metadata.total_frames,
        "tasks": [str(task) for task in metadata.tasks.index],
        "camera_keys": list(metadata.camera_keys),
        "state_shape": list(metadata.features["observation.state"]["shape"]),
        "action_shape": list(metadata.features["action"]["shape"]),
        "action_min": action.min(axis=0).tolist(),
        "action_max": action.max(axis=0).tolist(),
        "state_min": state.min(axis=0).tolist(),
        "state_max": state.max(axis=0).tolist(),
        "decoded_samples": decoded,
        "info_sha256": file_sha256(root / "meta" / "info.json"),
        "warnings": warnings,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/workspace/manipulationNet/peg_in_hole/15holes_merged"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decode-samples", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.source_root / "meta" / "info.json").is_file():
        roots = [args.source_root]
    else:
        roots = sorted(
            path for path in args.source_root.iterdir() if (path / "meta" / "info.json").is_file()
        )
    if not roots:
        raise FileNotFoundError(f"No LeRobot datasets found below {args.source_root}")
    datasets = [validate_dataset(root, args.decode_samples) for root in roots]
    reference = datasets[0]
    errors = []
    for dataset in datasets:
        for key in ("codebase_version", "robot_type", "fps", "camera_keys", "state_shape", "action_shape"):
            if dataset[key] != reference[key]:
                errors.append(f"{dataset['name']}: {key} differs from {reference['name']}")
        errors.extend(f"{dataset['name']}: {message}" for message in dataset["errors"])

    manifest = {
        "source_root": str(args.source_root.resolve()),
        "dataset_count": len(datasets),
        "total_episodes": sum(item["total_episodes"] for item in datasets),
        "total_frames": sum(item["total_frames"] for item in datasets),
        "datasets": datasets,
        "errors": errors,
        "valid": not errors,
    }
    rendered = json.dumps(manifest, indent=2, ensure_ascii=False)
    if args.quiet:
        print(
            f"valid={manifest['valid']} datasets={manifest['dataset_count']} "
            f"episodes={manifest['total_episodes']} frames={manifest['total_frames']} "
            f"errors={len(manifest['errors'])}"
        )
    else:
        print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
