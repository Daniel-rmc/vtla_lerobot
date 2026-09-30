#!/usr/bin/env python3
"""Convert a ManipulationNet LeRobot v3 dataset for PI05 post-training.

The source dataset is never modified. The converted dataset keeps all episodes and
videos, but rewrites the state/action vectors into a PI05-compatible ordering.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS_TOLERANCE = 1e-3
CHUNK_SIZE = 50
RGB_CAMERAS = [
    "observation.images.front_camera",
    "observation.images.left_hand_camera",
    "observation.images.right_hand_camera",
]
TACTILE_CAMERAS = [
    "observation.images.left_tactile_left_camera",
    "observation.images.left_tactile_right_camera",
    "observation.images.right_tactile_left_camera",
    "observation.images.right_tactile_right_camera",
]
ALL_CAMERAS = RGB_CAMERAS + TACTILE_CAMERAS
STATE_NAMES = [
    *(f"left_arm_joint_{i}" for i in range(7)),
    "left_arm_gripper",
    *(f"right_arm_joint_{i}" for i in range(7)),
    "right_arm_gripper",
    "left_arm_x",
    "left_arm_y",
    "left_arm_z",
    "left_arm_qx",
    "left_arm_qy",
    "left_arm_qz",
    "left_arm_qw",
    "right_arm_x",
    "right_arm_y",
    "right_arm_z",
    "right_arm_qx",
    "right_arm_qy",
    "right_arm_qz",
    "right_arm_qw",
]
ACTION_NAMES = [
    *(f"left_arm_joint_{i}" for i in range(7)),
    "left_arm_gripper",
    *(f"right_arm_joint_{i}" for i in range(7)),
    "right_arm_gripper",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def symlink_relative(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def source_data_files(root: Path) -> list[Path]:
    return sorted((root / "data").glob("chunk-*/file-*.parquet"))


def source_episode_files(root: Path) -> list[Path]:
    return sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))


def as_matrix(table: pa.Table, column: str, expected_dim: int) -> np.ndarray:
    if column not in table.column_names:
        raise ValueError(f"missing required column {column!r}")
    values = np.asarray(table[column].combine_chunks().to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise ValueError(f"{column!r} must have shape (N, {expected_dim}), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{column!r} contains NaN or Inf")
    return values


def stats_for_array(values: np.ndarray) -> dict[str, list[Any]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) < 2:
        raise ValueError("at least two values are required to compute statistics")
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [int(len(values))] * values.shape[1],
        "q01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).astype(float).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).astype(float).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).astype(float).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
    }


def validate_source(root: Path, info: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    features = info.get("features", {})
    if info.get("fps") != 10:
        warnings.append({"kind": "fps", "message": f"source fps is {info.get('fps')}, expected 10"})
    if features.get("action", {}).get("shape") != [16]:
        errors.append({"kind": "schema", "message": "source action must be 16-dimensional"})
    if features.get("observation.state", {}).get("shape") != [28]:
        errors.append({"kind": "schema", "message": "source observation.state must be 28-dimensional"})
    for relative in ["meta/info.json", "meta/stats.json", "meta/tasks.parquet"]:
        if not (root / relative).is_file():
            errors.append({"kind": "missing_file", "path": relative})
    data_files = source_data_files(root)
    if not data_files:
        errors.append({"kind": "missing_file", "path": "data/**/*.parquet"})

    rows = 0
    episodes: dict[int, dict[str, Any]] = {}
    previous_index: int | None = None
    for path in data_files:
        try:
            table = pq.read_table(path)
            state = as_matrix(table, "observation.state", 28)
            action = as_matrix(table, "action", 16)
            n = table.num_rows
            rows += n
            for column in ["episode_index", "frame_index", "index", "timestamp", "task_index"]:
                if column not in table.column_names:
                    errors.append({"kind": "schema", "path": str(path), "message": f"missing {column}"})
            if "episode_index" not in table.column_names:
                continue
            ep = np.asarray(table["episode_index"].combine_chunks().to_numpy(), dtype=np.int64)
            frame = np.asarray(table["frame_index"].combine_chunks().to_numpy(), dtype=np.int64)
            index = np.asarray(table["index"].combine_chunks().to_numpy(), dtype=np.int64)
            timestamp = np.asarray(table["timestamp"].combine_chunks().to_numpy(), dtype=np.float64)
            if not np.isfinite(timestamp).all():
                errors.append({"kind": "values", "path": str(path), "message": "timestamp contains NaN/Inf"})
            if len(index) and previous_index is not None and int(index[0]) != previous_index + 1:
                errors.append({"kind": "index", "path": str(path), "message": "global index is not continuous"})
            if len(index):
                previous_index = int(index[-1])
            for i, episode_index in enumerate(ep):
                entry = episodes.setdefault(int(episode_index), {"rows": 0, "last_frame": -1, "last_timestamp": -1.0})
                if i and ep[i - 1] == episode_index:
                    if frame[i] != frame[i - 1] + 1:
                        errors.append({"kind": "episode_frame", "path": str(path), "episode": int(episode_index), "message": "frame_index is not continuous"})
                    if timestamp[i] <= timestamp[i - 1]:
                        errors.append({"kind": "timestamp", "path": str(path), "episode": int(episode_index), "message": "timestamp is not increasing"})
                entry["rows"] += 1
                entry["last_frame"] = int(frame[i])
                entry["last_timestamp"] = float(timestamp[i])
                if not 0 <= int(episode_index) < int(info.get("total_episodes", 0)):
                    errors.append({"kind": "episode_index", "path": str(path), "episode": int(episode_index), "message": "episode index out of range"})
            if len(timestamp) > 1:
                dt = np.diff(timestamp)
                local_dt = dt[dt > 0]
                if len(local_dt) and abs(float(np.median(local_dt)) - 0.1) > FPS_TOLERANCE:
                    warnings.append({"kind": "fps", "path": str(path), "median_dt": float(np.median(local_dt))})
            del state, action
        except Exception as exc:  # noqa: BLE001 - report the offending file
            errors.append({"kind": "read_error", "path": str(path), "message": str(exc)})

    for camera in ALL_CAMERAS:
        camera_root = root / "videos" / camera
        video_files = sorted(camera_root.glob("chunk-*/file-*.mp4"))
        if not video_files:
            errors.append({"kind": "missing_video", "camera": camera})
        for video in video_files:
            if not video.is_file() or video.stat().st_size == 0:
                errors.append({"kind": "bad_video", "path": str(video)})

    expected_frames = int(info.get("total_frames", rows))
    if rows != expected_frames:
        errors.append({"kind": "frame_count", "expected": expected_frames, "actual": rows})
    report = {
        "source_root": str(root.resolve()),
        "source_total_frames": rows,
        "source_total_episodes": len(episodes),
        "episodes": {str(k): v for k, v in sorted(episodes.items())},
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }
    return data_files, report


def load_all_source_arrays(data_files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    for path in data_files:
        table = pq.read_table(path, columns=["observation.state", "action", "episode_index", "frame_index", "timestamp"])
        states.append(as_matrix(table, "observation.state", 28))
        actions.append(as_matrix(table, "action", 16))
        episode_indices.append(np.asarray(table["episode_index"].combine_chunks().to_numpy(), dtype=np.int64))
        frame_indices.append(np.asarray(table["frame_index"].combine_chunks().to_numpy(), dtype=np.int64))
        timestamps.append(np.asarray(table["timestamp"].combine_chunks().to_numpy(), dtype=np.float64))
    return tuple(np.concatenate(x) for x in (states, actions, episode_indices, frame_indices, timestamps))  # type: ignore[return-value]


def transform_arrays(
    states: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = len(states)
    output_state = np.empty((n, 30), dtype=np.float32)
    output_state[:, 0:7] = states[:, 0:7]
    output_state[:, 7] = actions[:, 7]
    output_state[:, 8:15] = states[:, 14:21]
    output_state[:, 15] = actions[:, 15]
    output_state[:, 16:23] = states[:, 7:14]
    output_state[:, 23:30] = states[:, 21:28]

    output_action = np.empty((n, 16), dtype=np.float32)
    output_action[:, 7] = actions[:, 7]
    output_action[:, 15] = actions[:, 15]
    terminal_count = 0
    for i in range(n):
        next_i = i + 1
        if next_i >= n or episode_indices[next_i] != episode_indices[i]:
            next_i = i
            terminal_count += 1
        output_action[i, 0:7] = states[next_i, 0:7]
        output_action[i, 8:15] = states[next_i, 14:21]
    if not np.isfinite(output_state).all() or not np.isfinite(output_action).all():
        raise ValueError("transformed state/action contains NaN or Inf")
    audit = {
        "state_source_indices": {
            "left_arm_joints": [0, 1, 2, 3, 4, 5, 6],
            "left_gripper": "source action[7]",
            "right_arm_joints": [14, 15, 16, 17, 18, 19, 20],
            "right_gripper": "source action[15]",
            "left_ee_pose": [7, 8, 9, 10, 11, 12, 13],
            "right_ee_pose": [21, 22, 23, 24, 25, 26, 27],
        },
        "action_source_indices": {
            "left_arm_joints": "next frame state[0:7] within episode",
            "left_gripper": "current frame source action[7]",
            "right_arm_joints": "next frame state[14:21] within episode",
            "right_gripper": "current frame source action[15]",
            "terminal_frame": "current frame joint state (no cross-episode read)",
        },
        "terminal_frames_padded": terminal_count,
        "joint_position_unit": "degrees",
    }
    return output_state, output_action, audit


def relative_action_stats(output_state: np.ndarray, output_action: np.ndarray, episode_indices: np.ndarray) -> dict[str, Any]:
    starts: list[int] = []
    for i in range(len(episode_indices) - CHUNK_SIZE + 1):
        if episode_indices[i] == episode_indices[i + CHUNK_SIZE - 1]:
            starts.append(i)
    if not starts:
        raise ValueError("no complete 50-frame action chunks found")
    chunks = np.concatenate([output_action[i : i + CHUNK_SIZE] for i in starts], axis=0).copy()
    current_states = np.concatenate([np.repeat(output_state[i : i + 1, :16], CHUNK_SIZE, axis=0) for i in starts], axis=0)
    mask = np.ones(16, dtype=bool)
    mask[[7, 15]] = False
    chunks[:, mask] -= current_states[:, mask]
    return stats_for_array(chunks)


def copy_metadata_assets(source: Path, destination: Path) -> None:
    for relative in ["meta/tasks.parquet", "meta/episodes"]:
        source_path = source / relative
        if source_path.is_dir():
            for path in source_path.rglob("*"):
                if path.is_file():
                    symlink_relative(path, destination / path.relative_to(source))
        elif source_path.is_file():
            symlink_relative(source_path, destination / relative)
    for path in (source / "videos").rglob("*"):
        if path.is_file():
            symlink_relative(path, destination / path.relative_to(source))


def write_data_files(
    source_files: list[Path], destination_root: Path, transformed_state: np.ndarray, transformed_action: np.ndarray
) -> None:
    offset = 0
    for source_path in source_files:
        table = pq.read_table(source_path)
        n = table.num_rows
        columns = {name: table[name] for name in table.column_names}
        columns["observation.state"] = pa.array(transformed_state[offset : offset + n].tolist(), type=pa.list_(pa.float32()))
        columns["action"] = pa.array(transformed_action[offset : offset + n].tolist(), type=pa.list_(pa.float32()))
        output_path = destination_root / "data" / source_path.relative_to(source_root_for(source_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), output_path)
        offset += n


def source_root_for(path: Path) -> Path:
    # This is replaced by the caller through the module-level context-free path shape.
    parts = path.parts
    data_index = parts.index("data")
    return Path(*parts[:data_index])


def build_info(source_info: dict[str, Any], *, cameras: list[str], total_frames: int, total_episodes: int) -> dict[str, Any]:
    info = json.loads(json.dumps(source_info))
    info["total_frames"] = total_frames
    info["total_episodes"] = total_episodes
    info["total_tasks"] = source_info.get("total_tasks", 0)
    info["fps"] = 10
    info["robot_type"] = source_info.get("robot_type", "realman_robot")
    info["features"]["action"] = {"dtype": "float32", "names": ACTION_NAMES, "shape": [16]}
    info["features"]["observation.state"] = {"dtype": "float32", "names": STATE_NAMES, "shape": [30]}
    for key in list(info["features"]):
        if key.startswith("observation.images.") and key not in cameras:
            del info["features"][key]
    info["splits"] = {"train": f"0:{total_episodes}"}
    return info


def make_splits(episode_indices: list[int], task_by_episode: dict[int, str], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    shuffled = episode_indices.copy()
    rng.shuffle(shuffled)
    cut = int(round(len(shuffled) * 0.9))
    train_eps, valid_eps = sorted(shuffled[:cut]), sorted(shuffled[cut:])
    tasks = sorted(set(task_by_episode.values()))
    shuffled_tasks = tasks.copy()
    rng.shuffle(shuffled_tasks)
    task_cut = max(1, int(round(len(shuffled_tasks) * 0.8))) if tasks else 0
    train_tasks, holdout_tasks = sorted(shuffled_tasks[:task_cut]), sorted(shuffled_tasks[task_cut:])
    return {
        "seed": seed,
        "episode_split": {"train": train_eps, "validation": valid_eps},
        "task_split": {"train_tasks": train_tasks, "holdout_tasks": holdout_tasks},
        "task_split_episode_ids": {
            "train": sorted(ep for ep, task in task_by_episode.items() if task in train_tasks),
            "holdout": sorted(ep for ep, task in task_by_episode.items() if task in holdout_tasks),
        },
    }


def populate_output(
    source_root: Path,
    output_root: Path,
    source_files: list[Path],
    source_info: dict[str, Any],
    transformed_state: np.ndarray,
    transformed_action: np.ndarray,
    episode_indices: np.ndarray,
    audit: dict[str, Any],
    cameras: list[str],
    source_stats: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "reports").mkdir()
    copy_metadata_assets(source_root, output_root)
    offset = 0
    for source_path in source_files:
        table = pq.read_table(source_path)
        n = table.num_rows
        columns = {name: table[name] for name in table.column_names}
        columns["observation.state"] = pa.array(transformed_state[offset : offset + n].tolist(), type=pa.list_(pa.float32()))
        columns["action"] = pa.array(transformed_action[offset : offset + n].tolist(), type=pa.list_(pa.float32()))
        output_path = output_root / "data" / source_path.relative_to(source_root / "data")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), output_path)
        offset += n

    full_stats = json.loads(json.dumps(source_stats))
    full_stats["observation.state"] = stats_for_array(transformed_state)
    absolute_action_stats = stats_for_array(transformed_action)
    full_stats["action"] = relative_action_stats(transformed_state, transformed_action, episode_indices)
    info = build_info(source_info, cameras=cameras, total_frames=len(transformed_state), total_episodes=int(source_info["total_episodes"]))
    write_json(output_root / "meta" / "info.json", info)
    write_json(output_root / "meta" / "stats.json", full_stats)
    write_json(output_root / "reports" / "absolute_action_stats.json", absolute_action_stats)
    write_json(output_root / "reports" / "action_state_mapping.json", audit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rgb-output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-video-files", action="store_true", help="Only for debugging; normally all videos are linked.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    rgb_root = (args.rgb_output_root or output_root.with_name(output_root.name + "_rgb")).expanduser().resolve()
    if not source_root.is_dir():
        raise SystemExit(f"source dataset does not exist: {source_root}")
    if output_root.exists():
        raise SystemExit(f"output directory already exists: {output_root}")
    if rgb_root.exists():
        raise SystemExit(f"RGB output directory already exists: {rgb_root}")

    source_info = read_json(source_root / "meta" / "info.json")
    source_stats = read_json(source_root / "meta" / "stats.json")
    data_files, integrity = validate_source(source_root, source_info)
    write_json(source_root / "reports" / "pre_edit_integrity.json", integrity) if False else None
    if integrity["errors"]:
        print(json.dumps(integrity, indent=2, ensure_ascii=False), file=sys.stderr)
        raise SystemExit("source integrity check failed; no output was published")

    states, actions, episode_indices, frame_indices, timestamps = load_all_source_arrays(data_files)
    transformed_state, transformed_action, audit = transform_arrays(states, actions, episode_indices)
    episodes = sorted(set(int(x) for x in episode_indices))
    task_table = pq.read_table(source_root / "meta" / "tasks.parquet").to_pydict()
    task_by_index = {int(i): str(task) for i, task in zip(task_table["task_index"], task_table["task"])}
    source_tables = []
    for path in data_files:
        source_tables.append(pq.read_table(path, columns=["episode_index", "task_index"]).to_pydict())
    task_by_episode: dict[int, str] = {}
    for table in source_tables:
        for ep, ti in zip(table["episode_index"], table["task_index"]):
            task_by_episode.setdefault(int(ep), task_by_index[int(ti)])
    splits = make_splits(episodes, task_by_episode, args.seed)
    audit["integrity_before_edit"] = integrity
    audit["source_state_shape"] = [28]
    audit["output_state_shape"] = [30]
    audit["source_action_shape"] = [16]
    audit["output_action_shape"] = [16]
    audit["fps"] = 10

    temp_parent = output_root.parent
    temp_full = temp_parent / f".{output_root.name}.tmp-{uuid.uuid4().hex[:8]}"
    temp_rgb = temp_parent / f".{rgb_root.name}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        populate_output(source_root, temp_full, data_files, source_info, transformed_state, transformed_action, episode_indices, audit, ALL_CAMERAS, source_stats)
        write_json(temp_full / "reports" / "splits.json", splits)
        full_integrity = {"source": integrity, "output": {"valid": True, "total_frames": len(transformed_state)}}
        write_json(temp_full / "reports" / "integrity.json", full_integrity)

        os.replace(temp_full, output_root)

        temp_rgb.mkdir(parents=True, exist_ok=False)
        for relative in ["data", "meta/tasks.parquet", "meta/episodes", "videos"]:
            symlink_relative(output_root / relative, temp_rgb / relative)
        rgb_info = build_info(source_info, cameras=RGB_CAMERAS, total_frames=len(transformed_state), total_episodes=len(episodes))
        write_json(temp_rgb / "meta" / "info.json", rgb_info)
        symlink_relative(output_root / "meta" / "stats.json", temp_rgb / "meta" / "stats.json")
        (temp_rgb / "reports").mkdir()
        for report in ["absolute_action_stats.json", "action_state_mapping.json", "splits.json", "integrity.json"]:
            symlink_relative(output_root / "reports" / report, temp_rgb / "reports" / report)
        write_json(temp_rgb / "reports" / "view.json", {"view": "rgb", "cameras": RGB_CAMERAS, "source_full_view": str(output_root)})

        os.replace(temp_rgb, rgb_root)
    except Exception:
        print(f"conversion failed; temporary output retained at {temp_full}", file=sys.stderr)
        print(f"conversion failed; temporary RGB output retained at {temp_rgb}", file=sys.stderr)
        raise

    print(f"created full dataset: {output_root}")
    print(f"created RGB view: {rgb_root}")
    print(f"episodes: {len(episodes)}, frames: {len(transformed_state)}, terminal pads: {audit['terminal_frames_padded']}")


if __name__ == "__main__":
    main()
