"""Training utilities shared by vtla_lerobot entrypoints."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _run_git(project_dir: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_dir), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def git_metadata(project_dir: str | Path) -> dict[str, Any]:
    project_dir = Path(project_dir)
    return {
        "root": str(project_dir.resolve()),
        "sha": _run_git(project_dir, "rev-parse", "HEAD"),
        "branch": _run_git(project_dir, "branch", "--show-current"),
        "dirty": bool(_run_git(project_dir, "status", "--short")),
    }


def lerobot_metadata(project_dir: str | Path) -> dict[str, Any]:
    project_dir = Path(project_dir)
    checkout = project_dir / "third_party" / "lerobot"
    metadata: dict[str, Any] = {"path": str(checkout)}
    if checkout.exists():
        metadata.update(git_metadata(checkout))
    try:
        import lerobot

        metadata["import_path"] = getattr(lerobot, "__file__", None)
        metadata["version"] = getattr(lerobot, "__version__", None)
    except Exception as exc:  # pragma: no cover - depends on environment
        metadata["import_error"] = str(exc)
    return metadata


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_run_metadata(output_dir: str | Path, metadata: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_metadata.json"
    path.write_text(
        json.dumps(_jsonable(metadata), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def runtime_metadata() -> dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pythonpath": os.environ.get("PYTHONPATH"),
    }
