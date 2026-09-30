#!/usr/bin/env python3
"""Zip a local dataset and upload the archive to a ModelScope dataset repository."""

# Usage examples:
#   cd /home/rmc/workspace/vtla_lerobot
#   export MODELSCOPE_API_TOKEN='your_modelscope_access_token'
#   python tool/zip_upload_to_modelscope.py \
#       /home/rmc/workspace/datasets/manipulationNet/peg_in_hole/15holes_v3_pi05_joint
#
# Preview the archive and upload target without uploading:
#   python tool/zip_upload_to_modelscope.py \
#       --dry-run \
#       /path/to/your/dataset
#
# Customize the remote path, local archive path, or overwrite an existing archive:
#   python tool/zip_upload_to_modelscope.py \
#       --remote-name archives/my_dataset.zip \
#       --output-zip /tmp/my_dataset.zip \
#       --overwrite \
#       /path/to/your/dataset

from __future__ import annotations

import argparse
import getpass
import os
import sys
import zipfile
from pathlib import Path

DEFAULT_REPO_ID = "Deeptouch_ICRA/manipulationNet-realman-xense"
DEFAULT_COMMIT_MESSAGE = "Upload dataset archive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a zip archive containing DATASET_DIR and upload it as one file "
            "to a ModelScope dataset repository."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Local dataset directory to archive and upload.",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("MODELSCOPE_REPO_ID", DEFAULT_REPO_ID),
        help=f"ModelScope repository ID (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--remote-name",
        help="Remote archive path; defaults to <dataset-directory-name>.zip.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        help=(
            "Local archive path; defaults to <dataset-parent>/<dataset-name>.zip. "
            "The archive must be outside DATASET_DIR."
        ),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("MODELSCOPE_API_TOKEN", ""),
        help="ModelScope access token; prefer MODELSCOPE_API_TOKEN or --prompt-token.",
    )
    parser.add_argument(
        "--prompt-token",
        action="store_true",
        help="Read the ModelScope access token without echoing it.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("MODELSCOPE_ENDPOINT"),
        help="Optional ModelScope endpoint override.",
    )
    parser.add_argument(
        "--commit-message",
        default=os.environ.get("MODELSCOPE_COMMIT_MESSAGE", DEFAULT_COMMIT_MESSAGE),
        help=f"Upload commit message (default: {DEFAULT_COMMIT_MESSAGE}).",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        choices=range(10),
        metavar="0-9",
        default=6,
        help="ZIP_DEFLATED compression level (default: 6).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing local archive at --output-zip.",
    )
    parser.add_argument(
        "--remove-zip",
        action="store_true",
        help="Delete the local archive after a successful upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create and inspect the archive, but do not upload it.",
    )
    return parser.parse_args()


def resolve_dataset(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Dataset directory does not exist: {path}")
    if not path.is_dir():
        raise SystemExit(f"Dataset path is not a directory: {path}")
    if not os.access(path, os.R_OK | os.X_OK):
        raise SystemExit(f"Dataset directory is not readable: {path}")
    try:
        next(path.iterdir())
    except StopIteration:
        raise SystemExit(f"Dataset directory is empty: {path}") from None
    return path


def resolve_output_zip(dataset_dir: Path, output_zip: Path | None) -> Path:
    if output_zip is None:
        output_zip = dataset_dir.parent / f"{dataset_dir.name}.zip"
    else:
        output_zip = output_zip.expanduser()
        if not output_zip.is_absolute():
            output_zip = Path.cwd() / output_zip
        output_zip = output_zip.resolve()

    try:
        output_zip.relative_to(dataset_dir)
    except ValueError:
        return output_zip
    raise SystemExit(f"Archive must be outside the dataset directory: {output_zip}")


def iter_dataset_files(dataset_dir: Path) -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_symlink():
            target = path.resolve(strict=False)
            if path.is_dir():
                raise SystemExit(f"Symlinked directories are not supported: {path}")
            if not target.is_file():
                raise SystemExit(f"Symlink does not resolve to a regular file: {path} -> {target}")
            # Store the file under its link name so the extracted dataset keeps
            # the same layout, while avoiding symlink entries in the archive.
            files.append((path, target))
        elif path.is_file():
            files.append((path, path))
    if not files:
        raise SystemExit(f"Dataset directory contains no regular files: {dataset_dir}")
    return files


def create_archive(dataset_dir: Path, output_zip: Path, compression_level: int) -> tuple[int, int]:
    files = iter_dataset_files(dataset_dir)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_zip,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compression_level,
    ) as archive:
        for archive_path, source_path in files:
            relative_path = archive_path.relative_to(dataset_dir).as_posix()
            archive.write(source_path, f"{dataset_dir.name}/{relative_path}")
    return len(files), output_zip.stat().st_size


def upload_archive(
    archive_path: Path,
    *,
    repo_id: str,
    remote_name: str,
    token: str | None,
    endpoint: str | None,
    commit_message: str,
) -> object:
    try:
        from modelscope.hub.api import HubApi
    except ImportError as error:
        raise SystemExit(
            "ModelScope is not installed. Install it with: python -m pip install modelscope"
        ) from error

    api = HubApi(endpoint=endpoint, token=token) if endpoint else HubApi(token=token)
    return api.upload_file(
        path_or_fileobj=archive_path,
        path_in_repo=remote_name,
        repo_id=repo_id,
        repo_type="dataset",
        token=token or None,
        commit_message=commit_message,
    )


def main() -> int:
    args = parse_args()
    dataset_dir = resolve_dataset(args.dataset_dir)
    output_zip = resolve_output_zip(dataset_dir, args.output_zip)
    remote_name = (args.remote_name or f"{dataset_dir.name}.zip").strip().lstrip("/")
    if not remote_name:
        raise SystemExit("--remote-name must not be empty")
    if args.prompt_token:
        args.token = getpass.getpass("ModelScope access token: ")
    if args.overwrite is False and output_zip.exists():
        raise SystemExit(
            f"Archive already exists: {output_zip}. Use --overwrite or choose --output-zip."
        )

    print(f"Dataset: {dataset_dir}")
    print(f"Creating archive: {output_zip}")
    file_count, archive_size = create_archive(
        dataset_dir, output_zip, args.compression_level
    )
    print(f"Archive created: {file_count} files, {archive_size:,} bytes")
    print(f"ModelScope target: dataset:{args.repo_id}/{remote_name}")

    if args.dry_run:
        print("Dry run: upload not performed; archive retained.")
        return 0

    result = upload_archive(
        output_zip,
        repo_id=args.repo_id,
        remote_name=remote_name,
        token=args.token or None,
        endpoint=args.endpoint,
        commit_message=args.commit_message,
    )
    print(f"ModelScope upload completed: {result}")
    if args.remove_zip:
        output_zip.unlink()
        print(f"Removed local archive: {output_zip}")
    else:
        print(f"Local archive retained: {output_zip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
