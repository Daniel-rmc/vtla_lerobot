from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.train_vtla import load_manifest, train


def make_args(tmp_path: Path, sample_weights: Path) -> argparse.Namespace:
    return argparse.Namespace(
        sample_weights=sample_weights,
        sampling_manifest=None,
        dataset_root=None,
        repo_id=None,
        output_dir=tmp_path / "run",
        batch_size=1,
        num_workers=0,
        steps=1,
        epoch_size=None,
        chunk_size=2,
        seed=0,
        video_backend="pyav",
        rank=0,
        world_size=1,
        validate_manifest=False,
        dry_run=False,
        synthetic_smoke=True,
        device="cpu",
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-4,
        grad_clip=10.0,
        warmup_steps=1,
        decay_start_step=1,
        decay_lr=1e-6,
        mixed_precision="no",
        log_freq=1,
        save_freq=0,
        image_size=(32, 32),
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        cross_attention_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        kl_weight=1.0,
        dropout=0.1,
        vision_backbone_weights=None,
        wandb_project="vtla_lerobot",
        wandb_entity=None,
        wandb_run_name=None,
        wandb_mode="disabled",
    )


def test_train_synthetic_smoke_writes_metrics_and_checkpoint(tmp_path: Path) -> None:
    sample_weights = tmp_path / "sample_weights.json"
    sample_weights.write_text(json.dumps({"/does/not/need/to/exist": 1.0}), encoding="utf-8")
    args = make_args(tmp_path, sample_weights)
    manifest = load_manifest(args)

    summary = train(args, manifest)

    assert summary["steps"] == 1
    assert summary["synthetic_smoke"] is True
    assert torch.isfinite(torch.tensor(summary["latest_metrics"]["loss"]))
    metrics_path = Path(summary["metrics_path"])
    checkpoint_path = Path(summary["checkpoint_path"])
    assert metrics_path.exists()
    assert checkpoint_path.exists()
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["step"] == 1
    assert records[0]["loss"] == summary["latest_metrics"]["loss"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 1
    assert checkpoint["sampling_manifest"]["datasets"][0]["weight"] == 1.0
    assert checkpoint["metrics"]["loss"] == summary["latest_metrics"]["loss"]
