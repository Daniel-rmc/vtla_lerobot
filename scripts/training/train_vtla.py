#!/usr/bin/env python3
"""VTLA training entrypoint for LeRobot v3 datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vtla_lerobot.data import SamplingManifest, sample_weight_file_to_manifest, single_dataset_manifest
from vtla_lerobot.training_utils import (
    default_project_root,
    git_metadata,
    lerobot_metadata,
    runtime_metadata,
    save_run_metadata,
)


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


def _maybe_init_wandb(args: argparse.Namespace, manifest: SamplingManifest, *, rank: int):
    if rank != 0 or args.wandb_mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={"args": _jsonable(vars(args)), "sampling_manifest": _jsonable(manifest.to_dict())},
        settings=wandb.Settings(start_method="thread"),
        reinit=True,
    )


DEFAULT_FEATURE_MAP = {
    "state": "observation.state",
    "action": "action",
    "cameras": {
        "front": "observation.images.front_camera",
        "left_hand": "observation.images.left_hand_camera",
        "right_hand": "observation.images.right_hand_camera",
    },
    "tactile": {
        "left_left": "observation.images.left_tactile_left_camera",
        "left_right": "observation.images.left_tactile_right_camera",
        "right_left": "observation.images.right_tactile_left_camera",
        "right_right": "observation.images.right_tactile_right_camera",
    },
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_weights", type=Path, help="JSON mapping LeRobot dataset directory -> sample weight")
    parser.add_argument("--sampling_manifest", type=Path, help="Resolved/internal sampling manifest JSON")
    parser.add_argument("--dataset_root", type=Path, help="Single LeRobot dataset root for quick smoke runs")
    parser.add_argument("--repo_id", default=None, help="Optional repo id for --dataset_root")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/train/debug"))
    parser.add_argument("--batch_size", type=positive_int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--steps", type=positive_int, default=10)
    parser.add_argument("--epoch_size", type=positive_int, default=None)
    parser.add_argument("--chunk_size", type=positive_int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video_backend", default="pyav")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--validate_manifest", action="store_true", help="Only resolve and save manifest metadata")
    parser.add_argument("--dry_run", action="store_true", help="Build the dataloader and fetch one batch")
    parser.add_argument(
        "--synthetic_smoke",
        action="store_true",
        help="Use deterministic in-memory tensors for a fast training smoke test instead of opening dataset roots",
    )

    parser.add_argument("--device", default="cuda", help="Training device, e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=positive_int, default=1000)
    parser.add_argument("--decay_start_step", type=positive_int, default=27000)
    parser.add_argument("--decay_lr", type=float, default=2.5e-6)
    parser.add_argument("--mixed_precision", choices=("no", "bf16", "fp16"), default="no")
    parser.add_argument("--log_freq", type=positive_int, default=10)
    parser.add_argument("--save_freq", type=positive_int, default=5000, help="Checkpoint every N steps; 0 disables periodic saves")

    parser.add_argument("--image_size", type=positive_int, nargs=2, default=(128, 128))
    parser.add_argument("--dim_model", type=positive_int, default=64)
    parser.add_argument("--n_heads", type=positive_int, default=4)
    parser.add_argument("--dim_feedforward", type=positive_int, default=128)
    parser.add_argument("--n_encoder_layers", type=positive_int, default=1)
    parser.add_argument("--n_decoder_layers", type=positive_int, default=1)
    parser.add_argument("--cross_attention_layers", type=positive_int, default=1)
    parser.add_argument("--n_vae_encoder_layers", type=positive_int, default=1)
    parser.add_argument("--latent_dim", type=positive_int, default=16)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vision_backbone_weights", default=None)
    parser.add_argument("--tactile_backbone_weights", default=None)

    parser.add_argument("--wandb_project", default="vtla_lerobot")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument(
        "--wandb_mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
        help="W&B logging mode; disabled by default so tests stay offline",
    )
    return parser.parse_args()


def load_manifest(args: argparse.Namespace) -> SamplingManifest:
    if args.sample_weights:
        return sample_weight_file_to_manifest(
            args.sample_weights,
            feature_map=DEFAULT_FEATURE_MAP,
            chunk_size=args.chunk_size,
            epoch_size=args.epoch_size,
            seed=args.seed,
            video_backend=args.video_backend,
        )
    if args.sampling_manifest:
        return SamplingManifest.from_json(args.sampling_manifest)
    if not args.dataset_root:
        raise SystemExit("Provide one of --sample_weights, --sampling_manifest, or --dataset_root")
    return single_dataset_manifest(
        root=args.dataset_root,
        repo_id=args.repo_id,
        feature_map=DEFAULT_FEATURE_MAP,
        chunk_size=args.chunk_size,
        epoch_size=args.epoch_size,
        seed=args.seed,
        video_backend=args.video_backend,
    )


def distributed_context(args: argparse.Namespace) -> tuple[int, int]:
    rank = args.rank
    world_size = args.world_size
    if rank is None:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def build_dataloader(
    manifest: SamplingManifest,
    *,
    batch_size: int,
    num_workers: int,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[Any, Any]:
    from torch.utils.data import DataLoader

    from vtla_lerobot.data import MixedLeRobotVTLADataset, WeightedMixedSampler

    dataset = MixedLeRobotVTLADataset.from_manifest(manifest)
    sampler = WeightedMixedSampler(dataset, seed=manifest.seed, rank=rank, world_size=world_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return loader, sampler


def _synthetic_batch(index: int, manifest: SamplingManifest, *, image_size: tuple[int, int]) -> dict[str, Any]:
    import torch

    spec = manifest.datasets[index % len(manifest.datasets)]
    feature_map = spec.feature_map
    state = torch.linspace(0.0, 1.0, 28) + float(index) / 100.0
    action_base = torch.linspace(-1.0, 1.0, 16) + float(index) / 50.0
    action = action_base.unsqueeze(0).repeat(int(spec.chunk_size or 1), 1)
    height, width = image_size
    sample: dict[str, Any] = {
        "observation.state": state,
        "action": action,
        "action_is_pad": torch.zeros(action.shape[0], dtype=torch.bool),
        "dataset_name": spec.name,
        "source_index": index,
    }
    for offset, source_name in enumerate(feature_map.cameras.values()):
        sample[source_name] = torch.full((3, height, width), fill_value=(index + offset + 1) / 255.0)
    for offset, source_name in enumerate(feature_map.tactile.values()):
        sample[source_name] = torch.full((3, height, width), fill_value=(index + offset + 11) / 255.0)
    return sample


class SyntheticVTLADataset:
    def __init__(self, manifest: SamplingManifest, *, length: int, image_size: tuple[int, int]) -> None:
        self.manifest = manifest
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _synthetic_batch(index, self.manifest, image_size=self.image_size)


def build_synthetic_dataloader(
    args: argparse.Namespace,
    manifest: SamplingManifest,
    *,
    rank: int,
    world_size: int,
) -> tuple[Any, Any]:
    from torch.utils.data import DataLoader, SequentialSampler

    total = max(args.batch_size * args.steps * max(world_size, 1), args.batch_size)
    dataset = SyntheticVTLADataset(manifest, length=total, image_size=tuple(args.image_size))
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=False)
    return loader, sampler


def dry_run(args: argparse.Namespace, manifest: SamplingManifest) -> dict[str, Any]:
    rank, world_size = distributed_context(args)
    if args.synthetic_smoke:
        loader, sampler = build_synthetic_dataloader(args, manifest, rank=rank, world_size=world_size)
    else:
        loader, sampler = build_dataloader(
            manifest,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            rank=rank,
            world_size=world_size,
        )
    batch = next(iter(loader))
    return {
        "rank": rank,
        "world_size": world_size,
        "num_datasets": len(manifest.datasets),
        "logical_epoch_size": len(sampler),
        "batch_keys": sorted(batch.keys()),
    }


def _move_to_device(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _tensor_feature_shape(tensor: Any, *, strip_batch: bool = True) -> tuple[int, ...]:
    import torch

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    shape = tuple(int(dim) for dim in tensor.shape)
    if strip_batch:
        shape = shape[1:]
    return shape


def build_policy_config(args: argparse.Namespace, manifest: SamplingManifest, batch: Mapping[str, Any]) -> Any:
    from lerobot.configs import FeatureType, PolicyFeature
    from vtla_lerobot.models import VTLAConfig

    feature_map = manifest.datasets[0].feature_map
    input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=_tensor_feature_shape(batch["observation.state"]),
        )
    }
    tactile_feature_names = list(feature_map.tactile.values())
    for source_name in feature_map.cameras.values():
        input_features[source_name] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=_tensor_feature_shape(batch[source_name]),
        )
    for source_name in tactile_feature_names:
        input_features[source_name] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=_tensor_feature_shape(batch[source_name]),
        )

    action_shape = _tensor_feature_shape(batch["action"])
    if len(action_shape) >= 2:
        action_shape = action_shape[-1:]

    return VTLAConfig(
        input_features=input_features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=action_shape)},
        chunk_size=int(batch["action"].shape[1]),
        n_action_steps=int(batch["action"].shape[1]),
        image_size=tuple(args.image_size),
        tactile_feature_names=tactile_feature_names,
        dtype="float32",
        vision_backbone_weights=args.vision_backbone_weights,
        tactile_backbone_weights=args.tactile_backbone_weights,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        cross_attention_layers=args.cross_attention_layers,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
        n_vae_encoder_layers=args.n_vae_encoder_layers,
        kl_weight=args.kl_weight,
        temporal_ensemble_coeff=None,
    )


def build_lr_scheduler(optimizer: Any, args: argparse.Namespace) -> Any:
    import torch

    if args.warmup_steps >= args.steps:
        raise ValueError("warmup_steps must be smaller than steps")
    if not 0 < args.decay_start_step <= args.steps:
        raise ValueError("decay_start_step must be in (0, steps]")
    if args.decay_start_step < args.warmup_steps:
        raise ValueError("decay_start_step must not precede warmup_steps")

    def lr_lambda(current_step: int) -> float:
        if current_step < args.warmup_steps:
            return float(current_step + 1) / float(args.warmup_steps)
        if current_step < args.decay_start_step:
            return 1.0
        decay_steps = max(args.steps - args.decay_start_step, 1)
        progress = min(current_step - args.decay_start_step, decay_steps) / decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor_ratio = args.decay_lr / args.lr
        return (1.0 - floor_ratio) * cosine + floor_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args: argparse.Namespace, manifest: SamplingManifest) -> dict[str, Any]:
    import torch
    from accelerate import Accelerator

    from vtla_lerobot.models import VTLAPolicy

    torch.manual_seed(args.seed)
    mixed_precision = args.mixed_precision
    if mixed_precision == "bf16" and not torch.cuda.is_available():
        mixed_precision = "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if args.synthetic_smoke:
        loader, sampler = build_synthetic_dataloader(args, manifest, rank=rank, world_size=world_size)
    else:
        loader, sampler = build_dataloader(
            manifest,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            rank=rank,
            world_size=world_size,
        )

    requested_device = accelerator.device
    iterator = iter(loader)
    config_batch = next(iterator)
    config = build_policy_config(args, manifest, config_batch)
    policy = VTLAPolicy(config)
    policy.train()
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, args)
    policy, optimizer, loader, scheduler = accelerator.prepare(policy, optimizer, loader, scheduler)

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.jsonl"
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    else:
        metrics_path = output_dir / "metrics.jsonl"
        checkpoints_dir = output_dir / "checkpoints"
    accelerator.wait_for_everyone()

    wandb_run = _maybe_init_wandb(args, manifest, rank=0 if accelerator.is_main_process else 1)
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "dataset_count": len(manifest.datasets),
                "logical_epoch_size": len(sampler),
                "step_budget": int(args.steps),
                "per_gpu_batch_size": int(args.batch_size),
                "global_batch_size": int(args.batch_size * world_size),
                "device": str(requested_device),
                "mixed_precision": mixed_precision,
            },
            allow_val_change=True,
        )

    latest_metrics: dict[str, Any] = {}
    final_checkpoint: Path | None = None
    try:
        batch = config_batch
        for step in range(1, args.steps + 1):
            if step > 1:
                try:
                    batch = next(iterator)
                except StopIteration:
                    if hasattr(sampler, "set_epoch"):
                        sampler.set_epoch(step)
                    iterator = iter(loader)
                    batch = next(iterator)

            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                loss, metrics = policy(batch)
            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            latest_metrics = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm),
                "lr": float(optimizer.param_groups[0]["lr"]),
                **{key: float(value) for key, value in metrics.items()},
            }
            if step % args.log_freq == 0 or step == args.steps:
                if accelerator.is_main_process:
                    with metrics_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(latest_metrics, ensure_ascii=False, sort_keys=True) + "\\n")
                    print(f"step={step} loss={latest_metrics['loss']:.6f} lr={latest_metrics['lr']:.8g}")
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": latest_metrics["loss"],
                                "train/grad_norm": latest_metrics["grad_norm"],
                                "train/lr": latest_metrics["lr"],
                                **{f"train/{key}": float(value) for key, value in metrics.items()},
                            },
                            step=step,
                        )

            if args.save_freq and step % args.save_freq == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    final_checkpoint = checkpoints_dir / f"step_{step:06d}.pt"
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": accelerator.get_state_dict(policy),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "args": vars(args),
                            "sampling_manifest": manifest.to_dict(),
                            "metrics": latest_metrics,
                        },
                        final_checkpoint,
                    )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            final_checkpoint = checkpoints_dir / "last.pt"
            torch.save(
                {
                    "step": args.steps,
                    "model_state_dict": accelerator.get_state_dict(policy),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": vars(args),
                    "sampling_manifest": manifest.to_dict(),
                    "metrics": latest_metrics,
                },
                final_checkpoint,
            )

        return {
            "rank": rank,
            "world_size": world_size,
            "device": str(requested_device),
            "steps": args.steps,
            "logical_epoch_size": len(sampler),
            "metrics_path": str(metrics_path),
            "checkpoint_path": str(final_checkpoint) if final_checkpoint else str(checkpoints_dir / "last.pt"),
            "latest_metrics": latest_metrics,
            "synthetic_smoke": bool(args.synthetic_smoke),
            "wandb_mode": args.wandb_mode,
            "mixed_precision": mixed_precision,
        }
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_sampling_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    project_root = default_project_root()
    metadata: dict[str, Any] = {
        "args": vars(args),
        "project": git_metadata(project_root),
        "lerobot": lerobot_metadata(project_root),
        "runtime": runtime_metadata(),
        "sampling_manifest": manifest.to_dict(),
        "wandb": {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "run_name": args.wandb_run_name,
            "mode": args.wandb_mode,
        },
    }

    if args.validate_manifest:
        metadata_path = save_run_metadata(output_dir, metadata)
        print(f"Manifest validation OK. Metadata: {metadata_path}")
        return

    if args.dry_run:
        metadata["dry_run"] = dry_run(args, manifest)
        metadata_path = save_run_metadata(output_dir, metadata)
        print(f"Dry run OK. Metadata: {metadata_path}")
        print(metadata["dry_run"])
        return

    metadata["train"] = train(args, manifest)
    metadata_path = save_run_metadata(output_dir, metadata)
    print(f"Training OK. Metadata: {metadata_path}")
    print(metadata["train"])


if __name__ == "__main__":
    main()
