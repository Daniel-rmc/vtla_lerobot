# vtla_lerobot

`vtla_lerobot` is the clean training repository for VTLA on LeRobot v3-format real-robot datasets.

The older `/home/rmc/workspace/VTLA_design` repository is intentionally preserved as the historical experiment codebase.  This repo ports the VTLA policy into a LeRobot-centered training stack and makes multi-dataset training explicit through a simple sample-weight JSON file.

## Scope

Phase 1 focuses on training:

- LeRobot v3 dataset loading
- VTLA model / policy plugin
- sample-weight controlled multi-dataset sampling
- single- and multi-GPU training entrypoints
- reproducible run metadata
- a dedicated training container

Deployment and robot-control integration are deferred until training is stable.

## Install

```bash
cd /home/rmc/workspace/vtla_lerobot
pip install -e .[dev]
# If using the pinned checkout instead of PyPI:
pip install -e third_party/lerobot
```

## Multi-dataset sample-weight file

Training should normally receive a JSON file whose keys are dataset directories and whose values are sampling weights:

```json
{
  "/data/task_a_lerobot_v3": 1.0,
  "/data/task_b_lerobot_v3": 2.0
}
```

Run with real LeRobot dataset paths:

```bash
python -m scripts.training.train_vtla \
  --sample_weights configs/sampling/sample_weights_example.json \
  --output_dir outputs/train/debug \
  --batch_size 2 \
  --steps 10
```

For a fast end-to-end optimizer/loss/checkpoint smoke test that does not open dataset paths, add `--synthetic_smoke`:

```bash
python -m scripts.training.train_vtla \
  --synthetic_smoke \
  --sample_weights configs/sampling/sample_weights_example.json \
  --output_dir outputs/train/synthetic_smoke \
  --batch_size 1 \
  --steps 1 \
  --chunk_size 2 \
  --device cpu \
  --dim_model 32 \
  --n_heads 4 \
  --dim_feedforward 64 \
  --image_size 32 32 \
  --latent_dim 8
```

The training program expands the weight file into an internal resolved manifest and saves it under the output directory for reproducibility.  Full training writes `metrics.jsonl`, `checkpoints/last.pt`, and `run_metadata.json`.

See [configs/sampling/sample_weights_example.json](configs/sampling/sample_weights_example.json) and [configs/sampling/README.md](configs/sampling/README.md).

## Upload a checkpoint to ModelScope

The host-side upload tool sends a completed checkpoint to the ModelScope model repository. The training container is not involved in the upload:

```bash
# Use the existing host-side `modelscope login` session.
tool/upload_modelscope_checkpoint.sh /path/to/checkpoints/last

# Or provide a token only for this invocation (keep it out of tracked files).
MODELSCOPE_TOKEN='paste_token_here' \
  tool/upload_modelscope_checkpoint.sh /path/to/checkpoints/last
```

The default repository is `Deeptouch_ICRA/manipulationNet-realman-xense`, and the checkpoint is stored under `checkpoints/pi05_rgb/last`. Override these with `--repo-id` and `--path-in-repo`. Use `--dry-run` to validate a checkpoint directory without uploading.


Use a dedicated VTLA training image.  The current Dockerfile derives from the existing local `vtla-train:lerobot-0.6.1` image so it can reuse the already-installed Python 3.12, CUDA-enabled torch, and pinned LeRobot environment.

Build example:

```bash
docker build -f docker/Dockerfile.train -t vtla_lerobot:train .
```

## Verification

Manifest-only validation does not open the placeholder dataset paths:

```bash
PYTHONPATH=src python -m scripts.training.train_vtla \
  --validate_manifest \
  --sample_weights configs/sampling/sample_weights_example.json
```

Unit tests require PyTorch, and policy tests additionally require a Python version supported by the pinned LeRobot checkout:

```bash
pytest -q
```
