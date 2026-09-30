# Sampling weights

Training takes a simple JSON sample-weight file as the user-facing multi-dataset interface.

Preferred format:

```json
{
  "/data/task_a_lerobot_v3": 1.0,
  "/data/task_b_lerobot_v3": 2.0,
  "/data/task_c_lerobot_v3": 0.5
}
```

Each key is a LeRobot dataset directory.  Each value is that dataset's sampling weight.  During training, the loader normalizes the weights and samples datasets accordingly.  For example, weights `1.0` and `2.0` produce an expected 1:2 sampling ratio.

An array form is also accepted when you want stable explicit names:

```json
[
  {"name": "task_a", "root": "/data/task_a_lerobot_v3", "weight": 1.0},
  {"name": "task_b", "root": "/data/task_b_lerobot_v3", "weight": 2.0}
]
```

The training script expands this sample-weight file into a resolved internal manifest and writes it to the output directory as `resolved_sampling_manifest.json`, together with `run_metadata.json`.  That resolved manifest records feature names, chunk size, seed, dataset names, and normalized sampling configuration for reproducibility.

## Usage

```bash
python -m scripts.training.train_vtla \
  --sample_weights configs/sampling/sample_weights_example.json \
  --output_dir outputs/train/debug \
  --batch_size 8 \
  --steps 1000
```

For a quick optimizer/loss/checkpoint smoke test that does not require the placeholder paths to exist:

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

Full training writes `resolved_sampling_manifest.json`, `metrics.jsonl`, `checkpoints/last.pt`, and `run_metadata.json`.

For manifest-only validation without opening dataset directories:

```bash
PYTHONPATH=src python -m scripts.training.train_vtla \
  --validate_manifest \
  --sample_weights configs/sampling/sample_weights_example.json
```

## DDP behavior

All ranks construct the same weighted logical epoch from the same seed and epoch number.  Each rank consumes a strided partition of logical indices, so ranks do not duplicate logical sample ids.  Call `sampler.set_epoch(epoch)` once per epoch to change the deterministic sequence.
