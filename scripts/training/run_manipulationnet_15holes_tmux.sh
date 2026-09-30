#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
container_name="${VTLA_CONTAINER_NAME:-vtla_lerobot_workspace}"
dataset_root="${VTLA_DATASET_ROOT:-/datasets/manipulationNet/peg_in_hole/15holes_merged}"
weights_host="${VTLA_SAMPLE_WEIGHTS_HOST:-${repo_root}/configs/sampling/manipulationnet_15holes_equal.json}"
output_dir="${VTLA_OUTPUT_DIR:-${repo_root}/outputs/train/$(date +%Y-%m-%d_%H-%M-%S)_manipulationnet_15holes}"
logs_dir="${output_dir}/logs"
run_script="${output_dir}/run_in_tmux.sh"
container_run_script="/workspace/vtla_lerobot/outputs/train/$(basename "${output_dir}")/run_in_tmux.sh"
container_output_dir="/workspace/vtla_lerobot/outputs/train/$(basename "${output_dir}")"
container_weights="${container_output_dir}/manipulationnet_15holes_equal.container.json"
session_name="${VTLA_TMUX_SESSION:-vtla_15holes_$(date +%H%M%S)}"
steps="${VTLA_STEPS:-30000}"
batch_size="${VTLA_BATCH_SIZE:-32}"
num_workers="${VTLA_NUM_WORKERS:-8}"
device="${VTLA_DEVICE:-cuda}"
wandb_project="${VTLA_WANDB_PROJECT:-vtla_lerobot}"
wandb_entity="${VTLA_WANDB_ENTITY:-}"
wandb_run_name="${VTLA_WANDB_RUN_NAME:-manipulationnet_15holes_30000}"
wandb_mode="${VTLA_WANDB_MODE:-online}"

mkdir -p "${logs_dir}"

if ! docker inspect -f '{{.State.Running}}' "${container_name}" 2>/dev/null | grep -q '^true$'; then
  echo "container is not running: ${container_name}" >&2
  exit 1
fi
if [[ ! -f "${weights_host}" ]]; then
  echo "missing sample weight file: ${weights_host}" >&2
  exit 1
fi

python3 - <<'PY' "${weights_host}" "${output_dir}/manipulationnet_15holes_equal.container.json"
import json
import sys
from pathlib import Path
weights_host = Path(sys.argv[1])
out_path = Path(sys.argv[2])
raw = json.loads(weights_host.read_text())
container_root = Path('/datasets/manipulationNet/peg_in_hole/15holes_merged')
converted = {str(container_root / Path(host_root).name): weight for host_root, weight in raw.items()}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(converted, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

# The output directory is bind-mounted at the same path inside the container,
# so the generated JSON is already available at container_weights.

echo "[preflight] container=${container_name} dataset=${dataset_root}" | tee -a "${logs_dir}/preflight.log"
docker exec "${container_name}" bash -lc "python - <<'PY'
import json
from pathlib import Path
weights = Path('${container_weights}')
raw = json.loads(weights.read_text())
missing = [task_root for task_root in raw if not Path(task_root).exists()]
if missing:
    raise SystemExit(f'missing roots: {missing}')
print(json.dumps({'tasks': len(raw), 'ok': True}, indent=2))
PY" 2>&1 | tee -a "${logs_dir}/preflight.log"

gpu_selector=/workspace/vtla_lerobot/scripts/training/with_free_gpus.sh
train_cmd="cd /workspace/vtla_lerobot && PYTHONPATH=src:/workspace/vtla_lerobot/third_party/lerobot/src ${gpu_selector} --accelerate -- -m scripts.training.train_vtla --sample_weights ${container_weights} --output_dir ${container_output_dir} --batch_size ${batch_size} --num_workers ${num_workers} --steps ${steps} --save_freq 5000 --lr 2.5e-5 --betas 0.9 0.95 --eps 1e-8 --weight_decay 0.01 --grad_clip 1.0 --warmup_steps 1000 --decay_start_step 27000 --decay_lr 2.5e-6 --mixed_precision bf16 --device ${device} --wandb_project ${wandb_project} ${wandb_entity:+--wandb_entity ${wandb_entity}} --wandb_run_name ${wandb_run_name} --wandb_mode ${wandb_mode} --dim_model 32 --n_heads 4 --dim_feedforward 64 --n_encoder_layers 1 --n_decoder_layers 1 --cross_attention_layers 1 --n_vae_encoder_layers 1 --latent_dim 8 --chunk_size 2 --image_size 128 128"

cat > "${run_script}" <<RUNNER
#!/usr/bin/env bash
set -euo pipefail
export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_API_KEY="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"
export WANDB_LOG_MODEL=false
export WANDB_DISABLE_CODE=true
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
${train_cmd}
RUNNER
chmod +x "${run_script}"

tmux new-session -d -s "${session_name}" "docker exec ${container_name} bash ${container_run_script} 2>&1 | tee -a ${logs_dir}/train.log"

echo "tmux session: ${session_name}"
echo "container: ${container_name}"
echo "output: ${output_dir}"
echo "logs: ${logs_dir}"
echo "attach: tmux attach -t ${session_name}"
echo "follow: tail -f ${logs_dir}/train.log"
