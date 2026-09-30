#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sample_weights="${VTLA_SAMPLE_WEIGHTS:-${repo_root}/configs/sampling/sample_weights_example.json}"
visible_gpus="${VTLA_GPUS:-0,1,2,3}"
num_processes="${VTLA_NUM_PROCESSES:-4}"
output_dir="${VTLA_OUTPUT_DIR:-${repo_root}/outputs/train/$(date +%Y-%m-%d_%H-%M-%S)_vtla_lerobot}"

export CUDA_VISIBLE_DEVICES="${visible_gpus}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_root}/src:${repo_root}/third_party/lerobot/src:${PYTHONPATH:-}"

common_args=(
  "--sample_weights=${sample_weights}"
  "--output_dir=${output_dir}"
)

if [[ "${VTLA_DRY_RUN:-0}" == "1" ]]; then
  common_args+=("--dry_run")
fi

if [[ "${num_processes}" -eq 1 ]]; then
  exec python -m scripts.training.train_vtla "${common_args[@]}" "$@"
fi

exec accelerate launch \
  --multi_gpu \
  --num_processes="${num_processes}" \
  -m scripts.training.train_vtla \
  "${common_args[@]}" \
  "$@"
