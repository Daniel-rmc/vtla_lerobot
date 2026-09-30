#!/usr/bin/env bash
# Select sufficiently idle GPUs inside a container, then run a training command.
# Usage: scripts/training/with_free_gpus.sh -- <command> [args...]
# Or:    scripts/training/with_free_gpus.sh --accelerate -- <module args...>
set -euo pipefail

min_free_mib="${VTLA_GPU_MIN_FREE_MIB:-40000}"
max_count="${VTLA_GPU_MAX_COUNT:-0}"
accelerate_mode=0

if [[ "${1:-}" == "--accelerate" ]]; then
  accelerate_mode=1
  shift
fi

if [[ "${1:-}" != "--" || "$#" -lt 2 ]]; then
  if [[ "${accelerate_mode}" -eq 1 ]]; then
    echo "usage: $0 --accelerate -- <module-or-script args...>" >&2
  else
    echo "usage: $0 -- <command> [args...]" >&2
  fi
  exit 2
fi
shift

mapfile -t free_gpus < <(
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | awk -F',' -v min_free_mib="${min_free_mib}" '
        {
          gpu_id = $1; free_mib = $2; utilization = $3
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", gpu_id)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", free_mib)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", utilization)
          if (free_mib >= min_free_mib && utilization == 0) print gpu_id
        }
      '
)

if [[ "${max_count}" -gt 0 && "${#free_gpus[@]}" -gt "${max_count}" ]]; then
  free_gpus=("${free_gpus[@]:0:max_count}")
fi

if [[ "${#free_gpus[@]}" -eq 0 ]]; then
  echo "No GPUs satisfy VTLA_GPU_MIN_FREE_MIB=${min_free_mib}." >&2
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${free_gpus[*]}")"
echo "Selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (min_free_mib=${min_free_mib}, count=${#free_gpus[@]})."

if [[ "${accelerate_mode}" -eq 1 ]]; then
  exec accelerate launch --multi_gpu --num_processes="${#free_gpus[@]}" "$@"
fi

exec "$@"
