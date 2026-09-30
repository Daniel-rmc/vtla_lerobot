#!/usr/bin/env bash
set -euo pipefail

# PI05 RGB baseline launcher for the converted ManipulationNet dataset.
# Defaults to the validated RGB view, batch size 16 per GPU, two-way gradient
# accumulation, and 30k optimizer steps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/home/rmc/workspace/datasets/manipulationNet/peg_in_hole/15holes_v3_pi05_joint_rgb}"
PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/pi05_base}"
NUM_PROCESSES="${NUM_PROCESSES:-${WORLD_SIZE:-2}}"
GPU_IDS="${GPU_IDS:-0,3}"
USE_FSDP="${USE_FSDP:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STEPS="${STEPS:-30000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/pi05_rgb}"
JOB_NAME="${JOB_NAME:-pi05_manipulationnet_rgb}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
RUN_IN_CONTAINER="${RUN_IN_CONTAINER:-auto}"
CONTAINER_NAME="${CONTAINER_NAME:-vtla_lerobot}"
REDIRECT_LOG="${REDIRECT_LOG:-1}"

# Hugging Face downloads use the persistent workspace cache. LeRobot caches its
# own Hub assets under ${HF_LEROBOT_HOME}, which is kept below ${HF_HOME} too.
# The host is already authenticated; training downloads through HF_ENDPOINT and
# intentionally does not inherit the local HTTP(S) proxy.
HF_TOKEN="${HF_TOKEN:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_LOGIN="${HF_LOGIN:-0}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/../hf_cache}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"

# Weights & Biases settings. Replace the placeholder with the real key or export
# WANDB_API_KEY before launching; metrics are logged, but model artifacts are not.
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-manipulationnet}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
WANDB_API_KEY="${WANDB_API_KEY:-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}"

# ModelScope upload runs on the host after container training succeeds. Paste the
# token locally or export MODELSCOPE_TOKEN; never pass it into the container.
MODELSCOPE_REPO_ID="${MODELSCOPE_REPO_ID:-Deeptouch_ICRA/manipulationNet-realman-xense}"
MODELSCOPE_PATH_IN_REPO="${MODELSCOPE_PATH_IN_REPO:-checkpoints/pi05_rgb/last}"
MODELSCOPE_TOKEN="${MODELSCOPE_TOKEN:-}"
UPLOAD_TO_MODELSCOPE="${UPLOAD_TO_MODELSCOPE:-1}"

LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/${JOB_NAME}.log}"

# Read the reproducible training episode split without depending on jq.
EPISODES="$(DATASET_ROOT="${DATASET_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path

splits = json.loads((Path(os.environ["DATASET_ROOT"]) / "reports" / "splits.json").read_text())
print(json.dumps(splits["episode_split"]["train"], separators=(",", ":")))
PY
)"

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "Missing dataset metadata: ${DATASET_ROOT}/meta/info.json" >&2
  exit 1
fi

if [[ "${REDIRECT_LOG}" == "1" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  printf '[%s] Starting PI05 RGB training; log: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${LOG_FILE}"
fi

cmd=(
  /opt/venv/bin/lerobot-train
  "--dataset.repo_id=local/15holes_v3_pi05_joint_rgb"
  "--dataset.root=${DATASET_ROOT}"
  "--dataset.episodes=${EPISODES}"
  "--dataset.video_backend=${VIDEO_BACKEND}"
  "--policy.type=pi05"
  "--policy.pretrained_path=${PRETRAINED_PATH}"
  "--policy.push_to_hub=false"
  "--policy.use_relative_actions=true"
  '--policy.relative_exclude_joints=["gripper"]'
  "--policy.chunk_size=50"
  "--policy.n_action_steps=10"
  "--policy.max_state_dim=32"
  "--policy.max_action_dim=32"
  "--policy.freeze_vision_encoder=false"
  "--policy.train_expert_only=false"
  "--policy.dtype=bfloat16"
  "--policy.device=cuda"
  "--policy.gradient_checkpointing=true"
  "--policy.compile_model=false"
  '--policy.normalization_mapping={"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"QUANTILES"}'
  "--batch_size=${BATCH_SIZE}"
  "--gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
  "--num_workers=${NUM_WORKERS}"
  "--steps=${STEPS}"
  "--save_freq=${SAVE_FREQ}"
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${JOB_NAME}"
  "--env_eval_freq=0"
  "--log_freq=${LOG_FREQ}"
  "--save_checkpoint=true"
  "--wandb.enable=${WANDB_ENABLE}"
  "--wandb.project=${WANDB_PROJECT}"
  "--wandb.disable_artifact=true"
  "--seed=42"
)

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  if [[ "${USE_FSDP}" == "1" ]]; then
    cmd=(
      accelerate launch
      --use_fsdp
      --num_processes="${NUM_PROCESSES}"
      --mixed_precision=bf16
      --fsdp_version=2
      --fsdp_reshard_after_forward=true
      --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP
      --fsdp_transformer_layer_cls_to_wrap=Linear,Embedding,Conv2d
      --fsdp_state_dict_type=FULL_STATE_DICT
      "${cmd[@]}"
    )
  else
    cmd=(accelerate launch --multi_gpu --num_processes="${NUM_PROCESSES}" "${cmd[@]}")
  fi
fi

run_inside() {
  export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
  export TOKENIZERS_PARALLELISM=false
  export PYTHONUNBUFFERED=1
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export WANDB_BASE_URL="${WANDB_BASE_URL}"
  export HF_TOKEN="${HF_TOKEN}"
  export HF_ENDPOINT="${HF_ENDPOINT}"
  export HF_HOME="${HF_HOME}"
  export HF_LEROBOT_HOME="${HF_LEROBOT_HOME}"
  export WANDB_API_KEY="${WANDB_API_KEY}"
  mkdir -p "${HF_HOME}" "${HF_LEROBOT_HOME}"
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  if [[ -n "${GPU_IDS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  fi

  if [[ "${HF_LOGIN}" == "1" ]]; then
    if [[ -z "${HF_TOKEN}" ]]; then
      echo "HF_LOGIN=1 requires HF_TOKEN to be set." >&2
      return 1
    fi
    hf auth login --token "${HF_TOKEN}" --add-to-git-credential
  fi

  echo "Hugging Face endpoint: ${HF_ENDPOINT}"
  echo "Hugging Face cache: ${HF_HOME}"
  echo "LeRobot cache: ${HF_LEROBOT_HOME}"
  "${cmd[@]}"
}

if [[ "${RUN_IN_CONTAINER}" == "1" || ( "${RUN_IN_CONTAINER}" == "auto" && -f /.dockerenv ) ]]; then
  run_inside
elif [[ "${RUN_IN_CONTAINER}" == "0" ]]; then
  run_inside
else
  # Re-run this script inside the existing container so host invocations use
  # the same mounted absolute paths as the training process.
  docker exec -it "${CONTAINER_NAME}" bash -lc \
    "RUN_IN_CONTAINER=1 REDIRECT_LOG=0 GPU_IDS=$(printf '%q' "${GPU_IDS}") DATASET_ROOT=$(printf '%q' "${DATASET_ROOT}") PRETRAINED_PATH=$(printf '%q' "${PRETRAINED_PATH}") NUM_PROCESSES=$(printf '%q' "${NUM_PROCESSES}") USE_FSDP=$(printf '%q' "${USE_FSDP}") BATCH_SIZE=$(printf '%q' "${BATCH_SIZE}") GRADIENT_ACCUMULATION_STEPS=$(printf '%q' "${GRADIENT_ACCUMULATION_STEPS}") NUM_WORKERS=$(printf '%q' "${NUM_WORKERS}") STEPS=$(printf '%q' "${STEPS}") SAVE_FREQ=$(printf '%q' "${SAVE_FREQ}") LOG_FREQ=$(printf '%q' "${LOG_FREQ}") OUTPUT_DIR=$(printf '%q' "${OUTPUT_DIR}") LOG_FILE=$(printf '%q' "${LOG_FILE}") JOB_NAME=$(printf '%q' "${JOB_NAME}") HF_TOKEN=$(printf '%q' "${HF_TOKEN}") HF_HOME=$(printf '%q' "${HF_HOME}") HF_LEROBOT_HOME=$(printf '%q' "${HF_LEROBOT_HOME}") HF_ENDPOINT=$(printf '%q' "${HF_ENDPOINT}") HF_LOGIN=$(printf '%q' "${HF_LOGIN}") WANDB_ENABLE=$(printf '%q' "${WANDB_ENABLE}") WANDB_PROJECT=$(printf '%q' "${WANDB_PROJECT}") WANDB_BASE_URL=$(printf '%q' "${WANDB_BASE_URL}") WANDB_API_KEY=$(printf '%q' "${WANDB_API_KEY}") bash $(printf '%q' "${REPO_ROOT}/scripts/training/run_pi05_rgb.sh")"

  if [[ "${UPLOAD_TO_MODELSCOPE}" == "1" ]]; then
    upload_args=(
      "${OUTPUT_DIR}/checkpoints/last"
      --repo-id "${MODELSCOPE_REPO_ID}"
      --path-in-repo "${MODELSCOPE_PATH_IN_REPO}"
    )
    if [[ -n "${MODELSCOPE_TOKEN}" ]]; then
      upload_args+=(--token "${MODELSCOPE_TOKEN}")
    fi
    "${REPO_ROOT}/tool/upload_modelscope_checkpoint.sh" "${upload_args[@]}"
  fi
fi
