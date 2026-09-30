#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   cd /home/rmc/workspace/vtla_lerobot
#   export MODELSCOPE_TOKEN='your_modelscope_access_token'
#   ./tool/upload_modelscope_checkpoint.sh \
#       /path/to/checkpoints/last
#
# Upload to a custom repository path:
# ./tool/upload_modelscope_checkpoint.sh \
#     --repo-id Deeptouch_ICRA/manipulationNet-realman-xense \
#     --path-in-repo checkpoints/pi05_rgb/025000 \
#     /home/rmc/workspace/vtla_lerobot/outputs/pi05_rgb/checkpoints/025000
#
# Preview the validation and upload target without uploading:
#   ./tool/upload_modelscope_checkpoint.sh \
#       --dry-run \
#       /path/to/checkpoints/last

# Upload a completed checkpoint from the host with the ModelScope CLI.
# The host may use an existing `modelscope login` session. Keep tokens out of
# tracked files and prefer MODELSCOPE_TOKEN over putting one in shell history.

REPO_ID="${MODELSCOPE_REPO_ID:-Deeptouch_ICRA/manipulationNet-realman-xense}"
PATH_IN_REPO="${MODELSCOPE_PATH_IN_REPO:-checkpoints/pi05_rgb/last}"
COMMIT_MESSAGE="${MODELSCOPE_COMMIT_MESSAGE:-Upload final checkpoint}"
TOKEN="${MODELSCOPE_TOKEN:-}"
DRY_RUN=0
CHECKPOINT_DIR=""

usage() {
  cat <<'EOF'
Upload a model checkpoint directory to ModelScope.
Usage:
  upload_modelscope_checkpoint.sh CHECKPOINT_DIR [options]

Options:
  --repo-id ID             ModelScope repository (default: Deeptouch_ICRA/manipulationNet-realman-xense)
  --path-in-repo PATH      Destination path (default: checkpoints/pi05_rgb/last)
  --token TOKEN            Access token; existing `modelscope login` is used when omitted
  --commit-message TEXT    Commit message (default: Upload final checkpoint)
  --dry-run                Validate and print the upload plan without uploading
  -h, --help               Show this help

Examples:
  tool/upload_modelscope_checkpoint.sh /path/to/checkpoints/last
  MODELSCOPE_TOKEN='paste_token_here' \
    tool/upload_modelscope_checkpoint.sh /path/to/checkpoints/last
EOF
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" ]]; then
    echo "${option} requires a non-empty value." >&2
    exit 2
  fi
}

while (($#)); do
  case "$1" in
    --repo-id)
      require_value "$1" "${2:-}"
      REPO_ID="$2"
      shift 2
      ;;
    --path-in-repo)
      require_value "$1" "${2:-}"
      PATH_IN_REPO="$2"
      shift 2
      ;;
    --token)
      require_value "$1" "${2:-}"
      TOKEN="$2"
      shift 2
      ;;
    --commit-message)
      require_value "$1" "${2:-}"
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "${CHECKPOINT_DIR}" ]]; then
        echo "Only one CHECKPOINT_DIR may be provided: ${CHECKPOINT_DIR}" >&2
        exit 2
      fi
      CHECKPOINT_DIR="$1"
      shift
      ;;
  esac
done

if (($#)); then
  if [[ -n "${CHECKPOINT_DIR}" ]]; then
    echo "Only one CHECKPOINT_DIR may be provided: ${CHECKPOINT_DIR}" >&2
    exit 2
  fi
  CHECKPOINT_DIR="$1"
  shift
fi
if (($#)); then
  echo "Unexpected argument: $1" >&2
  exit 2
fi

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "A checkpoint directory is required." >&2
  usage >&2
  exit 2
fi
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT_DIR}/pretrained_model" ]]; then
  echo "Missing pretrained_model directory: ${CHECKPOINT_DIR}/pretrained_model" >&2
  exit 1
fi
if ! command -v modelscope >/dev/null 2>&1; then
  echo "ModelScope CLI is not installed on the host or is not on PATH." >&2
  exit 1
fi

CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}" && pwd)"
printf 'ModelScope repository: %s\n' "${REPO_ID}"
printf 'Checkpoint source: %s\n' "${CHECKPOINT_DIR}"
printf 'Repository destination: %s\n' "${PATH_IN_REPO}"

if ((DRY_RUN)); then
  echo 'Dry run: upload not performed.'
  exit 0
fi

upload_cmd=(
  modelscope upload
  "${REPO_ID}"
  "${CHECKPOINT_DIR}"
  "${PATH_IN_REPO}"
  --repo-type model
  --commit-message "${COMMIT_MESSAGE}"
)
if [[ -n "${TOKEN}" ]]; then
  upload_cmd+=(--token "${TOKEN}")
else
  echo 'Using the existing host ModelScope login.'
fi

run_env=(
  env
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
  -u http_proxy -u https_proxy -u all_proxy
)

"${run_env[@]}" "${upload_cmd[@]}"
