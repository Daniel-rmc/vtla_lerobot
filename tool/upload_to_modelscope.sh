#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   cd /home/rmc/workspace/vtla_lerobot
#   export MODELSCOPE_API_TOKEN='your_modelscope_access_token'
#   ./tool/upload_to_modelscope.sh \
#       --repo-id Deeptouch_ICRA/manipulationNet-realman-xense \
#       --repo-type dataset \
#       /path/to/file.zip
#
# Upload a local dataset directory under a remote directory:
#   ./tool/upload_to_modelscope.sh \
#       --prompt-token \
#       --repo-id Deeptouch_ICRA/manipulationNet-realman-xense \
#       --repo-type dataset \
#       --remote-dir datasets \
#       /path/to/your/dataset
#
# Preview the upload command without uploading:
#   ./tool/upload_to_modelscope.sh \
#       --dry-run \
#       --repo-type dataset \
#       /path/to/file.zip

# Defaults can be overridden by CLI arguments or environment variables.
repo_id="${MODELSCOPE_REPO_ID:-Ruanminchi/Deeptouch_images}"
repo_type="${MODELSCOPE_REPO_TYPE:-dataset}"
endpoint="${MODELSCOPE_ENDPOINT:-https://modelscope.cn}"
token="${MODELSCOPE_API_TOKEN:-}"
commit_message="${MODELSCOPE_COMMIT_MESSAGE:-Upload artifacts}"
max_workers="${MODELSCOPE_MAX_WORKERS:-4}"
remote_dir=""
path_in_repo=""
prompt_token=false
dry_run=false
checksum=false
use_proxy=false
declare -a local_paths=()

usage() {
  cat <<'EOF'
Upload one or more files/directories to a ModelScope repository.

Usage:
  upload_to_modelscope.sh [options] LOCAL_PATH [LOCAL_PATH ...]

Options:
  --repo-id ID             Target repository (default: Ruanminchi/Deeptouch_images)
  --repo-type TYPE         dataset or model (default: dataset)
  --token TOKEN            ModelScope access token; prefer env or --prompt-token
  --prompt-token           Read the token without echoing it
  --remote-dir DIR         Upload every input under this remote directory
  --path-in-repo PATH      Exact remote path; valid only with one input
  --endpoint URL           ModelScope endpoint (default: https://modelscope.cn)
  --commit-message TEXT    Commit message (default: Upload artifacts)
  --max-workers N          Concurrent files for directory uploads (default: 4)
  --use-proxy              Inherit HTTP(S)/ALL proxy variables (default: direct)
  --checksum               Print SHA-256 for regular files before uploading
  --dry-run                Validate and print operations without uploading
  -h, --help               Show this help

Token precedence:
  --token / --prompt-token > MODELSCOPE_API_TOKEN > saved `modelscope login`

Examples:
  export MODELSCOPE_API_TOKEN='paste_token_here'
  ./tool/upload_to_modelscope.sh \
    --repo-id Ruanminchi/Deeptouch_images \
    /path/to/file.tar

  ./tool/upload_to_modelscope.sh \
    --prompt-token \
    --repo-id Ruanminchi/Deeptouch_images \
    --remote-dir docker-images \
    /path/to/file-a.tar /path/to/file-b.tar

  ./tool/upload_to_modelscope.sh \
    --repo-type dataset \
    --path-in-repo backups/run-01 \
    /path/to/local-directory
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
      repo_id="$2"
      shift 2
      ;;
    --repo-type)
      require_value "$1" "${2:-}"
      repo_type="$2"
      shift 2
      ;;
    --token)
      require_value "$1" "${2:-}"
      token="$2"
      shift 2
      ;;
    --prompt-token)
      prompt_token=true
      shift
      ;;
    --remote-dir)
      require_value "$1" "${2:-}"
      remote_dir="$2"
      shift 2
      ;;
    --path-in-repo)
      require_value "$1" "${2:-}"
      path_in_repo="$2"
      shift 2
      ;;
    --endpoint)
      require_value "$1" "${2:-}"
      endpoint="$2"
      shift 2
      ;;
    --commit-message)
      require_value "$1" "${2:-}"
      commit_message="$2"
      shift 2
      ;;
    --max-workers)
      require_value "$1" "${2:-}"
      max_workers="$2"
      shift 2
      ;;
    --use-proxy)
      use_proxy=true
      shift
      ;;
    --checksum)
      checksum=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      local_paths+=("$@")
      break
      ;;
    -* )
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      local_paths+=("$1")
      shift
      ;;
  esac
done

if ! command -v modelscope >/dev/null 2>&1; then
  echo "modelscope CLI is not installed or not on PATH." >&2
  exit 1
fi
if [[ "${repo_type}" != "dataset" && "${repo_type}" != "model" ]]; then
  echo "--repo-type must be dataset or model: ${repo_type}" >&2
  exit 2
fi
if [[ ! "${max_workers}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-workers must be a positive integer: ${max_workers}" >&2
  exit 2
fi
if ((${#local_paths[@]} == 0)); then
  echo "At least one LOCAL_PATH is required." >&2
  usage >&2
  exit 2
fi
if [[ -n "${path_in_repo}" && ${#local_paths[@]} -ne 1 ]]; then
  echo "--path-in-repo can only be used with exactly one LOCAL_PATH." >&2
  exit 2
fi
if [[ "${prompt_token}" == true ]]; then
  read -r -s -p "ModelScope access token: " token
  echo
  if [[ -z "${token}" ]]; then
    echo "The entered token is empty." >&2
    exit 2
  fi
fi

for local_path in "${local_paths[@]}"; do
  if [[ ! -e "${local_path}" ]]; then
    echo "Local path does not exist: ${local_path}" >&2
    exit 1
  fi
  if [[ ! -r "${local_path}" ]]; then
    echo "Local path is not readable: ${local_path}" >&2
    exit 1
  fi

  basename_path="$(basename "${local_path%/}")"
  if [[ -n "${path_in_repo}" ]]; then
    remote_path="${path_in_repo#/}"
  elif [[ -n "${remote_dir}" ]]; then
    remote_path="${remote_dir#/}"
    remote_path="${remote_path%/}/${basename_path}"
  else
    remote_path="${basename_path}"
  fi

  echo "Upload plan: ${local_path} -> ${repo_type}:${repo_id}/${remote_path}"
  if [[ "${checksum}" == true && -f "${local_path}" ]]; then
    sha256sum "${local_path}"
  fi

  command=(
    modelscope upload
    "${repo_id}"
    "${local_path}"
    "${remote_path}"
    --repo-type "${repo_type}"
    --endpoint "${endpoint}"
    --commit-message "${commit_message}"
    --max-workers "${max_workers}"
  )

  run_env=(env)
  if [[ "${use_proxy}" != true ]]; then
    run_env+=(
      -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
      -u http_proxy -u https_proxy -u all_proxy
    )
  fi

  if [[ "${dry_run}" == true ]]; then
    printf 'Command:'
    printf ' %q' "${run_env[@]}" "${command[@]}"
    printf '\n'
    continue
  fi

  if [[ -n "${token}" ]]; then
    "${run_env[@]}" MODELSCOPE_API_TOKEN="${token}" "${command[@]}"
  else
    "${run_env[@]}" "${command[@]}"
  fi
done
