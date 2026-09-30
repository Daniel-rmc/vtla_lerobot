#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
weights_host="${VTLA_SAMPLE_WEIGHTS_HOST:-${repo_root}/configs/sampling/manipulationnet_15holes_equal.json}"
output_dir="${VTLA_OUTPUT_DIR:-${repo_root}/outputs/train/$(date +%Y-%m-%d_%H-%M-%S)_manipulationnet_15holes}"
container_weights="/workspace/vtla_lerobot/outputs/train/$(basename "${output_dir}")/manipulationnet_15holes_equal.container.json"

echo "weights_host=${weights_host}"
echo "output_dir=${output_dir}"
python3 - <<'PY' "${weights_host}" "${output_dir}/manipulationnet_15holes_equal.container.json"
import json
import sys
from pathlib import Path

weights_host = Path(sys.argv[1])
out_path = Path(sys.argv[2])
raw = json.loads(weights_host.read_text())
container_root = Path('/data/manipulationnet_15holes')
converted = {str(container_root / Path(host_root).name): weight for host_root, weight in raw.items()}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(converted, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

echo "container_weights=${container_weights}"