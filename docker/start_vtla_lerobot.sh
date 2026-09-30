#!/usr/bin/env bash
set -euo pipefail

IMAGE="vtla_lerobot:train"
CONTAINER_NAME="vtla_lerobot"
WORKSPACE="$(pwd -P)"

if ! command -v docker >/dev/null 2>&1; then
    echo "错误：未找到 docker 命令。" >&2
    exit 1
fi

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    echo "容器已存在：${CONTAINER_NAME}"
    docker start "${CONTAINER_NAME}" >/dev/null 2>&1 || true
else
    echo "创建并启动持久容器：${CONTAINER_NAME}"
    echo "挂载目录：${WORKSPACE}:${WORKSPACE}"

    docker run \
        --detach \
        --restart unless-stopped \
        --name "${CONTAINER_NAME}" \
        --network host \
        --shm-size=120g \
        --gpus all \
        --volume "${WORKSPACE}:${WORKSPACE}" \
        --workdir "${WORKSPACE}" \
        "${IMAGE}" \
        sleep infinity
fi

echo "进入容器：${CONTAINER_NAME}"
exec docker exec -it "${CONTAINER_NAME}" bash
