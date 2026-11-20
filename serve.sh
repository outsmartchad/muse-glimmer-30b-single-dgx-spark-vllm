#!/usr/bin/env bash
set -euo pipefail

image=spark-catalog-muse:99a10304-cu130
target=RedHatAI/Muse-Glimmer-30B-NVFP4
revision=403f780a4d35194b7230bc2d809effa3c147f407
cache="${SPARK_HF_CACHE:-$HOME/.cache/spark-muse-glimmer/huggingface}"
snapshot="hub/models--${target//\//--}/snapshots/$revision"
name=spark-catalog-muse-target-only
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
  download)
    mkdir -p "$cache"
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download "$target" --revision "$revision" --cache-dir "$cache/hub"
    docker build --platform linux/arm64 -t "$image" "$recipe_dir"
    ;;
  start)
    test -f "$cache/$snapshot/config.json"
    docker run -d --name "$name" --network host --ipc host --gpus all \
      --mount "type=bind,source=$cache,target=/models,readonly" \
      --env HF_HOME=/models --env HF_HUB_OFFLINE=1 --env HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
      "$image" vllm serve "/models/$snapshot" \
      --served-model-name Muse-Glimmer-30B --host 127.0.0.1 --port 18896 \
      --trust-remote-code --max-model-len 131072 --max-num-seqs 8 \
      --gpu-memory-utilization 0.42 --reasoning-parser muse_glimmer --tool-call-parser muse_glimmer \
      --enable-auto-tool-choice --kv-cache-dtype fp8_e4m3 --no-enable-flashinfer-autotune
    ;;
  stop)
    docker stop "$name"
    docker rm "$name"
    ;;
  *)
    echo "Usage: $0 {download|start|stop}" >&2
    exit 2
    ;;
esac
