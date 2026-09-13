#!/usr/bin/env bash
# Same server/harness envelope as run_base.sh; only the LoRA changes.
set -euo pipefail
ROOT=/data/tb/runs/tb21-swift-internal-official-k5-20260906
CFG=$ROOT/tb21_swift_internal_official_k5.yaml
BASE=/data/models/Qwen3.8-27B
ADAPTER=/data/models/ukisai/Swift-Qwen3.8-27B-adapter
HARBOR=/data/tb/venv-harbor/bin/harbor
IMAGE=vllm/vllm-openai:v0.27.1
NAME=tb21-swift-internal-official
PORT=8595
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$ROOT/chain.log"; }
cleanup(){ docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
for g in 2 3 4 5 6 7; do
  used=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -le 1024 ] || { log "ABORT: GPU $g busy (${used} MiB)"; exit 7; }
done
test -s "$BASE/config.json"; test -s "$ADAPTER/adapter_model.safetensors"; test -s "$CFG"
docker rm -f "$NAME" >/dev/null 2>&1 || true
log "Booting BF16 Qwen3.8 + Swift on GPUs 2-7: TP=1 DP=6, cap 131072."
docker run -d --name "$NAME" --gpus all --ipc=host --shm-size=64g \
  -e CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 -e NVIDIA_VISIBLE_DEVICES=2,3,4,5,6,7 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -v "$BASE":/model:ro -v "$ADAPTER":/adapter:ro -p "$PORT":8000 "$IMAGE" \
  --model /model --served-model-name swift-tb21-official \
  --tensor-parallel-size 1 --data-parallel-size 6 --max-model-len 131072 \
  --gpu-memory-utilization .94 --reasoning-parser qwen3 --max-num-seqs 32 \
  --enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules swift-tb21-official=/adapter > "$ROOT/vllm.container.id"
for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/v1/models > "$ROOT/models.json" && break; sleep 10; done
test -s "$ROOT/models.json" || { docker logs "$NAME" > "$ROOT/vllm.boot.log" 2>&1; exit 8; }
log "Serving healthy. Validating resolved Harbor config."
OPENAI_API_KEY=EMPTY "$HARBOR" run -c "$CFG" --print-config > "$ROOT/resolved_config.json"
log "START: exact matched TB2.1 profile: 89 tasks x k=5, concurrency 89, agent 5h, LiteLLM 1h."
OPENAI_API_KEY=EMPTY "$HARBOR" run -c "$CFG" > "$ROOT/harbor.log" 2>&1
log "COMPLETE: preserve raw trials and aggregate into ALL_RESULTS / ALL_REPORT before comparison."
