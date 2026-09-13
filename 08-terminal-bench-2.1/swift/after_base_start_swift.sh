#!/usr/bin/env bash
# Starts Swift only after all 445 base trials are terminal.
set -euo pipefail
BASE_RESULT=/data/tb/runs/tb21-qwen38bf-base-internal-official-k5-20260906/result.json
TIH_ROOT=/data/tb/runs/tb21-swift-internal-official-k5-20260906
mkdir -p "$TIH_ROOT"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$TIH_ROOT/after_base.log"; }
log "Waiting for the base TB2.1 job to finish cleanly."
while true; do
  if [ -s "$BASE_RESULT" ]; then
    finished=$(jq -r '.finished_at // "null"' "$BASE_RESULT")
    [ "$finished" != null ] && break
  fi
  sleep 60
done
log "Base finished at $finished. Starting matched Swift run."
log "Waiting for the base launcher to remove its vLLM container and release GPUs 2-7."
while docker ps --format '{{.Names}}' | grep -qx 'tb21-q38bf-internal-official'; do
  sleep 15
done
for g in 2 3 4 5 6 7; do
  while true; do
    used=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -le 1024 ] && break
    sleep 15
  done
done
log "Base container is gone and GPUs 2-7 are free. Starting matched Swift run."
exec /data/tb/runs/tb21-swift-internal-official-k5-20260906/run_swift.sh
