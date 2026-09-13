#!/usr/bin/env bash
# Finalize the matched internal TB2.1 comparison after Swift completes.
set -euo pipefail
BASE=/data/tb/runs/tb21-qwen38bf-base-internal-official-k5-20260906
TIH=/data/tb/runs/tb21-swift-internal-official-k5-20260906
OUT=$TIH/analysis
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$TIH/finalize.log"; }
log "Waiting for Swift TB2.1 result.json finished_at."
while true; do
  if [ -s "$TIH/result.json" ]; then
    finished=$(jq -r '.finished_at // "null"' "$TIH/result.json")
    [ "$finished" != null ] && break
  fi
  sleep 60
done
log "Swift finished at $finished; producing durable paired token/accuracy analysis."
TB_STATS_OUT="$OUT" DRAWS=4000 SEED=20260906 \
  /data/tb/venv-train/bin/python /data/results/bin/tb_stats.py \
  base_qwen38_bf16="$BASE" swift_bf16_adapter="$TIH" | tee "$TIH/analysis.stdout"
{
  echo
  echo '===================================================================================================='
  echo 'TB2.1 INTERNAL QWEN-FAMILY MATCHED RUN — 2026-09-06'
  echo 'Controls: k=5; Harbor concurrency=89; agent cap=18000 s; LiteLLM=3600 s; model cap=131072.'
  echo 'Raw trial paths and machine-readable CSV live under the Swift run analysis/ directory.'
  cat "$OUT/tb21_stats.txt"
} >> /data/results/ALL_RESULTS.txt
{
  echo
  echo '===================================================================================================='
  echo 'TB2.1 INTERNAL QWEN-FAMILY MATCHED RUN — 2026-09-06'
  echo 'Frozen controls: k=5, concurrency 89, agent 5h, LiteLLM 1h, cap 131072.'
  cat "$OUT/tb21_stats.txt"
} >> /data/results/ALL_REPORT.txt
log "DONE: CSV/TXT saved under $OUT and both ALL_RESULTS.txt / ALL_REPORT.txt appended."
