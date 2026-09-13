#!/usr/bin/env bash
set -euo pipefail
BASE=/data/models/Qwen3.8-27B
ADAPTER=/data/models/ukisai/Swift-Qwen3.8-27B-adapter
MA=/data/tb/glm52_gap/tools/matharena
PY=$MA/.venv/bin/python
OUT=/data/ft/runs/qwen38_bf16_swift_ifbench_k5_81920_20260909
mkdir -p "$OUT" "$OUT/positive_control"
log(){ echo "[$(date -u)] $*" | tee -a "$OUT/chain.log"; }
cleanup(){ docker rm -f q38bf-ifb-base q38bf-ifb-swift >/dev/null 2>&1 || true; }
trap cleanup EXIT

for g in 2 3 4 5 6 7; do
  used=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -le 1024 ] || { log "ABORT GPU $g busy (${used} MiB)"; exit 7; }
done
test -s "$ADAPTER/adapter_model.safetensors"
test -x "$PY"

boot(){
  local name=$1 dev=$2 port=$3 extra=$4
  docker run -d --name "$name" --gpus all --shm-size 64g --ipc=host \
    -e CUDA_VISIBLE_DEVICES="$dev" -e NVIDIA_VISIBLE_DEVICES="$dev" -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -v "$BASE":/model:ro -v "$ADAPTER":/adapter:ro -p "127.0.0.1:$port":8000 vllm/vllm-openai:v0.27.1 \
    --model /model --served-model-name qwen38bf --tensor-parallel-size 1 --data-parallel-size 3 \
    --max-model-len 262144 --gpu-memory-utilization .94 --reasoning-parser qwen3 --max-num-seqs 48 --dtype bfloat16 --seed 0 $extra >/dev/null
}
/home/nvidia/qwen38-btl-compact/.venv/bin/python "$OUT/preflight.py" > "$OUT/preflight.log" 2>&1
boot q38bf-ifb-base '2,3,4' 8586 ''
boot q38bf-ifb-swift '5,6,7' 8587 '--enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules swift=/adapter'
ready=0
for attempt in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8586/v1/models > "$OUT/base.models.json"; then ready=1; break; fi
  sleep 10
done
[ "$ready" -eq 1 ] || { log "ERROR boot timeout base"; exit 12; }
ready=0
for attempt in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8587/v1/models > "$OUT/swift.models.json"; then ready=1; break; fi
  sleep 10
done
[ "$ready" -eq 1 ] || { log "ERROR boot timeout swift"; exit 12; }

/home/nvidia/qwen38-btl-compact/.venv/bin/python "$OUT/positive_control.py" \
  "$OUT/positive_control" http://127.0.0.1:8586/v1 qwen38bf http://127.0.0.1:8587/v1 swift | tee "$OUT/positive_control/result.log"
identical=$(awk '/identical_rows/{print $2}' "$OUT/positive_control/result.log")
[ "$identical" != 12 ] || { log 'ABORT adapter inactive'; exit 9; }
log "PASS BF16 DP3 LoRA positive control: changed $((12-identical))/12 outputs. GPUs0-1 remain free."



client(){
 local arm=$1 port=$2 model=$3 smoke=$4
 API_BASE=http://127.0.0.1:$port/v1 MODEL=$model TAG=$arm OUTDIR="$OUT/$arm" K=5 CONC=48 MAX_TOKENS=81920 REQ_TIMEOUT=10800 ATTEMPTS=3 IFB_DATA="$OUT/dataset.jsonl" SMOKE=$smoke /data/lcb/venv/bin/python "$OUT/generate.py"
}
log 'SMOKE: base first missing seed4 row; swift first seed0 row. Saved and reused.'
client base 8586 qwen38bf 1 > "$OUT/smoke_base.log" 2>&1 & p1=$!
client swift 8587 swift 1 > "$OUT/smoke_t20.log" 2>&1 & p2=$!
failed=0; wait "$p1" || failed=1; wait "$p2" || failed=1
[ "$failed" -eq 0 ] || exit 11
"$PY" - "$OUT" <<'CHECK'
import sys,json
from pathlib import Path
r=Path(sys.argv[1])
for a in ['base','swift']:
 x=json.loads((r/a/(a+'_raw.jsonl')).read_text().splitlines()[-1])
 assert x.get('response') is not None and not x.get('error'), x.get('error')
 assert x.get('reasoning_field_present') and x.get('reasoning'),'reasoning missing in smoke'
CHECK
log 'FULL START base remaining seed4; swift seeds0..4; cap81920 T1 xhigh-default; CONC48/arm'
client base 8586 qwen38bf 0 > "$OUT/full_base.log" 2>&1 & p1=$!
client swift 8587 swift 0 > "$OUT/full_t20.log" 2>&1 & p2=$!
failed=0; wait "$p1" || failed=1; wait "$p2" || failed=1
[ "$failed" -eq 0 ] || { log 'ERROR client failure, artifacts retained'; exit 11; }
"$PY" "$OUT/finalize.py" > "$OUT/finalize.log" 2>&1
log 'IFBENCH FINAL COMPLETE: official scoring, aggregate and both reports saved'
