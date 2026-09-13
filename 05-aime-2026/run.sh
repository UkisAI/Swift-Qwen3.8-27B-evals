#!/usr/bin/env bash
set -euo pipefail
BASE=/data/models/Qwen3.8-27B
ADAPTER=/data/models/ukisai/Swift-Qwen3.8-27B-adapter
MA=/data/tb/glm52_gap/tools/matharena
PY=$MA/.venv/bin/python
OUT=/data/ft/runs/qwen38_bf16_swift_aime2026_xhigh5_20260909
CFG=$OUT/configs
RESULTS=$OUT/matharena_outputs
mkdir -p "$OUT" "$RESULTS" "$OUT/positive_control"
log(){ echo "[$(date -u)] $*" | tee -a "$OUT/chain.log"; }
cleanup(){ docker rm -f q38bf-aime26-base q38bf-aime26-swift >/dev/null 2>&1 || true; }
trap cleanup EXIT

for g in 2 3 4 5 6 7; do
  used=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -le 1024 ] || { log "ABORT GPU $g busy (${used} MiB)"; exit 7; }
done
test -s "$ADAPTER/adapter_model.safetensors"
test -x "$PY"
test ! -e "$RESULTS/aime/aime_2026" || { log 'ABORT pre-existing AIME output tree'; exit 8; }

boot(){
  local name=$1 dev=$2 port=$3 extra=$4
  docker run -d --name "$name" --gpus all --shm-size 64g --ipc=host \
    -e CUDA_VISIBLE_DEVICES="$dev" -e NVIDIA_VISIBLE_DEVICES="$dev" -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -v "$BASE":/model:ro -v "$ADAPTER":/adapter:ro -p "127.0.0.1:$port":8000 vllm/vllm-openai:v0.27.1 \
    --model /model --served-model-name qwen38bf --tensor-parallel-size 1 --data-parallel-size 3 \
    --max-model-len 262144 --gpu-memory-utilization .94 --reasoning-parser qwen3 --max-num-seqs 64 --seed 0 $extra >/dev/null
}
boot q38bf-aime26-base '2,3,4' 8584 ''
boot q38bf-aime26-swift '5,6,7' 8585 '--enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules swift=/adapter'
until curl -sf http://127.0.0.1:8584/v1/models > "$OUT/base.models.json"; do sleep 10; done
until curl -sf http://127.0.0.1:8585/v1/models > "$OUT/swift.models.json"; do sleep 10; done

/home/nvidia/qwen38-btl-compact/.venv/bin/python /data/ft/runs/qwen38_bf16_swift_aime2026_xhigh5_20260909/positive_control.py \
  "$OUT/positive_control" http://127.0.0.1:8584/v1 qwen38bf http://127.0.0.1:8585/v1 swift | tee "$OUT/positive_control/result.log"
identical=$(awk '/identical_rows/{print $2}' "$OUT/positive_control/result.log")
[ "$identical" != 12 ] || { log 'ABORT adapter inactive'; exit 9; }
log "PASS BF16 DP3 LoRA positive control: changed $((12-identical))/12 outputs. GPUs0-1 remain free."

run_arm(){
  local arm=$1; shift
  cd "$MA"
  "$PY" scripts/run.py --comp aime/aime_2026 --models "$@" --n 1 \
    --comp-configs-dir "$MA/configs/competitions" --model-configs-dir "$CFG" --output-dir "$RESULTS"
}

log 'FULL start: AIME 2026 30 x 5 seeds per arm, all 10 clients parallel; 150 concurrent requests per arm.'
pids=()
for seed in 0 1 2 3 4; do
  run_arm base base_s$seed > "$OUT/base_s$seed.log" 2>&1 & pids+=($!)
  run_arm swift t20_s$seed > "$OUT/t20_s$seed.log" 2>&1 & pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { log 'CLIENT FAILURE; preserving artifacts, inspect seed logs'; exit 11; }
for arm in base swift; do
  for seed in 0 1 2 3 4; do
    count=$(find "$RESULTS/aime/aime_2026/${arm}_s${seed}" -type f | wc -l)
    [ "$count" -eq 30 ] || { log "INCOMPLETE ${arm}_s${seed}: files=$count expected=30"; exit 10; }
  done
done
log 'GENERATION COMPLETE: 150/150 artifacts per arm saved and canonically graded; aggregate/token audit pending.'

"$PY" "$OUT/aggregate.py" "$OUT" > "$OUT/finalize.log" 2>&1
"$PY" "$OUT/record.py"
log 'FINAL COMPLETE: aggregate and both central reports written.'
