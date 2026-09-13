#!/usr/bin/env python3
"""Deterministic vLLM LoRA positive control: two endpoints/models must differ."""
import concurrent.futures as cf
import hashlib
import json
import statistics
import sys
import urllib.request
from transformers import AutoTokenizer

out, base_api, base_model, lora_api, lora_model = sys.argv[1:]
tokenizer = AutoTokenizer.from_pretrained("/data/models/Qwen3.8-27B-INT4-RedHatAI")
prompts = []
for line in open("/data/ft/runs/int4_r0b_verified4096_20260829/lambda0_raw.jsonl"):
    row = json.loads(line)
    if row.get("reasoning"):
        prompts.append((len(row["reasoning"]), row["prompt"]))
prompts = [prompt for _, prompt in sorted(prompts, reverse=True)[:12]]

def ask(api, model, prompt):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "top_p": 1.0, "max_tokens": 1024}).encode()
    request = urllib.request.Request(api + "/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        message = json.load(response)["choices"][0]["message"]
    return message.get("reasoning_content") or message.get("reasoning") or "", message.get("content") or ""

def run(api, model):
    with cf.ThreadPoolExecutor(12) as executor:
        answers = list(executor.map(lambda p: ask(api, model, p), prompts))
    lengths = [len(ids) for ids in tokenizer([a + b for a, b in answers], add_special_tokens=False)["input_ids"]]
    hashes = [hashlib.sha256((a + "\n" + b).encode()).hexdigest()[:16] for a, b in answers]
    return {"thinking_tokens": lengths, "mean": statistics.mean(lengths), "hashes": hashes, "answers": answers}

base = run(base_api, base_model)
lora = run(lora_api, lora_model)
result = {"base": base, "lora": lora, "identical_rows": sum(a == b for a, b in zip(base["hashes"], lora["hashes"]))}
json.dump(result, open(out + "/positive.json", "w"))
print("base_mean", round(base["mean"], 1))
print("lora_mean", round(lora["mean"], 1))
print("identical_rows", result["identical_rows"])
