#!/usr/bin/env python3
import glob
import json
import os
import statistics
import sys

import zstandard as zstd
from transformers import AutoTokenizer

ROOT = sys.argv[1]
DATA = os.path.join(ROOT, "matharena_outputs", "aime", "aime_2026")
TOKENIZER = AutoTokenizer.from_pretrained("/data/models/Qwen3.8-27B", local_files_only=True)


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def load(path):
    with open(path, "rb") as f:
        raw = zstd.ZstdDecompressor().stream_reader(f).read()
    return json.loads(raw)


def tokens(text):
    return len(TOKENIZER.encode(text or "", add_special_tokens=False))


def summarize(arm):
    rows = []
    seed_counts = {}
    for seed in range(5):
        paths = sorted(glob.glob(os.path.join(DATA, f"{arm}_s{seed}", "*.json.zst")))
        seed_counts[str(seed)] = len(paths)
        for path in paths:
            x = load(path)
            msgs = x["messages"][0]
            cot = "".join(m.get("content", "") for m in msgs if m.get("type") == "cot")
            response = "".join(m.get("content", "") for m in msgs if m.get("type") == "response")
            dc = x["detailed_costs"][0]
            rows.append({
                "seed": seed,
                "problem": x["idx"],
                "correct": bool(x["correct"][0]),
                "api_output_tokens": int(dc["output_tokens"]),
                "thinking_tokens": tokens(cot),
                "response_tokens": tokens(response),
                "request_seconds": float(dc["request_time"]),
                "retries": int(dc.get("n_retries", 0)),
                "warning": x["warnings"][0],
            })
    if seed_counts != {str(i): 30 for i in range(5)}:
        raise SystemExit(f"REFUSING INCOMPLETE {arm}: {seed_counts}")
    def dist(key):
        vals = [r[key] for r in rows]
        return {
            "mean": statistics.fmean(vals),
            "median": statistics.median(vals),
            "p90": percentile(vals, .9),
            "max": max(vals),
            "sum": sum(vals),
        }
    return {
        "n": len(rows),
        "seed_counts": seed_counts,
        "correct": sum(r["correct"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "per_seed_accuracy": {
            str(s): sum(r["correct"] for r in rows if r["seed"] == s) / 30 for s in range(5)
        },
        "api_output_tokens": dist("api_output_tokens"),
        "thinking_tokens_retokenized": dist("thinking_tokens"),
        "response_tokens_retokenized": dist("response_tokens"),
        "request_seconds": dist("request_seconds"),
        "warnings_nonzero": sum(bool(r["warning"]) for r in rows),
        "retries": sum(r["retries"] for r in rows),
        "rows": rows,
    }


base = summarize("base")
swift = summarize("swift")
report = {
    "benchmark": "MathArena/aime_2026",
    "problems": 30,
    "seeds": [0, 1, 2, 3, 4],
    "base": base,
    "swift": swift,
    "delta": {
        "accuracy_points_t20_minus_base": 100 * (swift["accuracy"] - base["accuracy"]),
        "mean_api_output_tokens_pct": 100 * (swift["api_output_tokens"]["mean"] / base["api_output_tokens"]["mean"] - 1),
        "mean_thinking_tokens_pct": 100 * (swift["thinking_tokens_retokenized"]["mean"] / base["thinking_tokens_retokenized"]["mean"] - 1),
        "total_api_output_tokens_saved": base["api_output_tokens"]["sum"] - swift["api_output_tokens"]["sum"],
    },
}
json_path = os.path.join(ROOT, "aime2026_xhigh5_aggregate.json")
with open(json_path, "w") as f:
    json.dump(report, f, indent=2)

def line(name, x):
    return (f"{name}: {x['correct']}/{x['n']} = {100*x['accuracy']:.2f}%; "
            f"API output mean/p50/p90={x['api_output_tokens']['mean']:.1f}/"
            f"{x['api_output_tokens']['median']:.1f}/{x['api_output_tokens']['p90']:.1f}; "
            f"thinking mean/p50/p90={x['thinking_tokens_retokenized']['mean']:.1f}/"
            f"{x['thinking_tokens_retokenized']['median']:.1f}/"
            f"{x['thinking_tokens_retokenized']['p90']:.1f}; "
            f"warnings={x['warnings_nonzero']}, retries={x['retries']}")

text = "\n".join([
    "AIME 2026, BF16 xhigh, 30 problems x five explicit seeds",
    line("Base", base),
    line("Swift", swift),
    f"Delta accuracy (Swift-base): {report['delta']['accuracy_points_t20_minus_base']:+.2f} pp",
    f"Delta mean API output tokens: {report['delta']['mean_api_output_tokens_pct']:+.2f}%",
    f"Delta mean retokenized thinking tokens: {report['delta']['mean_thinking_tokens_pct']:+.2f}%",
]) + "\n"
txt_path = os.path.join(ROOT, "aime2026_xhigh5_aggregate.txt")
with open(txt_path, "w") as f:
    f.write(text)
print(text, end="")
