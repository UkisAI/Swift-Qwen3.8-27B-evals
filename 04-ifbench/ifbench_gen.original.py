#!/usr/bin/env python3
"""IFBench generation for Qwen3.8-27B, for the overthinking-penalty A/B.

WHY NOT IFBench's own generate_responses.py: it forwards only temperature and
max_tokens. Qwen3.8's documented thinking-mode config is temperature=1.0, top_p=0.95,
top_k=20, min_p=0.0 -- three of those cannot be expressed through that script, and
top_k in particular materially changes the sampling distribution. So we generate here
and score with IFBench's OFFICIAL run_eval.py, which is the part that must stay faithful.

WHAT COUNTS AS "THE RESPONSE": message.content ONLY, never the reasoning block.
IFBench constraints are things like "avoid these words", "exactly N sentences",
"end with this phrase". They apply to what the user is shown. Feeding the chain of
thought to the verifier would fail constraints the model actually satisfied, and would
do so ASYMMETRICALLY -- the base arm thinks longer, so it would be penalised harder.
That would manufacture exactly the result we are testing for.

max_tokens is deliberately 81920, the same as the GPQA runner that reproduced 89.2.
A cap is not an allocation in vLLM, so a high ceiling costs nothing, and it removes the
truncation confound: if the budget clipped thinking, the base arm would clip more than
the penalised arms and the penalty would look like it helped when it merely avoided the
wall. Truncation rate is recorded per sample and must be ~0 in every arm.

OUTPUT: one record-level jsonl with full token accounting, plus k response files in
IFBench's {"prompt","response"} shape -- one per sample index, because run_eval.py keys
on the prompt string and would collide on duplicates.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = os.environ.get("API_BASE", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
DATA = os.environ.get("IFB_DATA", "/data/lcb/IFBench/data/IFBench_test.jsonl")
OUTDIR = os.environ.get("OUTDIR", "/data/lcb/runs/ifbench")
TAG = os.environ.get("TAG", "base")
K = int(os.environ.get("K", "4"))
CONC = int(os.environ.get("CONC", "192"))
MAXTOK = int(os.environ.get("MAX_TOKENS", "81920"))
REQ_TIMEOUT = int(os.environ.get("REQ_TIMEOUT", "3600"))
ATTEMPTS = int(os.environ.get("ATTEMPTS", "3"))


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def post(payload, timeout):
    req = urllib.request.Request(
        API + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-noauth"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read().decode())


def done_keys(path):
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("response") is not None or d.get("error"):
                seen.add((d.get("key"), d.get("sample_index")))
    return seen


def one(row, si):
    t0 = time.time()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": row["prompt"]}],
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 0.0, "repetition_penalty": 1.0,
        "max_tokens": MAXTOK, "seed": si,
    }
    last = None
    for _ in range(ATTEMPTS):
        try:
            d = post(body, REQ_TIMEOUT)
            ch = d["choices"][0]
            msg = ch["message"]
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            content = msg.get("content") or ""
            us = d.get("usage") or {}
            return {
                "key": row["key"], "sample_index": si, "tag": TAG,
                "prompt": row["prompt"],
                "response": content,           # content ONLY -- see module docstring
                "reasoning": reasoning,
                "instruction_id_list": row.get("instruction_id_list"),
                "finish_reason": ch.get("finish_reason"),
                "truncated": ch.get("finish_reason") == "length",
                "prompt_tokens": us.get("prompt_tokens", 0),
                "completion_tokens": us.get("completion_tokens", 0),
                "seconds": round(time.time() - t0, 2),
            }
        except Exception as exc:                       # noqa: BLE001
            last = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            time.sleep(2)
    return {"key": row["key"], "sample_index": si, "tag": TAG,
            "prompt": row["prompt"], "response": None, "error": last,
            "seconds": round(time.time() - t0, 2)}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    raw = os.path.join(OUTDIR, "%s_raw.jsonl" % TAG)
    rows = load_rows(DATA)
    seen = done_keys(raw)
    jobs = [(r, si) for r in rows for si in range(K) if (r["key"], si) not in seen]
    print("IFBench %s: %d prompts x k=%d = %d samples; %d already done, %d to run"
          % (TAG, len(rows), K, len(rows) * K, len(seen), len(jobs)), flush=True)
    if not jobs:
        print("nothing to do", flush=True)
    t0 = time.time()
    n = 0
    with open(raw, "a", encoding="utf-8") as out, ThreadPoolExecutor(CONC) as ex:
        futs = [ex.submit(one, r, si) for r, si in jobs]
        for f in as_completed(futs):
            out.write(json.dumps(f.result(), ensure_ascii=False) + "\n")
            out.flush()
            n += 1
            if n % 50 == 0:
                el = time.time() - t0
                print("  %d/%d  %.1f/s  elapsed %.1f min" % (n, len(jobs), n / el, el / 60),
                      flush=True)

    # ---- emit IFBench-shaped response files, one per sample index ----
    recs = [json.loads(x) for x in open(raw, encoding="utf-8")]
    errs = sum(1 for r in recs if r.get("response") is None)
    trunc = sum(1 for r in recs if r.get("truncated"))
    for si in range(K):
        p = os.path.join(OUTDIR, "%s_responses_s%d.jsonl" % (TAG, si))
        with open(p, "w", encoding="utf-8") as fh:
            for r in recs:
                if r.get("sample_index") == si and r.get("response") is not None:
                    fh.write(json.dumps({"prompt": r["prompt"], "response": r["response"]},
                                        ensure_ascii=False) + "\n")
        print("  wrote %s (%d rows)" % (p, sum(1 for r in recs
                                               if r.get("sample_index") == si
                                               and r.get("response") is not None)))
    print("TOTAL records=%d errors=%d truncated=%d (%.2f%%)"
          % (len(recs), errs, trunc, 100.0 * trunc / max(1, len(recs))))
    if trunc > 0.005 * max(1, len(recs)):
        print("  !! TRUNCATION ABOVE 0.5%% -- raise MAX_TOKENS before trusting any delta")
    return 0


if __name__ == "__main__":
    sys.exit(main())
