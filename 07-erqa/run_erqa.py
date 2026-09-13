import json,time,urllib.request,concurrent.futures,statistics,sys,hashlib
from pathlib import Path
from transformers import AutoTokenizer
R=Path(__file__).parent
def save(p,x):
 tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(x,ensure_ascii=False));tmp.replace(p)
tok=AutoTokenizer.from_pretrained("/data/models/Qwen3.8-27B",local_files_only=True)
def ask(arm,seed,idx):
 p=R/arm/f"seed{seed}"/f"{idx}.json";p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): return json.loads(p.read_text())
 data=json.loads((R/"prompts"/f"{idx}.json").read_text())
 body={"model":"qwen38bf" if arm=="base" else "swift","messages":data["messages"],"temperature":1.0,"top_p":.95,"top_k":20,"min_p":0,"seed":seed,"presence_penalty":0,"repetition_penalty":1.0,"max_tokens":100000,"chat_template_kwargs":{"enable_thinking":True,"reasoning_effort":"xhigh"},"mm_processor_kwargs":{"min_pixels":602112,"max_pixels":4014080}}
 port=8586 if arm=="base" else 8587
 for attempt in range(3):
  try:
   t=time.monotonic()
   req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"})
   with urllib.request.urlopen(req,timeout=10800) as f: resp=json.load(f)
   msg=resp["choices"][0]["message"];content=msg.get("content") or ""
   reasoning=msg.get("reasoning_content") or msg.get("reasoning")
   row={"idx":idx,"seed":seed,"arm":arm,"gold":data["gold"],"type":data["type"],"n_images":data["n_images"],"correct":content.replace(".","").strip().lower()==data["gold"].strip().lower(),"seconds":time.monotonic()-t,"response":resp,"thinking_tokens":len(tok.encode(reasoning,add_special_tokens=False)) if reasoning else None,"visible_tokens":len(tok.encode(content,add_special_tokens=False)),"prompt_file":str(R/"prompts"/f"{idx}.json")}
   save(p,row);return row
  except Exception as e:
   with open(R/(arm+"_errors.jsonl"),"a") as f: f.write(json.dumps({"seed":seed,"idx":idx,"attempt":attempt,"error":str(e)})+"\n")
   if attempt==2: raise
   time.sleep(5)
def run(arm,smoke=False):
 jobs=[(0,0),(0,1)] if smoke else [(s,i) for s in range(5) for i in range(400)]
 with concurrent.futures.ThreadPoolExecutor(2 if smoke else 48) as pool:
  futures=[pool.submit(ask,arm,s,i) for s,i in jobs]
  for n,f in enumerate(concurrent.futures.as_completed(futures),1):
   f.result()
   if n%25==0 or n==len(jobs): print(arm,n,"/",len(jobs),flush=True)
def dist(v):
 if not v:return None
 v=sorted(v);k=(len(v)-1)*.9;l=int(k)
 return {"n":len(v),"sum":sum(v),"mean":statistics.fmean(v),"median":statistics.median(v),"p90":v[l]+(v[min(l+1,len(v)-1)]-v[l])*(k-l)}
if __name__=="__main__":
 if len(sys.argv)>1:
  run(sys.argv[1],len(sys.argv)>2);sys.exit()
 result={"benchmark":"ERQA","n_questions":400,"seeds":list(range(5)),"scoring":"official exact-match of final message.content after removing periods, stripping, lowercasing; no LLM judge","config":str(R/"MANIFEST.json")}
 for arm in ["base","swift"]:
  rows=[json.loads((R/arm/f"seed{s}"/f"{i}.json").read_text()) for s in range(5) for i in range(400)]
  assert len(rows)==2000 and len({(x["idx"],x["seed"]) for x in rows})==2000
  result[arm]={"n":2000,"correct":sum(x["correct"] for x in rows),"accuracy":sum(x["correct"] for x in rows)/2000,"per_seed_accuracy":{s:sum(x["correct"] for x in rows if x["seed"]==s)/400 for s in range(5)},"thinking":dist([x["thinking_tokens"] for x in rows if x["thinking_tokens"] is not None]),"missing_separate_reasoning":sum(x["thinking_tokens"] is None for x in rows),"visible_response":dist([x["visible_tokens"] for x in rows]),"api_completion":dist([x["response"]["usage"]["completion_tokens"] for x in rows]),"request_seconds":dist([x["seconds"] for x in rows]),"length_finishes":sum(x["response"]["choices"][0]["finish_reason"]=="length" for x in rows),"empty_content":sum(not(x["response"]["choices"][0]["message"].get("content") or "").strip() for x in rows)}
 save(R/"aggregate.json",result)
 for name in ["ALL_RESULTS.txt","ALL_REPORT.txt"]:
  with open("/data/results/"+name,"a") as f:f.write("\n\nERQA_BF16_BASE_SWIFT0_5X_20260909 FINAL\n"+json.dumps(result,indent=2)+"\n")
 print(json.dumps(result,indent=2))
