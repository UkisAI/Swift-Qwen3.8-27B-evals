import json,hashlib,io,base64,subprocess
from pathlib import Path
from PIL import Image
from tfrecord.reader import tfrecord_iterator
from tfrecord import example_pb2
R=Path("/data/mm_eval/erqa_bf16_t20_5x_20260909")
R.mkdir(exist_ok=True);(R/"prompts").mkdir(exist_ok=True)
rows=[]
for i,raw in enumerate(tfrecord_iterator("/data/mm_eval/ERQA/data/erqa.tfrecord")):
 e=example_pb2.Example();e.ParseFromString(bytes(raw));f=e.features.feature
 q=f["question"].bytes_list.value[0].decode();ans=f["answer"].bytes_list.value[0].decode()
 inds=list(f["visual_indices"].int64_list.value);images=list(f["image/encoded"].bytes_list.value)
 assert not inds or len(inds)==len(images)
 imgs=[];hashes=[]
 for b in images:
  im=Image.open(io.BytesIO(b));im.load();buf=io.BytesIO();im.save(buf,format="PNG")
  hashes.append(hashlib.sha256(b).hexdigest())
  imgs.append({"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()}})
 text=lambda s: {"type":"text","text":s}
 pairs=sorted(zip(imgs,inds),key=lambda x:x[1]);content=[]
 if not inds: content=imgs+[text(q)]
 elif all(x==0 for x in inds): content=[im for im,_ in pairs]+[text(q)]
 else:
  last=0
  for im,idx in pairs:
   if idx==0: content.append(im)
   elif idx<=len(q):
    if q[last:idx]: content.append(text(q[last:idx]))
    content.append(im);last=idx
   else: content.append(im)
  if last<len(q): content.append(text(q[last:]))
  if not content: content=[text(q)]+[im for im,_ in pairs]
 assert "".join(x["text"] for x in content if x["type"]=="text")==q
 assert sum(x["type"]=="image_url" for x in content)==len(images)
 assert ans in "ABCD" and len(ans)==1
 p={"idx":i,"gold":ans,"type":[v.decode() for v in f["question_type"].bytes_list.value],"n_images":len(images),"image_sha256":hashes,"visual_indices":inds,"messages":[{"role":"user","content":content}]}
 (R/"prompts"/f"{i}.json").write_text(json.dumps(p))
 rows.append({k:v for k,v in p.items() if k!="messages"})
assert len(rows)==400
manifest={"n":400,"seeds":[0,1,2,3,4],"base":"/data/models/Qwen3.8-27B","adapter":"/data/models/ukisai/Swift-Qwen3.8-27B-adapter","repo":"https://github.com/embodiedreasoning/ERQA","commit":subprocess.check_output(["git","-C","/data/mm_eval/ERQA","rev-parse","HEAD"],text=True).strip(),"dataset_sha256":hashlib.sha256(Path("/data/mm_eval/ERQA/data/erqa.tfrecord").read_bytes()).hexdigest(),"adapter_sha256":hashlib.sha256(Path("/data/models/ukisai/Swift-Qwen3.8-27B-adapter/adapter_model.safetensors").read_bytes()).hexdigest(),"max_images":max(x["n_images"] for x in rows),"rows":rows,"profile":"BF16 TP1 DP3 per arm; T1 top_p.95 top_k20 min_p0 xhigh cap100000 context262144; concurrency48 per arm; no tools/plugin; official text/image interleaving and exact-match scorer; transport, seed, temperature, cap, persistence adapted for vLLM, not exact Qwen published harness","mm_processor_kwargs":{"min_pixels":602112,"max_pixels":4014080}}
(R/"MANIFEST.json").write_text(json.dumps(manifest,indent=2))
print("PREP PASS",len(rows),"max_images",manifest["max_images"])
