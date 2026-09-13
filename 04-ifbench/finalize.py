import json,sys,subprocess,hashlib,fcntl
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
R=Path(__file__).parent
DATA=[json.loads(x) for x in (R/'dataset.jsonl').read_text().splitlines() if x.strip()]
keys={str(x['key']):x['prompt'] for x in DATA}
tok=AutoTokenizer.from_pretrained('/data/models/Qwen3.8-27B',local_files_only=True)
def stats(xs):
 return dict(n=len(xs),sum=sum(xs),mean=float(np.mean(xs)),median=float(np.median(xs)),p90=float(np.percentile(xs,90)),max=max(xs)) if xs else None
result={'benchmark':'IFBench','manifest':str(R/'MANIFEST.json'),'repeats':5,'questions':300,'base_reused_seeds':[0,1,2,3],'latency_comparability':'NO: historical DP8 base reused with new DP3'}
for arm in ['base','swift']:
 rs=[json.loads(x) for x in (R/arm/(arm+'_raw.jsonl')).read_text().splitlines() if x.strip()]
 assert len(rs)==1500 and len({(str(x['key']),x['sample_index']) for x in rs})==1500
 assert all(str(x['key']) in keys and x['prompt']==keys[str(x['key'])] and x['sample_index'] in range(5) and x.get('response') is not None and not x.get('error') for x in rs)
 a={'n':len(rs),'empty_final':sum(not x['response'].strip() for x in rs),'truncated':sum(bool(x.get('truncated')) for x in rs),'per_seed':{}}
 for seed in range(5):
  out=R/arm/'score'/('s'+str(seed));out.mkdir(parents=True,exist_ok=True)
  with (out/'eval.log').open('w') as f:
   subprocess.run(['/data/lcb/venv/bin/python','/data/lcb/IFBench/run_eval.py','--input_data='+str(R/'dataset.jsonl'),'--input_response_data='+str(R/arm/f'{arm}_responses_s{seed}.jsonl'),'--output_dir='+str(out)],stdout=f,stderr=subprocess.STDOUT,check=True,cwd='/data/lcb/IFBench')
  a['per_seed'][str(seed)]={}
  for mode in ['strict','loose']:
   paths=list(out.glob('*-eval_results_'+mode+'.jsonl'));assert len(paths)==1
   scored=[json.loads(x) for x in paths[0].read_text().splitlines() if x.strip()];assert len(scored)==300
   a['per_seed'][str(seed)][mode]=sum(bool(x['follow_all_instructions']) for x in scored)
 for mode in ['strict','loose']:
  correct=sum(x[mode] for x in a['per_seed'].values());a[mode]={'correct':correct,'n':1500,'accuracy':correct/1500}
 thinking=[];visible=[];missing=0
 for x in rs:
  if x.get('reasoning_field_present') is False or 'reasoning' not in x:missing+=1
  else:thinking.append(len(tok.encode(x['reasoning'],add_special_tokens=False)))
  visible.append(len(tok.encode(x['response'],add_special_tokens=False)))
 a['thinking_tokens']=stats(thinking);a['missing_reasoning']=missing;a['visible_tokens']=stats(visible)
 a['api_completion_tokens']=stats([x['completion_tokens'] for x in rs]);a['request_seconds_not_matched_latency']=stats([x['seconds'] for x in rs])
 result[arm]=a
(R/'aggregate.json').write_text(json.dumps(result,indent=2))
marker='IFBENCH_BF16_BASE_SWIFT0_K5_81920_20260909 FINAL'
entry='\n\n'+marker+'\n'+json.dumps(result,indent=2)+'\nCONFIG: '+(R/'MANIFEST.json').read_text()+'\n'
for name in ['ALL_RESULTS.txt','ALL_REPORT.txt']:
 with (Path('/data/results')/name).open('a+',encoding='utf-8',errors='replace') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
  if marker not in f.read():f.write(entry);f.flush()
print(json.dumps(result,indent=2))
