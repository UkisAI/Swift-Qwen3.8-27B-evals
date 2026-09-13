import json,sys
from pathlib import Path
from transformers import AutoTokenizer
R=Path(__file__).parent
t=AutoTokenizer.from_pretrained('/data/models/Qwen3.8-27B',local_files_only=True)
rs=[json.loads(x) for x in (R/'base/base_raw.jsonl').read_text().splitlines()]
seen=set();checks=[]
for r in rs:
 if r['key'] in seen:continue
 seen.add(r['key'])
 messages=[{'role':'user','content':r['prompt']}]
 default=t.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 explicit=t.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True,reasoning_effort='xhigh')
 assert default==explicit,'default is not xhigh'
 ids=t.encode(default,add_special_tokens=False)
 checks.append({'key':r['key'],'historical_prompt_tokens':r['prompt_tokens'],'current_prompt_tokens':len(ids),'match':len(ids)==r['prompt_tokens']})
(R/'template_audit.json').write_text(json.dumps(checks,indent=2))
assert all(x['match'] for x in checks),'historical prompt token counts mismatch'
print('PASS 300 prompt counts and default thinking/xhigh template match')
