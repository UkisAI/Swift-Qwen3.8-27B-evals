import json,time,subprocess,os,fcntl
from pathlib import Path
R=Path(__file__).parent
A=Path("/data/ft/runs/qwen38_bf16_swift_aime2026_xhigh5_20260909")
lock=open(R/"queue.lock","w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
print("WAITING for AIME final success, 150/150 per arm, then GPU2-7 release",flush=True)
while True:
 marker=A/"chain.log";ag=A/"aime2026_xhigh5_aggregate.json"
 if marker.exists() and "FINAL COMPLETE:" in marker.read_text() and ag.exists():
  d=json.loads(ag.read_text())
  assert d["base"]["n"]==d["swift"]["n"]==150
  break
 time.sleep(30)
while True:
 used=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
 if all(int(line.split(",")[1])<1024 for line in used.strip().splitlines() if int(line.split(",")[0]) in range(2,8)):break
 print("AIME complete; waiting for GPU2-7 to be free, no other workload killed",flush=True);time.sleep(30)
print("STARTING ERQA",flush=True)
result=subprocess.run(["bash",str(R/"run.sh")])
print("ERQA EXIT",result.returncode,flush=True)
