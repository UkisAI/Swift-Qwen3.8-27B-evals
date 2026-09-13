import json
from pathlib import Path
R=Path(__file__).parent
j=json.loads((R/"aime2026_xhigh5_aggregate.json").read_text())
for arm in ["base","swift"]: j[arm].pop("rows",None)
for name in ["ALL_RESULTS.txt","ALL_REPORT.txt"]:
 with open("/data/results/"+name,"a") as f:
  f.write("\n\nAIME2026_BF16_BASE_VS_SWIFT0_XHIGH5_20260909 FINAL\n"+json.dumps(j,indent=2)+"\nCONFIG AND RAW: "+str(R)+"\n")
