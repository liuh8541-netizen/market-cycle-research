import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_premarket_night_signal import TRAIN,TEST,select,wilson
from research_premarket_open_gap import prepare

def main():
    x=prepare(); x["prediction"]=np.where(x.tx_night_spread_per>=0,1,-1); x["confidence"]=x.tx_night_spread_per.abs(); records=[]; windows=[]; rows=0; start=0
    while start+TRAIN+TEST<=len(x):
        train=x.iloc[start:start+TRAIN]; test=x.iloc[start+TRAIN:start+TRAIN+TEST]; cfg=select(train); rows+=len(test)
        if cfg:
            _,threshold,tacc,tn=cfg; use=test.confidence>=threshold; hit=test.loc[use,"prediction"]==test.loc[use,"target"]; records+=hit.astype(int).tolist(); windows.append({"test_end":str(test.signal_date.max().date()),"threshold":threshold,"train_accuracy":tacc,"train_cases":tn,"test_cases":int(use.sum()),"test_accuracy":float(hit.mean()) if len(hit) else 0})
        start+=TEST
    n=len(records); hits=sum(records); acc=hits/n if n else 0; cov=n/rows if rows else 0; lower=wilson(hits,n); actual=x.iloc[TRAIN:TRAIN+rows].target; up=float((actual==1).mean()); baseline=max(up,1-up); passed=acc>=.9 and n>=100 and cov>=.1 and lower>=.8 and acc>baseline
    result={"cases":n,"hits":hits,"accuracy":acc,"coverage":cov,"baseline":baseline,"edge":acc-baseline,"wilson_95_lower":lower,"passed":passed,"windows":windows}; payload={"scope":"Opening-gap direction only.","method":"Official same-trading-day attribution; after-market spread_per sign; 756/126 walk-forward threshold selection.","result":result}
    out=ROOT/"reports"; (out/"premarket_spread_gap.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); (out/"premarket_spread_gap.md").write_text("\n".join(["# Night-futures spread to cash opening gap","",payload['method'],"",f"OOS: {hits}/{n} = {acc:.2%}; coverage {cov:.2%}; baseline {baseline:.2%}; edge {acc-baseline:.2%}; Wilson lower {lower:.2%}; passed={passed}."])+"\n",encoding="utf-8"); print(json.dumps({k:v for k,v in result.items() if k!='windows'},indent=2))

if __name__=="__main__":main()
