import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_premarket_night_signal import TRAIN,TEST,select,wilson

def prepare():
    tw=pd.read_csv(ROOT/"data"/"processed"/"twii_daily.csv",usecols=["date","open","close"]); tw["date"]=pd.to_datetime(tw.date); tw["gap_return"]=pd.to_numeric(tw.open,errors="coerce")/pd.to_numeric(tw.close,errors="coerce").shift(1)-1
    night=pd.read_csv(ROOT/"data"/"processed"/"taiwan_futures_night.csv"); night["signal_date"]=pd.to_datetime(night.signal_date); night["tx_night_return"]=pd.to_numeric(night.tx_night_return,errors="coerce")
    x=night.merge(tw[["date","gap_return"]],left_on="signal_date",right_on="date",how="inner").dropna().sort_values("signal_date").reset_index(drop=True)
    x["prediction"]=np.where(x.tx_night_return>=0,1,-1); x["target"]=np.where(x.gap_return>=0,1,-1); x["confidence"]=x.tx_night_return.abs(); return x

def main():
    x=prepare(); records=[]; windows=[]; rows=0; start=0
    while start+TRAIN+TEST<=len(x):
        train=x.iloc[start:start+TRAIN]; test=x.iloc[start+TRAIN:start+TRAIN+TEST]; cfg=select(train); rows+=len(test)
        if cfg:
            _,threshold,tacc,tn=cfg; use=test.confidence>=threshold; hit=test.loc[use,"prediction"]==test.loc[use,"target"]; records+=hit.astype(int).tolist(); windows.append({"test_end":str(test.signal_date.max().date()),"threshold":threshold,"train_accuracy":tacc,"train_cases":tn,"test_cases":int(use.sum()),"test_accuracy":float(hit.mean()) if len(hit) else 0})
        start+=TEST
    n=len(records); hits=sum(records); acc=hits/n if n else 0; cov=n/rows if rows else 0; lower=wilson(hits,n); actual=x.iloc[TRAIN:TRAIN+rows].target; up=float((actual==1).mean()); baseline=max(up,1-up); passed=acc>=.9 and n>=100 and cov>=.1 and lower>=.8 and acc>baseline
    result={"cases":n,"hits":hits,"accuracy":acc,"coverage":cov,"baseline":baseline,"edge":acc-baseline,"wilson_95_lower":lower,"passed":passed,"windows":windows}
    payload={"scope":"Premarket opening-gap direction only; this is not a cash-close or macro-cycle forecast.","method":"Night-session return sign predicts next cash open versus prior cash close; 756/126 walk-forward threshold selection.","aligned_days":len(x),"result":result}
    out=ROOT/"reports"; (out/"premarket_open_gap.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); (out/"premarket_open_gap.md").write_text("\n".join(["# Premarket opening-gap research","",payload['scope'],payload['method'],"",f"Aligned days: {len(x)}",f"OOS: {hits}/{n} = {acc:.2%}; coverage {cov:.2%}; baseline {baseline:.2%}; edge {acc-baseline:.2%}; Wilson lower {lower:.2%}; passed={passed}."])+"\n",encoding="utf-8")
    print(json.dumps({"aligned_days":len(x),"result":{k:v for k,v in result.items() if k!='windows'}},indent=2))

if __name__=="__main__":main()
