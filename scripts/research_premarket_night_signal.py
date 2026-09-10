import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TRAIN=756; TEST=126

def wilson(h,n,z=1.959963984540054):
    if not n:return 0.0
    p=h/n; d=1+z*z/n; return (p+z*z/(2*n)-z*np.sqrt((p*(1-p)+z*z/(4*n))/n))/d

def prepare():
    tw=pd.read_csv(ROOT/"data"/"processed"/"twii_daily.csv",usecols=["date","close"]); tw["date"]=pd.to_datetime(tw.date); tw["cash_return"]=tw.close.pct_change()
    night=pd.read_csv(ROOT/"data"/"processed"/"taiwan_futures_night.csv"); night["signal_date"]=pd.to_datetime(night.signal_date); night["tx_night_return"]=pd.to_numeric(night.tx_night_return,errors="coerce")
    x=night.merge(tw[["date","cash_return"]],left_on="signal_date",right_on="date",how="inner").dropna(subset=["tx_night_return","cash_return"]).sort_values("signal_date").reset_index(drop=True)
    x["prediction"]=np.where(x.tx_night_return>=0,1,-1); x["target"]=np.where(x.cash_return>=0,1,-1); x["confidence"]=x.tx_night_return.abs()
    return x

def select(train):
    best=None
    for q in np.linspace(0,.9,19):
        threshold=float(train.confidence.quantile(q)); use=train.confidence>=threshold; n=int(use.sum())
        if n<50 or n/len(train)<.10:continue
        acc=float((train.loc[use,"prediction"]==train.loc[use,"target"]).mean()); score=acc+.001*min(n,100)
        if best is None or score>best[0]:best=(score,threshold,acc,n)
    return best

def main():
    x=prepare(); records=[]; windows=[]; rows=0; start=0
    while start+TRAIN+TEST<=len(x):
        train=x.iloc[start:start+TRAIN]; test=x.iloc[start+TRAIN:start+TRAIN+TEST]; cfg=select(train); rows+=len(test)
        if cfg:
            _,threshold,tacc,tn=cfg; use=test.confidence>=threshold; hits=test.loc[use,"prediction"]==test.loc[use,"target"]; records+=hits.astype(int).tolist(); windows.append({"test_end":str(test.signal_date.max().date()),"threshold":threshold,"train_accuracy":tacc,"train_cases":tn,"test_cases":int(use.sum()),"test_accuracy":float(hits.mean()) if len(hits) else 0})
        start+=TEST
    n=len(records); hits=sum(records); acc=hits/n if n else 0; cov=n/rows if rows else 0; lower=wilson(hits,n); actual=x.iloc[TRAIN:TRAIN+rows].target; up=float((actual==1).mean()); baseline=max(up,1-up); passed=acc>=.9 and n>=100 and cov>=.1 and lower>=.8 and acc>baseline
    payload={"hypothesis":"The completed Taiwan index-futures night session has causal information for the following cash-session close direction.","method":"Expanding/rolling 756-day train and 126-day test; threshold selected using training only; prediction is the sign of night return.","aligned_days":len(x),"result":{"cases":n,"hits":hits,"accuracy":acc,"coverage":cov,"baseline":baseline,"edge":acc-baseline,"wilson_95_lower":lower,"passed":passed,"windows":windows}}
    out=ROOT/"reports"; (out/"premarket_night_signal.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    r=payload['result']; (out/"premarket_night_signal.md").write_text("\n".join(["# Premarket night-futures signal research","",payload['hypothesis'],payload['method'],"",f"Aligned days: {len(x)}",f"OOS: {hits}/{n} = {acc:.2%}; coverage {cov:.2%}; baseline {baseline:.2%}; edge {acc-baseline:.2%}; Wilson lower {lower:.2%}; passed={passed}."])+"\n",encoding="utf-8")
    print(json.dumps({"aligned_days":len(x),"result":{k:v for k,v in r.items() if k!='windows'}},indent=2))

if __name__=="__main__":main()
