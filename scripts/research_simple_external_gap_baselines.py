import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_afterhours_institutional as study
import research_afterhours_with_complexion as extended

SIGNALS=["sp500_return_1d","nasdaq_return_1d","sox_return_1d","dow_return_1d","tsm_adr_return_1d","micron_return_1d"]

def select(train,signal):
    valid=train.dropna(subset=[signal]);best=None
    for q in np.linspace(0,.9,10):
        threshold=float(valid[signal].abs().quantile(q));use=valid[signal].abs()>=threshold;n=int(use.sum())
        if n<13 or n/len(valid)<.10:continue
        pred=np.where(valid.loc[use,signal]>=0,1,-1);acc=float((pred==valid.loc[use,"target"].to_numpy()).mean());score=acc+.001*min(n,50)
        if best is None or score>best[0]:best=(score,threshold,acc,n)
    return best

def evaluate(data,signal):
    records=[];rows=0;start=0
    while start+study.TRAIN+study.TEST<=len(data):
        train=data.iloc[start:start+study.TRAIN];test=data.iloc[start+study.TRAIN:start+study.TRAIN+study.TEST];cfg=select(train,signal);rows+=len(test)
        if cfg:
            _,threshold,_,_=cfg;use=test[signal].notna()&(test[signal].abs()>=threshold);pred=np.where(test.loc[use,signal]>=0,1,-1);records+=(pred==test.loc[use,"target"].to_numpy()).astype(int).tolist()
        start+=study.TEST
    n=len(records);hits=sum(records);acc=hits/n if n else 0;cov=n/rows if rows else 0;lower=study.wilson(hits,n);return {"signal":signal,"cases":n,"hits":hits,"accuracy":acc,"coverage":cov,"wilson_95_lower":lower,"passes_numeric_90_gate":acc>=.9 and n>=100 and cov>=.1 and lower>=.8}

def main():
    data=extended.prepare();data["target"]=np.where(data.gap_return>=0,1,-1);results=[evaluate(data,s) for s in SIGNALS]
    payload={"method":"Each simple prior-US-market sign selects its own absolute-return threshold on the preceding 756 aligned days, then predicts the next 126 days.","results":results}
    out=ROOT/"reports";(out/"simple_external_gap_baselines.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# Simple external opening-gap baselines","",payload['method'],"","| Signal | Cases | Accuracy | Coverage | Wilson lower | Numeric 90 gate |","| --- | ---: | ---: | ---: | ---: | --- |"]
    for r in results:lines.append(f"| {r['signal']} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['wilson_95_lower']:.2%} | {r['passes_numeric_90_gate']} |")
    (out/"simple_external_gap_baselines.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps(results,indent=2))

if __name__=="__main__":main()
