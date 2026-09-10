import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_premarket_night_signal import TRAIN,TEST,select,wilson
from research_premarket_open_gap import prepare
import research_afterhours_with_complexion as extdef

def exact(a,b):
    n=a+b
    if not n:return 1.0
    return min(1.0,2*sum(comb(n,k) for k in range(min(a,b)+1))/(2**n))

def evaluate(x,signal):
    y=x.copy();y["prediction"]=np.where(y[signal]>=0,1,-1);y["confidence"]=y[signal].abs();records=[];tested=0;start=0
    while start+TRAIN+TEST<=len(y):
        train=y.iloc[start:start+TRAIN];test=y.iloc[start+TRAIN:start+TRAIN+TEST];tested+=len(test);cfg=select(train)
        if cfg:
            _,threshold,tacc,tn=cfg;use=test.confidence>=threshold
            for pos in np.flatnonzero(use):
                r=test.iloc[pos];records.append({"date":str(r.signal_date.date()),"target":int(r.target),"model":int(r.prediction),"night_return":1 if r.tx_night_return>=0 else -1,"night_spread":1 if r.tx_night_spread_per>=0 else -1,"external_majority":int(r.external_majority)})
        start+=TEST
    n=len(records);mh=sum(r["model"]==r["target"] for r in records);comparisons={}
    competitors=[c for c in ["night_return","night_spread","external_majority"] if c!={"tx_night_return":"night_return","tx_night_spread_per":"night_spread"}[signal]]
    for c in competitors:
        h=sum(r[c]==r["target"] for r in records);mo=sum(r["model"]==r["target"] and r[c]!=r["target"] for r in records);bo=sum(r["model"]!=r["target"] and r[c]==r["target"] for r in records);comparisons[c]={"hits":h,"accuracy":h/n,"model_edge":mh/n-h/n,"model_only_correct":mo,"competitor_only_correct":bo,"mcnemar_exact_p":exact(mo,bo)}
    strongest=max(comparisons,key=lambda c:comparisons[c]["accuracy"]);s=comparisons[strongest];acc=mh/n;return {"signal":signal,"cases":n,"hits":mh,"accuracy":acc,"coverage":n/tested,"wilson_95_lower":wilson(mh,n),"comparisons":comparisons,"strongest_competitor":strongest,"numeric_gate":n>=100 and n/tested>=.1 and acc>=.9 and wilson(mh,n)>=.8 and acc>s["accuracy"],"significant_strong_baseline_gate":acc>s["accuracy"] and s["mcnemar_exact_p"]<.05,"records":records}

def main():
    x=prepare();external=pd.read_csv(ROOT/"data"/"processed"/"external_markets.csv");external["external_date"]=pd.to_datetime(external.date);signals=["sp500_return_1d","nasdaq_return_1d","sox_return_1d","dow_return_1d","tsm_adr_return_1d"]
    x=pd.merge_asof(x.sort_values("signal_date"),external[["external_date"]+signals].sort_values("external_date"),left_on="signal_date",right_on="external_date",direction="backward",allow_exact_matches=False,tolerance=pd.Timedelta("4D"));votes=np.sign(x[signals].fillna(0)).sum(axis=1);x["external_majority"]=np.where(votes>=0,1,-1)
    results=[evaluate(x,"tx_night_return"),evaluate(x,"tx_night_spread_per")];payload={"status":"formal identical-date strong-baseline audit after canonical FinMind trading-day rebuild","method":"Each simple night signal retains its own causal rolling 756/126 threshold selection; comparisons use exactly the dates selected by that signal.","results":results};out=ROOT/"reports";(out/"simple_night_gap_baseline_audit.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Simple night-gap strong-baseline audit","",payload["status"],payload["method"],"","| Model signal | Cases | Accuracy | Coverage | Wilson lower | Strongest competitor | Competitor accuracy | Edge | McNemar p | Numeric gate | Significant gate |","| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |"]
    for r in results:
        c=r["comparisons"][r["strongest_competitor"]];lines.append(f"| {r['signal']} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['wilson_95_lower']:.2%} | {r['strongest_competitor']} | {c['accuracy']:.2%} | {c['model_edge']:.2%} | {c['mcnemar_exact_p']:.6g} | {r['numeric_gate']} | {r['significant_strong_baseline_gate']} |")
    (out/"simple_night_gap_baseline_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps([{k:v for k,v in r.items() if k!="records"} for r in results],indent=2))

if __name__=="__main__":main()
