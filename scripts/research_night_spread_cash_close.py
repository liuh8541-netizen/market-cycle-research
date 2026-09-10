"""Strict OOS test of canonical night spread for same-day cash-close direction."""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_premarket_night_signal import TRAIN,TEST,select,wilson

def exact(a,b):
    n=a+b
    if not n:return 1.0
    return min(1.0,2*sum(comb(n,k) for k in range(min(a,b)+1))/(2**n))

def main():
    tw=pd.read_csv(ROOT/"data"/"processed"/"twii_daily.csv",usecols=["date","close"]);tw["date"]=pd.to_datetime(tw.date);tw["cash_return"]=pd.to_numeric(tw.close).pct_change()
    night=pd.read_csv(ROOT/"data"/"processed"/"taiwan_futures_night.csv");night["signal_date"]=pd.to_datetime(night.signal_date)
    x=night.merge(tw[["date","cash_return"]],left_on="signal_date",right_on="date",how="inner").dropna(subset=["cash_return","tx_night_spread_per","tx_night_return"]).sort_values("signal_date").reset_index(drop=True);x["target"]=np.where(x.cash_return>=0,1,-1);x["prediction"]=np.where(x.tx_night_spread_per>=0,1,-1);x["confidence"]=x.tx_night_spread_per.abs();x["night_return_baseline"]=np.where(x.tx_night_return>=0,1,-1)
    ext=pd.read_csv(ROOT/"data"/"processed"/"external_markets.csv");ext["external_date"]=pd.to_datetime(ext.date);cols=["sp500_return_1d","nasdaq_return_1d","sox_return_1d","dow_return_1d","tsm_adr_return_1d"];x=pd.merge_asof(x,ext[["external_date"]+cols].sort_values("external_date"),left_on="signal_date",right_on="external_date",direction="backward",allow_exact_matches=False,tolerance=pd.Timedelta("4D"));x["external_majority"]=np.where(np.sign(x[cols].fillna(0)).sum(axis=1)>=0,1,-1)
    records=[];windows=[];tested=0;start=0
    while start+TRAIN+TEST<=len(x):
        train=x.iloc[start:start+TRAIN];test=x.iloc[start+TRAIN:start+TRAIN+TEST];tested+=len(test);cfg=select(train)
        if cfg:
            _,threshold,tacc,tn=cfg;use=test.confidence>=threshold
            for pos in np.flatnonzero(use):
                r=test.iloc[pos];records.append({"date":str(r.signal_date.date()),"target":int(r.target),"model":int(r.prediction),"night_return":int(r.night_return_baseline),"external_majority":int(r.external_majority)})
            windows.append({"test_end":str(test.signal_date.max().date()),"threshold":threshold,"cases":int(use.sum()),"accuracy":float((test.loc[use,"prediction"]==test.loc[use,"target"]).mean()) if use.sum() else 0,"training_accuracy":tacc,"training_cases":tn})
        start+=TEST
    n=len(records);hits=sum(r["model"]==r["target"] for r in records);acc=hits/n if n else 0;comparisons={}
    for name in ["night_return","external_majority"]:
        h=sum(r[name]==r["target"] for r in records);mo=sum(r["model"]==r["target"] and r[name]!=r["target"] for r in records);bo=sum(r["model"]!=r["target"] and r[name]==r["target"] for r in records);comparisons[name]={"hits":h,"accuracy":h/n,"model_edge":acc-h/n,"model_only_correct":mo,"baseline_only_correct":bo,"mcnemar_exact_p":exact(mo,bo)}
    strongest=max(comparisons,key=lambda c:comparisons[c]["accuracy"]);s=comparisons[strongest];result={"cases":n,"hits":hits,"accuracy":acc,"coverage":n/tested,"wilson_95_lower":wilson(hits,n),"comparisons":comparisons,"strongest_simple_baseline":strongest,"numeric_gate":n>=100 and n/tested>=.1 and acc>=.9 and wilson(hits,n)>=.8 and acc>s["accuracy"],"significant_gate":acc>s["accuracy"] and s["mcnemar_exact_p"]<.05}
    payload={"status":"corrected canonical night-date strict OOS research","scope":"same-day TWII cash close versus prior cash close","method":"Rolling 756-day threshold selection and frozen 126-day chronological OOS tests; prediction is sign of night spread_per; comparisons on identical selected dates.","result":result,"windows":windows,"records":records};out=ROOT/"reports";(out/"night_spread_cash_close.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    c=comparisons[strongest];lines=["# Night spread to same-day cash-close direction","",payload["status"],payload["scope"],payload["method"],"",f"OOS: {hits}/{n} = {acc:.2%}; coverage {n/tested:.2%}; Wilson lower {wilson(hits,n):.2%}.",f"Strongest identical-date baseline: {strongest} {c['accuracy']:.2%}; edge {c['model_edge']:.2%}; McNemar p={c['mcnemar_exact_p']:.6g}.",f"Numeric gate: {result['numeric_gate']}; significant gate: {result['significant_gate']}."];(out/"night_spread_cash_close.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps(result,indent=2))

if __name__=="__main__":main()
