"""Long-history frozen nonlinear model using only night futures and prior US data."""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_afterhours_institutional as common
import research_afterhours_with_complexion as extdef
import research_nonlinear_premarket_state as nonlinear

TRAIN=756;VALID=126;TEST=126
EXTERNAL=extdef.EXTERNAL

def prepare():
    tw=pd.read_csv(ROOT/"data"/"processed"/"twii_daily.csv",usecols=["date","open","close"]);tw["date"]=pd.to_datetime(tw.date);tw["gap_return"]=pd.to_numeric(tw.open)/pd.to_numeric(tw.close).shift(1)-1
    night=pd.read_csv(ROOT/"data"/"processed"/"taiwan_futures_night.csv");night["date"]=pd.to_datetime(night.signal_date);night=night[["date","tx_night_return","tx_night_range","tx_night_volume","tx_night_spread_per"]]
    x=tw.merge(night,on="date",how="inner").sort_values("date")
    external=pd.read_csv(ROOT/"data"/"processed"/"external_markets.csv");external["external_date"]=pd.to_datetime(external.date)
    x=pd.merge_asof(x,external[["external_date"]+EXTERNAL].sort_values("external_date"),left_on="date",right_on="external_date",direction="backward",allow_exact_matches=False,tolerance=pd.Timedelta("4D"))
    votes=np.sign(x[["sp500_return_1d","nasdaq_return_1d","sox_return_1d","dow_return_1d","tsm_adr_return_1d"]].fillna(0));x["baseline"]=np.where(votes.sum(axis=1)>=0,1,-1);x["gap_direction"]=np.where(x.gap_return>=0,1,-1);x["target"]=x.gap_direction
    x["external_lag_days"]=(x.date-x.external_date).dt.days;x["weekday"]=x.date.dt.weekday;x["external_consensus_abs"]=votes.sum(axis=1).abs()/5;x["external_return_mean"]=x[EXTERNAL].mean(axis=1);x["external_return_dispersion"]=x[EXTERNAL].std(axis=1)
    # Causal pulse context: today's completed night session relative to strictly past sessions.
    for c in ["tx_night_return","tx_night_range","tx_night_volume","tx_night_spread_per"]:
        s=pd.to_numeric(x[c],errors="coerce");mean=s.shift(1).rolling(252,min_periods=63).mean();std=s.shift(1).rolling(252,min_periods=63).std().replace(0,np.nan);x[c+"_z"]=((s-mean)/std).clip(-8,8)
    x["night_external_divergence"]=x.tx_night_return_z*x.baseline;x["night_range_external_state"]=x.tx_night_range_z*x.external_consensus_abs
    return x.reset_index(drop=True)

def mcnemar(a,b):
    n=a+b
    if not n:return 1.0
    return min(1.0,2*sum(comb(n,k) for k in range(min(a,b)+1))/(2**n))

def main():
    x=prepare();features=["tx_night_return","tx_night_range","tx_night_volume","tx_night_spread_per","tx_night_return_z","tx_night_range_z","tx_night_volume_z","tx_night_spread_per_z","night_external_divergence","night_range_external_state"]+EXTERNAL+["external_lag_days","weekday","external_consensus_abs","external_return_mean","external_return_dispersion"]
    records=[];windows=[];tested=0;start=0
    while start+TRAIN+TEST<=len(x):
        hist=x.iloc[start:start+TRAIN];fit=hist.iloc[:-VALID];valid=hist.iloc[-VALID:];test=x.iloc[start+TRAIN:start+TRAIN+TEST];tested+=len(test);cfg=nonlinear.select(fit,valid,features)
        if cfg:
            _,model,threshold,rounds,lr,vacc,vbacc,vn=cfg;pred,conf=nonlinear.prediction(model,test,features);use=conf>=threshold;actual=test.gap_direction.to_numpy(int);baseline=test.baseline.to_numpy(int)
            for pos in np.flatnonzero(use):records.append({"date":str(test.iloc[pos].date.date()),"target":int(actual[pos]),"model":int(pred[pos]),"baseline":int(baseline[pos]),"confidence":float(conf[pos])})
            windows.append({"test_start":str(test.date.min().date()),"test_end":str(test.date.max().date()),"cases":int(use.sum()),"model_accuracy":float((pred[use]==actual[use]).mean()) if use.sum() else 0,"baseline_accuracy":float((baseline[use]==actual[use]).mean()) if use.sum() else 0,"rounds":rounds,"learning_rate":lr,"threshold":threshold,"validation_accuracy":vacc,"validation_baseline":vbacc,"validation_cases":vn})
        start+=TEST
    n=len(records);mh=sum(r["model"]==r["target"] for r in records);bh=sum(r["baseline"]==r["target"] for r in records);mo=sum(r["model"]==r["target"] and r["baseline"]!=r["target"] for r in records);bo=sum(r["model"]!=r["target"] and r["baseline"]==r["target"] for r in records);acc=mh/n if n else 0;bacc=bh/n if n else 0;cov=n/tested if tested else 0;lower=common.wilson(mh,n);p=mcnemar(mo,bo);passed=n>=100 and cov>=.1 and acc>=.9 and lower>=.8 and acc>bacc and p<.05
    result={"cases":n,"hits":mh,"accuracy":acc,"coverage":cov,"wilson_95_lower":lower,"baseline_hits":bh,"baseline_accuracy":bacc,"edge":acc-bacc,"model_only_correct":mo,"baseline_only_correct":bo,"mcnemar_exact_p":p,"passed":passed}
    payload={"status":"exploratory; related target and external data were previously inspected","hypothesis":"Night-session pulse and strictly past overseas complexion form a nonlinear premarket state that transfers across post-2017 market cycles.","method":"Only features available since 2017 are used. Each 126-day OOS block freezes a stump ensemble fitted on 630 days and calibrated on the next 126 days. Past-only rolling night standardization; exact-date paired external-majority baseline.","aligned_days":len(x),"tested_days":tested,"features":features,"result":result,"windows":windows,"records":records}
    out=ROOT/"reports";(out/"long_history_night_external.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Long-history night-pulse plus overseas-complexion research","",payload["status"],"",payload["hypothesis"],payload["method"],"","| Cases | Accuracy | Coverage | Wilson lower | Majority baseline | Edge | Model-only / baseline-only | McNemar p | Passed |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",f"| {n} | {acc:.2%} | {cov:.2%} | {lower:.2%} | {bacc:.2%} | {acc-bacc:.2%} | {mo} / {bo} | {p:.4f} | {passed} |","","## Walk-forward windows","","| Test period | Cases | Model | Baseline |","| --- | ---: | ---: | ---: |"]
    for w in windows:lines.append(f"| {w['test_start']} to {w['test_end']} | {w['cases']} | {w['model_accuracy']:.2%} | {w['baseline_accuracy']:.2%} |")
    (out/"long_history_night_external.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps({"aligned_days":len(x),"tested_days":tested,"windows":len(windows),"result":result},indent=2))

if __name__=="__main__":main()
