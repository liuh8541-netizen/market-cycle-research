"""Frozen nonlinear premarket-state model with nested chronological calibration."""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_afterhours_amount_structure as amount
import research_afterhours_institutional as common
import research_afterhours_with_complexion as ext
from research_purged_boosting import fit_stumps, predict as stump_predict

TRAIN=756;VALID=126;TEST=126
PARAMS=[(10,.05),(25,.05),(50,.05),(10,.10),(25,.10),(50,.10),(25,.20)]

def prepare():
    x=amount.prepare().sort_values("date").reset_index(drop=True)
    x["external_lag_days"]=(x.date-x.external_date).dt.days
    x["weekday"]=x.date.dt.weekday
    votes=np.sign(x[["sp500_return_1d","nasdaq_return_1d","sox_return_1d","dow_return_1d","tsm_adr_return_1d"]].fillna(0))
    x["external_consensus_abs"]=votes.sum(axis=1).abs()/5
    x["external_return_mean"]=x[ext.EXTERNAL].mean(axis=1)
    x["external_return_dispersion"]=x[ext.EXTERNAL].std(axis=1)
    x["target"]=x.gap_direction
    return x

def fit_model(frame,features,rounds,lr):
    return fit_stumps(frame,features,rounds,lr)

def prediction(model,frame,features):
    return stump_predict(model,frame,features)

def select(fit,valid,features):
    best=None
    for rounds,lr in PARAMS:
        model=fit_model(fit,features,rounds,lr);pred,conf=prediction(model,valid,features);actual=valid.gap_direction.to_numpy(int);baseline=valid.baseline.to_numpy(int)
        for q in np.linspace(0,.9,10):
            threshold=float(np.quantile(conf,q));use=conf>=threshold;n=int(use.sum())
            if n<13 or n/len(valid)<.10:continue
            acc=float((pred[use]==actual[use]).mean());bacc=float((baseline[use]==actual[use]).mean());score=(acc>=.90 and acc>bacc,acc,acc-bacc,.001*min(n,50),-rounds)
            if best is None or score>best[0]:best=(score,model,threshold,rounds,lr,acc,bacc,n)
    return best

def mcnemar(a,b):
    n=a+b
    if not n:return 1.0
    return min(1.0,2*sum(comb(n,k) for k in range(min(a,b)+1))/(2**n))

def main():
    x=prepare();features=[c for c in x.columns if c.startswith("futures_") or c.startswith("option_")]+["tx_night_return","tx_night_range","tx_night_volume","tx_night_spread_per"]+ext.EXTERNAL+["external_lag_days","weekday","external_consensus_abs","external_return_mean","external_return_dispersion"]
    records=[];windows=[];tested=0;start=0
    while start+TRAIN+TEST<=len(x):
        train=x.iloc[start:start+TRAIN];fit=train.iloc[:-VALID];valid=train.iloc[-VALID:];test=x.iloc[start+TRAIN:start+TRAIN+TEST];tested+=len(test);cfg=select(fit,valid,features)
        if cfg:
            _,model,threshold,rounds,lr,vacc,vbacc,vn=cfg;pred,conf=prediction(model,test,features);use=conf>=threshold;actual=test.gap_direction.to_numpy(int);baseline=test.baseline.to_numpy(int)
            for pos in np.flatnonzero(use):records.append({"date":str(test.iloc[pos].date.date()),"target":int(actual[pos]),"model":int(pred[pos]),"baseline":int(baseline[pos]),"confidence":float(conf[pos])})
            windows.append({"test_end":str(test.date.max().date()),"cases":int(use.sum()),"model_accuracy":float((pred[use]==actual[use]).mean()) if use.sum() else 0,"baseline_accuracy":float((baseline[use]==actual[use]).mean()) if use.sum() else 0,"rounds":rounds,"learning_rate":lr,"threshold":threshold,"validation_accuracy":vacc,"validation_baseline":vbacc,"validation_cases":vn})
        start+=TEST
    n=len(records);mh=sum(r["model"]==r["target"] for r in records);bh=sum(r["baseline"]==r["target"] for r in records);mo=sum(r["model"]==r["target"] and r["baseline"]!=r["target"] for r in records);bo=sum(r["model"]!=r["target"] and r["baseline"]==r["target"] for r in records);acc=mh/n if n else 0;bacc=bh/n if n else 0;cov=n/tested if tested else 0;lower=common.wilson(mh,n);p=mcnemar(mo,bo);passed=n>=100 and cov>=.1 and acc>=.9 and lower>=.8 and acc>bacc and p<.05
    result={"cases":n,"hits":mh,"accuracy":acc,"coverage":cov,"wilson_95_lower":lower,"baseline_hits":bh,"baseline_accuracy":bacc,"edge":acc-bacc,"model_only_correct":mo,"baseline_only_correct":bo,"mcnemar_exact_p":p,"passed":passed}
    payload={"status":"exploratory; historical period already inspected","hypothesis":"A nonlinear combination of overseas disagreement, volatility, institutional amount structure and night-session pulse identifies the opening-gap state.","method":"For every 126-day OOS block: 630 earlier days fit, following 126 days select hyperparameters and absolute confidence threshold with the registered 90%-and-baseline gate prioritized, then freeze both model and threshold for test. Paired external-majority comparison on identical dates.","features":features,"aligned_days":len(x),"tested_days":tested,"result":result,"windows":windows,"records":records}
    out=ROOT/"reports";(out/"nonlinear_premarket_state.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Frozen nonlinear premarket-state research","",payload["status"],"",payload["hypothesis"],payload["method"],"","| Cases | Accuracy | Coverage | Wilson lower | Majority baseline | Edge | Model-only / baseline-only | McNemar p | Passed |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",f"| {n} | {acc:.2%} | {cov:.2%} | {lower:.2%} | {bacc:.2%} | {acc-bacc:.2%} | {mo} / {bo} | {p:.4f} | {passed} |","","## Walk-forward windows","","| Test end | Cases | Model | Baseline | Rounds / learning rate |","| --- | ---: | ---: | ---: | --- |"]
    for w in windows:lines.append(f"| {w['test_end']} | {w['cases']} | {w['model_accuracy']:.2%} | {w['baseline_accuracy']:.2%} | {w['rounds']} / {w['learning_rate']} |")
    (out/"nonlinear_premarket_state.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps({"aligned_days":len(x),"feature_count":len(features),"tested_days":tested,"result":result},indent=2))

if __name__=="__main__":main()
