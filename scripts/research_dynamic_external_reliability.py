"""Causal dynamic reliability weighting of prior-dated overseas signals."""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_afterhours_with_complexion as extended
import research_afterhours_institutional as common

SIGNALS=["sp500_return_1d","nasdaq_return_1d","sox_return_1d","dow_return_1d","tsm_adr_return_1d","micron_return_1d"]
LOOKBACKS=[63,126,252,504]
TRAIN=756;VALID=126;TEST=126

def prepare():
    x=extended.prepare().sort_values("date").reset_index(drop=True)
    x["target"]=np.where(x.gap_return>=0,1,-1)
    signs=np.sign(x[SIGNALS].fillna(0));x["baseline"]=np.where(signs.sum(axis=1)>=0,1,-1)
    for lookback in LOOKBACKS:
        weighted=np.zeros(len(x));denom=np.zeros(len(x))
        for signal in SIGNALS:
            s=np.sign(pd.to_numeric(x[signal],errors="coerce")).replace(0,np.nan)
            hit=(s==x.target).astype(float).where(s.notna())
            reliability=hit.shift(1).rolling(lookback,min_periods=max(32,lookback//3)).mean()
            # Convert accuracy to signed edge. A historically inverse signal gets a negative weight.
            weight=(2*reliability-1).clip(-.5,.5).fillna(0)
            weighted += s.fillna(0).to_numpy()*weight.to_numpy();denom += weight.abs().to_numpy()
        score=np.divide(weighted,denom,out=np.zeros_like(weighted),where=denom>0)
        x[f"pred_{lookback}"]=np.where(score>=0,1,-1);x[f"confidence_{lookback}"]=np.abs(score)
    return x

def select(train):
    fit=train.iloc[:-VALID];valid=train.iloc[-VALID:];best=None
    for lookback in LOOKBACKS:
        fit_conf=fit[f"confidence_{lookback}"].dropna()
        for q in np.linspace(0,.9,10):
            threshold=float(fit_conf.quantile(q));use=valid[f"confidence_{lookback}"].to_numpy()>=threshold;n=int(use.sum())
            if n<13 or n/len(valid)<.10:continue
            pred=valid[f"pred_{lookback}"].to_numpy(int);actual=valid.target.to_numpy(int);baseline=valid.baseline.to_numpy(int)
            acc=float((pred[use]==actual[use]).mean());bacc=float((baseline[use]==actual[use]).mean());score=(acc-bacc,acc,min(n,50)*.001)
            if best is None or score>best[0]:best=(score,lookback,threshold,acc,bacc,n)
    return best

def mcnemar(a,b):
    n=a+b
    if not n:return 1.0
    return min(1.0,2*sum(comb(n,k) for k in range(min(a,b)+1))/(2**n))

def main():
    x=prepare();records=[];windows=[];tested=0;start=0
    while start+TRAIN+TEST<=len(x):
        train=x.iloc[start:start+TRAIN];test=x.iloc[start+TRAIN:start+TRAIN+TEST];tested+=len(test);cfg=select(train)
        if cfg:
            _,lookback,threshold,vacc,vbacc,vn=cfg;use=test[f"confidence_{lookback}"].to_numpy()>=threshold;pred=test[f"pred_{lookback}"].to_numpy(int)
            for pos in np.flatnonzero(use):records.append({"date":str(test.iloc[pos].date.date()),"target":int(test.iloc[pos].target),"model":int(pred[pos]),"baseline":int(test.iloc[pos].baseline),"lookback":lookback,"confidence":float(test.iloc[pos][f"confidence_{lookback}"])})
            actual=test.target.to_numpy(int);baseline=test.baseline.to_numpy(int);windows.append({"test_end":str(test.date.max().date()),"lookback":lookback,"threshold":threshold,"cases":int(use.sum()),"model_accuracy":float((pred[use]==actual[use]).mean()) if use.sum() else 0,"baseline_accuracy":float((baseline[use]==actual[use]).mean()) if use.sum() else 0,"validation_accuracy":vacc,"validation_baseline":vbacc,"validation_cases":vn})
        start+=TEST
    n=len(records);mh=sum(r["model"]==r["target"] for r in records);bh=sum(r["baseline"]==r["target"] for r in records);mo=sum(r["model"]==r["target"] and r["baseline"]!=r["target"] for r in records);bo=sum(r["model"]!=r["target"] and r["baseline"]==r["target"] for r in records);acc=mh/n if n else 0;bacc=bh/n if n else 0;cov=n/tested if tested else 0;lower=common.wilson(mh,n);p=mcnemar(mo,bo);passed=n>=100 and cov>=.1 and acc>=.9 and lower>=.8 and acc>bacc and p<.05
    result={"cases":n,"hits":mh,"accuracy":acc,"coverage":cov,"wilson_95_lower":lower,"baseline_hits":bh,"baseline_accuracy":bacc,"edge":acc-bacc,"model_only_correct":mo,"baseline_only_correct":bo,"mcnemar_exact_p":p,"passed":passed}
    payload={"status":"exploratory; historical period already inspected","hypothesis":"Overseas complexion is stable, but each market's directional coefficient changes with its strictly past reliability for the TWII opening gap.","method":"Past-only rolling hit-rate weights; nested 756/126/126 walk-forward selects reliability lookback and confidence threshold; paired comparison with unweighted overseas majority on identical OOS dates.","signals":SIGNALS,"lookbacks":LOOKBACKS,"aligned_days":len(x),"tested_days":tested,"result":result,"windows":windows,"records":records}
    out=ROOT/"reports";(out/"dynamic_external_reliability.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Dynamic overseas-reliability research","",payload["status"],"",payload["hypothesis"],payload["method"],"","| Cases | Accuracy | Coverage | Wilson lower | Majority baseline | Edge | Model-only / baseline-only | McNemar p | Passed |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",f"| {n} | {acc:.2%} | {cov:.2%} | {lower:.2%} | {bacc:.2%} | {acc-bacc:.2%} | {mo} / {bo} | {p:.4f} | {passed} |","","## Walk-forward windows","","| Test end | Lookback | Cases | Model | Baseline |","| --- | ---: | ---: | ---: | ---: |"]
    for w in windows:lines.append(f"| {w['test_end']} | {w['lookback']} | {w['cases']} | {w['model_accuracy']:.2%} | {w['baseline_accuracy']:.2%} |")
    (out/"dynamic_external_reliability.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps({"aligned_days":len(x),"tested_days":tested,"result":result},indent=2))

if __name__=="__main__":main()
