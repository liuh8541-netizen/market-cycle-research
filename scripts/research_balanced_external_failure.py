"""Balanced causal classifier for rare external-majority failures."""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_long_history_night_external as source
import research_afterhours_institutional as common

TRAIN=756;VALID=126;TEST=126;ROUNDS=[10,25,50];LRS=[.05,.10,.20]

def fit_balanced(frame,features,rounds,lr):
    raw=frame[features].replace([np.inf,-np.inf],np.nan);med=raw.median();X=raw.fillna(med).fillna(0).to_numpy(float);y=frame.failure_target.to_numpy(int);weights=np.zeros(len(y),float)
    for cls in [-1,1]:
        mask=y==cls;weights[mask]=.5/max(int(mask.sum()),1)
    weights/=weights.sum();thresholds=[np.unique(np.quantile(X[:,j],np.linspace(.1,.9,9))) for j in range(X.shape[1])];stumps=[]
    for _ in range(rounds):
        best=None
        for j,values in enumerate(thresholds):
            for t in values:
                base=np.where(X[:,j]>=t,1,-1)
                for polarity in [1,-1]:
                    pred=base*polarity;err=float(weights[pred!=y].sum())
                    if best is None or err<best[0]:best=(err,j,float(t),polarity,pred)
        if best is None:break
        err,j,t,polarity,pred=best;err=min(max(err,1e-8),1-1e-8)
        if err>=.5:break
        alpha=lr*.5*np.log((1-err)/err);weights*=np.exp(-alpha*y*pred);weights/=weights.sum();stumps.append((j,t,polarity,float(alpha)))
    return {"stumps":stumps,"median":med}

def score(model,frame,features):
    X=frame[features].replace([np.inf,-np.inf],np.nan).fillna(model["median"]).fillna(0).to_numpy(float);s=np.zeros(len(frame))
    for j,t,p,a in model["stumps"]:s+=a*np.where(X[:,j]>=t,1,-1)*p
    return s

def select(fit,valid,features):
    best=None;actual=valid.gap_direction.to_numpy(int);baseline=valid.baseline.to_numpy(int)
    for lr in LRS:
        full=fit_balanced(fit,features,max(ROUNDS),lr)
        for rounds in ROUNDS:
            model={"stumps":full["stumps"][:rounds],"median":full["median"]};s=score(model,valid,features)
            for strength in [.2,.6,1.0]:
                eligible=valid.external_consensus_abs.to_numpy()>=strength
                if eligible.sum()<13:continue
                # Negative score means the balanced model expects baseline failure.
                negative=(-s[s<0])
                cuts=np.unique(np.quantile(negative,[0,.25,.5,.75])) if len(negative) else [np.inf]
                for cut in cuts:
                    flip=s<=-float(cut);pred=baseline.copy();pred[flip]*=-1;use=eligible;n=int(use.sum());acc=float((pred[use]==actual[use]).mean());bacc=float((baseline[use]==actual[use]).mean());mo=int(((pred==actual)&(baseline!=actual)&use).sum());bo=int(((pred!=actual)&(baseline==actual)&use).sum());gate=acc>=.9 and acc>bacc
                    rank=(gate,acc,acc-bacc,mo-bo,-int((flip&use).sum()),n)
                    if best is None or rank>best[0]:best=(rank,model,float(cut),strength,rounds,lr,acc,bacc,n,mo,bo)
    return best

def mcnemar(a,b):
    n=a+b
    if not n:return 1.0
    return min(1.0,2*sum(comb(n,k) for k in range(min(a,b)+1))/(2**n))

def main():
    x=source.prepare();x["failure_target"]=np.where(x.gap_direction==x.baseline,1,-1)
    features=["tx_night_return","tx_night_range","tx_night_volume","tx_night_spread_per","tx_night_return_z","tx_night_range_z","tx_night_volume_z","tx_night_spread_per_z","night_external_divergence","night_range_external_state"]+source.EXTERNAL+["external_lag_days","weekday","external_consensus_abs","external_return_mean","external_return_dispersion"]
    records=[];windows=[];tested=0;start=0
    while start+TRAIN+TEST<=len(x):
        hist=x.iloc[start:start+TRAIN];fit=hist.iloc[:-VALID];valid=hist.iloc[-VALID:];test=x.iloc[start+TRAIN:start+TRAIN+TEST];tested+=len(test);cfg=select(fit,valid,features)
        if cfg:
            _,model,cut,strength,rounds,lr,vacc,vbacc,vn,vmo,vbo=cfg;s=score(model,test,features);eligible=test.external_consensus_abs.to_numpy()>=strength;flip=s<=-cut;baseline=test.baseline.to_numpy(int);pred=baseline.copy();pred[flip]*=-1;actual=test.gap_direction.to_numpy(int)
            for pos in np.flatnonzero(eligible):records.append({"date":str(test.iloc[pos].date.date()),"target":int(actual[pos]),"model":int(pred[pos]),"baseline":int(baseline[pos]),"flipped":bool(flip[pos]),"failure_score":float(s[pos])})
            windows.append({"test_start":str(test.date.min().date()),"test_end":str(test.date.max().date()),"cases":int(eligible.sum()),"model_accuracy":float((pred[eligible]==actual[eligible]).mean()),"baseline_accuracy":float((baseline[eligible]==actual[eligible]).mean()),"flips":int((flip&eligible).sum()),"rounds":rounds,"learning_rate":lr,"flip_threshold":cut,"minimum_consensus":strength,"validation_accuracy":vacc,"validation_baseline":vbacc,"validation_cases":vn,"validation_model_only":vmo,"validation_baseline_only":vbo})
        start+=TEST
    n=len(records);mh=sum(r["model"]==r["target"] for r in records);bh=sum(r["baseline"]==r["target"] for r in records);mo=sum(r["model"]==r["target"] and r["baseline"]!=r["target"] for r in records);bo=sum(r["model"]!=r["target"] and r["baseline"]==r["target"] for r in records);acc=mh/n if n else 0;bacc=bh/n if n else 0;cov=n/tested if tested else 0;lower=common.wilson(mh,n);p=mcnemar(mo,bo);passed=n>=100 and cov>=.1 and acc>=.9 and lower>=.8 and acc>bacc and p<.05
    result={"cases":n,"hits":mh,"accuracy":acc,"coverage":cov,"wilson_95_lower":lower,"baseline_hits":bh,"baseline_accuracy":bacc,"edge":acc-bacc,"model_only_correct":mo,"baseline_only_correct":bo,"mcnemar_exact_p":p,"passed":passed}
    payload={"status":"exploratory; target and history previously inspected","hypothesis":"Rare failures of the overseas-majority opening-gap signal are identifiable from night-session divergence and overseas state when failure examples receive balanced training weight.","method":"630-day balanced AdaBoost fit, 126-day chronological rule/threshold calibration, frozen 126-day OOS test. Model normally retains the external baseline and flips only when the learned failure score crosses its past-selected threshold.","aligned_days":len(x),"tested_days":tested,"features":features,"result":result,"windows":windows,"records":records}
    out=ROOT/"reports";(out/"balanced_external_failure.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Balanced external-failure research","",payload["status"],"",payload["hypothesis"],payload["method"],"","| Cases | Accuracy | Coverage | Wilson lower | Majority baseline | Edge | Model-only / baseline-only | McNemar p | Passed |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",f"| {n} | {acc:.2%} | {cov:.2%} | {lower:.2%} | {bacc:.2%} | {acc-bacc:.2%} | {mo} / {bo} | {p:.4f} | {passed} |","","## Walk-forward windows","","| Test period | Cases | Model | Baseline | Flips |","| --- | ---: | ---: | ---: | ---: |"]
    for w in windows:lines.append(f"| {w['test_start']} to {w['test_end']} | {w['cases']} | {w['model_accuracy']:.2%} | {w['baseline_accuracy']:.2%} | {w['flips']} |")
    (out/"balanced_external_failure.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps({"aligned_days":len(x),"tested_days":tested,"windows":len(windows),"result":result},indent=2))

if __name__=="__main__":main()
