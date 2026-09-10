import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_causal_phase import wilson_lower
from research_latent_dynamics_purged import dynamic_matrix,fit_ridge,pred
from research_latent_factor_purged import pca_fit
from research_three_layer_purged import build_frame,nonoverlap

HORIZONS=[20,60,120]; HISTORY=1512; VALID=252; TEST=252; K=3
LOOKBACKS=[126,252,504,756]; LAMBDAS=[.1,1,10,100]

def choose(history,features,h):
    split=len(history)-VALID
    pca_train=history.iloc[:split-h]
    valid=history.iloc[split:len(history)-h]
    model=pca_fit(pca_train,features,K)
    Xvalid=dynamic_matrix(model,pca_train,valid,features)
    best=None
    for lookback in LOOKBACKS:
        map_train=pca_train.tail(lookback)
        Xtrain=dynamic_matrix(model,pca_train.iloc[:0],map_train,features)
        for lam in LAMBDAS:
            prediction,confidence=pred(Xvalid,fit_ridge(Xtrain,map_train.target.to_numpy(float),lam)); actual=valid.target.to_numpy(int)
            for q in np.linspace(0,.9,10):
                threshold=float(np.quantile(confidence,q)); use=confidence>=threshold; n=int(use.sum())
                if n<25 or n/len(valid)<.10: continue
                acc=float((prediction[use]==actual[use]).mean()); score=acc+.001*min(n,75)
                if best is None or score>best[0]: best=(score,lookback,lam,threshold,acc,n)
    return best

def evaluate(base,features,h):
    x=base.copy(); future=x.close.shift(-h)/x.close-1; x["target"]=np.where(future>0,1,-1).astype(float); x.loc[future.isna(),"target"]=np.nan
    records=[]; rows=0; windows=[]; start=0
    while start+HISTORY+TEST+h<=len(x):
        history=x.iloc[start:start+HISTORY]; test=x.iloc[start+HISTORY:start+HISTORY+TEST]; cfg=choose(history,features,h); rows+=len(test)
        if cfg:
            _,lookback,lam,threshold,vacc,vn=cfg
            observable=history.iloc[:-h]; model=pca_fit(observable,features,K); map_train=observable.tail(lookback); Xtrain=dynamic_matrix(model,observable.iloc[:0],map_train,features); Xtest=dynamic_matrix(model,observable,test,features); prediction,confidence=pred(Xtest,fit_ridge(Xtrain,map_train.target.to_numpy(float),lam)); actual=test.target.to_numpy(); use=(confidence>=threshold)&~np.isnan(actual); positions=np.flatnonzero(use); hits=prediction[use]==actual[use]
            records += [{"index":int(test.index[p]),"hit":int(prediction[p]==actual[p])} for p in positions]; windows.append({"test_end":str(test.date.max().date()),"lookback_days":lookback,"lambda":lam,"validation_accuracy":vacc,"validation_cases":vn,"test_cases":len(positions),"test_accuracy":float(hits.mean()) if len(hits) else 0})
        start+=TEST
    n=len(records); hits=sum(r['hit'] for r in records); acc=hits/n if n else 0; cov=n/rows if rows else 0; lower=wilson_lower(hits,n); independent=nonoverlap(records,h); en=len(independent); eh=sum(r['hit'] for r in independent); eacc=eh/en if en else 0; elower=wilson_lower(eh,en)
    actual=x.iloc[HISTORY:HISTORY+rows].target.dropna(); up=float((actual==1).mean()); baseline=max(up,1-up); passed=acc>=.9 and n>=100 and cov>=.1 and lower>=.8 and acc>baseline and en>=100 and elower>=.8
    return {"horizon_days":h,"cases":n,"hits":hits,"accuracy":acc,"coverage":cov,"baseline":baseline,"edge":acc-baseline,"wilson_95_lower":lower,"nonoverlap_cases":en,"nonoverlap_accuracy":eacc,"nonoverlap_wilson_95_lower":elower,"passed":passed,"windows":windows}

def main():
    base,features=build_frame(); results=[evaluate(base,features,h) for h in HORIZONS]
    payload={"hypothesis":"The latent stress axis is stable, but its continuation/reversal coefficient can be learned from the most recent completed regime.","method":"Purged nested selection of 126/252/504/756-day mapping lookback, using three causal PCA factors and their 5/20-day dynamics.","results":results,"passing":[r['horizon_days'] for r in results if r['passed']]}
    out=ROOT/"reports"; (out/"adaptive_latent_regime.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# Adaptive latent-regime research","",payload['hypothesis'],payload['method'],"","| Horizon | Cases | Accuracy | Coverage | Baseline | Edge | Nonoverlap N/accuracy | Passed |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results: lines.append(f"| {r['horizon_days']} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['baseline']:.2%} | {r['edge']:.2%} | {r['nonoverlap_cases']} / {r['nonoverlap_accuracy']:.2%} | {r['passed']} |")
    (out/"adaptive_latent_regime.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"passing":payload['passing'],"results":[{k:v for k,v in r.items() if k!='windows'} for r in results]},indent=2))

if __name__=="__main__": main()
