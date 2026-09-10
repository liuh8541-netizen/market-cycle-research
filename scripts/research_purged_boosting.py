import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_causal_phase import wilson_lower
from research_three_layer_purged import build_frame,nonoverlap

HORIZONS=[20,60,120]; MATERIAL={20:.03,60:.06,120:.10}
HISTORY=1512; VALID=252; TEST=252; ROUNDS=[10,25,50]; LRS=[.05,.10,.20]

def matrix_fit(frame,features):
    raw=frame[features].replace([np.inf,-np.inf],np.nan); med=raw.median(); raw=raw.fillna(med).fillna(0)
    return raw.to_numpy(float),med

def matrix_apply(frame,features,med):
    return frame[features].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0).to_numpy(float)

def fit_stumps(frame,features,rounds,lr):
    labeled=frame[frame.target!=0]
    X,med=matrix_fit(labeled,features); y=labeled.target.to_numpy(int); n=len(y); weights=np.ones(n)/max(n,1); stumps=[]
    thresholds=[np.unique(np.quantile(X[:,j],np.linspace(.1,.9,9))) for j in range(X.shape[1])]
    for _ in range(rounds):
        best=None
        for j,values in enumerate(thresholds):
            for t in values:
                base=np.where(X[:,j]>=t,1,-1)
                for polarity in [1,-1]:
                    pred=base*polarity; err=float(weights[pred!=y].sum())
                    if best is None or err<best[0]: best=(err,j,float(t),polarity,pred)
        if best is None: break
        err,j,t,polarity,pred=best; err=min(max(err,1e-8),1-1e-8)
        if err>=.5: break
        alpha=lr*.5*np.log((1-err)/err); weights*=np.exp(-alpha*y*pred); weights/=weights.sum()
        stumps.append((j,t,polarity,float(alpha)))
    return {"stumps":stumps,"median":med}

def predict(model,frame,features):
    X=matrix_apply(frame,features,model["median"]); score=np.zeros(len(frame))
    for j,t,p,a in model["stumps"]: score+=a*np.where(X[:,j]>=t,1,-1)*p
    return np.where(score>=0,1,-1),np.abs(score)

def choose(history,features,h):
    split=len(history)-VALID; fit=history.iloc[:split-h]; valid=history.iloc[split:len(history)-h]; best=None
    for lr in LRS:
        full=fit_stumps(fit,features,max(ROUNDS),lr)
        for rounds in ROUNDS:
            model={"stumps":full["stumps"][:rounds],"median":full["median"]}; pred,conf=predict(model,valid,features); target=valid.target.to_numpy(int)
            for q in np.linspace(0,.9,10):
                threshold=float(np.quantile(conf,q)); use=conf>=threshold; n=int(use.sum())
                if n<25 or n/len(valid)<.10: continue
                acc=float((pred[use]==target[use]).mean()); score=acc+.001*min(n/len(valid),.3)
                if best is None or score>best[0]: best=(score,lr,rounds,threshold,acc,n)
    return best

def evaluate(base,features,h):
    x=base.copy(); future=x.close.shift(-h)/x.close-1; neutral=MATERIAL[h]
    x["target"]=np.where(future>neutral,1,np.where(future<-neutral,-1,0)); x.loc[future.isna(),"target"]=np.nan
    records=[]; rows=0; windows=[]; start=0
    while start+HISTORY+TEST+h<=len(x):
        hist=x.iloc[start:start+HISTORY]; test=x.iloc[start+HISTORY:start+HISTORY+TEST]; cfg=choose(hist,features,h); rows+=len(test)
        if cfg:
            _,lr,rounds,threshold,vacc,vn=cfg; train=hist.iloc[:-h]; model=fit_stumps(train,features,rounds,lr); pred,conf=predict(model,test,features); actual=test.target.to_numpy(); use=(conf>=threshold)&~np.isnan(actual)
            pos=np.flatnonzero(use); wh=int((pred[use]==actual[use]).sum()); wn=len(pos)
            records += [{"index":int(test.index[p]),"hit":int(pred[p]==actual[p])} for p in pos]
            windows.append({"test_end":str(test.date.max().date()),"lr":lr,"rounds":rounds,"threshold":threshold,"validation_accuracy":vacc,"validation_cases":vn,"test_cases":wn,"test_accuracy":wh/wn if wn else 0})
        start+=TEST
    n=len(records); hits=sum(r['hit'] for r in records); acc=hits/n if n else 0; cov=n/rows if rows else 0; lower=wilson_lower(hits,n)
    effective=nonoverlap(records,h); en=len(effective); eh=sum(r['hit'] for r in effective); eacc=eh/en if en else 0; elower=wilson_lower(eh,en)
    actual=x.iloc[HISTORY:HISTORY+rows].target.dropna(); counts=actual.value_counts(normalize=True); baseline=float(counts.max()) if len(counts) else 0
    passed=acc>=.9 and n>=100 and cov>=.1 and lower>=.8 and acc>baseline and en>=100 and elower>=.8
    return {"horizon_days":h,"material_move":neutral,"cases":n,"hits":hits,"accuracy":acc,"coverage":cov,"baseline":baseline,"edge":acc-baseline,"wilson_95_lower":lower,"nonoverlap_cases":en,"nonoverlap_accuracy":eacc,"nonoverlap_wilson_95_lower":elower,"passed":passed,"windows":windows}

def main():
    base,features=build_frame(); results=[evaluate(base,features,h) for h in HORIZONS]
    payload={"method":"Purged nested AdaBoost stumps on continuous three-layer interactions; moves smaller than the horizon material threshold count as wrong when a direction is issued.","results":results,"passing":[r['horizon_days'] for r in results if r['passed']]}
    out=ROOT/"reports"; (out/"purged_boosting_research.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# Purged nonlinear boosting research","",payload['method'],"","| Horizon | Material move | Cases | Accuracy | Coverage | Baseline | Edge | Wilson | Nonoverlap N | Nonoverlap accuracy | Nonoverlap Wilson | Passed |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results: lines.append(f"| {r['horizon_days']} | {r['material_move']:.1%} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['baseline']:.2%} | {r['edge']:.2%} | {r['wilson_95_lower']:.2%} | {r['nonoverlap_cases']} | {r['nonoverlap_accuracy']:.2%} | {r['nonoverlap_wilson_95_lower']:.2%} | {r['passed']} |")
    (out/"purged_boosting_research.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"passing":payload['passing'],"results":[{k:v for k,v in r.items() if k!='windows'} for r in results]},indent=2))

if __name__=="__main__": main()
