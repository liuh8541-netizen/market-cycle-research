import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_causal_phase import wilson_lower
from research_three_layer_purged import build_frame,nonoverlap

HORIZONS=[20,60,120]; HISTORY=1512; VALID=252; TEST=252
COMPONENTS=[1,2,3,5,8]; LAMBDAS=[.1,1,10,100]

def pca_fit(frame,features,k):
    raw=frame[features].replace([np.inf,-np.inf],np.nan); med=raw.median(); raw=raw.fillna(med).fillna(0); mean=raw.mean(); std=raw.std().replace(0,1); Z=((raw-mean)/std).to_numpy(float)
    _,_,vt=np.linalg.svd(Z,full_matrices=False); loadings=vt[:min(k,vt.shape[0])].T
    return med,mean,std,loadings

def transform(model,frame,features):
    med,mean,std,loadings=model; raw=frame[features].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0); return ((raw-mean)/std).to_numpy(float)@loadings

def ridge_fit(pca,frame,lam):
    X=np.column_stack([np.ones(len(pca)),pca]); y=frame.target.to_numpy(float); penalty=np.eye(X.shape[1])*lam; penalty[0,0]=0; return np.linalg.pinv(X.T@X+penalty)@X.T@y

def predict(pca,w):
    score=np.column_stack([np.ones(len(pca)),pca])@w; return np.where(score>=0,1,-1),np.abs(score)

def select(history,features,h):
    split=len(history)-VALID; fit_frame=history.iloc[:split-h]; valid=history.iloc[split:len(history)-h]; best=None
    for k in COMPONENTS:
        model=pca_fit(fit_frame,features,k); Xfit=transform(model,fit_frame,features); Xvalid=transform(model,valid,features)
        for lam in LAMBDAS:
            pred,conf=predict(Xvalid,ridge_fit(Xfit,fit_frame,lam)); actual=valid.target.to_numpy(int)
            for q in np.linspace(0,.9,10):
                threshold=float(np.quantile(conf,q)); use=conf>=threshold; n=int(use.sum())
                if n<25 or n/len(valid)<.10: continue
                acc=float((pred[use]==actual[use]).mean()); score=acc+.001*min(n,75)
                if best is None or score>best[0]: best=(score,k,lam,threshold,acc,n)
    return best

def evaluate(base,features,h):
    x=base.copy(); future=x.close.shift(-h)/x.close-1; x["target"]=np.where(future>0,1,-1).astype(float); x.loc[future.isna(),"target"]=np.nan
    records=[]; rows=0; windows=[]; first_vectors=[]; start=0
    while start+HISTORY+TEST+h<=len(x):
        hist=x.iloc[start:start+HISTORY]; test=x.iloc[start+HISTORY:start+HISTORY+TEST]; cfg=select(hist,features,h); rows+=len(test)
        if cfg:
            _,k,lam,threshold,vacc,vn=cfg; train=hist.iloc[:-h]; model=pca_fit(train,features,k); first_vectors.append(model[3][:,0]); Xtrain=transform(model,train,features); Xtest=transform(model,test,features); pred,conf=predict(Xtest,ridge_fit(Xtrain,train,lam)); actual=test.target.to_numpy(); use=(conf>=threshold)&~np.isnan(actual); positions=np.flatnonzero(use); hits=pred[use]==actual[use]
            records += [{"index":int(test.index[p]),"hit":int(pred[p]==actual[p])} for p in positions]
            windows.append({"test_end":str(test.date.max().date()),"components":k,"lambda":lam,"validation_accuracy":vacc,"validation_cases":vn,"test_cases":len(positions),"test_accuracy":float(hits.mean()) if len(hits) else 0})
        start+=TEST
    n=len(records); hits=sum(r['hit'] for r in records); acc=hits/n if n else 0; cov=n/rows if rows else 0; lower=wilson_lower(hits,n); independent=nonoverlap(records,h); en=len(independent); eh=sum(r['hit'] for r in independent); eacc=eh/en if en else 0; elower=wilson_lower(eh,en)
    similarities=[float(abs(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))) for a,b in zip(first_vectors,first_vectors[1:]) if np.linalg.norm(a)>0 and np.linalg.norm(b)>0]; stability=float(np.median(similarities)) if similarities else 0
    actual=x.iloc[HISTORY:HISTORY+rows].target.dropna(); up=float((actual==1).mean()); baseline=max(up,1-up); passed=acc>=.9 and n>=100 and cov>=.1 and lower>=.8 and acc>baseline and en>=100 and elower>=.8
    return {"horizon_days":h,"cases":n,"hits":hits,"accuracy":acc,"coverage":cov,"baseline":baseline,"edge":acc-baseline,"wilson_95_lower":lower,"nonoverlap_cases":en,"nonoverlap_accuracy":eacc,"nonoverlap_wilson_95_lower":elower,"latent_pc1_median_consecutive_cosine":stability,"passed":passed,"windows":windows}

def main():
    base,features=build_frame(); results=[evaluate(base,features,h) for h in HORIZONS]
    descriptive=pca_fit(base,features,1)[3][:,0]
    top_loadings=sorted([{"feature":name,"loading":float(value),"absolute_loading":float(abs(value))} for name,value in zip(features,descriptive)],key=lambda x:x["absolute_loading"],reverse=True)[:12]
    payload={"hypothesis":"A low-dimensional unobserved common state links constitution, pulse and complexion and remains stable across time.","method":"PCA fitted inside each purged training window; component count, ridge penalty and abstention selected on purged validation only; PC1 stability measured across adjacent windows.","descriptive_full_sample_pc1_top_loadings":top_loadings,"results":results,"passing":[r['horizon_days'] for r in results if r['passed']]}
    out=ROOT/"reports"; (out/"latent_factor_purged.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# Purged latent-factor research","",payload['hypothesis'],payload['method'],"","| Horizon | Cases | Accuracy | Coverage | Baseline | Edge | Nonoverlap N/accuracy | PC1 stability | Passed |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results: lines.append(f"| {r['horizon_days']} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['baseline']:.2%} | {r['edge']:.2%} | {r['nonoverlap_cases']} / {r['nonoverlap_accuracy']:.2%} | {r['latent_pc1_median_consecutive_cosine']:.3f} | {r['passed']} |")
    lines += ["", "## Descriptive full-sample PC1 loadings (not predictive evidence)"] + [f"- {item['feature']}: {item['loading']:.4f}" for item in top_loadings]
    (out/"latent_factor_purged.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"passing":payload['passing'],"results":[{k:v for k,v in r.items() if k!='windows'} for r in results]},indent=2))

if __name__=="__main__": main()
