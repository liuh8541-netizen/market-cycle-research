import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_causal_phase import add_phase_features,wilson_lower
from research_internal_constitution import prepare as prepare_constitution

HORIZONS=[20,60,120]; TRAIN=1260; VALID=252; TEST=252
CONFIGS=[(mc,cf) for mc in [20,40,60] for cf in [.60,.70,.80,.90]]

def mapping(train,min_count,confidence):
    out={}
    for key,g in train.dropna(subset=["target"]).groupby("clinical_state"):
        if len(g)<min_count: continue
        c=g.target.value_counts(); purity=c.max()/len(g)
        if purity>=confidence: out[key]=int(c.idxmax())
    return out

def score(train,test,config):
    predicted=test.clinical_state.map(mapping(train,*config)); use=predicted.notna()&test.target.notna()
    n=int(use.sum()); hits=int((predicted[use].astype(int)==test.loc[use,"target"].astype(int)).sum())
    return hits,n,len(test)

def choose(history):
    fit=history.iloc[:-VALID]; valid=history.iloc[-VALID:]
    candidates=[]
    for cfg in CONFIGS:
        h,n,rows=score(fit,valid,cfg); cov=n/rows
        if cov>=.10 and n>=25: candidates.append((h/n,n,cfg))
    return max(candidates,key=lambda x:(x[0],x[1])) if candidates else None

def evaluate(base,horizon):
    x=base.copy(); future=x.close.shift(-horizon)/x.close-1
    x["target"]=np.where(future>0,1,-1).astype(float); x.loc[future.isna(),"target"]=np.nan
    hits=cases=rows=0; windows=[]; start=0
    while start+TRAIN+VALID+TEST+horizon<=len(x):
        hist=x.iloc[start:start+TRAIN+VALID]; test=x.iloc[start+TRAIN+VALID:start+TRAIN+VALID+TEST]
        selected=choose(hist); rows+=len(test)
        if selected:
            val_acc,val_cases,cfg=selected; wh,wn,_=score(hist,test,cfg); hits+=wh; cases+=wn
            windows.append({"test_end":str(test.date.max().date()),"validation_accuracy":val_acc,"validation_cases":val_cases,"min_count":cfg[0],"confidence":cfg[1],"test_cases":wn,"test_accuracy":wh/wn if wn else 0})
        else: windows.append({"test_end":str(test.date.max().date()),"test_cases":0,"test_accuracy":0})
        start+=TEST
    acc=hits/cases if cases else 0; cov=cases/rows if rows else 0; lower=wilson_lower(hits,cases)
    actual=x.iloc[TRAIN+VALID:TRAIN+VALID+rows].target.dropna(); up=float((actual==1).mean()) if len(actual) else .5; baseline=max(up,1-up)
    return {"horizon_days":horizon,"cases":cases,"hits":hits,"accuracy":acc,"coverage":cov,"baseline":baseline,"edge":acc-baseline,"wilson_95_lower":lower,"passed":acc>=.9 and cases>=100 and cov>=.1 and lower>=.8 and acc>baseline,"windows":windows}

def main():
    base=prepare_constitution(); results=[evaluate(base,h) for h in HORIZONS]
    payload={"method":"Nested walk-forward: each test year uses a prior validation year to select minimum state sample and confidence; constitution publication lag retained.","results":results,"passing":[r['horizon_days'] for r in results if r['passed']]}
    out=ROOT/"reports"; (out/"constitution_selective_research.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# Selective constitution research","",payload['method'],"","| Horizon | Cases | Accuracy | Coverage | Baseline | Edge | Wilson lower | Passed |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results: lines.append(f"| {r['horizon_days']} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['baseline']:.2%} | {r['edge']:.2%} | {r['wilson_95_lower']:.2%} | {r['passed']} |")
    (out/"constitution_selective_research.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"passing":payload['passing'],"results":[{k:v for k,v in r.items() if k!='windows'} for r in results]},indent=2))

if __name__=="__main__": main()
