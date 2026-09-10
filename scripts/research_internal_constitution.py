import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_causal_phase import add_phase_features,wilson_lower

HORIZONS=[20,60,120]; TRAIN_DAYS=1260; TEST_DAYS=252

def prepare():
    price=pd.read_csv(ROOT/"data"/"processed"/"twii_daily.csv")
    price["date"]=pd.to_datetime(price["date"])
    price=add_phase_features(price,20,120,12)
    body=pd.read_csv(ROOT/"data"/"processed"/"factors"/"business_indicator.csv")
    body["date"]=pd.to_datetime(body["date"])
    # Conservative point-in-time rule: month M becomes usable only at the end of M+1.
    body["effective_date"]=body["date"]+pd.offsets.MonthEnd(2)
    lead=pd.to_numeric(body["leading_notrend"],errors="coerce")
    coinc=pd.to_numeric(body["coincident_notrend"],errors="coerce")
    monitor=pd.to_numeric(body["monitoring"],errors="coerce")
    lead_delta=lead.diff(3)
    state=pd.Series("NEUTRAL",index=body.index,dtype="object")
    state[(lead>=100)&(coinc>=100)&(lead_delta>0)]="ROBUST"
    state[(lead<100)&(coinc<100)&(lead_delta<0)]="WEAK"
    state[(lead<100)&(lead_delta>0)]="RECOVERING"
    state[(lead>=100)&(lead_delta<0)]="DETERIORATING"
    state[(monitor>=32)&(lead_delta>0)]="OVERHEATED"
    body["constitution"]=state
    body["lead_level"]=lead; body["lead_delta3"]=lead_delta
    x=pd.merge_asof(price.sort_values("date"),body[["effective_date","constitution","lead_level","lead_delta3"]].sort_values("effective_date"),left_on="date",right_on="effective_date",direction="backward")
    x["clinical_state"]=x["state"]+"|"+x["constitution"].fillna("UNKNOWN")
    return x

def state_map(train):
    out={}
    for key,g in train.dropna(subset=["target"]).groupby("clinical_state"):
        if len(g)<30: continue
        c=g.target.value_counts(); purity=c.max()/len(g)
        if purity>=.60: out[key]=int(c.idxmax())
    return out

def evaluate(base,h):
    x=base.copy(); future=x.close.shift(-h)/x.close-1
    x["target"]=np.where(future>0,1,-1).astype(float); x.loc[future.isna(),"target"]=np.nan
    rec=[]; tested=0; start=0
    while start+TRAIN_DAYS+TEST_DAYS+h<=len(x):
        train=x.iloc[start:start+TRAIN_DAYS]; test=x.iloc[start+TRAIN_DAYS:start+TRAIN_DAYS+TEST_DAYS].copy()
        test["prediction"]=test.clinical_state.map(state_map(train)); use=test.prediction.notna()&test.target.notna()
        for r in test.loc[use].itertuples(): rec.append({"constitution":r.constitution,"hit":int(r.prediction==r.target)})
        tested+=len(test); start+=TEST_DAYS
    f=pd.DataFrame(rec); n=len(f); hits=int(f.hit.sum()) if n else 0; acc=hits/n if n else 0; cov=n/tested if tested else 0
    actual=x.iloc[TRAIN_DAYS:TRAIN_DAYS+tested].target.dropna(); up=float((actual==1).mean()) if len(actual) else .5; baseline=max(up,1-up); lower=wilson_lower(hits,n)
    groups={}
    if n:
        for k,g in f.groupby("constitution"):
            gh=int(g.hit.sum()); gn=len(g); groups[str(k)]={"cases":gn,"hits":gh,"accuracy":gh/gn,"wilson_95_lower":wilson_lower(gh,gn)}
    return {"horizon_days":h,"cases":n,"hits":hits,"accuracy":acc,"coverage":cov,"baseline":baseline,"edge":acc-baseline,"wilson_95_lower":lower,"passed":acc>=.9 and n>=100 and cov>=.1 and lower>=.8 and acc>baseline,"by_constitution":groups}

def main():
    results=[evaluate(prepare(),h) for h in HORIZONS]
    payload={"hypothesis":"Published business-cycle indicators represent internal constitution and determine whether a price-cycle disturbance recovers or deteriorates.","point_in_time_rule":"Indicator month M is usable only from the end of month M+1.","method":"Fixed constitution states and yearly walk-forward phase mapping.","results":results,"passing":[r['horizon_days'] for r in results if r['passed']]}
    out=ROOT/"reports"; (out/"internal_constitution_research.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# Internal constitution research","",payload["point_in_time_rule"],"","| Horizon | Cases | Accuracy | Coverage | Baseline | Edge | Wilson lower | Passed |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results: lines.append(f"| {r['horizon_days']} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['baseline']:.2%} | {r['edge']:.2%} | {r['wilson_95_lower']:.2%} | {r['passed']} |")
    for r in results:
        lines += ["",f"## {r['horizon_days']} days by constitution"]+[f"- {k}: {v['accuracy']:.2%} ({v['hits']}/{v['cases']}; lower {v['wilson_95_lower']:.2%})" for k,v in sorted(r['by_constitution'].items())]
    (out/"internal_constitution_research.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"passing":payload['passing'],"results":[{k:v for k,v in r.items() if k!='by_constitution'} for r in results]},indent=2))

if __name__=="__main__": main()
