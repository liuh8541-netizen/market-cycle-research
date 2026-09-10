import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_causal_phase import wilson_lower
from research_cross_market_event_transfer import FEATURES,HORIZON,MARKETS,make_events

KS=[5,11,21,41]

def prep(train,query):
    raw=train[FEATURES].replace([np.inf,-np.inf],np.nan); med=raw.median(); raw=raw.fillna(med).fillna(0); mean=raw.mean(); std=raw.std().replace(0,1)
    X=((raw-mean)/std).to_numpy(float); q=((query[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0)-mean)/std).to_numpy(float)
    return X,q

def knn_predict(train,query,k):
    X,q=prep(train,query); out=[]; confidence=[]; labels=train.target.to_numpy(int)
    for row in q:
        nearest=np.argsort(np.sum((X-row)**2,axis=1))[:min(k,len(X))]; vote=float(labels[nearest].mean()); out.append(1 if vote>=0 else -1); confidence.append(abs(vote))
    return np.array(out),np.array(confidence)

def choose_k(train):
    dates=sorted(train.outcome_date.unique()); cut=dates[int(len(dates)*.8)]; fit=train[train.outcome_date<cut]; valid=train[train.outcome_date>=cut]
    if len(fit)<40 or len(valid)<15: return 11
    best=None
    for k in KS:
        pred,_=knn_predict(fit,valid,k); acc=float((pred==valid.target.to_numpy()).mean())
        if best is None or acc>best[0]: best=(acc,k)
    return best[1]

def main():
    ext=pd.read_csv(ROOT/"data"/"processed"/"external_markets.csv"); ext["date"]=pd.to_datetime(ext.date)
    overseas=pd.concat([make_events(ext[["date",col]].dropna().rename(columns={col:"close"}),market) for market,col in MARKETS.items()],ignore_index=True)
    tw=pd.read_csv(ROOT/"data"/"processed"/"twii_daily.csv",usecols=["date","close"]); tw["date"]=pd.to_datetime(tw.date); tests=make_events(tw,"TWII")
    records=[]
    for row in tests.itertuples(index=False):
        train=overseas[overseas.outcome_date<row.date]
        if len(train)<50: continue
        k=choose_k(train); one=pd.DataFrame([{f:getattr(row,f) for f in FEATURES}]); pred,conf=knn_predict(train,one,k)
        records.append({"date":str(row.date.date()),"event":row.event,"training_episodes":len(train),"k":k,"prediction":int(pred[0]),"target":int(row.target),"hit":int(pred[0]==row.target),"confidence":float(conf[0])})
    n=len(records); hits=sum(r['hit'] for r in records); acc=hits/n if n else 0; coverage=n/len(tests); lower=wilson_lower(hits,n); up=float((tests.target==1).mean()); baseline=max(up,1-up); passed=acc>=.9 and n>=100 and coverage>=.1 and lower>=.8 and acc>baseline
    payload={"method":"Causal cross-market case matching; k selected using only the latest completed 20% of prior overseas episodes; TWII test episodes non-overlapping.","twii_events":len(tests),"result":{"cases":n,"hits":hits,"accuracy":acc,"coverage":coverage,"baseline":baseline,"edge":acc-baseline,"wilson_95_lower":lower,"passed":passed},"predictions":records}
    out=ROOT/"reports"; (out/"cross_market_event_knn.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    r=payload['result']; (out/"cross_market_event_knn.md").write_text("\n".join(["# Cross-market event nearest-neighbor research","",payload['method'],"",f"TWII events: {len(tests)}",f"OOS result: {r['hits']}/{r['cases']} = {r['accuracy']:.2%}; coverage {r['coverage']:.2%}; baseline {r['baseline']:.2%}; edge {r['edge']:.2%}; Wilson lower {r['wilson_95_lower']:.2%}; passed={r['passed']}."])+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in payload.items() if k!='predictions'},indent=2))

if __name__=="__main__": main()
