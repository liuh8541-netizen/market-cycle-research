import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_causal_phase import wilson_lower

HORIZON=20; LAMBDAS=[.1,1,10,100,1000]
MARKETS={"SP500":"sp500_close","NASDAQ":"nasdaq_close","SOX":"sox_close","DOW":"dow_close"}
FEATURES=["ret5","ret20","ret60","ret240","vol_ratio","drawdown120","pulse_rate","event_panic","event_euphoria","event_breakout"]

def price_features(frame):
    x=frame.copy().sort_values("date").reset_index(drop=True); c=x.close; r=np.log(c).diff()
    x["ret5"]=c.pct_change(5); x["ret20"]=c.pct_change(20); x["ret60"]=c.pct_change(60); x["ret240"]=c.pct_change(240)
    x["vol_ratio"]=r.rolling(20).std()/r.rolling(120).std().replace(0,np.nan)
    x["drawdown120"]=c/c.rolling(120).max()-1
    x["pulse_rate"]=r.gt(0).ne(r.shift().gt(0)).rolling(20).mean()
    x["range20"]=c.rolling(20).max()/c.rolling(20).min()-1
    return x

def make_events(frame,market):
    x=price_features(frame)
    masks={
        "PANIC":(x.ret5<=-.03)&(x.vol_ratio>=1.15),
        "EUPHORIA":(x.ret20>=.10)&(x.vol_ratio>=1.0),
    }
    stagnant=(x.ret20.abs()<=.02)&(x.range20<=.06)&(x.vol_ratio<=.85)
    masks["BREAKOUT"]=stagnant.shift(1).fillna(False)&~stagnant&(x.ret5.abs()>=.02)
    candidates=[]
    for event,mask in masks.items():
        onset=mask&~mask.shift(1).fillna(False)
        for pos in x.index[onset]: candidates.append((pos,event))
    candidates.sort(); kept=[]; last=-10**9
    for pos,event in candidates:
        if pos-last>=HORIZON: kept.append((pos,event)); last=pos
    rows=[]
    for pos,event in kept:
        if pos+HORIZON>=len(x): continue
        row={f:float(x.at[pos,f]) if pd.notna(x.at[pos,f]) else np.nan for f in FEATURES[:7]}
        row.update({"event_panic":int(event=="PANIC"),"event_euphoria":int(event=="EUPHORIA"),"event_breakout":int(event=="BREAKOUT")})
        future=float(x.at[pos+HORIZON,"close"]/x.at[pos,"close"]-1)
        row.update({"market":market,"event":event,"date":x.at[pos,"date"],"outcome_date":x.at[pos+HORIZON,"date"],"target":1 if future>0 else -1,"future_return":future})
        rows.append(row)
    return pd.DataFrame(rows)

def fit(train,lam):
    raw=train[FEATURES].replace([np.inf,-np.inf],np.nan); med=raw.median(); raw=raw.fillna(med).fillna(0); mean=raw.mean(); std=raw.std().replace(0,1)
    X=((raw-mean)/std).to_numpy(float); X=np.column_stack([np.ones(len(X)),X]); y=train.target.to_numpy(float)
    p=np.eye(X.shape[1])*lam; p[0,0]=0; w=np.linalg.pinv(X.T@X+p)@X.T@y
    return w,med,mean,std

def predict(model,frame):
    w,med,mean,std=model; raw=frame[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0); X=((raw-mean)/std).to_numpy(float); X=np.column_stack([np.ones(len(X)),X]); score=X@w
    return np.where(score>=0,1,-1),np.abs(score)

def choose_lambda(train):
    dates=sorted(train.outcome_date.unique()); cut=dates[int(len(dates)*.8)]
    fit_part=train[train.outcome_date<cut]; valid=train[train.outcome_date>=cut]
    if len(fit_part)<50 or len(valid)<20: return 10.0
    best=None
    for lam in LAMBDAS:
        pred,_=predict(fit(fit_part,lam),valid); acc=float((pred==valid.target.to_numpy()).mean())
        if best is None or acc>best[0]: best=(acc,lam)
    return best[1]

def main():
    ext=pd.read_csv(ROOT/"data"/"processed"/"external_markets.csv"); ext["date"]=pd.to_datetime(ext.date)
    panels=[]
    for market,col in MARKETS.items():
        frame=ext[["date",col]].dropna().rename(columns={col:"close"}); panels.append(make_events(frame,market))
    overseas=pd.concat(panels,ignore_index=True)
    tw=pd.read_csv(ROOT/"data"/"processed"/"twii_daily.csv",usecols=["date","close"]); tw["date"]=pd.to_datetime(tw.date); test_events=make_events(tw,"TWII")
    records=[]
    for row in test_events.itertuples(index=False):
        # At event onset, only overseas episodes whose forward outcome has already completed are admissible.
        train=overseas[overseas.outcome_date<row.date]
        # Ten shared features; require at least five completed episodes per feature.
        if len(train)<50: continue
        lam=choose_lambda(train); one=pd.DataFrame([{f:getattr(row,f) for f in FEATURES}]); pred,conf=predict(fit(train,lam),one)
        records.append({"date":str(row.date.date()),"event":row.event,"training_episodes":len(train),"lambda":lam,"prediction":int(pred[0]),"target":int(row.target),"hit":int(pred[0]==row.target),"confidence":float(conf[0]),"future_return":float(row.future_return)})
    n=len(records); hits=sum(r['hit'] for r in records); acc=hits/n if n else 0; lower=wilson_lower(hits,n); coverage=n/len(test_events) if len(test_events) else 0
    targets=pd.Series([int(r.target) for r in test_events.itertuples()]); up=float((targets==1).mean()); baseline=max(up,1-up)
    passed=acc>=.9 and n>=100 and coverage>=.1 and lower>=.8 and acc>baseline
    payload={"method":"For each non-overlapping TWII event, fit only completed prior overseas index episodes; nested temporal lambda selection; no contemporaneous/future overseas outcomes.","overseas_episodes":{m:int((overseas.market==m).sum()) for m in MARKETS},"twii_events":len(test_events),"result":{"cases":n,"hits":hits,"accuracy":acc,"coverage":coverage,"baseline":baseline,"edge":acc-baseline,"wilson_95_lower":lower,"passed":passed},"predictions":records}
    out=ROOT/"reports"; (out/"cross_market_event_transfer.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    r=payload['result']; lines=["# Cross-market event transfer","",payload['method'],"",f"Overseas training episodes: {payload['overseas_episodes']}",f"TWII independent events: {len(test_events)}",f"OOS result: {r['hits']}/{r['cases']} = {r['accuracy']:.2%}; coverage {r['coverage']:.2%}; baseline {r['baseline']:.2%}; edge {r['edge']:.2%}; Wilson lower {r['wilson_95_lower']:.2%}; passed={r['passed']}." ]
    (out/"cross_market_event_transfer.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in payload.items() if k!='predictions'},indent=2))

if __name__=="__main__": main()
