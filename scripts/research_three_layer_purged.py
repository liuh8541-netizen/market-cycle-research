import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts")); sys.path.insert(0,str(ROOT/"src"))
from market_lifecycle.factor_features import add_factor_features
from research_causal_phase import wilson_lower
from research_internal_constitution import prepare as prepare_constitution

HORIZONS=[20,60,120]; HISTORY=1512; VALID=252; TEST=252
LAMBDAS=[.1,1.,10.,100.,1000.]

def causal_z(s,w=252):
    return (s-s.rolling(w,min_periods=80).mean().shift(1))/s.rolling(w,min_periods=80).std().shift(1).replace(0,np.nan)

def build_frame():
    x=prepare_constitution()
    x=add_factor_features(x,str(ROOT/"data"/"processed"/"factors"))
    close=x.close; logp=np.log(close); r=logp.diff()
    for d in [5,20,60,120,240]: x[f"ret_{d}"]=close.pct_change(d)
    x["vol_ratio"]=r.rolling(20).std()/r.rolling(120).std().replace(0,np.nan)
    x["drawdown_120"]=close/close.rolling(120).max()-1
    x["pulse_rate"]=r.gt(0).ne(r.shift().gt(0)).rolling(20).mean()
    x["body_strength"]=(x.lead_level-100)/10
    x["body_momentum"]=x.lead_delta3/3
    ext=pd.read_csv(ROOT/"data"/"processed"/"external_markets.csv"); ext["date"]=pd.to_datetime(ext.date)
    color=[]
    for col in ["sp500_close","nasdaq_close","sox_close","dow_close","tsm_adr_close","micron_close"]:
        v=pd.to_numeric(ext[col],errors="coerce")
        ext[col+"_ret20_lag1"]=v.pct_change(20).shift(1); color.append(ext[col+"_ret20_lag1"])
    ext["color_breadth"]=pd.concat(color,axis=1).mean(axis=1)
    ext["vix_pressure"]=causal_z(pd.to_numeric(ext.vix_close,errors="coerce")).shift(1)
    cols=["date","color_breadth","vix_pressure"]+[c+"_ret20_lag1" for c in ["sp500_close","nasdaq_close","sox_close","dow_close","tsm_adr_close","micron_close"]]
    x=pd.merge_asof(x.sort_values("date"),ext[cols].sort_values("date"),on="date",direction="backward",tolerance=pd.Timedelta("5D"))
    factor_cols=[
        "inst_net_20d","inst_foreign_net_20d","inst_trust_net_20d","inst_dealer_net_20d",
        "margin_change_20d","futures_inst_net_20d","futures_foreign_net_20d",
        "futures_trust_net_20d","futures_dealer_net_20d","option_put_call_20d_z",
        "option_inst_net_20d","option_foreign_net_20d","option_trust_net_20d",
        "option_dealer_net_20d","option_vix_20d_z"
    ]
    for c in factor_cols:
        if c not in x: x[c]=np.nan
        x[c+"_z"]=causal_z(pd.to_numeric(x[c],errors="coerce"))
    x["pulse_severity"]=(x.vol_ratio-1)+x.vix_pressure.clip(lower=0).fillna(0)
    x["external_pressure"]=-x.color_breadth+x.vix_pressure.fillna(0)*.1
    x["body_x_pulse"]=x.body_strength*x.pulse_severity
    x["body_x_external"]=x.body_strength*x.external_pressure
    x["pulse_x_external"]=x.pulse_severity*x.external_pressure
    features=[f"ret_{d}" for d in [5,20,60,120,240]]+["vol_ratio","drawdown_120","pulse_rate","body_strength","body_momentum","color_breadth","vix_pressure","pulse_severity","external_pressure","body_x_pulse","body_x_external","pulse_x_external"]+[c+"_z" for c in factor_cols]
    return x,features

def fit_model(frame,features,lam):
    raw=frame[features].replace([np.inf,-np.inf],np.nan)
    med=raw.median(); raw=raw.fillna(med).fillna(0); mean=raw.mean(); std=raw.std().replace(0,1)
    X=((raw-mean)/std).to_numpy(float); X=np.column_stack([np.ones(len(X)),X]); y=frame.target.to_numpy(float)
    penalty=np.eye(X.shape[1])*lam; penalty[0,0]=0
    w=np.linalg.pinv(X.T@X+penalty)@X.T@y
    return w,med,mean,std

def predict(model,frame,features):
    w,med,mean,std=model; raw=frame[features].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0)
    X=((raw-mean)/std).to_numpy(float); X=np.column_stack([np.ones(len(X)),X]); score=X@w
    return np.where(score>=0,1,-1),np.abs(score)

def select_config(history,features,h):
    split=len(history)-VALID
    fit=history.iloc[:max(0,split-h)]
    valid=history.iloc[split:len(history)-h]
    best=None
    for lam in LAMBDAS:
        model=fit_model(fit,features,lam); pred,conf=predict(model,valid,features); target=valid.target.to_numpy(int)
        for q in np.linspace(0,.9,19):
            threshold=float(np.quantile(conf,q)); use=conf>=threshold; n=int(use.sum())
            if n<25 or n/len(valid)<.10: continue
            acc=float((pred[use]==target[use]).mean()); score=acc+.002*min(n/len(valid),.3)
            if best is None or score>best[0]: best=(score,lam,threshold,acc,n)
    return best

def nonoverlap(records,h):
    if not records: return []
    chosen=[]; last=-10**9
    for rec in records:
        if rec["index"]-last>=h: chosen.append(rec); last=rec["index"]
    return chosen

def evaluate(base,features,h):
    x=base.copy(); future=x.close.shift(-h)/x.close-1
    x["target"]=np.where(future>0,1,-1).astype(float); x.loc[future.isna(),"target"]=np.nan
    rec=[]; rows=0; windows=[]; start=0
    while start+HISTORY+TEST+h<=len(x):
        hist=x.iloc[start:start+HISTORY].copy(); test=x.iloc[start+HISTORY:start+HISTORY+TEST].copy(); cfg=select_config(hist,features,h); rows+=len(test)
        if cfg:
            _,lam,threshold,vacc,vn=cfg
            # Labels in the last horizon before test are not yet observable and are purged.
            train=hist.iloc[:-h]; model=fit_model(train,features,lam); pred,conf=predict(model,test,features); use=conf>=threshold
            actual=test.target.to_numpy(); valid=use&~np.isnan(actual); wh=int((pred[valid]==actual[valid]).sum()); wn=int(valid.sum())
            positions=np.flatnonzero(valid)
            rec += [{"index":int(test.index[pos]),"hit":int(pred[pos]==actual[pos])} for pos in positions]
            windows.append({"test_end":str(test.date.max().date()),"lambda":lam,"threshold":threshold,"validation_accuracy":vacc,"validation_cases":vn,"test_cases":wn,"test_accuracy":wh/wn if wn else 0})
        start+=TEST
    n=len(rec); hits=sum(r["hit"] for r in rec); acc=hits/n if n else 0; cov=n/rows if rows else 0; lower=wilson_lower(hits,n)
    independent=nonoverlap(rec,h); en=len(independent); eh=sum(r["hit"] for r in independent); eacc=eh/en if en else 0; elower=wilson_lower(eh,en)
    actual=x.iloc[HISTORY:HISTORY+rows].target.dropna(); up=float((actual==1).mean()) if len(actual) else .5; baseline=max(up,1-up)
    passed=acc>=.9 and n>=100 and cov>=.1 and lower>=.8 and acc>baseline and en>=100 and elower>=.8
    return {"horizon_days":h,"cases":n,"hits":hits,"accuracy":acc,"coverage":cov,"baseline":baseline,"edge":acc-baseline,"wilson_95_lower":lower,"nonoverlap_cases":en,"nonoverlap_accuracy":eacc,"nonoverlap_wilson_95_lower":elower,"passed":passed,"windows":windows}

def main():
    base,features=build_frame(); results=[evaluate(base,features,h) for h in HORIZONS]
    payload={"method":"Purged nested walk-forward ridge model with continuous constitution, pulse and lagged complexion interactions.","features":features,"gate":"90% accuracy, >=100 rows, >=10% coverage, Wilson lower >=80%, above majority baseline, and >=100 non-overlapping outcomes with Wilson lower >=80%.","results":results,"passing":[r['horizon_days'] for r in results if r['passed']]}
    out=ROOT/"reports"; (out/"three_layer_purged_research.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# Three-layer purged research","",payload['method'],"",payload['gate'],"","| Horizon | Cases | Accuracy | Coverage | Baseline | Edge | Wilson | Nonoverlap cases | Nonoverlap accuracy | Nonoverlap Wilson | Passed |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results: lines.append(f"| {r['horizon_days']} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['baseline']:.2%} | {r['edge']:.2%} | {r['wilson_95_lower']:.2%} | {r['nonoverlap_cases']} | {r['nonoverlap_accuracy']:.2%} | {r['nonoverlap_wilson_95_lower']:.2%} | {r['passed']} |")
    (out/"three_layer_purged_research.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"passing":payload['passing'],"results":[{k:v for k,v in r.items() if k!='windows'} for r in results]},indent=2))

if __name__=="__main__": main()
