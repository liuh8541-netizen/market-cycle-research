"""Low-degree causal premarket ensemble for 1/5/20-day close direction."""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_premarket_night_signal import wilson

TRAIN=756;VALID=126;TEST=126;HORIZONS=[1,5,20]
SETS={"night_external_trend20":["night_spread","external","trend20"],"night_pair_external":["night_spread","night_return","external"],"night_external_trend5_trend20":["night_spread","external","trend5","trend20"],"all_five":["night_spread","night_return","external","trend5","trend20"]}

def prepare():
    tw=pd.read_csv(ROOT/"data"/"processed"/"twii_daily.csv",usecols=["date","close"]);tw["date"]=pd.to_datetime(tw.date);close=pd.to_numeric(tw.close);tw["prior_close"]=close.shift(1);tw["trend5_value"]=tw.prior_close/close.shift(6)-1;tw["trend20_value"]=tw.prior_close/close.shift(21)-1;tw["trend5"]=np.where(tw.trend5_value>=0,1,-1);tw["trend20"]=np.where(tw.trend20_value>=0,1,-1)
    for h in HORIZONS:
        future=close.shift(-(h-1));ret=future/tw.prior_close-1;tw[f"target_{h}"]=np.where(ret>=0,1,-1);tw.loc[ret.isna(),f"target_{h}"]=np.nan
    night=pd.read_csv(ROOT/"data"/"processed"/"taiwan_futures_night.csv");night["date"]=pd.to_datetime(night.signal_date);night["night_spread"]=np.where(night.tx_night_spread_per>=0,1,-1);night["night_return"]=np.where(night.tx_night_return>=0,1,-1);night["night_confidence"]=night.tx_night_spread_per.abs()
    x=night.merge(tw,on="date",how="inner").sort_values("date")
    ext=pd.read_csv(ROOT/"data"/"processed"/"external_markets.csv");ext["external_date"]=pd.to_datetime(ext.date);cols=["sp500_return_1d","nasdaq_return_1d","sox_return_1d","dow_return_1d","tsm_adr_return_1d"];x=pd.merge_asof(x,ext[["external_date"]+cols].sort_values("external_date"),left_on="date",right_on="external_date",direction="backward",allow_exact_matches=False,tolerance=pd.Timedelta("4D"));x["external"]=np.where(np.sign(x[cols].fillna(0)).sum(axis=1)>=0,1,-1);return x.reset_index(drop=True)

def ensemble(frame,components):
    votes=frame[components].to_numpy(float);score=votes.mean(axis=1);return np.where(score>=0,1,-1),np.abs(score)

def select(history,h):
    fit=history.iloc[:-VALID];valid=history.iloc[-VALID:-(h-1) if h>1 else None];actual=valid[f"target_{h}"].to_numpy(int);best=None
    for name,components in SETS.items():
        pred,agreement=ensemble(valid,components);baselines={c:valid[c].to_numpy(int) for c in ["night_spread","night_return","external","trend5","trend20"]}
        for q in [0,.3,.5,.7,.8,.9]:
            threshold=float(fit.night_confidence.quantile(q))
            for min_agreement in [.5,.6,1.0]:
                use=(valid.night_confidence.to_numpy()>=threshold)&(agreement>=min_agreement);n=int(use.sum())
                if n<13 or n/len(valid)<.10:continue
                acc=float((pred[use]==actual[use]).mean());bacc=max(float((b[use]==actual[use]).mean()) for b in baselines.values());rank=(acc>=.9 and acc>bacc,acc,acc-bacc,min(n,50),-len(components))
                if best is None or rank>best[0]:best=(rank,name,threshold,min_agreement,acc,bacc,n)
    return best

def exact(a,b):
    n=a+b
    if not n:return 1.0
    return min(1.0,2*sum(comb(n,k) for k in range(min(a,b)+1))/(2**n))

def evaluate(x,h):
    records=[];windows=[];tested=0;start=0
    while start+TRAIN+TEST+h-1<=len(x):
        history=x.iloc[start:start+TRAIN];test=x.iloc[start+TRAIN:start+TRAIN+TEST];tested+=len(test);cfg=select(history,h)
        if cfg:
            _,name,threshold,min_agreement,vacc,vbacc,vn=cfg;components=SETS[name];pred,agreement=ensemble(test,components);actual=test[f"target_{h}"].to_numpy();use=(test.night_confidence.to_numpy()>=threshold)&(agreement>=min_agreement)&~np.isnan(actual)
            for pos in np.flatnonzero(use):
                r=test.iloc[pos];records.append({"index":int(start+TRAIN+pos),"date":str(r.date.date()),"target":int(actual[pos]),"model":int(pred[pos]),**{c:int(r[c]) for c in ["night_spread","night_return","external","trend5","trend20"]}})
            windows.append({"test_end":str(test.date.max().date()),"ensemble":name,"threshold":threshold,"minimum_agreement":min_agreement,"cases":int(use.sum()),"accuracy":float((pred[use]==actual[use]).mean()) if use.sum() else 0,"validation_accuracy":vacc,"validation_strongest_baseline":vbacc,"validation_cases":vn})
        start+=TEST
    n=len(records);hits=sum(r["model"]==r["target"] for r in records);acc=hits/n if n else 0;comparisons={}
    for c in ["night_spread","night_return","external","trend5","trend20"]:
        bh=sum(r[c]==r["target"] for r in records);mo=sum(r["model"]==r["target"] and r[c]!=r["target"] for r in records);bo=sum(r["model"]!=r["target"] and r[c]==r["target"] for r in records);comparisons[c]={"hits":bh,"accuracy":bh/n if n else 0,"model_edge":acc-bh/n if n else 0,"model_only_correct":mo,"baseline_only_correct":bo,"mcnemar_p":exact(mo,bo)}
    strongest=max(comparisons,key=lambda c:comparisons[c]["accuracy"]);s=comparisons[strongest]
    effective=[];last=-10**9
    for r in sorted(records,key=lambda z:z["index"]):
        if r["index"]>=last+h:effective.append(r);last=r["index"]
    en=len(effective);eh=sum(r["model"]==r["target"] for r in effective);eacc=eh/en if en else 0;elower=wilson(eh,en)
    passed=n>=100 and n/tested>=.1 and acc>=.9 and wilson(hits,n)>=.8 and acc>s["accuracy"] and en>=100 and eacc>=.9 and elower>=.8
    return {"horizon":h,"cases":n,"hits":hits,"accuracy":acc,"coverage":n/tested if tested else 0,"wilson_lower":wilson(hits,n),"strongest_baseline":strongest,"strongest_baseline_accuracy":s["accuracy"],"edge":acc-s["accuracy"],"strongest_baseline_mcnemar_p":s["mcnemar_p"],"nonoverlap_cases":en,"nonoverlap_hits":eh,"nonoverlap_accuracy":eacc,"nonoverlap_wilson_lower":elower,"passed":passed,"comparisons":comparisons,"windows":windows,"records":records}

def main():
    x=prepare();results=[evaluate(x,h) for h in HORIZONS];payload={"status":"exploratory registered low-degree ensemble after canonical night-date correction","method":"756-day history with last 126 days used for purged configuration validation, frozen 126-day OOS blocks; prediction components are causal night spread/return, strictly prior overseas majority, and prior-close 5/20-day trends. Multi-day effective results greedily retain non-overlapping origins.","sets":SETS,"results":results};out=ROOT/"reports";(out/"multihorizon_premarket_ensemble.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Multi-horizon causal premarket ensemble","",payload["status"],payload["method"],"","| Horizon | Cases | Accuracy | Coverage | Wilson | Strongest baseline | Baseline acc. | Edge | Nonoverlap N | Nonoverlap acc. | Nonoverlap Wilson | Passed |","| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results:lines.append(f"| {r['horizon']} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['wilson_lower']:.2%} | {r['strongest_baseline']} | {r['strongest_baseline_accuracy']:.2%} | {r['edge']:.2%} | {r['nonoverlap_cases']} | {r['nonoverlap_accuracy']:.2%} | {r['nonoverlap_wilson_lower']:.2%} | {r['passed']} |")
    (out/"multihorizon_premarket_ensemble.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps([{k:v for k,v in r.items() if k not in ["records","windows","comparisons"]} for r in results],indent=2))

if __name__=="__main__":main()
