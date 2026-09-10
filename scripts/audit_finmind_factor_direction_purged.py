"""Purged univariate audit of FinMind factor polarity and stability."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_three_layer_purged import build_frame
from research_causal_phase import wilson_lower

HORIZONS=[20,60];HISTORY=1512;VALID=252;TEST=252
FACTORS=[
 "inst_net_20d_z","inst_foreign_net_20d_z","inst_trust_net_20d_z","inst_dealer_net_20d_z",
 "margin_change_20d_z","futures_inst_net_20d_z","futures_foreign_net_20d_z","futures_trust_net_20d_z","futures_dealer_net_20d_z",
 "option_put_call_20d_z_z","option_inst_net_20d_z","option_foreign_net_20d_z","option_trust_net_20d_z","option_dealer_net_20d_z","option_vix_20d_z_z"
]

def select(history,factor,h):
    split=len(history)-VALID;fit=history.iloc[:split-h];valid=history.iloc[split:len(history)-h];fit_abs=fit[factor].abs().dropna();best=None
    if len(fit_abs)<80:return None
    for q in [0,.5,.7,.8,.9]:
        threshold=float(fit_abs.quantile(q))
        for polarity in [1,-1]:
            use=valid[factor].notna()&(valid[factor].abs()>=threshold)&valid.target.notna();n=int(use.sum())
            if n<25 or n/len(valid)<.10:continue
            pred=np.where(valid.loc[use,factor]>=0,1,-1)*polarity;actual=valid.loc[use,"target"].to_numpy(int);acc=float((pred==actual).mean());rank=(acc,acc>=.9,n)
            if best is None or rank>best[0]:best=(rank,threshold,polarity,acc,n)
    return best

def evaluate(base,factor,h):
    x=base.copy();future=x.close.shift(-h)/x.close-1;x["target"]=np.where(future>=0,1,-1).astype(float);x.loc[future.isna(),"target"]=np.nan;records=[];windows=[];tested=0;start=0
    while start+HISTORY+TEST+h<=len(x):
        hist=x.iloc[start:start+HISTORY];test=x.iloc[start+HISTORY:start+HISTORY+TEST];tested+=len(test);cfg=select(hist,factor,h)
        if cfg:
            _,threshold,polarity,vacc,vn=cfg;use=test[factor].notna()&(test[factor].abs()>=threshold)&test.target.notna();pred=np.where(test.loc[use,factor]>=0,1,-1)*polarity;actual=test.loc[use,"target"].to_numpy(int)
            for idx,p,a in zip(test.index[use],pred,actual):records.append({"index":int(idx),"hit":int(p==a),"prediction":int(p),"target":int(a)})
            windows.append({"test_end":str(test.date.max().date()),"polarity":"same" if polarity==1 else "inverse","threshold":threshold,"validation_accuracy":vacc,"validation_cases":vn,"test_cases":int(use.sum()),"test_accuracy":float((pred==actual).mean()) if len(actual) else 0})
        start+=TEST
    n=len(records);hits=sum(r["hit"] for r in records);acc=hits/n if n else 0;actual=[r["target"] for r in records];up=sum(a==1 for a in actual)/n if n else .5;baseline=max(up,1-up);effective=[];last=-10**9
    for r in records:
        if r["index"]>=last+h:effective.append(r);last=r["index"]
    en=len(effective);eh=sum(r["hit"] for r in effective);same=sum(w["polarity"]=="same" for w in windows);inverse=len(windows)-same
    return {"factor":factor,"horizon":h,"cases":n,"hits":hits,"accuracy":acc,"coverage":n/tested if tested else 0,"baseline":baseline,"edge":acc-baseline,"wilson_lower":wilson_lower(hits,n),"nonoverlap_cases":en,"nonoverlap_accuracy":eh/en if en else 0,"nonoverlap_wilson_lower":wilson_lower(eh,en),"same_polarity_windows":same,"inverse_polarity_windows":inverse,"polarity_stability":max(same,inverse)/len(windows) if windows else 0,"passed":n>=100 and n/tested>=.1 and acc>=.9 and wilson_lower(hits,n)>=.8 and acc>baseline and en>=100 and wilson_lower(eh,en)>=.8,"windows":windows}

def main():
    base,_=build_frame();available=[f for f in FACTORS if f in base.columns];results=[evaluate(base,f,h) for h in HORIZONS for f in available];payload={"status":"formal univariate polarity audit after cash-institution split-feature fix","method":"For each 1512-day history, last 252 days select same/inverse polarity and an extreme absolute-z threshold after purging the horizon; freeze into the next 252-day OOS block. Multi-day effective samples are non-overlapping.","available_factors":available,"results":results};out=ROOT/"reports";(out/"finmind_factor_direction_purged.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# FinMind factor direction purged audit","",payload["status"],payload["method"],"","| H | Factor | Cases | Accuracy | Baseline | Edge | Nonoverlap N | Nonoverlap acc. | Same/inverse windows | Stability | Passed |","| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results:lines.append(f"| {r['horizon']} | {r['factor']} | {r['cases']} | {r['accuracy']:.2%} | {r['baseline']:.2%} | {r['edge']:.2%} | {r['nonoverlap_cases']} | {r['nonoverlap_accuracy']:.2%} | {r['same_polarity_windows']}/{r['inverse_polarity_windows']} | {r['polarity_stability']:.2%} | {r['passed']} |")
    (out/"finmind_factor_direction_purged.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps({"available":available,"best":[{k:v for k,v in r.items() if k!="windows"} for r in sorted(results,key=lambda z:z["accuracy"],reverse=True)[:8]]},indent=2))

if __name__=="__main__":main()
