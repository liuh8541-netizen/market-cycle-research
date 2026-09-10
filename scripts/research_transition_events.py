import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_causal_phase import wilson_lower
from research_three_layer_purged import build_frame

HORIZONS=[20,60,120]

def candidate_events(x):
    work=x.copy(); r=np.log(work.close).diff(); vol20=r.rolling(20).std(); vol120=r.rolling(120).std()
    work["ret5"]=work.close.pct_change(5); work["ret20_event"]=work.close.pct_change(20)
    work["vol_ratio_event"]=vol20/vol120.replace(0,np.nan)
    work["range20"]=work.close.rolling(20).max()/work.close.rolling(20).min()-1
    shock=(work.ret5<=-.03)&(work.vol_ratio_event>=1.15)
    euphoria=(work.ret20_event>=.10)&(work.vol_ratio_event>=1.0)
    stagnation=(work.ret20_event.abs()<=.02)&(work.range20<=.06)&(work.vol_ratio_event<=.85)
    breakout=stagnation.shift(1).fillna(False)&~stagnation&(work.ret5.abs()>=.02)
    rows=[]
    for name,mask,prediction in [
        ("PANIC_RECOVERY",shock,1),
        ("EUPHORIA_CORRECTION",euphoria,-1),
        ("STAGNATION_BREAKOUT",breakout,None),
    ]:
        entered=mask&~mask.shift(1).fillna(False)
        for idx in work.index[entered]:
            direction=prediction if prediction is not None else int(np.sign(work.at[idx,"ret5"]))
            if direction: rows.append({"position":int(idx),"date":work.at[idx,"date"],"event":name,"prediction":direction})
    return pd.DataFrame(rows).sort_values(["date","event"]).reset_index(drop=True)

def nonoverlap_by_event(events,horizon):
    keep=[]; last=-10**9
    for row in events.sort_values("position").itertuples(index=False):
        if row.position-last>=horizon:
            keep.append(row._asdict()); last=row.position
    return pd.DataFrame(keep)

def evaluate(price,events,h):
    sample=nonoverlap_by_event(events,h)
    future=price.close.shift(-h)/price.close-1
    sample["future_return"]=sample["position"].map(future)
    sample=sample.dropna(subset=["future_return"])
    sample["target"]=np.where(sample.future_return>0,1,-1)
    sample["hit"]=sample.prediction==sample.target
    n=len(sample); hits=int(sample.hit.sum()); acc=hits/n if n else 0; lower=wilson_lower(hits,n)
    up=float((future.dropna()>0).mean()); baseline=max(up,1-up)
    groups={}
    for key,g in sample.groupby("event"):
        gh=int(g.hit.sum()); gn=len(g); groups[key]={"cases":gn,"hits":gh,"accuracy":gh/gn,"wilson_95_lower":wilson_lower(gh,gn)}
    return {"horizon_days":h,"cases":n,"hits":hits,"accuracy":acc,"baseline":baseline,"edge":acc-baseline,"wilson_95_lower":lower,"passed":acc>=.9 and n>=100 and lower>=.8 and acc>baseline,"by_event":groups,"events":[{"date":str(r.date.date()),"event":r.event,"prediction":int(r.prediction),"future_return":float(r.future_return),"hit":bool(r.hit)} for r in sample.itertuples()]}

def main():
    price,_=build_frame(); events=candidate_events(price); results=[evaluate(price,events,h) for h in HORIZONS]
    payload={"method":"Causally triggered fixed transition events; episodes are separated by at least one full forecast horizon; no fitted parameters.","candidate_events":len(events),"results":results,"passing":[r['horizon_days'] for r in results if r['passed']]}
    out=ROOT/"reports"; (out/"transition_event_research.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# Non-overlapping transition event research","",payload['method'],"",f"Raw causal event onsets: {len(events)}","","| Horizon | Episodes | Accuracy | Baseline | Edge | Wilson lower | Passed |","| ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results: lines.append(f"| {r['horizon_days']} | {r['cases']} | {r['accuracy']:.2%} | {r['baseline']:.2%} | {r['edge']:.2%} | {r['wilson_95_lower']:.2%} | {r['passed']} |")
    for r in results:
        lines += ["",f"## {r['horizon_days']} days by event"]+[f"- {k}: {v['accuracy']:.2%} ({v['hits']}/{v['cases']}; lower {v['wilson_95_lower']:.2%})" for k,v in sorted(r['by_event'].items())]
    (out/"transition_event_research.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"candidate_events":len(events),"passing":payload['passing'],"results":[{k:v for k,v in r.items() if k not in ['events','by_event']}|{"by_event":r['by_event']} for r in results]},indent=2))

if __name__=="__main__": main()
