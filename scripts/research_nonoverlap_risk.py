import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_causal_phase import wilson_lower
from research_three_layer_purged import build_frame,fit_model,predict

H=20; TRAIN=100; VALID=30; TEST=30; LAMBDAS=[.1,1,10,100,1000]; DRAWDOWN=-.05

def blocks(base):
    positions=np.arange(240,len(base)-H,H); out=base.iloc[positions].copy(); targets=[]; worst=[]
    for p in positions:
        path=base.close.iloc[p+1:p+H+1]/base.close.iloc[p]-1; low=float(path.min()); worst.append(low); targets.append(-1 if low<=DRAWDOWN else 1)
    out["target"]=targets; out["future_worst_return"]=worst; return out.reset_index(drop=True)

def choose(train,valid,features):
    best=None
    for lam in LAMBDAS:
        model=fit_model(train,features,lam); pred,conf=predict(model,valid,features); actual=valid.target.to_numpy(int)
        for q in np.linspace(0,.4,5):
            threshold=float(np.quantile(conf,q)); use=conf>=threshold; n=int(use.sum())
            if n<18 or n/len(valid)<.60: continue
            acc=float((pred[use]==actual[use]).mean()); score=acc+.001*n
            if best is None or score>best[0]: best=(score,lam,threshold,acc,n)
    return best

def main():
    base,features=build_frame(); data=blocks(base); rec=[]; total=0; windows=[]; start=0
    while start+TRAIN+VALID+TEST<=len(data):
        train=data.iloc[start:start+TRAIN]; valid=data.iloc[start+TRAIN:start+TRAIN+VALID]; test=data.iloc[start+TRAIN+VALID:start+TRAIN+VALID+TEST]; cfg=choose(train,valid,features); total+=len(test)
        if cfg:
            _,lam,threshold,vacc,vn=cfg; history=data.iloc[start:start+TRAIN+VALID]; pred,conf=predict(fit_model(history,features,lam),test,features); actual=test.target.to_numpy(int); use=conf>=threshold; hits=pred[use]==actual[use]
            rec+=hits.astype(int).tolist(); windows.append({"test_end":str(test.date.max().date()),"lambda":lam,"threshold":threshold,"validation_accuracy":vacc,"validation_cases":vn,"test_cases":int(use.sum()),"test_accuracy":float(hits.mean()) if len(hits) else 0})
        start+=TEST
    n=len(rec); hits=sum(rec); acc=hits/n if n else 0; coverage=n/total if total else 0; lower=wilson_lower(hits,n)
    actual=data.iloc[TRAIN+VALID:TRAIN+VALID+total].target; safe=float((actual==1).mean()); baseline=max(safe,1-safe); passed=acc>=.9 and n>=100 and coverage>=.1 and lower>=.8 and acc>baseline
    result={"cases":n,"hits":hits,"accuracy":acc,"coverage":coverage,"safe_rate_baseline":baseline,"edge":acc-baseline,"wilson_95_lower":lower,"passed":passed,"windows":windows}
    payload={"target":"Whether the next 20 trading days avoid a drawdown of 5% or more from the current close.","method":"Fixed non-overlapping 20-day blocks; nested 100/30/30 three-layer ridge; validation coverage forced above 60%.","available_blocks":len(data),"result":result}
    out=ROOT/"reports"; (out/"nonoverlap_risk_research.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    (out/"nonoverlap_risk_research.md").write_text("\n".join(["# Non-overlapping 20-day risk research","",payload['target'],payload['method'],"",f"Available blocks: {len(data)}",f"OOS: {hits}/{n} = {acc:.2%}; coverage {coverage:.2%}; always-safe/risk baseline {baseline:.2%}; edge {acc-baseline:.2%}; Wilson lower {lower:.2%}; passed={passed}."])+"\n",encoding="utf-8")
    print(json.dumps({"available_blocks":len(data),"result":{k:v for k,v in result.items() if k!='windows'}},indent=2))

if __name__=="__main__": main()
