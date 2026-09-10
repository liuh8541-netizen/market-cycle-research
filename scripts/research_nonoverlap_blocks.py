import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_causal_phase import wilson_lower
from research_three_layer_purged import build_frame,fit_model,predict

HORIZON=20; TRAIN_BLOCKS=100; VALID_BLOCKS=30; TEST_BLOCKS=30; LAMBDAS=[.1,1,10,100,1000]

def make_blocks(base):
    # Fixed cadence registered independently of outcomes. Consecutive forward windows do not overlap.
    positions=np.arange(240,len(base)-HORIZON,HORIZON)
    blocks=base.iloc[positions].copy()
    future=base.close.shift(-HORIZON)/base.close-1
    blocks["target"]=np.where(future.iloc[positions].to_numpy()>0,1,-1)
    blocks["position"]=positions
    return blocks.reset_index(drop=True)

def choose(train,valid,features):
    best=None
    for lam in LAMBDAS:
        model=fit_model(train,features,lam); pred,conf=predict(model,valid,features); target=valid.target.to_numpy(int)
        for q in np.linspace(0,.9,10):
            threshold=float(np.quantile(conf,q)); use=conf>=threshold; n=int(use.sum())
            # Force enough independent out-of-sample decisions to make the 100-case gate attainable.
            if n<18 or n/len(valid)<.60: continue
            acc=float((pred[use]==target[use]).mean()); score=acc+.001*n
            if best is None or score>best[0]: best=(score,lam,threshold,acc,n)
    return best

def main():
    base,features=build_frame(); blocks=make_blocks(base); records=[]; windows=[]; start=0; total=0
    while start+TRAIN_BLOCKS+VALID_BLOCKS+TEST_BLOCKS<=len(blocks):
        train=blocks.iloc[start:start+TRAIN_BLOCKS]; valid=blocks.iloc[start+TRAIN_BLOCKS:start+TRAIN_BLOCKS+VALID_BLOCKS]; test=blocks.iloc[start+TRAIN_BLOCKS+VALID_BLOCKS:start+TRAIN_BLOCKS+VALID_BLOCKS+TEST_BLOCKS]
        cfg=choose(train,valid,features); total+=len(test)
        if cfg:
            _,lam,threshold,vacc,vn=cfg; history=blocks.iloc[start:start+TRAIN_BLOCKS+VALID_BLOCKS]; model=fit_model(history,features,lam); pred,conf=predict(model,test,features); use=conf>=threshold; actual=test.target.to_numpy(int); hits=(pred[use]==actual[use])
            records += hits.astype(int).tolist(); windows.append({"test_end":str(test.date.max().date()),"lambda":lam,"threshold":threshold,"validation_accuracy":vacc,"validation_cases":vn,"test_cases":int(use.sum()),"test_accuracy":float(hits.mean()) if len(hits) else 0})
        start+=TEST_BLOCKS
    n=len(records); hits=sum(records); acc=hits/n if n else 0; cov=n/total if total else 0; lower=wilson_lower(hits,n)
    actual=blocks.iloc[TRAIN_BLOCKS+VALID_BLOCKS:TRAIN_BLOCKS+VALID_BLOCKS+total].target; up=float((actual==1).mean()); baseline=max(up,1-up)
    passed=acc>=.9 and n>=100 and cov>=.1 and lower>=.8 and acc>baseline
    payload={"method":"Fixed 20-trading-day non-overlapping blocks; nested 100-block train, 30-block validation, 30-block test; continuous three-layer ridge with abstention.","available_blocks":len(blocks),"result":{"cases":n,"hits":hits,"accuracy":acc,"coverage":cov,"baseline":baseline,"edge":acc-baseline,"wilson_95_lower":lower,"passed":passed,"windows":windows}}
    out=ROOT/"reports"; (out/"nonoverlap_block_research.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    r=payload['result']; lines=["# Non-overlapping 20-day block research","",payload['method'],"",f"Available independent blocks: {len(blocks)}",f"Out-of-sample: {r['hits']}/{r['cases']} = {r['accuracy']:.2%}; coverage {r['coverage']:.2%}; baseline {r['baseline']:.2%}; edge {r['edge']:.2%}; Wilson lower {r['wilson_95_lower']:.2%}; passed={r['passed']}." ]
    (out/"nonoverlap_block_research.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"available_blocks":len(blocks),"result":{k:v for k,v in r.items() if k!='windows'}},indent=2))

if __name__=="__main__": main()
