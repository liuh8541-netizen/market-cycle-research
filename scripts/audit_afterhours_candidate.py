import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_afterhours_institutional as study
import research_afterhours_with_complexion as extended

def exact_two_sided_binomial(a,b):
    n=a+b
    if not n:return 1.0
    k=min(a,b);tail=sum(math.comb(n,i) for i in range(k+1))/(2**n);return min(1.0,2*tail)

def metrics(records,key):
    valid=[r for r in records if r.get(key) is not None];n=len(valid);hits=sum(int(r[key]==r["target"]) for r in valid);return {"cases":n,"hits":hits,"accuracy":hits/n if n else 0}

def main():
    data=extended.prepare();study.FEATURES=study.FEATURES+extended.EXTERNAL;study.MIN_VALID_COVERAGE=.10;target="gap_direction";data[target]=np.where(data.gap_return>=0,1,-1)
    records=[];start=0
    while start+study.TRAIN+study.TEST<=len(data):
        train=data.iloc[start:start+study.TRAIN];test=data.iloc[start+study.TRAIN:start+study.TRAIN+study.TEST].copy();cfg=study.select(train,target)
        if cfg:
            _,lam,threshold,_,_=cfg;prediction,confidence=study.predict(study.fit(train,target,lam),test);use=confidence>=threshold
            for pos in np.flatnonzero(use):
                row=test.iloc[pos]; values=[row.get(c) for c in ["sp500_return_1d","nasdaq_return_1d","sox_return_1d","dow_return_1d","tsm_adr_return_1d"]]; signs=[np.sign(v) for v in values if v is not None and np.isfinite(v) and v!=0]; majority=1 if signs and sum(signs)>=0 else (-1 if signs else None)
                records.append({"date":str(row.date.date()),"target":int(row[target]),"model":int(prediction[pos]),"external_majority":majority,"sp500":int(np.sign(row.sp500_return_1d)) if np.isfinite(row.sp500_return_1d) and row.sp500_return_1d!=0 else None,"nasdaq":int(np.sign(row.nasdaq_return_1d)) if np.isfinite(row.nasdaq_return_1d) and row.nasdaq_return_1d!=0 else None,"sox":int(np.sign(row.sox_return_1d)) if np.isfinite(row.sox_return_1d) and row.sox_return_1d!=0 else None,"tsm_adr":int(np.sign(row.tsm_adr_return_1d)) if np.isfinite(row.tsm_adr_return_1d) and row.tsm_adr_return_1d!=0 else None})
        start+=study.TEST
    comparisons={k:metrics(records,k) for k in ["model","external_majority","sp500","nasdaq","sox","tsm_adr"]}
    discord_model=sum(r["model"]==r["target"] and r["external_majority"]!=r["target"] for r in records if r["external_majority"] is not None);discord_baseline=sum(r["model"]!=r["target"] and r["external_majority"]==r["target"] for r in records if r["external_majority"] is not None)
    payload={"status":"candidate audit on previously inspected historical OOS windows; forward confirmation still required","time_causality":{"external_date_rule":"strictly earlier than TWII date","same_or_future_external_rows":int((data.external_date>=data.date).sum()),"afterhours_trade_date_rule":"FinMind/TAIFEX after_market for D ends by 05:00 on D"},"comparisons_on_identical_selected_cases":comparisons,"model_vs_external_majority":{"model_only_correct":discord_model,"baseline_only_correct":discord_baseline,"mcnemar_exact_two_sided_p":exact_two_sided_binomial(discord_model,discord_baseline)},"records":records}
    out=ROOT/"reports";(out/"afterhours_candidate_audit.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# After-hours candidate audit","",payload['status'],"",f"Same/future external rows: {payload['time_causality']['same_or_future_external_rows']}","","| Predictor | Cases | Hits | Accuracy |","| --- | ---: | ---: | ---: |"]
    for k,v in comparisons.items():lines.append(f"| {k} | {v['cases']} | {v['hits']} | {v['accuracy']:.2%} |")
    lines += ["",f"Model-only correct vs external-majority-only correct: {discord_model} vs {discord_baseline}; exact McNemar p={payload['model_vs_external_majority']['mcnemar_exact_two_sided_p']:.6g}."]
    (out/"afterhours_candidate_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps({k:v for k,v in payload.items() if k!='records'},indent=2))

if __name__=="__main__":main()
