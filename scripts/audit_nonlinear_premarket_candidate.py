import json
import sys
from math import comb
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_nonlinear_premarket_state as modeldef

def exact(a,b):
    n=a+b
    if not n:return 1.0
    return min(1.0,2*sum(comb(n,k) for k in range(min(a,b)+1))/(2**n))

def main():
    research=json.loads((ROOT/"reports"/"nonlinear_premarket_state.json").read_text(encoding="utf-8"));frame=modeldef.prepare();frame.index=frame.date.dt.date.astype(str);rows=[]
    for r in research["records"]:
        x=frame.loc[r["date"]];rows.append({**r,"night_return":1 if x.tx_night_return>=0 else -1,"night_spread":1 if x.tx_night_spread_per>=0 else -1})
    comparisons={};model_acc=research["result"]["accuracy"]
    for name in ["baseline","night_return","night_spread"]:
        n=len(rows);hits=sum(r[name]==r["target"] for r in rows);mo=sum(r["model"]==r["target"] and r[name]!=r["target"] for r in rows);bo=sum(r["model"]!=r["target"] and r[name]==r["target"] for r in rows);comparisons[name]={"cases":n,"hits":hits,"accuracy":hits/n,"model_edge":model_acc-hits/n,"model_only_correct":mo,"baseline_only_correct":bo,"mcnemar_exact_p":exact(mo,bo)}
    strongest=max(comparisons,key=lambda k:comparisons[k]["accuracy"]);c=comparisons[strongest];passed=model_acc>c["accuracy"] and c["mcnemar_exact_p"]<.05
    payload={"status":"formal strong-baseline audit on exact nonlinear-model OOS dates","model":{"cases":len(rows),"hits":research["result"]["hits"],"accuracy":model_acc},"comparisons":comparisons,"strongest_simple_baseline":strongest,"passes_strongest_simple_baseline":passed,"records":rows};out=ROOT/"reports";(out/"nonlinear_premarket_candidate_audit.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Nonlinear premarket candidate strong-baseline audit","",payload["status"],"","| Signal | Cases | Hits | Accuracy | Model edge | Model-only / signal-only | McNemar p |","| --- | ---: | ---: | ---: | ---: | ---: | ---: |",f"| model | {len(rows)} | {research['result']['hits']} | {model_acc:.2%} | - | - | - |"]
    for name,v in comparisons.items():lines.append(f"| {name} | {v['cases']} | {v['hits']} | {v['accuracy']:.2%} | {v['model_edge']:.2%} | {v['model_only_correct']} / {v['baseline_only_correct']} | {v['mcnemar_exact_p']:.6g} |")
    lines += ["",f"Strongest simple baseline: {strongest}.",f"Model passes strongest simple baseline: {passed}."];(out/"nonlinear_premarket_candidate_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps({k:v for k,v in payload.items() if k!="records"},indent=2))

if __name__=="__main__":main()
