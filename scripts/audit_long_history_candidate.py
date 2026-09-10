"""Compare the long-history model with every strong simple signal on identical dates."""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_long_history_night_external as source

def mcnemar(a,b):
    n=a+b
    if not n:return 1.0
    return min(1.0,2*sum(comb(n,k) for k in range(min(a,b)+1))/(2**n))

def main():
    research=json.loads((ROOT/"reports"/"long_history_night_external.json").read_text(encoding="utf-8"));data=source.prepare().set_index(data_key(source.prepare()))
    rows=[]
    for r in research["records"]:
        row=data.loc[r["date"]];night_return=1 if row.tx_night_return>=0 else -1;night_spread=1 if row.tx_night_spread_per>=0 else -1
        rows.append({**r,"night_return":night_return,"night_spread":night_spread})
    comparisons={}
    for name in ["baseline","night_return","night_spread"]:
        n=len(rows);hits=sum(r[name]==r["target"] for r in rows);mo=sum(r["model"]==r["target"] and r[name]!=r["target"] for r in rows);bo=sum(r["model"]!=r["target"] and r[name]==r["target"] for r in rows);comparisons[name]={"cases":n,"hits":hits,"accuracy":hits/n,"model_only_correct":mo,"baseline_only_correct":bo,"model_edge":research["result"]["accuracy"]-hits/n,"mcnemar_exact_p":mcnemar(mo,bo)}
    strongest=max(comparisons,key=lambda k:comparisons[k]["accuracy"]);c=comparisons[strongest];passed=research["result"]["accuracy"]>c["accuracy"] and c["mcnemar_exact_p"]<.05
    payload={"status":"formal strong-baseline audit on the exact model-selected OOS dates","model":{"cases":research["result"]["cases"],"hits":research["result"]["hits"],"accuracy":research["result"]["accuracy"]},"comparisons":comparisons,"strongest_simple_baseline":strongest,"passes_strongest_simple_baseline":passed,"records":rows}
    out=ROOT/"reports";(out/"long_history_candidate_audit.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Long-history candidate strong-baseline audit","",payload["status"],"","| Signal | Cases | Hits | Accuracy | Model edge | Model-only / signal-only | McNemar p |","| --- | ---: | ---: | ---: | ---: | ---: | ---: |",f"| model | {payload['model']['cases']} | {payload['model']['hits']} | {payload['model']['accuracy']:.2%} | - | - | - |"]
    for name,v in comparisons.items():lines.append(f"| {name} | {v['cases']} | {v['hits']} | {v['accuracy']:.2%} | {v['model_edge']:.2%} | {v['model_only_correct']} / {v['baseline_only_correct']} | {v['mcnemar_exact_p']:.6g} |")
    lines += ["",f"Strongest simple baseline: {strongest}.",f"Model passes strongest simple baseline: {passed}."]
    (out/"long_history_candidate_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps({k:v for k,v in payload.items() if k!="records"},indent=2))

def data_key(frame):
    return frame.date.dt.date.astype(str)

if __name__=="__main__":main()
