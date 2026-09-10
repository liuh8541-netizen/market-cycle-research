import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_purged_boosting as boosting
from research_three_layer_purged import build_frame

def main():
    boosting.MATERIAL={20:0.0,60:0.0,120:0.0}
    base,features=build_frame(); results=[boosting.evaluate(base,features,h) for h in [20,60,120]]
    payload={"hypothesis":"A nonlinear gate formed by constitution, pulse and complexion selects continuation versus reversal of the stable stress factor.","method":"Purged nested AdaBoost stumps; all future directions are targets (no material-move neutral class).","results":results,"passing":[r['horizon_days'] for r in results if r['passed']]}
    out=ROOT/"reports"; (out/"purged_regime_boosting.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# Purged nonlinear regime gate","",payload['hypothesis'],payload['method'],"","| Horizon | Cases | Accuracy | Coverage | Baseline | Edge | Nonoverlap N/accuracy | Passed |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results: lines.append(f"| {r['horizon_days']} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['baseline']:.2%} | {r['edge']:.2%} | {r['nonoverlap_cases']} / {r['nonoverlap_accuracy']:.2%} | {r['passed']} |")
    (out/"purged_regime_boosting.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"passing":payload['passing'],"results":[{k:v for k,v in r.items() if k!='windows'} for r in results]},indent=2))

if __name__=="__main__": main()
