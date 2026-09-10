import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_afterhours_institutional as study

def main():
    study.MIN_VALID_COVERAGE=.30
    data=study.prepare();results=[study.evaluate(data,"cash_direction"),study.evaluate(data,"gap_direction")]
    payload={"status":"post-hoc sensitivity only; not independent confirmation","change":"Minimum validation coverage raised from 10% to 30% after seeing the original 78-case gap result.","results":results}
    out=ROOT/"reports";(out/"afterhours_highcoverage_sensitivity.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# After-hours high-coverage sensitivity","",payload['status'],payload['change'],"","| Target | Cases | Accuracy | Coverage | Baseline | Wilson lower | Passed by numeric gate |","| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results:lines.append(f"| {r['target']} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['baseline']:.2%} | {r['wilson_95_lower']:.2%} | {r['passed']} |")
    (out/"afterhours_highcoverage_sensitivity.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps({"status":payload['status'],"results":[{k:v for k,v in r.items() if k!='windows'} for r in results]},indent=2))

if __name__=="__main__":main()
