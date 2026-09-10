import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_afterhours_institutional as study

def main():
    data=study.prepare(); rows=[]
    for coverage in [.10,.125,.15,.175,.20,.225,.25,.275,.30]:
        study.MIN_VALID_COVERAGE=coverage
        for target in ["cash_direction","gap_direction"]:
            result=study.evaluate(data,target); rows.append({"minimum_validation_coverage":coverage,**{k:v for k,v in result.items() if k!="windows"}})
    payload={"status":"post-hoc precision/coverage sensitivity; not independent confirmation","rows":rows}
    out=ROOT/"reports";(out/"afterhours_coverage_curve.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# After-hours precision/coverage curve","",payload['status'],"","| Target | Min validation coverage | OOS cases | OOS coverage | Accuracy | Wilson lower | Numeric gate |","| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:lines.append(f"| {r['target']} | {r['minimum_validation_coverage']:.1%} | {r['cases']} | {r['coverage']:.2%} | {r['accuracy']:.2%} | {r['wilson_95_lower']:.2%} | {r['passed']} |")
    (out/"afterhours_coverage_curve.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps(rows,indent=2))

if __name__=="__main__":main()
