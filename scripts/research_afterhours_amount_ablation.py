import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_afterhours_institutional as study
import research_afterhours_with_complexion as ext
import research_afterhours_amount_structure as amount

def main():
    x=amount.prepare();core=[c for c in x.columns if c.startswith("futures_") or c.startswith("option_")];pulse=core+["tx_night_return","tx_night_range","tx_night_volume","tx_night_spread_per"]
    variants={"pulse_only":pulse,"external_only":ext.EXTERNAL,"combined":pulse+ext.EXTERNAL};results={};study.MIN_VALID_COVERAGE=.10
    for name,features in variants.items():
        study.FEATURES=features;r,w,records=amount.evaluate(x);results[name]={"features":features,"result":r,"windows":w,"records":records}
    payload={"status":"exploratory ablation on previously inspected history","method":"Identical nested walk-forward protocol for pulse-only, external-only and combined feature families.","variants":results}
    out=ROOT/"reports";(out/"afterhours_amount_ablation.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# After-hours amount feature ablation","",payload["status"],payload["method"],"","| Variant | Cases | Accuracy | Coverage | Wilson lower | External baseline | Edge | Passed |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name,v in results.items():
        r=v["result"];lines.append(f"| {name} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['wilson_95_lower']:.2%} | {r['baseline_accuracy']:.2%} | {r['edge']:.2%} | {r['passed']} |")
    (out/"afterhours_amount_ablation.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps({k:v["result"] for k,v in results.items()},indent=2))

if __name__=="__main__":main()
