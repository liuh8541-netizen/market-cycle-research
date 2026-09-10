import json
import sys
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_afterhours_institutional as study

EXTERNAL=["sp500_return_1d","nasdaq_return_1d","sox_return_1d","dow_return_1d","vix_return_1d","tsm_adr_return_1d","micron_return_1d"]

def prepare():
    base=study.prepare().sort_values("date")
    ext=pd.read_csv(ROOT/"data"/"processed"/"external_markets.csv");ext["external_date"]=pd.to_datetime(ext.date)
    # Strictly earlier date prevents using a US close that was not known by Taiwan premarket.
    x=pd.merge_asof(base,ext[["external_date"]+EXTERNAL].sort_values("external_date"),left_on="date",right_on="external_date",direction="backward",allow_exact_matches=False,tolerance=pd.Timedelta("4D"))
    return x

def main():
    data=prepare();study.FEATURES=study.FEATURES+EXTERNAL;study.MIN_VALID_COVERAGE=.10;results=[study.evaluate(data,"cash_direction"),study.evaluate(data,"gap_direction")]
    payload={"status":"candidate feature extension on a previously inspected historical period; forward confirmation required","method":"After-hours institutional pulse plus strictly prior-dated US/SOX/VIX/ADR complexion; same nested 756/126/126 walk-forward.","features":study.FEATURES,"results":results}
    out=ROOT/"reports";(out/"afterhours_with_complexion.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# After-hours pulse plus overseas complexion","",payload['status'],payload['method'],"","| Target | Cases | Accuracy | Coverage | Baseline | Edge | Wilson lower | Numeric gate |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results:lines.append(f"| {r['target']} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['baseline']:.2%} | {r['edge']:.2%} | {r['wilson_95_lower']:.2%} | {r['passed']} |")
    (out/"afterhours_with_complexion.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps({"status":payload['status'],"results":[{k:v for k,v in r.items() if k!='windows'} for r in results]},indent=2))

if __name__=="__main__":main()
