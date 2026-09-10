import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_afterhours_institutional as study
import research_afterhours_with_complexion as extended

START=np.datetime64("2026-06-16")

def main():
    manifest=json.loads((ROOT/"config"/"locked_afterhours_candidate.json").read_text(encoding="utf-8"));data=extended.prepare();study.FEATURES=manifest["features"];study.MIN_VALID_COVERAGE=.10;data["gap_direction"]=np.where(data.gap_return>=0,1,-1);records=[]
    for idx,row in data[data.date>=START].iterrows():
        prior=data.loc[:idx-1].tail(study.TRAIN)
        if len(prior)<study.TRAIN:continue
        cfg=study.select(prior,"gap_direction")
        if not cfg:continue
        _,lam,threshold,vacc,vn=cfg;prediction,confidence=study.predict(study.fit(prior,"gap_direction",lam),data.loc[[idx]]);values=[row.get(c) for c in ["sp500_return_1d","nasdaq_return_1d","sox_return_1d","dow_return_1d","tsm_adr_return_1d"]];signs=[np.sign(v) for v in values if v is not None and np.isfinite(v) and v!=0];baseline=1 if signs and sum(signs)>=0 else -1
        records.append({"date":str(row.date.date()),"actionable":bool(confidence[0]>=threshold),"prediction":int(prediction[0]),"target":int(row.gap_direction),"hit":bool(prediction[0]==row.gap_direction),"external_majority":int(baseline),"baseline_hit":bool(baseline==row.gap_direction),"confidence":float(confidence[0]),"threshold":float(threshold),"validation_accuracy":float(vacc),"validation_cases":int(vn)})
    action=[r for r in records if r['actionable']];n=len(action);hits=sum(r['hit'] for r in action);bh=sum(r['baseline_hit'] for r in action);acc=hits/n if n else 0;bacc=bh/n if n else 0
    payload={"status":"reserved-tail diagnostic, not post-lock prospective evidence","period_start":"2026-06-16","all_days":len(records),"actionable_cases":n,"model_hits":hits,"model_accuracy":acc,"external_majority_hits":bh,"external_majority_accuracy":bacc,"model_edge":acc-bacc,"records":records}
    out=ROOT/"reports";(out/"locked_afterhours_tail_validation.json").write_text(json.dumps(payload,indent=2),encoding="utf-8");(out/"locked_afterhours_tail_validation.md").write_text("\n".join(["# Locked candidate reserved-tail diagnostic","",payload['status'],f"Period start: {payload['period_start']}",f"All/aligned actionable days: {payload['all_days']}/{n}",f"Model: {hits}/{n} = {acc:.2%}",f"External majority: {bh}/{n} = {bacc:.2%}",f"Model edge: {acc-bacc:.2%}"])+"\n",encoding="utf-8");print(json.dumps({k:v for k,v in payload.items() if k!='records'},indent=2))

if __name__=="__main__":main()
