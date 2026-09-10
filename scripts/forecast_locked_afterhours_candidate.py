import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_afterhours_institutional as study
import research_afterhours_with_complexion as extended

LOG=ROOT/"reports"/"locked_afterhours_forecast_log.json"

def feature_frame():
    fut=pd.read_csv(ROOT/"data"/"processed"/"factors"/"futures_institutional_after_hours.csv");fut["date"]=pd.to_datetime(fut.date);fut["net"]=pd.to_numeric(fut.long_deal_volume)-pd.to_numeric(fut.short_deal_volume);fut["kind"]=fut.institutional_investors.map(study.investor_label);fp=fut.pivot_table(index="date",columns="kind",values="net",aggfunc="sum",fill_value=0);fp["total"]=fp.sum(axis=1);fp=fp.rename(columns={c:f"futures_{c}" for c in fp.columns})
    opt=pd.read_csv(ROOT/"data"/"processed"/"factors"/"option_institutional_after_hours.csv");opt["date"]=pd.to_datetime(opt.date);net=pd.to_numeric(opt.long_deal_volume)-pd.to_numeric(opt.short_deal_volume);put=opt.call_put.astype(str).str.strip().str.lower().isin(["put","p","賣權"]);opt["net"]=net.where(~put,-net);opt["kind"]=opt.institutional_investors.map(study.investor_label);op=opt.pivot_table(index="date",columns="kind",values="net",aggfunc="sum",fill_value=0);op["total"]=op.sum(axis=1);op=op.rename(columns={c:f"option_{c}" for c in op.columns})
    night=pd.read_csv(ROOT/"data"/"processed"/"taiwan_futures_night.csv");night["date"]=pd.to_datetime(night.signal_date);night=night.set_index("date")[["tx_night_return","tx_night_spread_per"]]
    base=fp.join(op,how="inner").join(night,how="left").reset_index().sort_values("date")
    ext=pd.read_csv(ROOT/"data"/"processed"/"external_markets.csv");ext["external_date"]=pd.to_datetime(ext.date)
    return pd.merge_asof(base,ext[["external_date"]+extended.EXTERNAL].sort_values("external_date"),left_on="date",right_on="external_date",direction="backward",allow_exact_matches=False,tolerance=pd.Timedelta("4D"))

def wilson_lower(hits,cases,z=1.959963984540054):
    if not cases:return 0.0
    p=hits/cases;den=1+z*z/cases
    return (p+z*z/(2*cases)-z*np.sqrt((p*(1-p)+z*z/(4*cases))/cases))/den

def external_majority(row):
    values=[row.get(c) for c in ["sp500_return_1d","nasdaq_return_1d","sox_return_1d","dow_return_1d","tsm_adr_return_1d"]]
    signs=[np.sign(v) for v in values if v is not None and np.isfinite(v) and v != 0]
    return 1 if signs and sum(signs) >= 0 else -1


def main():
    manifest=json.loads((ROOT/"config"/"locked_afterhours_candidate.json").read_text(encoding="utf-8"));history=extended.prepare();study.FEATURES=manifest["features"];study.MIN_VALID_COVERAGE=.10;history["gap_direction"]=np.where(history.gap_return>=0,1,-1);train=history.tail(study.TRAIN);cfg=study.select(train,"gap_direction")
    cash=pd.read_csv(ROOT/"data"/"processed"/"twii_daily.csv",usecols=["date","open","close"]);cash["date"]=pd.to_datetime(cash.date);cash["gap_return"]=pd.to_numeric(cash.open,errors="coerce")/pd.to_numeric(cash.close,errors="coerce").shift(1)-1;latest_cash=cash.date.max();future=feature_frame();future=future[future.date>latest_cash]
    existing=json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
    outcomes=cash.set_index(cash.date.dt.date.astype(str))["gap_return"]
    for record in existing:
        if record.get("outcome") is None and record.get("signal_date") in outcomes.index and pd.notna(outcomes.loc[record["signal_date"]]):
            actual_return=float(outcomes.loc[record["signal_date"]]);actual_value=1 if actual_return>=0 else -1;record["outcome"]="gap_up" if actual_value>0 else "gap_down";record["actual_gap_return"]=actual_return;record["hit"]=bool(record["prediction_value"]==actual_value) if record.get("actionable") else None;record["baseline_hit"]=bool(record["baseline_prediction_value"]==actual_value) if record.get("actionable") and record.get("baseline_prediction_value") is not None else None
    known={r["signal_date"] for r in existing};created=[]
    if cfg:
        _,lam,threshold,vacc,vn=cfg;model=study.fit(train,"gap_direction",lam)
        for _,row in future.iterrows():
            day=str(row.date.date())
            if day in known:continue
            one=pd.DataFrame([row]);prediction,confidence=study.predict(model,one);baseline=external_majority(row);record={"candidate_id":manifest["candidate_id"],"created_from_locked_algorithm":True,"signal_date":day,"prediction":"gap_up" if prediction[0]>0 else "gap_down","prediction_value":int(prediction[0]),"baseline":"external_majority","baseline_prediction_value":int(baseline),"confidence_score":float(confidence[0]),"threshold":float(threshold),"actionable":bool(confidence[0]>=threshold),"lambda":float(lam),"validation_accuracy_at_run":float(vacc),"validation_cases_at_run":int(vn),"outcome":None};existing.append(record);created.append(record)
    LOG.write_text(json.dumps(existing,ensure_ascii=False,indent=2),encoding="utf-8")
    resolved=[r for r in existing if r.get("outcome") is not None];actionable=[r for r in resolved if r.get("actionable")];hits=sum(bool(r.get("hit")) for r in actionable);cases=len(actionable);accuracy=hits/cases if cases else 0.0;baseline_hits=sum(bool(r.get("baseline_hit")) for r in actionable);baseline_accuracy=baseline_hits/cases if cases else 0.0;coverage=cases/len(resolved) if resolved else 0.0;lower=wilson_lower(hits,cases);gate=manifest["forward_gate"];passed=cases>=gate["minimum_new_cases"] and accuracy>=gate["minimum_accuracy"] and coverage>=gate["minimum_coverage"] and lower>=gate["minimum_wilson_95_lower"] and (not gate.get("must_beat_external_majority") or accuracy>baseline_accuracy)
    status=ROOT/"reports"/"locked_afterhours_candidate_status.md";status.write_text("\n".join(["# Locked after-hours candidate status","",f"Candidate: {manifest['candidate_id']}",f"Latest cash date: {latest_cash.date()}",f"New eligible after-hours dates: {len(future)}",f"New forecasts written: {len(created)}",f"Total prospective records: {len(existing)}",f"Resolved/actionable: {len(resolved)}/{cases}",f"Forward hits/accuracy: {hits}/{cases} = {accuracy:.2%}",f"External-majority baseline: {baseline_hits}/{cases} = {baseline_accuracy:.2%}",f"Model edge over baseline: {accuracy-baseline_accuracy:.2%}",f"Forward coverage: {coverage:.2%}",f"Forward Wilson 95% lower: {lower:.2%}",f"Forward gate passed: {passed}","",manifest['warning']])+"\n",encoding="utf-8")
    print(f"Locked candidate: {len(created)} new forecast(s); {len(existing)} total prospective record(s).")

if __name__=="__main__":main()
