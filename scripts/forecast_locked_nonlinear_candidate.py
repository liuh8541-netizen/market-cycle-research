"""Prospective-only logger for the locked nonlinear premarket candidate."""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_afterhours_amount_structure as amount
import research_nonlinear_premarket_state as modeldef

MANIFEST=ROOT/"config"/"locked_nonlinear_premarket_candidate.json"
LOG=ROOT/"reports"/"locked_nonlinear_forecast_log.json"

def feature_frame():
    factors=ROOT/"data"/"processed"/"factors"
    f=amount.institutional_frame(factors/"futures_institutional_after_hours.csv","futures")
    o=amount.institutional_frame(factors/"option_institutional_after_hours.csv","option",option=True)
    night=pd.read_csv(ROOT/"data"/"processed"/"taiwan_futures_night.csv");night["date"]=pd.to_datetime(night.signal_date);night=night.set_index("date")[["tx_night_return","tx_night_range","tx_night_volume","tx_night_spread_per"]]
    x=f.join(o,how="inner").join(night,how="left").reset_index().sort_values("date")
    external=pd.read_csv(ROOT/"data"/"processed"/"external_markets.csv");external["external_date"]=pd.to_datetime(external.date)
    x=pd.merge_asof(x,external[["external_date"]+modeldef.ext.EXTERNAL].sort_values("external_date"),left_on="date",right_on="external_date",direction="backward",allow_exact_matches=False,tolerance=pd.Timedelta("4D"))
    votes=np.sign(x[["sp500_return_1d","nasdaq_return_1d","sox_return_1d","dow_return_1d","tsm_adr_return_1d"]].fillna(0));x["baseline"]=np.where(votes.sum(axis=1)>=0,1,-1)
    x["external_lag_days"]=(x.date-x.external_date).dt.days;x["weekday"]=x.date.dt.weekday;x["external_consensus_abs"]=votes.sum(axis=1).abs()/5;x["external_return_mean"]=x[modeldef.ext.EXTERNAL].mean(axis=1);x["external_return_dispersion"]=x[modeldef.ext.EXTERNAL].std(axis=1)
    return x

def features(frame):
    return [c for c in frame.columns if c.startswith("futures_") or c.startswith("option_")]+["tx_night_return","tx_night_range","tx_night_volume","tx_night_spread_per"]+modeldef.ext.EXTERNAL+["external_lag_days","weekday","external_consensus_abs","external_return_mean","external_return_dispersion"]

def wilson(h,n,z=1.959963984540054):
    if not n:return 0.0
    p=h/n;d=1+z*z/n;return (p+z*z/(2*n)-z*np.sqrt((p*(1-p)+z*z/(4*n))/n))/d

def mcnemar(a,b):
    n=a+b
    if not n:return 1.0
    return min(1.0,2*sum(comb(n,k) for k in range(min(a,b)+1))/(2**n))

def main():
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"));history=modeldef.prepare();train=history.tail(modeldef.TRAIN);fit=train.iloc[:-modeldef.VALID];valid=train.iloc[-modeldef.VALID:];cols=features(history);cfg=modeldef.select(fit,valid,cols)
    cash=pd.read_csv(ROOT/"data"/"processed"/"twii_daily.csv",usecols=["date","open","close"]);cash["date"]=pd.to_datetime(cash.date);cash["gap_return"]=pd.to_numeric(cash.open,errors="coerce")/pd.to_numeric(cash.close,errors="coerce").shift(1)-1;latest_cash=cash.date.max();future=feature_frame();future=future[future.date>latest_cash]
    existing=json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else [];outcomes=cash.set_index(cash.date.dt.date.astype(str))["gap_return"]
    for r in existing:
        if r.get("outcome") is None and r["signal_date"] in outcomes.index and pd.notna(outcomes.loc[r["signal_date"]]):
            ret=float(outcomes.loc[r["signal_date"]]);actual=1 if ret>=0 else -1;r["outcome"]="gap_up" if actual>0 else "gap_down";r["actual_gap_return"]=ret;r["hit"]=bool(r["prediction_value"]==actual) if r["actionable"] else None;r["baseline_hit"]=bool(r["baseline_prediction_value"]==actual) if r["actionable"] else None
    known={r["signal_date"] for r in existing};created=[]
    if cfg:
        _,model,threshold,rounds,lr,vacc,vbacc,vn=cfg
        for _,row in future.iterrows():
            day=str(row.date.date())
            if day in known:continue
            one=pd.DataFrame([row]);pred,conf=modeldef.prediction(model,one,cols);actionable=bool(conf[0]>=threshold);record={"candidate_id":manifest["candidate_id"],"created_from_locked_algorithm":True,"signal_date":day,"prediction":"gap_up" if pred[0]>0 else "gap_down","prediction_value":int(pred[0]),"baseline":"external_majority","baseline_prediction_value":int(row.baseline),"confidence_score":float(conf[0]),"threshold":float(threshold),"actionable":actionable,"rounds":rounds,"learning_rate":lr,"validation_accuracy_at_run":vacc,"validation_baseline_at_run":vbacc,"validation_cases_at_run":vn,"outcome":None};existing.append(record);created.append(record)
    LOG.write_text(json.dumps(existing,ensure_ascii=False,indent=2),encoding="utf-8")
    resolved=[r for r in existing if r.get("outcome") is not None];action=[r for r in resolved if r.get("actionable")];n=len(action);hits=sum(bool(r.get("hit")) for r in action);bh=sum(bool(r.get("baseline_hit")) for r in action);acc=hits/n if n else 0;bacc=bh/n if n else 0;coverage=n/len(resolved) if resolved else 0;lower=wilson(hits,n);mo=sum(r.get("hit") and not r.get("baseline_hit") for r in action);bo=sum(not r.get("hit") and r.get("baseline_hit") for r in action);p=mcnemar(mo,bo);gate=manifest["forward_gate"];passed=n>=gate["minimum_new_cases"] and acc>=gate["minimum_accuracy"] and coverage>=gate["minimum_coverage"] and lower>=gate["minimum_wilson_95_lower"] and acc>bacc and p<=gate["maximum_paired_mcnemar_p"]
    status=ROOT/"reports"/"locked_nonlinear_candidate_status.md";status.write_text("\n".join(["# Locked nonlinear premarket candidate status","",f"Candidate: {manifest['candidate_id']}",f"Latest cash date: {latest_cash.date()}",f"New forecasts written: {len(created)}",f"Prospective records: {len(existing)}",f"Resolved/actionable: {len(resolved)}/{n}",f"Model: {hits}/{n} = {acc:.2%}",f"External majority: {bh}/{n} = {bacc:.2%}",f"Edge: {acc-bacc:.2%}",f"Wilson 95% lower: {lower:.2%}",f"Model-only / baseline-only: {mo}/{bo}; McNemar p={p:.4f}",f"Forward gate passed: {passed}","",manifest["warning"]])+"\n",encoding="utf-8");print(f"Locked nonlinear candidate: {len(created)} new; {len(existing)} total prospective record(s).")

if __name__=="__main__":main()
