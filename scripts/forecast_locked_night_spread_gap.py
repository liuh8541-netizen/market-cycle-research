"""Prospective logger for the canonical night-spread opening-gap rule."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_premarket_night_signal import TRAIN,select,wilson
from research_premarket_open_gap import prepare
import research_afterhours_with_complexion as extdef

MANIFEST=ROOT/"config"/"locked_night_spread_gap_candidate.json";LOG=ROOT/"reports"/"locked_night_spread_gap_forecast_log.json"

def external_votes(frame):
    external=pd.read_csv(ROOT/"data"/"processed"/"external_markets.csv");external["external_date"]=pd.to_datetime(external.date);cols=["sp500_return_1d","nasdaq_return_1d","sox_return_1d","dow_return_1d","tsm_adr_return_1d"]
    x=pd.merge_asof(frame.sort_values("signal_date"),external[["external_date"]+cols].sort_values("external_date"),left_on="signal_date",right_on="external_date",direction="backward",allow_exact_matches=False,tolerance=pd.Timedelta("4D"));x["external_majority"]=np.where(np.sign(x[cols].fillna(0)).sum(axis=1)>=0,1,-1);return x

def main():
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"));history=prepare();history["prediction"]=np.where(history.tx_night_spread_per>=0,1,-1);history["confidence"]=history.tx_night_spread_per.abs();cfg=select(history.tail(TRAIN))
    cash=pd.read_csv(ROOT/"data"/"processed"/"twii_daily.csv",usecols=["date","open","close"]);cash["date"]=pd.to_datetime(cash.date);cash["gap_return"]=pd.to_numeric(cash.open)/pd.to_numeric(cash.close).shift(1)-1;cash["cash_close_return"]=pd.to_numeric(cash.close).pct_change();latest=cash.date.max();night=pd.read_csv(ROOT/"data"/"processed"/"taiwan_futures_night.csv");night["signal_date"]=pd.to_datetime(night.signal_date);future=external_votes(night[night.signal_date>latest].copy())
    existing=json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else [];outcomes=cash.set_index(cash.date.dt.date.astype(str))[["gap_return","cash_close_return"]]
    today=pd.Timestamp.now(tz="Asia/Taipei").date()
    for r in existing:
        # Canonical daily data intentionally excludes the still-open session.
        # Withdraw any outcome that an older runner may have derived from a
        # Yahoo intraday candle before this integrity guard existed.
        if pd.Timestamp(r["signal_date"]).date()>=today:
            r["outcome"]=None
            for key in ["actual_gap_return","hit","external_baseline_hit","night_return_baseline_hit","cash_close_outcome","actual_cash_close_return","cash_close_hit"]:
                r.pop(key,None)
            continue
        # Predictions, thresholds and actionability are immutable. Realized
        # OHLC may be revised by the source, so reconcile outcomes on every run.
        if r["signal_date"] in outcomes.index and pd.notna(outcomes.loc[r["signal_date"],"gap_return"]):
            ret=float(outcomes.loc[r["signal_date"],"gap_return"]);actual=1 if ret>=0 else -1;cash_ret=float(outcomes.loc[r["signal_date"],"cash_close_return"]);cash_actual=1 if cash_ret>=0 else -1;r["outcome"]="gap_up" if actual>0 else "gap_down";r["actual_gap_return"]=ret;r["hit"]=bool(r["prediction_value"]==actual) if r["actionable"] else None;r["external_baseline_hit"]=bool(r["external_baseline_value"]==actual) if r["actionable"] else None;r["night_return_baseline_hit"]=bool(r["night_return_baseline_value"]==actual) if r["actionable"] else None;r["cash_close_outcome"]="up" if cash_actual>0 else "down";r["actual_cash_close_return"]=cash_ret;r["cash_close_hit"]=bool(r["prediction_value"]==cash_actual) if r["actionable"] else None
    known={r["signal_date"] for r in existing};created=[]
    if cfg:
        _,threshold,tacc,tn=cfg
        for _,row in future.iterrows():
            day=str(row.signal_date.date())
            if day in known:continue
            pred=1 if row.tx_night_spread_per>=0 else -1;record={"candidate_id":manifest["candidate_id"],"created_from_locked_algorithm":True,"signal_date":day,"prediction":"gap_up" if pred>0 else "gap_down","prediction_value":pred,"confidence_score":abs(float(row.tx_night_spread_per)),"threshold":float(threshold),"actionable":bool(abs(float(row.tx_night_spread_per))>=threshold),"external_baseline_value":int(row.external_majority),"night_return_baseline_value":1 if row.tx_night_return>=0 else -1,"training_accuracy_at_run":tacc,"training_cases_at_run":tn,"outcome":None};existing.append(record);created.append(record)
    LOG.write_text(json.dumps(existing,ensure_ascii=False,indent=2),encoding="utf-8");resolved=[r for r in existing if r.get("outcome") is not None];action=[r for r in resolved if r["actionable"]];n=len(action);hits=sum(bool(r.get("hit")) for r in action);eh=sum(bool(r.get("external_baseline_hit")) for r in action);rh=sum(bool(r.get("night_return_baseline_hit")) for r in action);acc=hits/n if n else 0;eacc=eh/n if n else 0;racc=rh/n if n else 0;coverage=n/len(resolved) if resolved else 0;lower=wilson(hits,n);g=manifest["forward_gate"];passed=n>=g["minimum_new_cases"] and acc>=g["minimum_accuracy"] and coverage>=g["minimum_coverage"] and lower>=g["minimum_wilson_95_lower"] and acc>eacc
    cash_hits=sum(bool(r.get("cash_close_hit")) for r in action);cash_acc=cash_hits/n if n else 0
    status=ROOT/"reports"/"locked_night_spread_gap_status.md";status.write_text("\n".join(["# Locked night-spread opening-gap status","",f"Candidate: {manifest['candidate_id']}",f"Latest cash date: {latest.date()}",f"New forecasts written: {len(created)}",f"Prospective records: {len(existing)}",f"Resolved/actionable: {len(resolved)}/{n}",f"Opening-gap model: {hits}/{n} = {acc:.2%}",f"External majority: {eh}/{n} = {eacc:.2%}",f"Night-return baseline: {rh}/{n} = {racc:.2%}",f"Secondary cash-close tracking: {cash_hits}/{n} = {cash_acc:.2%}",f"Coverage: {coverage:.2%}",f"Wilson 95% lower: {lower:.2%}",f"Forward gate passed: {passed}","",manifest["warning"]])+"\n",encoding="utf-8");print(f"Locked night-spread gap: {len(created)} new; {len(existing)} total prospective record(s).")

if __name__=="__main__":main()
