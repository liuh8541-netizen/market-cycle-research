"""Rebuild canonical TX night-session features from saved FinMind daily rows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"data"/"processed"/"factors"/"futures_daily.csv"
OUTPUT=ROOT/"data"/"processed"/"taiwan_futures_night.csv"
AUDIT=ROOT/"reports"/"taiwan_futures_night_rebuild_audit.json"

def main():
    old=pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame();raw=pd.read_csv(SOURCE)
    x=raw[raw.trading_session.astype(str).str.strip().eq("after_market")].copy();x=x[~x.contract_date.astype(str).str.contains("/",regex=False)]
    for c in ["open","max","min","close","spread_per","volume"]:x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x[(x.volume>0)&(x.open>0)&(x["max"]>0)&(x["min"]>0)&(x.close>0)].copy();x["date"]=pd.to_datetime(x.date)
    x=x.sort_values(["date","volume"]).groupby("date",as_index=False).tail(1).sort_values("date")
    out=pd.DataFrame({"night_date":x.date.dt.date.astype(str),"signal_date":x.date.dt.date.astype(str),"contract_date":x.contract_date.astype(str),"tx_night_open":x.open,"tx_night_high":x["max"],"tx_night_low":x["min"],"tx_night_close":x.close,"tx_night_volume":x.volume.astype(int),"tx_night_return":x.close/x.open-1,"tx_night_range":x["max"]/x["min"]-1,"tx_night_spread_per":x.spread_per/100})
    old_mismatch=int((pd.to_datetime(old.signal_date)!=pd.to_datetime(old.night_date)).sum()) if len(old) and "signal_date" in old else 0
    old_dates=set(old.night_date.astype(str)) if len(old) else set();new_dates=set(out.night_date);audit={"source":str(SOURCE),"official_semantics":"TaiwanFuturesDaily after_market date D covers prior business day 15:00 through D 05:00 and is causal for D cash open.","old_rows":len(old),"old_signal_date_mismatches":old_mismatch,"rebuilt_rows":len(out),"first_date":out.night_date.iloc[0],"last_date":out.night_date.iloc[-1],"dates_removed_as_invalid_or_nonoutright":len(old_dates-new_dates),"dates_added_from_raw":len(new_dates-old_dates),"rebuilt_signal_date_mismatches":int((out.signal_date!=out.night_date).sum()),"zero_or_invalid_selected_rows":int(((out.tx_night_open<=0)|(out.tx_night_high<=0)|(out.tx_night_low<=0)|(out.tx_night_close<=0)|(out.tx_night_volume<=0)).sum())}
    out.to_csv(OUTPUT,index=False,encoding="utf-8");AUDIT.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
