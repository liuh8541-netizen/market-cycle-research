import json
import sys
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from market_lifecycle.factor_features import (
    build_futures_institutional_features,
    build_institutional_features,
    build_option_daily_features,
    build_option_institutional_features,
)

FACTOR=ROOT/"data"/"processed"/"factors"

def main():
    option_inst=pd.read_csv(FACTOR/"option_institutional.csv")
    labels=option_inst.call_put.astype(str).str.strip().str.lower()
    is_put=labels.isin(["put","p","賣權"])
    derived=build_option_institutional_features(option_inst)
    net=pd.to_numeric(option_inst.long_open_interest_balance_volume,errors="coerce")-pd.to_numeric(option_inst.short_open_interest_balance_volume,errors="coerce")
    manual=net.where(~is_put,-net).groupby(option_inst.date).sum().sort_index()
    actual=derived.set_index(derived.date.astype(str)).option_inst_net_proxy.sort_index()

    option_daily=pd.read_csv(FACTOR/"option_daily.csv")
    option_daily_derived=build_option_daily_features(option_daily)
    futures=pd.read_csv(FACTOR/"futures_institutional.csv")
    futures_derived=build_futures_institutional_features(futures)
    institution=pd.read_csv(FACTOR/"institutional_total.csv")
    institution_derived=build_institutional_features(institution)

    checks={
        "option_call_rows":int((labels=="買權").sum()),
        "option_put_rows":int(is_put.sum()),
        "option_unknown_labels":sorted(labels[~labels.isin(["call","c","買權","put","p","賣權"])].unique().tolist()),
        "option_manual_direction_matches_builder":bool(actual.equals(manual.reindex(actual.index))),
        "option_daily_both_sides_positive_days":int(((pd.to_numeric(option_daily.put_volume)>0)&(pd.to_numeric(option_daily.call_volume)>0)).sum()),
        "option_daily_days":len(option_daily),
        "option_pcr_rows":int(option_daily_derived.option_put_call_proxy.notna().sum()),
        "futures_institution_types":sorted(futures.institutional_investors.astype(str).str.strip().unique().tolist()),
        "futures_source_rows":len(futures),
        "futures_derived_days":len(futures_derived),
        "institution_source_names":sorted(institution.name.astype(str).str.strip().unique().tolist()),
        "institution_source_rows":len(institution),
        "institution_derived_days":len(institution_derived),
    }
    checks["passed"]=bool(checks["option_put_rows"]>0 and not checks["option_unknown_labels"] and checks["option_manual_direction_matches_builder"] and checks["option_daily_both_sides_positive_days"]==checks["option_daily_days"])
    out=ROOT/"reports"; (out/"finmind_factor_semantics_audit.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# FinMind factor semantics audit","",f"Passed: {checks['passed']}","",f"- Option call rows: {checks['option_call_rows']}",f"- Option put rows classified and sign-inverted: {checks['option_put_rows']}",f"- Unknown option labels: {checks['option_unknown_labels'] or 'none'}",f"- Manual option direction equals builder: {checks['option_manual_direction_matches_builder']}",f"- Option daily both-side positive days: {checks['option_daily_both_sides_positive_days']}/{checks['option_daily_days']}",f"- Futures institution types: {checks['futures_institution_types']}",f"- Futures rows/days: {checks['futures_source_rows']}/{checks['futures_derived_days']}",f"- Cash institutional source names: {checks['institution_source_names']}",f"- Cash institutional rows/days: {checks['institution_source_rows']}/{checks['institution_derived_days']}"]
    (out/"finmind_factor_semantics_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(checks,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
