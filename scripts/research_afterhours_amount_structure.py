"""Test whether after-hours transaction amount structure adds premarket information."""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_afterhours_institutional as study
import research_afterhours_with_complexion as extstudy


def institutional_frame(path, prefix, option=False):
    x = pd.read_csv(path); x["date"] = pd.to_datetime(x.date); x["kind"] = x.institutional_investors.map(study.investor_label)
    lv = pd.to_numeric(x.long_deal_volume, errors="coerce").fillna(0); sv = pd.to_numeric(x.short_deal_volume, errors="coerce").fillna(0)
    la = pd.to_numeric(x.long_deal_amount, errors="coerce").fillna(0); sa = pd.to_numeric(x.short_deal_amount, errors="coerce").fillna(0)
    sign = np.where(x.call_put.astype(str).str.strip().str.lower().isin(["put", "p", "賣權"]), -1, 1) if option else 1
    x["volume_net"] = (lv - sv) * sign; x["amount_net"] = (la - sa) * sign
    x["volume_ratio"] = ((lv - sv) / (lv + sv).replace(0, np.nan) * sign).fillna(0)
    x["amount_ratio"] = ((la - sa) / (la + sa).replace(0, np.nan) * sign).fillna(0)
    pieces = []
    for value in ["volume_net", "amount_net", "volume_ratio", "amount_ratio"]:
        p = x.pivot_table(index="date", columns="kind", values=value, aggfunc="sum", fill_value=0)
        if value.endswith("net"): p["total"] = p.sum(axis=1)
        else: p["total"] = x.groupby("date")[value].mean()
        p = p.rename(columns={c: f"{prefix}_{value}_{c}" for c in p.columns}); pieces.append(p)
    return pd.concat(pieces, axis=1)


def prepare():
    factors = ROOT / "data" / "processed" / "factors"
    f = institutional_frame(factors / "futures_institutional_after_hours.csv", "futures")
    o = institutional_frame(factors / "option_institutional_after_hours.csv", "option", option=True)
    night = pd.read_csv(ROOT / "data" / "processed" / "taiwan_futures_night.csv")
    night["date"] = pd.to_datetime(night.signal_date); night = night.set_index("date")[["tx_night_return", "tx_night_range", "tx_night_volume", "tx_night_spread_per"]]
    tw = pd.read_csv(ROOT / "data" / "processed" / "twii_daily.csv", usecols=["date", "open", "close"])
    tw["date"] = pd.to_datetime(tw.date); tw["gap_return"] = tw.open / tw.close.shift(1) - 1
    x = tw.set_index("date").join(f, how="inner").join(o, how="inner").join(night, how="left").reset_index().sort_values("date")
    external = pd.read_csv(ROOT / "data" / "processed" / "external_markets.csv"); external["external_date"] = pd.to_datetime(external.date)
    x = pd.merge_asof(x, external[["external_date"] + extstudy.EXTERNAL].sort_values("external_date"), left_on="date", right_on="external_date", direction="backward", allow_exact_matches=False, tolerance=pd.Timedelta("4D"))
    votes = np.sign(x[["sp500_return_1d", "nasdaq_return_1d", "sox_return_1d", "dow_return_1d", "tsm_adr_return_1d"]].fillna(0)).sum(axis=1)
    x["baseline"] = np.where(votes >= 0, 1, -1); x["gap_direction"] = np.where(x.gap_return >= 0, 1, -1)
    return x


def mcnemar(a, b):
    n = a+b
    if not n:return 1.0
    return min(1.0, 2*sum(comb(n,k) for k in range(min(a,b)+1))/(2**n))


def evaluate(x):
    records=[]; windows=[]; rows=0; start=0
    while start+study.TRAIN+study.TEST <= len(x):
        train=x.iloc[start:start+study.TRAIN]; test=x.iloc[start+study.TRAIN:start+study.TRAIN+study.TEST]; cfg=study.select(train,"gap_direction"); rows += len(test)
        if cfg:
            _,lam,threshold,vacc,vn=cfg; pred,conf=study.predict(study.fit(train,"gap_direction",lam),test); use=conf>=threshold
            for pos in np.flatnonzero(use): records.append({"date":str(test.iloc[pos].date.date()),"target":int(test.iloc[pos].gap_direction),"model":int(pred[pos]),"baseline":int(test.iloc[pos].baseline)})
            windows.append({"test_end":str(test.date.max().date()),"cases":int(use.sum()),"model_accuracy":float((pred[use]==test.gap_direction.to_numpy(int)[use]).mean()) if use.sum() else 0,"baseline_accuracy":float((test.baseline.to_numpy(int)[use]==test.gap_direction.to_numpy(int)[use]).mean()) if use.sum() else 0,"lambda":lam,"threshold":threshold,"validation_accuracy":vacc,"validation_cases":vn})
        start += study.TEST
    n=len(records); mh=sum(r["model"]==r["target"] for r in records); bh=sum(r["baseline"]==r["target"] for r in records); mo=sum(r["model"]==r["target"] and r["baseline"]!=r["target"] for r in records); bo=sum(r["model"]!=r["target"] and r["baseline"]==r["target"] for r in records)
    acc=mh/n if n else 0; bacc=bh/n if n else 0; cov=n/rows if rows else 0; lower=study.wilson(mh,n); p=mcnemar(mo,bo)
    return {"cases":n,"hits":mh,"accuracy":acc,"coverage":cov,"wilson_95_lower":lower,"baseline_hits":bh,"baseline_accuracy":bacc,"edge":acc-bacc,"model_only_correct":mo,"baseline_only_correct":bo,"mcnemar_exact_p":p,"passed":n>=100 and cov>=.1 and acc>=.9 and lower>=.8 and acc>bacc and p<.05},windows,records


def main():
    x=prepare(); core=[c for c in x.columns if c.startswith("futures_") or c.startswith("option_")]
    study.FEATURES=core+["tx_night_return","tx_night_range","tx_night_volume","tx_night_spread_per"]+extstudy.EXTERNAL; study.MIN_VALID_COVERAGE=.10
    result,windows,records=evaluate(x); payload={"status":"exploratory; feature family was proposed after earlier historical inspection and requires prospective locking if it passes","hypothesis":"Transaction amount and normalized long-short structure distinguish directional institutional conviction from low-premium contract counts.","method":"Strict 756/126/126 nested rolling walk-forward; same-day after-hours data plus strictly prior-dated overseas closes; paired external-majority benchmark on identical selected dates.","features":study.FEATURES,"aligned_days":len(x),"result":result,"windows":windows,"records":records}
    out=ROOT/"reports";(out/"afterhours_amount_structure.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    r=result; lines=["# After-hours amount-structure research","",payload["status"],"",payload["hypothesis"],payload["method"],"","| Cases | Accuracy | Coverage | Wilson lower | External baseline | Edge | Model-only / baseline-only | McNemar p | Passed |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",f"| {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['wilson_95_lower']:.2%} | {r['baseline_accuracy']:.2%} | {r['edge']:.2%} | {r['model_only_correct']} / {r['baseline_only_correct']} | {r['mcnemar_exact_p']:.4f} | {r['passed']} |","","## Walk-forward windows","","| Test end | Cases | Model | Baseline |","| --- | ---: | ---: | ---: |"]
    for w in windows:lines.append(f"| {w['test_end']} | {w['cases']} | {w['model_accuracy']:.2%} | {w['baseline_accuracy']:.2%} |")
    (out/"afterhours_amount_structure.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps({"aligned_days":len(x),"feature_count":len(study.FEATURES),"result":result},indent=2))


if __name__=="__main__":main()
