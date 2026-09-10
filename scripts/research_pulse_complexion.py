import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_causal_phase import wilson_lower
from research_pulse_causes import prepare


HORIZONS = [20, 60, 120]
TRAIN_DAYS = 756
TEST_DAYS = 126
COLOR_ASSETS = ["sp500_close", "nasdaq_close", "sox_close", "dow_close", "tsm_adr_close", "micron_close", "samsung_close", "sk_hynix_close"]


def add_complexion(internal):
    ext = pd.read_csv(ROOT / "data" / "processed" / "external_markets.csv")
    ext["date"] = pd.to_datetime(ext["date"])
    signals = []
    for col in COLOR_ASSETS:
        if col in ext:
            signals.append(np.sign(pd.to_numeric(ext[col], errors="coerce").pct_change(20)).rename(col))
    breadth = pd.concat(signals, axis=1).mean(axis=1, skipna=True)
    vix_change = pd.to_numeric(ext["vix_close"], errors="coerce").pct_change(20)
    ext["breadth"] = breadth
    ext["vix_change"] = vix_change
    # Taiwanese decisions on date t use overseas information through t-1 only.
    ext[["breadth", "vix_change"]] = ext[["breadth", "vix_change"]].shift(1)
    x = pd.merge_asof(internal.sort_values("date"), ext[["date", "breadth", "vix_change"]].sort_values("date"), on="date", direction="backward", tolerance=pd.Timedelta("5D"))
    color = pd.Series("NEUTRAL", index=x.index, dtype="object")
    color[(x["breadth"] >= .50) & (x["vix_change"] <= 0)] = "RUDDY"
    color[(x["breadth"] <= -.50) & (x["vix_change"] > 0)] = "PALE"
    color[(x["breadth"] >= .50) & (x["vix_change"] > 0)] = "FEVERISH"
    color[(x["breadth"] <= -.50) & (x["vix_change"] <= 0)] = "COLD"
    x["complexion"] = color
    x["clinical_state"] = x["state"] + "|" + x["cause"] + "|" + x["complexion"]
    return x


def mapping(train):
    out = {}
    for key, group in train.dropna(subset=["target"]).groupby("clinical_state"):
        if len(group) < 15:
            continue
        counts = group["target"].value_counts()
        if counts.max() / len(group) >= .65:
            out[key] = int(counts.idxmax())
    return out


def evaluate(data, horizon):
    x = data.copy()
    future = x["close"].shift(-horizon) / x["close"] - 1
    x["target"] = np.where(future > 0, 1, -1).astype(float)
    x.loc[future.isna(), "target"] = np.nan
    records=[]; tested=0; start=0
    while start+TRAIN_DAYS+TEST_DAYS+horizon<=len(x):
        train=x.iloc[start:start+TRAIN_DAYS]
        test=x.iloc[start+TRAIN_DAYS:start+TRAIN_DAYS+TEST_DAYS].copy()
        test["prediction"]=test["clinical_state"].map(mapping(train))
        use=test["prediction"].notna()&test["target"].notna()
        for r in test.loc[use].itertuples(): records.append({"cause":r.cause,"complexion":r.complexion,"hit":int(r.prediction==r.target)})
        tested+=len(test); start+=TEST_DAYS
    frame=pd.DataFrame(records); n=len(frame); hits=int(frame.hit.sum()) if n else 0
    actual=x.iloc[TRAIN_DAYS:TRAIN_DAYS+tested].target.dropna(); up=float((actual==1).mean()) if len(actual) else .5; baseline=max(up,1-up)
    accuracy=hits/n if n else 0; coverage=n/tested if tested else 0; lower=wilson_lower(hits,n)
    groups={}
    if n:
        for (cause,color),g in frame.groupby(["cause","complexion"]):
            gh=int(g.hit.sum()); gn=len(g); groups[f"{cause}|{color}"]={"cases":gn,"hits":gh,"accuracy":gh/gn,"wilson_95_lower":wilson_lower(gh,gn)}
    return {"horizon_days":horizon,"cases":n,"hits":hits,"accuracy":accuracy,"coverage":coverage,"baseline":baseline,"edge":accuracy-baseline,"wilson_95_lower":lower,"passed":accuracy>=.9 and n>=100 and coverage>=.1 and lower>=.8 and accuracy>baseline,"by_diagnosis":groups}


def main():
    data=add_complexion(prepare())
    results=[evaluate(data,h) for h in HORIZONS]
    payload={"hypothesis":"Lagged overseas complexion distinguishes temporary external illness from a true change in Taiwan's internal cycle.","method":"Overseas breadth and VIX lagged one trading day; fixed clinical states; 3-year/6-month walk-forward.","results":results,"passing":[r['horizon_days'] for r in results if r['passed']]}
    out=ROOT/"reports"; (out/"pulse_complexion_research.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# Pulse and complexion research","",payload["method"],"","| Horizon | Cases | Accuracy | Coverage | Baseline | Edge | Wilson lower | Passed |","| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results: lines.append(f"| {r['horizon_days']} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['baseline']:.2%} | {r['edge']:.2%} | {r['wilson_95_lower']:.2%} | {r['passed']} |")
    for r in results:
        lines += ["",f"## {r['horizon_days']} days by pulse cause and complexion"]+[f"- {k}: {v['accuracy']:.2%} ({v['hits']}/{v['cases']}; lower {v['wilson_95_lower']:.2%})" for k,v in sorted(r['by_diagnosis'].items())]
    (out/"pulse_complexion_research.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"passing":payload["passing"],"results":[{k:v for k,v in r.items() if k!='by_diagnosis'} for r in results]},indent=2))


if __name__=="__main__": main()
