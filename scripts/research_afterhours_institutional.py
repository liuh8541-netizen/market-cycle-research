import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TRAIN=756; VALID=126; TEST=126; LAMBDAS=[.1,1,10,100,1000]
MIN_VALID_COVERAGE=.10
FEATURES=["futures_total","futures_foreign","futures_trust","futures_dealer","option_total","option_foreign","option_trust","option_dealer","tx_night_return","tx_night_spread_per"]

def investor_label(value):
    v=str(value).strip().lower()
    if v in ["外資","foreign","foreign_investor"]:return "foreign"
    if v in ["投信","investment_trust","trust"]:return "trust"
    if v in ["自營商","dealer"]:return "dealer"
    return "other"

def prepare():
    fut=pd.read_csv(ROOT/"data"/"processed"/"factors"/"futures_institutional_after_hours.csv"); fut["date"]=pd.to_datetime(fut.date); fut["net"]=pd.to_numeric(fut.long_deal_volume)-pd.to_numeric(fut.short_deal_volume); fut["kind"]=fut.institutional_investors.map(investor_label)
    fp=fut.pivot_table(index="date",columns="kind",values="net",aggfunc="sum",fill_value=0); fp["total"]=fp.sum(axis=1); fp=fp.rename(columns={c:f"futures_{c}" for c in fp.columns})
    opt=pd.read_csv(ROOT/"data"/"processed"/"factors"/"option_institutional_after_hours.csv"); opt["date"]=pd.to_datetime(opt.date); net=pd.to_numeric(opt.long_deal_volume)-pd.to_numeric(opt.short_deal_volume); is_put=opt.call_put.astype(str).str.strip().str.lower().isin(["put","p","賣權"]); opt["net"]=net.where(~is_put,-net); opt["kind"]=opt.institutional_investors.map(investor_label)
    op=opt.pivot_table(index="date",columns="kind",values="net",aggfunc="sum",fill_value=0); op["total"]=op.sum(axis=1); op=op.rename(columns={c:f"option_{c}" for c in op.columns})
    night=pd.read_csv(ROOT/"data"/"processed"/"taiwan_futures_night.csv"); night["date"]=pd.to_datetime(night.signal_date); night=night.set_index("date")[["tx_night_return","tx_night_spread_per"]]
    tw=pd.read_csv(ROOT/"data"/"processed"/"twii_daily.csv",usecols=["date","open","close"]); tw["date"]=pd.to_datetime(tw.date); tw["cash_return"]=tw.close.pct_change(); tw["gap_return"]=tw.open/tw.close.shift(1)-1; tw=tw.set_index("date")
    x=tw.join(fp,how="inner").join(op,how="inner").join(night,how="left").reset_index().sort_values("date")
    for c in FEATURES:
        if c not in x:x[c]=0.0
    return x

def fit(frame,target,lam):
    raw=frame[FEATURES].replace([np.inf,-np.inf],np.nan); med=raw.median(); raw=raw.fillna(med).fillna(0); mean=raw.mean(); std=raw.std().replace(0,1); X=((raw-mean)/std).to_numpy(float); X=np.column_stack([np.ones(len(X)),X]); y=frame[target].to_numpy(float); p=np.eye(X.shape[1])*lam;p[0,0]=0;w=np.linalg.pinv(X.T@X+p)@X.T@y;return w,med,mean,std

def predict(model,frame):
    w,med,mean,std=model; raw=frame[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0);X=((raw-mean)/std).to_numpy(float);score=np.column_stack([np.ones(len(X)),X])@w;return np.where(score>=0,1,-1),np.abs(score)

def select(train,target):
    fit_part=train.iloc[:-VALID];valid=train.iloc[-VALID:];best=None
    for lam in LAMBDAS:
        p,c=predict(fit(fit_part,target,lam),valid);actual=valid[target].to_numpy(int)
        for q in np.linspace(0,.9,10):
            threshold=float(np.quantile(c,q));use=c>=threshold;n=int(use.sum())
            if n<13 or n/len(valid)<MIN_VALID_COVERAGE:continue
            acc=float((p[use]==actual[use]).mean());score=acc+.001*min(n,50)
            if best is None or score>best[0]:best=(score,lam,threshold,acc,n)
    return best

def evaluate(base,target):
    x=base.copy();source="cash_return" if target=="cash_direction" else "gap_return";x[target]=np.where(x[source]>=0,1,-1);records=[];rows=0;windows=[];start=0
    while start+TRAIN+TEST<=len(x):
        train=x.iloc[start:start+TRAIN];test=x.iloc[start+TRAIN:start+TRAIN+TEST];cfg=select(train,target);rows+=len(test)
        if cfg:
            _,lam,threshold,vacc,vn=cfg;p,c=predict(fit(train,target,lam),test);actual=test[target].to_numpy(int);use=c>=threshold;hit=p[use]==actual[use];records+=hit.astype(int).tolist();windows.append({"test_end":str(test.date.max().date()),"lambda":lam,"threshold":threshold,"validation_accuracy":vacc,"validation_cases":vn,"test_cases":int(use.sum()),"test_accuracy":float(hit.mean()) if len(hit) else 0})
        start+=TEST
    n=len(records);hits=sum(records);acc=hits/n if n else 0;cov=n/rows if rows else 0;actual=x.iloc[TRAIN:TRAIN+rows][target];up=float((actual==1).mean());baseline=max(up,1-up);lower=wilson(hits,n);passed=acc>=.9 and n>=100 and cov>=.1 and lower>=.8 and acc>baseline
    return {"target":target,"cases":n,"hits":hits,"accuracy":acc,"coverage":cov,"baseline":baseline,"edge":acc-baseline,"wilson_95_lower":lower,"passed":passed,"windows":windows}

def wilson(h,n,z=1.959963984540054):
    if not n:return 0.0
    p=h/n;d=1+z*z/n;return (p+z*z/(2*n)-z*np.sqrt((p*(1-p)+z*z/(4*n))/n))/d

def main():
    data=prepare();results=[evaluate(data,"cash_direction"),evaluate(data,"gap_direction")];payload={"method":"Same-trading-day after-hours institutional futures/options flows, available by 05:00; nested 756-day train with 126-day validation and 126-day OOS tests.","aligned_days":len(data),"features":FEATURES,"results":results,"passing":[r['target'] for r in results if r['passed']]};out=ROOT/"reports";(out/"afterhours_institutional_research.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# After-hours institutional premarket research","",payload['method'],"","| Target | Cases | Accuracy | Coverage | Baseline | Edge | Wilson lower | Passed |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in results:lines.append(f"| {r['target']} | {r['cases']} | {r['accuracy']:.2%} | {r['coverage']:.2%} | {r['baseline']:.2%} | {r['edge']:.2%} | {r['wilson_95_lower']:.2%} | {r['passed']} |")
    (out/"afterhours_institutional_research.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps({"aligned_days":len(data),"passing":payload['passing'],"results":[{k:v for k,v in r.items() if k!='windows'} for r in results]},indent=2))

if __name__=="__main__":main()
