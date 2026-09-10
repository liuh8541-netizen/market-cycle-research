import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from research_latent_factor_purged import pca_fit,transform
from research_three_layer_purged import build_frame

HISTORY=1512; TEST=252; HORIZONS=[20,60,120]

def safe_corr(a,b):
    mask=np.isfinite(a)&np.isfinite(b)
    return float(np.corrcoef(a[mask],b[mask])[0,1]) if mask.sum()>=20 and np.std(a[mask])>0 and np.std(b[mask])>0 else None

def main():
    base,features=build_frame(); windows=[]; start=0
    while start+HISTORY+TEST+max(HORIZONS)<=len(base):
        train=base.iloc[start:start+HISTORY]; test=base.iloc[start+HISTORY:start+HISTORY+TEST]; model=pca_fit(train,features,1)
        combined=pd.concat([train.tail(20),test]); pc=transform(model,combined,features)[:,0]; level=pc[-len(test):]; d5=(pc[5:]-pc[:-5])[-len(test):]; d20=(pc[20:]-pc[:-20])[-len(test):]
        row={"test_start":str(test.date.min().date()),"test_end":str(test.date.max().date())}
        for h in HORIZONS:
            future=(base.close.shift(-h)/base.close-1).loc[test.index].to_numpy(float)
            row[f"level_corr_{h}"]=safe_corr(level,future); row[f"d5_corr_{h}"]=safe_corr(d5,future); row[f"d20_corr_{h}"]=safe_corr(d20,future)
        windows.append(row); start+=TEST
    summary={}
    for h in HORIZONS:
        for name in ["level","d5","d20"]:
            vals=[w[f"{name}_corr_{h}"] for w in windows if w[f"{name}_corr_{h}"] is not None]; positives=sum(v>0 for v in vals); negatives=sum(v<0 for v in vals)
            summary[f"{name}_{h}"]={"windows":len(vals),"positive_windows":positives,"negative_windows":negatives,"sign_flip_present":positives>0 and negatives>0,"median_correlation":float(np.median(vals)),"min_correlation":float(min(vals)),"max_correlation":float(max(vals))}
    payload={"question":"Is the latent stress factor stable while its effect on future returns changes sign?","method":"Fit PC1 on each prior six-year window, project the next year, then measure its level/5d/20d-change correlation with future returns. No future data enters PCA.","summary":summary,"windows":windows}
    out=ROOT/"reports"; (out/"latent_effect_drift_audit.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# Latent-factor effect drift audit","",payload['method'],"","| Signal / horizon | Windows | Positive | Negative | Median corr | Range | Sign flips |","| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for key,s in summary.items(): lines.append(f"| {key} | {s['windows']} | {s['positive_windows']} | {s['negative_windows']} | {s['median_correlation']:.3f} | {s['min_correlation']:.3f} to {s['max_correlation']:.3f} | {s['sign_flip_present']} |")
    (out/"latent_effect_drift_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
