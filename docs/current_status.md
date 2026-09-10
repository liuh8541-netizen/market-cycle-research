# 專案目前狀態

## 2026-07-24 正式結案狀態

- 固定歷史研究計畫已完成，結案完整性稽核通過。
- 原始實用門檻維持不變：命中率90%、至少100件、覆蓋率10%、Wilson 95%下限80%，且必須優於最強簡單基準。
- 大盤多日方向最佳正式結果為66.18%（1,986／3,001），低於68.42%基準；研究假設未獲支持。
- 正式程式已停用大盤多日方向訊號；1、5、20、60日歷史相似分布只保留為探索統計。
- 通過驗證的限定模型只有：
  - 同日現貨開盤缺口方向：97.38%（335／344）。
  - 開盤缺口重大振幅風險：92.66%（240／259），不預測漲跌。
- FinMind多日方向係數維持0；籌碼、期權、市值、估值、廣度、借券、外資持股與信用結構只作描述及新資料研究。
- 完整目標與事實差距見 `reports/final_research_conclusion.md`。
- 前瞻紀錄仍可每日增量追蹤，但不會用新結果回改已結案的歷史模型。

## 已完成

- 建立大盤生命周期六階段框架。
- 建立資料欄位規格。
- 建立 Google Apps Script 整合設計。
- 建立 FinMind Sponsor 方案下的配額與快取策略。
- 建立外掛式 token 讀取方式。
- 建立短期快取與 Drive 長期快取樣板。
- 建立 Google Sheets 批次寫入與去重樣板。
- 建立 Python baseline：
  - 特徵計算
  - 生命周期打分
  - 風險狀態判斷
  - 策略回測
  - walk-forward 驗證
  - Markdown / JSON 報告輸出
- 使用合成資料跑通完整流程。
- 接入 Yahoo Finance `^TWII` 台股加權指數歷史資料。
- 建立失敗因子分析與逐一排除流程。
- 建立 walk-forward 失敗因子排除驗證。
- 建立 FinMind 非價格因子接入計畫。
- 建立 FinMind 因子下載設定與下載腳本。
- 建立非價格因子特徵工程：
  - `chip_score`
  - `derivative_score`
  - `non_price_score`
- CLI 支援 `--factor-dir`，有因子資料時自動合併，沒有資料時維持價格模型。
- Google Apps Script 支援批次更新 FinMind 非價格因子資料表。
- 新增指定日期與當日指數的前後期間走勢分析：
  - 檢查輸入指數與資料收盤是否一致
  - 分析前 N 個交易日走勢
  - 分析後 N 個交易日走勢
  - 判斷高點反轉、低點反轉、趨勢延續或盤整
  - 驗證模型當日預測是否命中後續方向

## 合成資料流程測試

測試檔案：

- `data/processed/sample_index_daily.csv`
- `reports/sample_lifecycle_scored.csv`
- `reports/sample_backtest_report.json`
- `reports/sample_backtest_report.md`

測試結果摘要：

- 全期間 5 日方向命中率：約 54.00%
- walk-forward 平均 5 日方向命中率：約 49.78%
- 全期間策略最大回撤低於大盤
- 分段驗證顯示模型穩定性仍不足

這代表管線可執行，但不能宣稱已通過真實市場驗證。

## 台股加權指數真實資料驗證

資料來源：

- Yahoo Finance `^TWII`
- 期間：2000-01-04 至 2026-07-14
- 筆數：6503

輸出檔案：

- `data/raw/yahoo_twii_chart.json`
- `data/processed/twii_daily.csv`
- `reports/twii_lifecycle_scored.csv`
- `reports/twii_backtest_report.md`
- `reports/twii_failure_analysis_20d.md`
- `reports/twii_lifecycle_corrected.csv`

Baseline 結果：

- 全期間 5 日方向命中率：約 53.98%
- walk-forward 平均 5 日方向命中率：約 50.84%
- 策略最大回撤：約 -17.74%
- 大盤最大回撤：約 -66.22%
- walk-forward 正報酬視窗數：33 / 45

20 日方向預測失敗因子排除：

- 排除前可行動成功率：約 51.65%
- 找出的主要失敗因子：
  - `momentum_score = -2`
  - `stage_score_bucket = negative`
  - `lifecycle_stage = early_bear`
  - `risk_regime = risk_off`
- 逐一排除後，全樣本可行動成功率提升至約 55.12%
- 但 walk-forward 未看過區間中，排除前平均可行動成功率約 49.41%，排除後降至約 38.56%

結論：

- 失敗因子排除在全樣本上有效，但無法通過 walk-forward 泛化驗證。
- 目前不能把失敗因子排除規則視為正式模型。
- Baseline 具有降低回撤的風控價值，但方向預測能力仍不足。
- 下一輪應加入市場廣度、籌碼、期貨與選擇權資料，而不是只靠價格技術指標硬調。

## 下一步

1. 將 `FINMIND_TOKEN` 設為環境變數或 Google Apps Script Script Properties。
2. 執行 `scripts/fetch_finmind_factors.py` 下載真實非價格因子。
3. 用 `--factor-dir data/processed/factors` 重新跑 5 日、20 日、60 日 walk-forward 驗證。
4. 比較價格 baseline 與價格 + 非價格因子的改善幅度。
5. 將「方向預測」和「風險控管」拆成兩個模型，不混用同一套規則。
6. 補市場廣度資料，例如上漲下跌家數、創新高新低家數、成分股站上均線比例。

## FinMind 因子接入狀態

已完成：

- `docs/finmind_factor_plan.md`
- `config/finmind_factor_datasets.json`
- `scripts/fetch_finmind_factors.py`
- `src/market_lifecycle/factor_features.py`
- `scripts/generate_sample_factors.py`

已用合成非價格因子完成管線測試：

- `data/processed/sample_factors/`
- `reports/twii_sample_factor_lifecycle_scored.csv`
- `reports/twii_sample_factor_backtest_report.md`

注意：合成非價格因子只用來確認管線可執行，不代表真實 FinMind 因子已驗證有效。

## 指定日期前後走勢分析測試

測試案例：

- 日期：2022-01-05
- 輸入指數：18499.96
- 前後期間：各 20 個交易日
- 報告：`reports/context_2022-01-05.md`

結果：

- 輸入指數與資料收盤一致。
- 前 20 個交易日報酬：約 +3.95%。
- 後 20 個交易日報酬：約 -1.02%。
- 轉折型態：`top_reversal`。
- 模型當日判斷：`risk_on`，預測方向 `up`。
- 實際後續方向：`down`。
- 驗證結果：未命中。

另以錯誤輸入指數 18000 測試，系統可偵測與資料收盤差異約 -2.70%，並標記為不一致。

## 機率預報與自我模擬

已新增類似天氣預報的機率式輸出：

- 輸入預報日期。
- 只使用該日期以前的歷史資料。
- 找出相似生命周期、風險狀態、分數區間的歷史案例。
- 輸出未來 1、5、20、60 個交易日的上漲、下跌、盤整機率。
- 逐日歷史回放做自我模擬，統計命中率。

測試案例：

- 預報日期：2026-07-14
- 報告：`reports/twii_probability_forecast_2026-07-14.md`

預報結果：

- 1 日：預測盤整，上漲機率約 29.44%，下跌約 23.36%，盤整約 47.20%。
- 5 日：預測上漲，上漲機率約 49.52%。
- 20 日：預測上漲，上漲機率約 82.05%。
- 60 日：預測上漲，上漲機率約 95.48%。

自我模擬結果：

- 1 日命中率：約 42.71%。
- 5 日命中率：約 45.43%。
- 20 日命中率：約 53.25%。
- 60 日命中率：約 59.05%。
- 平均命中率：約 50.11%。
- 平均高信心命中率：約 51.91%。

結論：

- 60 日預報具有初步有效性。
- 20 日預報略高於隨機，但仍需更嚴格驗證。
- 1 日與 5 日短線預報目前失敗。
- 整體機率預報功能已完成，但模型尚未達到全週期穩定成功。

## 90% 高命中率情境研究

已新增通用市場情境庫，掃描：

- 單日急跌 / 急漲
- 5 日急跌 / 急漲
- 20 日波段跌 / 波段漲
- 接近歷史高點
- 中度 / 深度 / 崩跌級回撤
- 強多頭 / 弱多頭
- 強空頭 / 弱空頭
- 均線上方 / 下方
- RSI 過熱 / 超跌
- 高波動 / 低波動
- 生命周期階段

測試設定：

- 目標命中率：90%
- 最少樣本數：80
- 週期：1、5、20、60 個交易日
- 報告：`reports/twii_scenario_research_90.md`

結果：

- 達標情境數：0
- 最佳全樣本情境：`twenty_day_surge`，即 20 日漲幅超過 10% 後，未來 60 日上漲機率約 70.07%
- Walk-forward 平均命中率：約 47.24%
- Walk-forward 中含 90% 規則的視窗數：0

結論：

- 只用價格與技術狀態，找不到 90% 且樣本數足夠的穩定情境。
- 「大跌後」、「急漲後」、「強趨勢」等價格情境最高只到約 70% 等級。
- 若目標堅持 90%，必須加入非價格因子，例如三大法人、融資融券、期貨、選擇權、VIX、市場廣度。

## 因果情境資料庫

已建立每日因果案例資料庫：

- 輸出：`reports/twii_causal_database.csv`
- 摘要：`reports/twii_causal_database_report.md`
- 筆數：6503

每一筆包含：

- 基準日 OHLCV
- 生命周期階段
- 風險狀態
- 階段分數
- 前 1、5、20、60 日報酬與方向
- 後 1、5、20、60 日報酬與方向
- 前因情境
- 市場模式
- 事件強度
- 因果簽名

目前找到的主要因果線索：

- `deep_drawdown|strong_risk_off|normal|early_bear`
- 後 60 日預測：上漲
- 樣本數：108
- 命中率：約 82.41%
- 平均後續報酬：約 6.13%

這代表「深度回撤 + 強風險關閉 + early_bear」之後，中期反彈機率很高，是目前最有價值的研究方向。

但它仍未達 90%，因此不能宣稱已達高確定性預報標準。

## 最強因果線索失敗樣本分析

針對目前最強線索：

```text
deep_drawdown|strong_risk_off|normal|early_bear
```

已完成成功組與失敗組比較：

- 報告：`reports/twii_causal_edge_refinement.md`
- 週期：後 60 日
- 目標方向：上漲
- 原始樣本：108
- 成功：89
- 失敗：19
- 命中率：約 82.41%

成功組與失敗組差異較大的價格因子：

- 成功組平均成交量較高。
- 成功組 RSI 較低。
- 成功組趨勢分數與階段分數更弱。
- 成功組回撤更深。

嘗試用價格條件排除失敗樣本後：

- 無任何條件能達到 90% 且保留至少 50 筆樣本。
- 最佳排除條件只剩 14 筆，命中率約 85.71%，樣本太少且未達標。

結論：

- 此因果線索已接近價格資料能提供的上限。
- 要辨識剩下 19 筆失敗，必須加入非價格資料：
  - 三大法人
  - 融資融券
  - 期貨法人
  - 選擇權 Put/Call
  - VIX
  - 市場廣度
