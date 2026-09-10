# 大盤週期研究變更紀錄

## 2026-09-10

### Sponsor 資料源修正

- `scripts/predict_market.py` 直接執行時會自動讀取 `config/.secrets/finmind_token.txt`，避免只在 PowerShell 包裝器中才使用 FinMind Sponsor token。
- 加權指數正式日K改為 FinMind Sponsor `TaiwanStockPrice / TAIEX` 優先；TWSE 作官方成交量與備援補強；Yahoo 僅作長歷史與盤中快照備援。
- `scripts/fetch_finmind_factors.py` 補上相同 token fallback，直接執行也能使用 Sponsor 權限。
- 驗證結果：`2026-09-10` 報告確認 `finmind_sponsor_taiex_used: true`，正式台股現貨與成交量更新至 `2026-09-10`。

### 資料意圖與兵法心理層

- 在 `scripts/predict_market.py` 的心理戰模型加入「資料即意圖痕跡」：
  - 夜盤看試探與預期差。
  - 日盤收盤看現貨承接與真偽裁判。
  - 成交量與族群廣度看換手是否健康。
  - 期現差與法人期貨看避險、套利、壓力測試或空單回補。
  - 新聞與生活心理面只作觸發按鈕，必須回到日盤與收盤驗證。
- `reports/today_market_forecast.md` 與 `reports/today_market_forecast_detail.md` 已輸出資料意圖判讀。
- `research/HUMAN_NATURE_CYCLE_FRAMEWORK.md` 補上「資料即意圖痕跡」章節，作為跨電腦與每日自動任務的共同思想框架。

### 自動修正排程治理

- 檢查 `C:/Users/User/.codex/automations/automation/automation.toml`，確認「台股大盤每日自主進化監控」為 `ACTIVE`。
- 清除排程中殘留的 `2026-09-08` 臨時盤中條件，改為通用戰術雷達規則。
- 排程已加入「資料是意圖表現」規則，要求每日回到收盤與病歷驗證，不得把未驗證研究升級為投資命令。

### 驗證

- `py -X utf8 -m py_compile scripts/predict_market.py scripts/fetch_finmind_factors.py` 通過。
- `py -X utf8 -m unittest tests.test_production_validation_policy tests.test_market_health tests.test_clinical_ledger` 通過，總計 77 tests OK。

### GitHub 狀態

- 目前 `G:/我的雲端硬碟/codex/大盤週期研究` 尚不是 Git repository，因此本次變更只能先登記在本地變更紀錄，尚未 commit / push 到 GitHub。
- 若要比照 `個股研究` 中央羅盤流程，需要先初始化 Git repo 或接上既有 GitHub remote，再提交本變更紀錄與程式修正。
