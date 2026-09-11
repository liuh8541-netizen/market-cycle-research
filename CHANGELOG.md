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

- 已建立 Git repository，remote 指向 `https://github.com/liuh8541-netizen/market-cycle-research.git`。
- 初始治理版本已推送到 GitHub `main`，commit `58d117b163343068de9d4cfa84a9fce6608072b7`。

### 跨電腦接續

- 新增 `docs/cross_device_continuity.md`，記錄換電腦後的接續 SOP：clone/pull、安裝套件、恢復 FinMind Sponsor token、執行預測、安裝 Windows 每日排程、重建 Codex heartbeat 自主進化任務。
- `README.md` 補上跨電腦接手入口，確保另一台電腦可依 GitHub、handoff 文件與 SOP 延續目前模型設定。

## 2026-09-11

### 對談觀察整理成模型改善手段

- 新增 `research/MODEL_IMPROVEMENT_ROADMAP_FROM_DIALOGUE.md`，將使用者長期對談中累積的觀察整理為可實作規格。
- 核心整理項目包含：劇本不是答案而是檢查表、階梯式緩升/緩降與雙週換檔、47,000 關卡攻防病歷、夜盤前哨與日盤裁判、資料即意圖痕跡、心律健康監控、病灶基因/觸發按鈕/即效藥、孫子兵法與八卦模型化、每日自主修正流程。
- `research/HANDOFF_CURRENT.md` 與 `research/RESEARCH_INDEX.md` 補上此路線圖入口，確保跨電腦接手時不會遺失對談累積出的模型智慧。

### 46,000 嚇阻線戰術病歷

- 新增 `research/TACTICAL_CASE_2026-09-11_DETERRENCE_LINE.md`，記錄 2026-09-11 日盤圍繞 46,000 的戰術演進。
- 案例命名包含「藏線嚇阻，留退伏攻」、「借敵死線，轉己生命」、「太極守線，借空補多」、「守線待變，以逸待勞」。
- `research/TACTICAL_PRESSURE_TEST_46000_2026-09.md`、`research/MODEL_IMPROVEMENT_ROADMAP_FROM_DIALOGUE.md` 與 `research/RESEARCH_INDEX.md` 已補上此案例入口；後續須由夜盤與下週一走勢驗證，不得直接升級為投資命令。

### 主因與觸發開關分離

- `research/HUMAN_NATURE_CYCLE_FRAMEWORK.md` 新增「主因期限判斷」：外部新聞是引爆器，不是炸藥本身；炸藥是早已存在的病灶。
- `research/MODEL_IMPROVEMENT_ROADMAP_FROM_DIALOGUE.md` 補上主因期限模型，要求用病灶深度、觸發強度、日盤吸收力與復發次數判斷影響期。
- `research/HANDOFF_CURRENT.md` 補上交接規則：油價、通膨、利率、匯率、地緣政治與科技鏈估值不得被當成單日臨時主因，必須區分長期病灶與當日觸發。
