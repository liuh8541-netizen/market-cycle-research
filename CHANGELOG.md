# 大盤週期研究變更紀錄

## 2026-09-12

### 對談核心價值觀

- 新增 `research/CORE_VALUES_FROM_DIALOGUE.md`，把長期對談中可長期保留的內容整理成模型最高層價值觀。
- 核心價值觀包含：市場是有生命的統計系統、預測追求勝率優勢、知己知彼是風險預知管控、預防勝於治療、資料是意圖痕跡、哲學必須科學化、複雜推理最後收斂為 0/1 主劇本、不可測因素只能風控、錯誤是進化燃料、AI 是高空視野與濾波器。
- `research/RESEARCH_INDEX.md` 與 `research/HANDOFF_CURRENT.md` 已加入入口，確保跨電腦接續時先讀取這份核心準則。

### 主波段週期調查

- 新增 `research/MAJOR_SWING_CYCLE_AUDIT_2026-09-12.md`，整理年度可交易波段與 4 月至今峰谷路徑。
- 專案資料顯示 2005 至今可交易波段歷史平均每年約 4.14 段，中位數 4 段；2026 年目前 5 段，近 252 交易日 6 段，屬高頻波段。
- 4 月至今主要峰谷為：2026-06-03 峰、2026-06-11 谷、2026-06-22 峰、2026-07-30 谷、2026-09-07 峰候選；目前屬 7/30 谷底後反攻的高檔回測/峰後驗證，不是單日漲跌問題。

### AI 濾波層與對談核心規則入模

- `scripts/predict_market.py` 新增 `build_dialogue_core_rules`，把長期對談中反覆出現的核心觀念正式轉成模型規則。
- 新增 `dialogue_core_rules_v1` 欄位：`chronic_conditions`、`trigger_conditions`、`transmission_checks`、`model_rules`、`report_policy`。
- `analyze_practical_cause_arbitration` 會讀取對談核心規則，將「慢性病灶不會一天消失、單日劇變要找觸發鈕與倉位重定價、夜盤是前哨日盤是裁判」納入實務病因仲裁。
- 主報告 `證據分層 / 實務病因` 改用 `dialogue_focus`，只輸出濾波後重點；細部推理保留在 JSON 與 detail。
- `tests/test_production_validation_policy.py` 新增測試，確認慢性病灶與單日觸發會被分離，且模型規則要求主報告只輸出仲裁結果。

### 技術方程式答案與推理式對照

- `scripts/predict_market.py` 新增 `technical_equation_answer_v1`，把卦位、均線、K線、破線與波動方程式合成為一個技術答案。
- 新增 `equation_reasoning_audit_v1`，用推理程序核對技術公式答案是否與 0/1 偏向、實務病因、日夜盤傳導與市場保護層一致。
- 主報告新增「公式/推理對照」與「技術/卦位答案」，讓報告先講答案，再列出算式來源。
- 「公式/推理對照」新增誤差追因、缺漏變數、候選變數與數學策略；原則是先補期現差、未平倉、選擇權、族群廣度、成交量與事件日曆等變數，再考慮權重、貝葉斯或狀態轉移，避免用高等數學掩蓋缺資料。
- 「公式/推理對照」新增邊際搖擺因子，分別累計偏1/偏0分數與淨差，保留小數，避免把攻上/攻不上簡化成各50%；真正沒有公開前兆且瞬間變卦的事件則標為不可測邊界，只能用風控防呆處理。
- 公式答案與公式/推理對照皆標記 `research_status: experimental_hypothesis`；所有討論與公式都視為候選假說與前瞻實驗，不得升級為定論。
- 測試確認技術層不只輸出固定公式步驟，也輸出方程式答案與公式/推理一致性。

### 驗證

- `py -X utf8 -m py_compile scripts/predict_market.py` 通過。
- `py -X utf8 -m unittest tests.test_production_validation_policy` 通過，64 tests OK。

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

### 市場防呆保護層

- `scripts/predict_market.py` 新增 `analyze_market_protection_layers`，把市場臨界保護機制正式模型化。
- `scripts/predict_market.py` 新增 `analyze_crisis_opportunity_interface`，每日標示危機線、轉機線、峰谷信號與可執行的風控作業。
- 六層檢查包含：價格線防呆、資金承接防呆、心理嚇阻防呆、跨盤/制度風控防呆、政策底線防呆、基本面體質防呆。
- `reports/today_market_forecast.md` 新增「市場防呆保護層」與「危機與轉機界面」重點答案；detail 報告保留六層表格、證據、失效條件與峰谷界面。
- 判讀語固定為：合理呼吸、臨界壓測、防呆偏弱、防呆失靈；目的在回答波動是否仍在可控制範圍，不產生投資命令。
- `research/HUMAN_NATURE_CYCLE_FRAMEWORK.md`、`research/MODEL_IMPROVEMENT_ROADMAP_FROM_DIALOGUE.md` 與 `research/HANDOFF_CURRENT.md` 已同步補上此規則，確保跨電腦接續時不遺失。

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
