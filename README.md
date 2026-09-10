# 大盤生命周期預測研究

本專題目標是建立一套可驗證、可迭代的大盤生命周期判讀框架，用來辨識市場目前處於哪一個階段，並評估未來一段時間轉入下一階段的機率。

## 基本思想框架

本專案的核心不是把股市看成純數學亂數，而是把股市視為人類心理、物質慾望、資金目的與風險承受度的總合。價格、成交量、K線、期貨、法人、融資與族群廣度都是表層結果；真正推動結果的是恐懼、貪婪、猶豫、試探、誘惑、承接、換手與出清等人性行為。

因此模型固定採用三層架構：

1. 底層因：人性、利益、恐懼、貪婪、物極必反與周而復始。
2. 中層法：八卦生命週期、五行消長、孫子兵法攻防、主趨勢與殘餘風險清洗。
3. 表層果：價格、量能、K線、日夜盤、外部市場、籌碼、族群廣度與風險值。

數據不是被否定，而是用來驗證人性與週期框架是否成立。主趨勢不因單日波動被推翻；單日波動主要用來判斷節奏、時機、換手、清洗是否有效，以及潛伏病灶是否被外在條件觸發。

## 研究問題

- 現在的大盤比較像生命周期中的哪一個階段？
- 哪些指標能最早提示階段轉換？
- 不同周期長度（日、週、月）下，階段判斷是否一致？
- 預測結果是否能在歷史資料中穩定通過驗證？

## 生命周期階段假設

初版採用六階段框架：

1. 築底期：下跌動能衰竭，波動仍高，成交或廣度開始改善。
2. 初升期：價格突破中期均線，廣度擴散，風險偏好回升。
3. 主升期：趨勢明確，均線多頭排列，類股輪動健康。
4. 高檔震盪期：價格創高但動能、廣度或量能出現背離。
5. 初跌期：關鍵支撐跌破，廣度惡化，波動放大。
6. 主跌期：趨勢轉空，反彈無法修復結構，防禦資產相對占優。

## 資料面向

### 價格與趨勢

- 指數收盤價、開高低收、成交量
- 均線斜率與均線排列
- 高低點結構
- 報酬率與最大回撤

### 動能與波動

- RSI、MACD、KD
- ATR、歷史波動率
- 上漲與下跌動能變化

### 市場廣度

- 上漲家數與下跌家數
- 創新高與創新低家數
- 成分股高於均線比例
- 類股輪動與擴散程度

### 資金與籌碼

- 外資、投信、自營商買賣超
- 融資融券變化
- 成交值與量價背離

### 總體與風險偏好

- 利率、匯率、通膨、景氣指標
- 信用利差
- VIX 或區域波動指標
- 美股、美元、債券、原物料等跨市場訊號

## 初版模型方向

先從可解釋模型開始，再逐步加入機器學習：

- 規則式打分：每個指標轉成多空分數，彙總成生命周期分數。
- 隱馬可夫模型：讓市場階段作為隱藏狀態，由價格、波動、廣度推估。
- 分類模型：以歷史標註階段訓練 Random Forest、XGBoost 或 Logistic Regression。
- 轉換機率模型：預測未來 1、4、12 週進入下一階段的機率。

## 驗證方法

- Walk-forward validation，避免偷看未來資料。
- 比較不同市場環境：多頭、空頭、盤整、金融危機、升息循環。
- 評估指標：
  - 階段分類準確率
  - 轉折提前量
  - 錯誤警報率
  - 策略化後的報酬、回撤、勝率

## 專案結構

```text
data/
  raw/          原始資料
  processed/    清理後資料
notebooks/      研究筆記與實驗
src/            可重複執行的程式
reports/        圖表、結論、週期判讀報告
docs/           方法論與研究紀錄
```

## 下一步

1. 決定研究標的：台股加權指數、S&P 500、NASDAQ、或多市場比較。
2. 收集日線與週線資料。
3. 建立第一版生命周期標註規則。
4. 產出第一份歷史週期切分圖。
5. 用簡單規則式模型做 baseline。

## 執行預測與驗證

準備一份歷史日線 CSV，至少包含：

```text
date,open,high,low,close,volume
```

執行：

```bash
python -m market_lifecycle.cli \
  --input data/processed/index_daily.csv \
  --output reports/lifecycle_scored.csv \
  --report reports/backtest_report.json \
  --markdown-report reports/backtest_report.md
```

輸出內容：

- `lifecycle_scored.csv`：每日生命周期階段、分數、風險狀態與預測方向。
- `backtest_report.json`：機器可讀的驗證指標。
- `backtest_report.md`：人工可讀的研究報告。

若已下載 FinMind 非價格因子，可加入：

```bash
python -m market_lifecycle.cli \
  --input data/processed/twii_daily.csv \
  --factor-dir data/processed/factors \
  --output reports/twii_lifecycle_scored.csv \
  --report reports/twii_backtest_report.json \
  --markdown-report reports/twii_backtest_report.md
```

非價格因子接入計畫見 `docs/finmind_factor_plan.md`。

## 雲端資料夾的一鍵預測與前瞻紀錄

在任何同步此專案的 Windows 電腦上，執行專案根目錄的 `run_predict_market.bat`。程式會更新資料、產生今日報告，並維護不可回填的前瞻研究紀錄。

跨電腦接手時，先讀 `research/HANDOFF_CURRENT.md`。該檔保存目前唯一結論協議：八卦生命週期、46,000 灘頭堡攻防、夜日盤驗證、人類行為模式、健康價值判斷、買賣風控標註、孫子兵法戰略語言與每日自主進化邊界。換電腦後若 Codex App heartbeat automation 未同步，依該檔重建每日自主進化監控，不需重新向使用者詢問完整邏輯。

每日自主監控必須上網覽讀國際重大財經政治消息，整理 `reports/daily_global_news_risk.json`，再併入 `reports/today_market_forecast.md` 的外部事件重置與風險評估；若新聞風險缺漏，報告必須明確標示缺口。

報告重點頁先看 `總仲裁`，再看單一分數。總仲裁會把大盤健康指數、基本面命格、外部事件重置、夜盤三基準、族群廣度、心理戰與模型可靠度放在同一裁判層；若健康分數偏高但外部風險已啟動，短線劇本必須降權為防守觀察。

台指期夜盤必須分開顯示三種基準：官方漲跌、夜盤開收到收盤報酬、相對前一有效夜盤收盤。三者不得混用；官方漲跌偏開盤缺口，開收報酬偏夜盤路徑，前夜比較偏夜盤自身連續性。

每日流程包含 `scripts/fetch_global_news_risk.py` 新聞風險初篩骨架；RSS/自動抓取失敗時保留既有 `reports/daily_global_news_risk.json` 並標示缺口，不可把缺資料解讀成無風險。

每日流程也會產生 `reports/previous_day_forecast_error_analysis.md`，核對最近一筆已到期的一日誤差。`one_day_psychology_overlay_v1` 僅在心理固定方向分數絕對值至少為 2 時修正一日分布；原歷史分布會保留在 `fixed_factor_review`，5、20、60 日預測不受影響。弱分數但夜盤收近低點、回收不足時只標示尾部風險，不強制改方向。

固定一日覆寫因數可用 `python scripts/calibrate_one_day_psychology_overlay.py` 完整重算，結果寫入 `reports/one_day_psychology_overlay_calibration.md`。週五伏筆與週一脈象的長歷史驗證可用 `python scripts/research_monday_weekly_pulse.py` 重建；該研究目前只允許病程分流監測，不是固定週方向訊號。

Q85開盤缺口候選的預測時點是09:00後，因此早晨執行批次檔時會啟動隱藏監測程序：它等到Yahoo可取得當日開盤價後寫入一次性紀錄，成功後自行停止。13:30後不允許補寫當日預測；結果只在下一日核對。狀態見 `reports/prospective_open_gap_cash_close_status.md`。

## 指定日期驗證

若要輸入某一天的大盤指數資料，並驗證模型當天預測是否符合後續走勢，可使用：

```bash
python -m market_lifecycle.cli \
  --input data/processed/index_daily.csv \
  --output reports/lifecycle_scored.csv \
  --report reports/backtest_report.json \
  --validate-date 2022-01-28 \
  --validate-horizon-days 20 \
  --validation-report reports/point_validation_2022-01-28.md
```

驗證方式：

- 使用指定日期當天的收盤資料；若該日非交易日，使用該日之前最近一個交易日。
- 依當天生命周期階段產生預測方向。
- 往後看指定交易天數的實際報酬。
- 比對預測方向與實際方向，輸出命中或未命中。

## 指定日期與當日指數的前後走勢分析

若要輸入日期與當日大盤指數，並分析前後期間走勢：

```bash
python -m market_lifecycle.cli \
  --input data/processed/twii_daily.csv \
  --output reports/twii_lifecycle_scored.csv \
  --report reports/twii_backtest_report.json \
  --context-date 2022-01-05 \
  --context-index 18103.33 \
  --lookback-days 20 \
  --forward-days 20 \
  --context-report reports/context_2022-01-05.md
```

輸出會包含：

- 輸入指數與資料庫收盤值是否一致
- 前 N 個交易日報酬與趨勢
- 後 N 個交易日報酬與趨勢
- 是否形成高點反轉、低點反轉、趨勢延續或盤整
- 當日模型生命周期階段與預測方向
- 後續走勢是否驗證模型預測

## 目前狀態

真實市場固定歷史研究已於2026-07-24結案。大盤多日方向最佳正式結果為66.18%，未達90%門檻且低於68.42%基準，因此正式多日方向訊號已停用。通過驗證的功能只限於同日開盤缺口方向與開盤缺口振幅風險，不能延伸為收盤或多日趨勢。

日報中的1、5、20、60日相似情境百分比已明確列為探索性歷史分布，不是正式交易訊號。詳細狀態見：

- `docs/current_status.md`
- `reports/final_research_conclusion.md`
- `reports/README.md`
