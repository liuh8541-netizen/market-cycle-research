# Google Apps 整合設計

本專題採用 Google Apps Script 作為輕量資料更新與排程層，Google Sheets 作為人工檢視、參數設定與週期報告入口。核心原則是：金鑰外掛化、資料呼叫去重、快取優先、配額保守使用。

## 方案基礎

依照目前採用方案規格：

- FinMind Sponsor
- API / 下載限制：6,000 次 / 小時
- 91 種資料集
- 可用於個人、學術、web、app 開發等非商業用途
- 資料需標示來源為 FinMind

實作時仍要同時考慮 Google Apps Script 自身的 UrlFetch、執行時間、觸發器與試算表讀寫限制，因此系統配額採「較小者為準」。

## 架構

```text
Google Sheets
  - 參數設定
  - 資料檢視
  - 週期判讀報告

Google Apps Script
  - 排程觸發
  - API 呼叫管理
  - 快取與去重
  - 金鑰讀取
  - 錯誤重試

FinMind API
  - 指數資料
  - 籌碼資料
  - 即時資訊
  - 衍生性金融商品資料

Drive / Script Properties
  - 長期快取索引
  - 歷史資料 JSON 快取
  - 呼叫紀錄
  - 配額計數
```

## 密碼與 Token 外掛式設計

不把 API token、密碼或敏感參數寫死在程式碼裡。

建議使用三層設計：

1. `SecretProvider`：統一讀取密碼與 token。
2. `Script Properties`：Google Apps Script 內建的安全設定位置。
3. 可替換外掛：未來若改用 Google Secret Manager、外部 API Gateway 或其他密碼服務，只需替換 `SecretProvider`。

### 金鑰名稱

| Key | 用途 |
| --- | --- |
| FINMIND_TOKEN | FinMind API token |
| ADMIN_EMAIL | 錯誤通知收件人 |
| CACHE_FOLDER_ID | Drive 長期快取資料夾 |
| FACTOR_CSV_FOLDER_ID | 雲端硬碟中 `data/processed/factors` 對應資料夾 ID |

## 雲端硬碟版設定

本專案不使用 Windows 環境變數保存 FinMind Token。Token 僅存於 Google Apps Script 的 Script Properties，CSV 則輸出至 Google 雲端硬碟並由 Drive 同步。

1. 在 Google Drive 建立 `data/processed/factors` 資料夾。
2. 從資料夾網址複製資料夾 ID。
3. 開啟綁定試算表的 Apps Script 專案。
4. 在「專案設定 → 指令碼屬性」新增：
   - `FINMIND_TOKEN`：FinMind Sponsor API Token。
   - `FACTOR_CSV_FOLDER_ID`：上述雲端資料夾 ID。
   - `FACTOR_START_DATE`：第一次歷史回補起日，例如 `2000-01-01`（可選）。
   - `TAIEX_START_DATE`：指數歷史回補起日（可選）。
5. 將 `src/google_apps_script` 中的 `.gs` 與 `appsscript.json` 部署至該專案。
6. 先執行 `verifyCloudConfiguration()`，再手動執行一次 `runDailyCloudUpdate()`。
7. 確認 CSV 已寫入雲端資料夾後，執行 `installDailyCloudTrigger()`。

日常排程會在台北時間約 15:45 執行，並輸出：

```text
institutional_total.csv
margin_total.csv
futures_daily.csv
futures_institutional.csv
option_daily.csv
option_institutional.csv
option_vix.csv
twii_daily_finmind.csv
```

桌機批次檔只讀取同步後的 `data/processed/factors`，不需要取得或保存 FinMind Token。

## 呼叫去重策略

每次資料請求先正規化成 request key：

```text
dataset + data_id + start_date + end_date + params_hash
```

處理順序：

1. 產生 request key。
2. 檢查短期快取 `CacheService`。
3. 檢查長期快取 `Drive`。
4. 檢查本小時配額。
5. 必要時才呼叫 FinMind API。
6. 成功後寫入快取與呼叫紀錄。

## 配額控管

雖然 Sponsor 方案提供 6,000 次 / 小時，實作上不直接打滿。

建議預設：

- 硬上限：6,000 次 / 小時
- 軟上限：5,000 次 / 小時
- 警戒線：4,500 次 / 小時
- 單次排程上限：依資料集分批，例如 200 至 500 次

如果達到軟上限：

- 停止新資料請求
- 保留已快取資料輸出
- 將未完成請求寫入待補佇列
- 下一個排程再繼續

## 資料更新節奏

| 資料類型 | 建議頻率 |
| --- | --- |
| 日線價格 | 每日收盤後 1 次 |
| 週線資料 | 每週收盤後 1 次 |
| 籌碼資料 | 每日收盤後 1 次 |
| 即時資訊 | 只在儀表板開啟或手動刷新時呼叫 |
| 歷史資料回補 | 分批排程，不與日常更新混用 |

## 避免浪費呼叫的規則

- 同一份資料在同一天已成功取得，不重複呼叫。
- 歷史區間資料固定後，以長期快取為主。
- 即時資料使用短 TTL 快取，例如 60 至 300 秒。
- 報表讀取只讀本地資料表，不直接打 API。
- 排程更新和使用者手動更新共用同一個請求紀錄。
- 失敗重試最多 2 次，且使用退避等待。
- 快取命中不逐筆寫入試算表，避免為了省 API 卻浪費 Google Apps 讀寫額度。

## Google Sheets 分頁建議

| 分頁 | 用途 |
| --- | --- |
| config | 參數設定 |
| request_log | API 請求紀錄 |
| quota_log | 每小時配額統計 |
| raw_index | 指數原始資料 |
| processed_features | 衍生指標 |
| lifecycle_score | 生命周期分數 |
| dashboard | 儀表板 |
| factor_institutional_total | 整體三大法人 |
| factor_margin_total | 整體融資融券 |
| factor_futures_daily | 期貨日成交 |
| factor_futures_institutional | 期貨三大法人 |
| factor_option_daily | 選擇權日成交 |
| factor_option_institutional | 選擇權三大法人 |
| factor_option_vix | 臺指選擇權波動率 |

## 批次寫入原則

- 試算表資料寫入使用 `setValues` 一次批次完成。
- 以 `date + stock_id` 作為原始行情去重鍵。
- 已存在的歷史資料不覆寫，避免重複消耗 Apps Script 讀寫額度。
- 日常更新只補新增日期，歷史回補才抓長區間。
- `request_log` 只記錄 API 呼叫、批次摘要、錯誤與配額阻擋，不記錄每一筆快取命中。
