# API 配額與快取策略

## 設計目標

- 避免重複呼叫 FinMind API。
- 避免觸及 Google Apps Script 呼叫上限。
- 保留可追蹤的請求紀錄。
- 讓資料更新可以中斷後續跑，不需要重頭開始。

## Request Key

所有 API 請求都必須先轉為穩定的 request key。

```text
finmind:v4:data:{dataset}:{data_id}:{start_date}:{end_date}:{params_hash}
```

範例：

```text
finmind:v4:data:TaiwanStockPrice:TAIEX:2020-01-01:2026-07-14:a1b2c3
```

## 快取層級

### L1：短期快取

- 工具：Google Apps Script `CacheService`
- 適合：即時資訊、當日查詢、短時間重複刷新
- TTL：60 秒至 6 小時

### L2：長期快取

- 工具：Google Drive JSON 檔
- 適合：歷史日線、籌碼、已結束交易日資料
- TTL：固定資料可視為永久有效

### L3：衍生資料

- 工具：`processed_features` 分頁或匯出的 CSV
- 適合：均線、RSI、波動率、生命周期分數
- 原則：衍生資料由 L2 原始資料計算，不直接增加 API 呼叫。

## 去重流程

```text
收到資料需求
  -> 建立 request key
  -> 查 L1 快取
  -> 查 L2 長期快取
  -> 檢查 quota
  -> 呼叫 API
  -> 寫入 L1 / L2 / request_log
  -> 回傳資料
```

## 配額紀錄欄位

| 欄位 | 說明 |
| --- | --- |
| hour_bucket | 小時區間，例如 2026-07-14T09 |
| provider | finmind |
| quota_limit | 6000 |
| soft_limit | 5000 |
| used_count | 已使用次數 |
| skipped_by_cache | 因快取省下的次數 |
| blocked_by_quota | 因配額阻擋的次數 |

## 請求紀錄欄位

| 欄位 | 說明 |
| --- | --- |
| timestamp | 請求時間 |
| request_key | 請求指紋 |
| dataset | 資料集 |
| data_id | 標的代碼 |
| start_date | 開始日期 |
| end_date | 結束日期 |
| source | cache_l1、cache_l2、api |
| status | success、failed、quota_blocked |
| response_rows | 回傳筆數 |
| error_message | 錯誤訊息 |

注意：快取命中不建議逐筆寫入 `request_log`，否則儀表板刷新時會消耗大量 Google Sheets 寫入次數。實作上只記錄 API 呼叫、錯誤、配額阻擋與批次摘要。

## 排程建議

### 日常更新

- 每日收盤後執行。
- 只抓最近需要更新的日期。
- 已存在且完整的日期不重抓。

### 歷史回補

- 單獨排程。
- 分市場、分資料集、分年份批次執行。
- 每批完成後寫入進度，避免重跑。

### 手動刷新

- 儀表板按鈕只觸發必要資料。
- 若資料仍在 TTL 內，直接回傳快取。
- 若當小時接近警戒線，顯示最後更新資料，不打新 API。
