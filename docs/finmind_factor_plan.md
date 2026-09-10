# FinMind 非價格因子接入計畫

## 目標

價格技術指標的方向預測已接近隨機邊界，因此下一階段加入非價格因子：

- 市場廣度
- 籌碼
- 融資融券
- 期貨
- 選擇權

## 官方資料集依據

FinMind 官方資料列表目前列出台灣市場 86 種資料集，包含：

- 籌碼面：`TaiwanStockTotalMarginPurchaseShortSale`、`TaiwanStockTotalInstitutionalInvestors`
- 衍生性商品：`TaiwanFuturesDaily`、`TaiwanOptionDaily`、`TaiwanFuturesInstitutionalInvestors`、`TaiwanOptionInstitutionalInvestors`
- 選擇權波動：`TaiwanOptionVix`

參考：

- https://finmind.github.io/tutor/TaiwanMarket/DataList/
- https://finmind.github.io/tutor/TaiwanMarket/Derivative/
- https://finmind.github.io/WhatIsNew/

## 第一批接入資料

| 類型 | FinMind dataset | 用途 |
| --- | --- | --- |
| 籌碼 | TaiwanStockTotalInstitutionalInvestors | 整體三大法人買賣超 |
| 融資融券 | TaiwanStockTotalMarginPurchaseShortSale | 整體市場槓桿與散戶風險偏好 |
| 期貨 | TaiwanFuturesDaily | 臺指期成交、未平倉、基差輔助 |
| 期貨法人 | TaiwanFuturesInstitutionalInvestors | 法人期貨部位偏向 |
| 選擇權 | TaiwanOptionDaily | 選擇權成交與未平倉 |
| 選擇權法人 | TaiwanOptionInstitutionalInvestors | 法人選擇權部位偏向 |
| 波動率 | TaiwanOptionVix | 臺指選擇權波動率指數 |

## 資料放置規格

下載後統一放在：

```text
data/processed/factors/
  institutional_total.csv
  margin_total.csv
  futures_daily.csv
  futures_institutional.csv
  option_daily.csv
  option_institutional.csv
  option_vix.csv
```

所有檔案至少需要 `date` 欄位。欄位名稱不一致時，由特徵工程模組做寬鬆辨識。

## 預期新增特徵

### 籌碼

- 法人買賣超總額
- 外資買賣超
- 投信買賣超
- 自營商買賣超
- 法人買賣超 5 日、20 日累計

### 融資融券

- 融資餘額變化率
- 融券餘額變化率
- 融資融券比
- 槓桿過熱分數

### 期貨

- 臺指期近月成交量
- 臺指期未平倉量
- 期貨法人淨部位
- 期貨淨部位 5 日、20 日變化

### 選擇權

- Put/Call 成交量比
- Put/Call 未平倉比
- 選擇權法人淨部位
- VIX 水準與變化

## 驗證方式

每次加入一類資料都要跑：

1. 價格 baseline
2. 價格 + 該類因子
3. 價格 + 全部非價格因子

比較：

- 5 日方向命中率
- 20 日方向命中率
- walk-forward 命中率
- 最大回撤
- 可行動樣本覆蓋率

