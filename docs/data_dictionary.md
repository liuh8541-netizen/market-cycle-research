# 資料欄位規格

## 價格資料

| 欄位 | 說明 |
| --- | --- |
| date | 交易日期 |
| open | 開盤價 |
| high | 最高價 |
| low | 最低價 |
| close | 收盤價 |
| volume | 成交量 |
| turnover | 成交金額 |

## 衍生指標

| 欄位 | 說明 |
| --- | --- |
| return_1d | 1 日報酬率 |
| return_1w | 1 週報酬率 |
| return_4w | 4 週報酬率 |
| ma_20 | 20 日均線 |
| ma_60 | 60 日均線 |
| ma_120 | 120 日均線 |
| ma_240 | 240 日均線 |
| ma_20_slope | 20 日均線斜率 |
| ma_60_slope | 60 日均線斜率 |
| rsi_14 | 14 日 RSI |
| atr_14 | 14 日 ATR |
| volatility_20 | 20 日歷史波動率 |
| drawdown | 自高點回撤 |

## 市場廣度

| 欄位 | 說明 |
| --- | --- |
| advancers | 上漲家數 |
| decliners | 下跌家數 |
| new_highs | 創新高家數 |
| new_lows | 創新低家數 |
| pct_above_ma20 | 成分股高於 20 日均線比例 |
| pct_above_ma60 | 成分股高於 60 日均線比例 |

## 籌碼與資金

| 欄位 | 說明 |
| --- | --- |
| foreign_net_buy | 外資買賣超 |
| investment_trust_net_buy | 投信買賣超 |
| dealer_net_buy | 自營商買賣超 |
| margin_balance | 融資餘額 |
| short_balance | 融券餘額 |

## 模型標籤

| 欄位 | 說明 |
| --- | --- |
| lifecycle_stage | 人工或規則標註的生命周期階段 |
| stage_score | 階段總分 |
| risk_regime | 風險狀態：risk_on、neutral、risk_off |
| next_1w_transition | 未來 1 週是否轉換階段 |
| next_4w_transition | 未來 4 週是否轉換階段 |
| next_12w_risk_off | 未來 12 週是否進入 risk_off |

