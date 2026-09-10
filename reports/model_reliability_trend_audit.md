# 模型可靠度趨勢稽核

- 產生時間: 2026-09-10T15:13:00
- 最新大盤資料日: 2026-09-10
- 評估門檻: 漲跌超過 0.50% 才算方向，上下未超過門檻算盤整
- 預測紀錄數: 75
- 已成熟可驗證預測數: 150
- 整體命中率: 85/150 = 56.67%

## 逐段可靠度
- 前段: 30/50 = 60.00%
- 中段: 29/50 = 58.00%
- 後段: 26/50 = 52.00%

## 近期香港命中率
- 最近 10 筆: 7/10 = 70.00%
- 最近 20 筆: 13/20 = 65.00%
- 最近 30 筆: 14/30 = 46.67%

## 1日預測逐段
- 前段: 8/18 = 44.44%
- 中段: 8/18 = 44.44%
- 後段: 7/18 = 38.89%

## 正式驗證狀態
- 多日方向正式上線: False
- 生產閘門通過: False
- 最佳正式模型: three_layer_purged_ridge，準確率 66.18%，基準 68.42%

## 子模組可靠度
- canonical_night_spread_gap_v1: scope=same-day TWII cash open versus prior cash close only，accuracy=97.38%，cases=344
- daily_night_gap_amplitude_v1: scope=before TWII cash open, classify whether absolute opening gap exceeds the frozen prior-history 75th percentile; not a direction forecast，accuracy=92.66%，cases=259
- 心理狀態回測整體方向: 71.25%
- 內生調節脈動: hit_rate=74.75%，cases=404，passed=True
- 夜盤對日盤開盤20/60/252日對齊: 65.00% / 78.33% / 84.13%
- 夜盤對日盤收盤20/60/252日對齊: 65.00% / 65.00% / 76.59%

## 結論
- 診斷與風控可靠度已有提升，因為夜盤/日盤、心理狀態、內生調節、資料時點稽查已拆成可驗證模組。
- 純粹多日方向預測尚未證明逐次穩定提高；正式生產閘門仍未通過，不能升級成投資命令。
