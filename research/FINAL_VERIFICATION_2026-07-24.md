# 研究證據鏈最終驗證

日期：2026-07-24

## 驗證範圍

- `monthly_revenue_breadth_v1`
- `credit_short_inventory_v1`
- `market_absorption_geometry_v1`
- 正式預測驗證政策與夜盤微結構測試

## 結果

三套 locked experiment 的獨立稽核均重新執行並通過：

- 正式執行回條為 `complete`。
- 結果檔、快照及鎖定輸入 SHA-256 均一致。
- 確認期、唯一日期、20 列間距、案例數、命中率與 Wilson 下限均一致。
- 目標、決策日期、基準命中及控制／增量門檻均可獨立重建。
- 配對日期、McNemar 計數與 p 值均一致。

專案測試以 Python `unittest` 執行，共 13 項，全部通過：

- 未達標的主要多日方向訊號維持停用。
- 正式驗證範圍只包含開盤缺口方向與缺口幅度風險兩個窄目標。
- 05:00 前後夜盤日期語意正確。
- 未完成夜盤、錯誤契約及 OHLC 容忍規則正確。

所有 `src/`、`scripts/` 及三個隔離研究目錄的 Python 檔案亦通過 `compileall`。

## 最終決策

- 三個新增因數實驗均未通過預先鎖定門檻，不得併入正式程式。
- 不得依結果改參數、切子群或重跑既有 locked experiment。
- 正式多日方向模型保持停用。
- 本輪歷史研究與證據鏈驗證完成，後續只允許依 `ONLINE_HISTORY_PROTOCOL.md` 累積真正未見資料，或依 `STOP_POLICY.md` 另立具獨立確認樣本的新因數計畫。
