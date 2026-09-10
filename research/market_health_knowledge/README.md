# 市場臨床知識庫

本目錄為隔離研究區，不修改正式模型。

- `pathology_cases.jsonl`：程式執行時追加的不可回寫病例。
- `complete_market_records.jsonl`：逐日重建的完整市場病歷；分離盤前證據、因果鏈、診斷、既往病例預後、處方、禁忌、收盤轉歸與誤差檢討。
- `prospective_predictions.jsonl`：盤前鎖定、只追加且具雜湊鏈的前瞻診斷。
- `prospective_outcomes.jsonl`：與預測分離、收盤後只追加的前瞻轉歸。
- `reference_profiles.json`：具市場、頻率、目標與段位範圍的常模註冊。
- `prescriptions.json`：具診斷、信心、禁忌與處置強度的處方註冊。

找不到完全符合範圍的常模時回傳 `unsupported_scope`，不得套用通用常數。多份設定同時符合時回傳 `ambiguous_scope`，不得自行選擇。

目前常模只完成範圍註冊，尚未鎖定生命力、合理變異與轉段常數，因此狀態為 `pending_independent_validation`，不可作正式診斷。處方目前只包含限制傷害的安全政策，不包含提高槓桿、攤平或正式方向交易。

完整病歷另輸出 `reports/market_clinical_ledger.csv`（篩選與統計）、`reports/market_clinical_ledger.json`（摘要及最新病例）與 `reports/market_clinical_ledger.md`（人讀報告）。歷史病例的預後只使用決策日前同狀態、同方向且已結案的病例，禁止用未來轉歸回填過去診斷。
