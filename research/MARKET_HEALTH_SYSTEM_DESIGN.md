# 市場生命診療系統設計

更新日：2026-07-25

## 核心定位

市場生命週期模型扮演醫生，不扮演預言家。系統先辨識生命徵象與病兆，再提出可被後續資料推翻的暫定診斷，最後依證據強度限制處置。第一原則是避免傷害。

## 三層責任

### 症狀

只記錄已觀察到的事實與資料限制：

- 價格及K線損傷；
- 洗盤是否失敗；
- 外部、夜盤、籌碼與產業風險；
- 支持修復的證據；
- 訊號矛盾；
- 資料是否過期或尚未完成。

症狀本身不得直接成為交易命令。

### 診斷

診斷必須包含：

- `status`：暫定、需鑑別或資料不足；
- `primary`：主要市場健康狀態；
- `lifecycle_stage`：生命週期段位；
- `confidence`：診斷信心；
- `alternatives`：其他可能解釋及辨別條件；
- `confirmation_conditions`：康復或惡化確認；
- `invalidation_conditions`：原診斷翻案條件。

廣義方向模型未通過正式驗證，因此診斷只描述狀態與風險，不具正式方向權限。

### 處方

目前允許的處置：

- `observe_and_recheck`
- `hold_risk_constant`
- `reduce_risk`
- `observe_repair_without_directional_authority`
- `staged_reentry_only_after_confirmation`

正式多日方向停用期間，研究性修復診斷不得授權重新進場，只能等待確認。

## 不可違反的安全規則

1. `validated=false` 時，`formal_direction_signal` 必須是 `null`。
2. 診斷信心低時，處置只能是 `none` 或 `light`。
3. 資料過期、缺漏或夜盤未完成時，只能觀察與複查。
4. 邏輯出現硬矛盾時，必須進入鑑別診斷，不能開方向處方。
5. 增加任何風險前必須先滿足確認條件。
6. 系統不得建議提高槓桿。
7. 系統不得建議虧損攤平。
8. 失效條件觸發後，原診斷與處置必須停止。

## 程式介面

主要函式：

```python
from market_lifecycle.market_health import build_market_health_assessment

assessment = build_market_health_assessment(payload)
```

回傳結構：

```json
{
  "framework": "preventive_market_medicine_v1",
  "principle": "first_do_no_harm",
  "symptoms": [],
  "diagnosis": {
    "status": "provisional",
    "primary": "deterioration_watch",
    "confidence": "low",
    "alternatives": [],
    "confirmation_conditions": [],
    "invalidation_conditions": [],
    "is_formal_direction_diagnosis": false
  },
  "prescription": {
    "action": "reduce_risk",
    "intensity": "light",
    "formal_direction_signal": null,
    "may_increase_leverage": false,
    "may_average_down": false,
    "requires_confirmation_before_risk_increase": true
  },
  "safety_checks": {}
}
```

正式預測JSON的欄位名稱為 `market_health`，Markdown報告則新增「市場健康診療」段落。

## 對話到程式的演進流程

往後形成新的市場觀念時，依下列順序處理：

1. 將觀念寫成可被證偽的規則。
2. 判定它屬於症狀、診斷、處方或安全限制。
3. 明確列出適用範圍及不得使用的目標。
4. 先加入測試，描述最危險的誤診或錯藥情境。
5. 實作最小程式變更。
6. 用歷史封存資料驗證，不依結果事後改門檻。
7. 未通過正式門檻時，只能保留為研究觀察或風控限制。
8. 保存失敗案例，讓程式學會何時不應出手。

## 現有測試保障

`tests/test_market_health.py`目前驗證：

- 未驗證方向不能產生正式處方；
- 低信心不得使用中度或重度處置；
- 資料不完整必須停止診斷；
- 矛盾證據必須進入鑑別診斷；
- 研究性偏多不得授權進場；
- 不得提高槓桿或攤平。

這套機制的目的不是讓程式假裝永遠正確，而是讓它在不確定時保持誠實，在犯錯時限制傷害，並透過不可回寫的病例紀錄逐步改善。

## 自我偵測與修護閉環

主要函式：

```python
from market_lifecycle.self_repair import build_self_repair_assessment

repair = build_self_repair_assessment(payload)
```

每次執行會保存：

- 輸入日期與訊號日期；
- 資料時效狀態；
- 邏輯矛盾數；
- 市場健康診斷與處方；
- 已成熟病例的原預判及實際結果；
- 正式方向權限；
- 上述證據的穩定SHA-256指紋。

目前錯誤分類包括：

- `data_quality_error`：資料過期、缺漏或尚未完成；
- `diagnostic_conflict`：不同診斷證據存在硬矛盾；
- `scope_authority_error`：未驗證模型越權產生正式訊號；
- `outcome_mismatch`：到期病例與原探索性判斷不符。

允許自動執行的修護：

- 阻擋未驗證正式輸出；
- 強制轉為觀察與複查；
- 降低診斷或處方權限；
- 追加錯誤病例與根因分析候選；
- 在沒有錯誤時持續監測。

禁止自動執行的變更：

- 修改模型係數；
- 修改驗證或信心門檻；
- 擴張模型適用範圍；
- 改寫歷史病歷；
- 重跑已完成的locked experiment。

受保護變更只能建立候選版本，先固定適用範圍、設定與雜湊，再使用獨立病例驗證。通過後從新的生效日啟用，不得回頭美化舊結果。

正式預測JSON的欄位名稱為 `self_repair`，Markdown報告則新增「自我偵測與修護」段落。

## 病理、藥理與常模知識層

主要函式：

```python
from market_lifecycle.clinical_knowledge import update_clinical_knowledge

knowledge = update_clinical_knowledge(
    payload,
    "research/market_health_knowledge",
)
```

知識層包含：

- `pathology_cases.jsonl`：依證據指紋建立的不可回寫病例；
- `reference_profiles.json`：綁定市場、頻率、目標與段位的常模；
- `prescriptions.json`：綁定診斷、信心、禁忌及處置強度的處方；
- 相似病例搜索：只在相同市場、頻率與目標內搜索。

範圍路由規則：

1. 市場與資料頻率必須完全相同。
2. 目標及生命週期段位必須符合註冊範圍。
3. 找不到常模時回傳 `unsupported_scope`。
4. 多份常模同時符合時回傳 `ambiguous_scope`。
5. 未經獨立驗證的常模即使範圍符合，仍回傳 `pending_validation`及`usable=false`。
6. 不得退回全域通用常數。
7. 處方若觸及任何禁忌，必須回傳`no_safe_prescription`並改為觀察複查。

相似病例以症狀集合交集及段位相符度排序，但市場、頻率或目標不同的病例在計分前即排除，不能因表面相似而跨範圍套用。

目前常模只完成適用範圍註冊，尚未估計生命力、合理變異及轉段常數。這是刻意的安全狀態：先建立正確容器及治理，再用封存歷史病例估計常數；不能為了讓功能看似完整而填入未驗證數字。

正式預測JSON的欄位名稱為 `clinical_knowledge`，Markdown報告則新增「病理與藥理知識庫」段落。
