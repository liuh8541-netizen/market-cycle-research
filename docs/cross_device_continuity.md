# 跨電腦自動接續與復原 SOP

本專案以 GitHub 作為唯一程式與模型規則來源。換到另一台電腦時，不需要重講模型邏輯；先拉下此倉庫，再依本文件恢復資料權限與每日排程。

## 1. 下載專案

```powershell
git clone https://github.com/liuh8541-netizen/market-cycle-research.git
cd market-cycle-research
```

若該電腦已經有舊資料夾：

```powershell
git pull
```

## 2. 安裝 Python 套件

```powershell
py -m pip install -r requirements.txt
```

## 3. 恢復 FinMind Sponsor token

Sponsor token 不會推上 GitHub。新電腦需自行建立：

```powershell
New-Item -ItemType Directory -Force config/.secrets
Set-Content -Path config/.secrets/finmind_token.txt -Value "你的 FinMind token"
```

也可使用環境變數：

```powershell
setx FINMIND_TOKEN "你的 FinMind token"
```

## 4. 驗證資料與模型能跑

```powershell
.\run_predict_market.bat
```

確認報告中至少要看到：

- 台股分析基準日為最新有效交易日。
- 資料來源包含 FinMind Sponsor 或 TWSE official fallback。
- `reports/today_market_forecast.md` 有輸出大盤健康指數、風險值、夜日盤關係、八卦生命週期與資料意圖判讀。

## 5. 恢復 Windows 每日排程

本機 Windows 工作排程不會隨 GitHub 自動搬移。新電腦要執行一次：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_daily_market_tasks.ps1
```

安裝後會建立：

- `TaiwanMarket-NightUpdate`：交易日 05:20 更新夜盤與盤前資料。
- `TaiwanMarket-CloseUpdate`：交易日 14:20 更新收盤檢討與日報。

## 6. Codex heartbeat 自主進化

Codex App 的 heartbeat automation 屬於該台電腦的 Codex 設定，未必會由 GitHub 自動同步。若換電腦後沒有看到每日自主進化任務，依 `research/HANDOFF_CURRENT.md` 重新建立同名 heartbeat：

- 名稱：台股大盤每日自主進化監控
- 目標：每日更新 `reports/today_market_forecast.md`、`reports/today_market_forecast_detail.md`，並執行前一日誤差檢討、病歷表、戰術雷達、新聞風險與模型修正。
- 邊界：只做風險預警、資料核對、研究留底、模型降權、候選規則追加與前瞻驗證；不得做投資命令。

## 7. 接續原則

另一台電腦接手時，優先閱讀：

- `README.md`
- `research/HANDOFF_CURRENT.md`
- `CHANGELOG.md`
- `docs/current_status.md`
- `docs/model_validation_requirements.md`
- `docs/cross_device_continuity.md`

核心原則是：GitHub 保存模型思想、程式、規則、測試與文件；本機只補上 token、資料快取與排程。這樣就能讓模型跨電腦接續，而不是重新開始。
