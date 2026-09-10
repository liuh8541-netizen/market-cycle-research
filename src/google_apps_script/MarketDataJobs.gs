function updateTaiexDaily() {
  var today = Utilities.formatDate(new Date(), "Asia/Taipei", "yyyy-MM-dd");
  var startDate = incrementalStartDate("raw_index", getConfigValue("TAIEX_START_DATE", "2000-01-01"), 7);

  var rows = FinMindClient.fetchData({
    dataset: "TaiwanStockPrice",
    dataId: "TAIEX",
    startDate: startDate,
    endDate: today,
    cacheTtlSeconds: 21600
  });

  var normalized = rows.map(function (row) {
    return {
      date: row.date,
      stock_id: row.stock_id || "TAIEX",
      open: Number(row.open),
      high: Number(row.max),
      low: Number(row.min),
      close: Number(row.close),
      volume: Number(row.Trading_Volume || row.trading_volume || 0),
      turnover: Number(row.Trading_money || row.trading_money || 0)
    };
  });

  var inserted = SheetStore.upsertRows("raw_index", ["date", "stock_id"], normalized);
  RequestLog.append(
    { dataset: "TaiwanStockPrice", dataId: "TAIEX", startDate: startDate, endDate: today },
    "sheet:raw_index:TAIEX",
    "sheet",
    "success",
    inserted,
    ""
  );
  DriveCsvExporter.exportSheet("raw_index", "twii_daily_finmind.csv");
}

function updateFinMindFactorDatasets() {
  var today = Utilities.formatDate(new Date(), "Asia/Taipei", "yyyy-MM-dd");
  var datasets = [
    { name: "institutional_total", sheet: "factor_institutional_total", dataset: "TaiwanStockTotalInstitutionalInvestors", dataId: "", keys: ["date", "name"] },
    { name: "margin_total", sheet: "factor_margin_total", dataset: "TaiwanStockTotalMarginPurchaseShortSale", dataId: "", keys: ["date"] },
    { name: "futures_daily", sheet: "factor_futures_daily", dataset: "TaiwanFuturesDaily", dataId: "TX", keys: ["date", "contract_date", "trading_session"] },
    { name: "futures_institutional", sheet: "factor_futures_institutional", dataset: "TaiwanFuturesInstitutionalInvestors", dataId: "TX", keys: ["date", "name", "contract_date"] },
    { name: "option_daily", sheet: "factor_option_daily", dataset: "TaiwanOptionDaily", dataId: "TXO", keys: ["date", "contract_date", "strike_price", "call_put", "trading_session"] },
    { name: "option_institutional", sheet: "factor_option_institutional", dataset: "TaiwanOptionInstitutionalInvestors", dataId: "TXO", keys: ["date", "name", "call_put", "contract_date"] },
    { name: "option_vix", sheet: "factor_option_vix", dataset: "TaiwanOptionVix", dataId: "", keys: ["date"] }
  ];

  datasets.forEach(function (item) {
    var startDate = incrementalStartDate(
      item.sheet,
      getConfigValue("FACTOR_START_DATE", "2000-01-01"),
      7
    );
    var rows = FinMindClient.fetchData({
      dataset: item.dataset,
      dataId: item.dataId,
      startDate: startDate,
      endDate: today,
      cacheTtlSeconds: 21600
    });

    if (!rows || rows.length === 0) {
      return;
    }

    var inserted = SheetStore.upsertRows(item.sheet, item.keys, rows);
    RequestLog.append(
      { dataset: item.dataset, dataId: "", startDate: startDate, endDate: today },
      "sheet:" + item.sheet,
      "sheet",
      "success",
      inserted,
      ""
    );
    DriveCsvExporter.exportSheet(item.sheet, item.name + ".csv");
  });
}

function runDailyCloudUpdate() {
  updateTaiexDaily();
  updateFinMindFactorDatasets();
}

function incrementalStartDate(sheetName, fallback, overlapDays) {
  var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = spreadsheet.getSheetByName(sheetName);
  if (!sheet || sheet.getLastRow() < 2) {
    return fallback;
  }
  var headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
  var dateIndex = headers.indexOf("date");
  if (dateIndex < 0) {
    return fallback;
  }
  var values = sheet.getRange(2, dateIndex + 1, sheet.getLastRow() - 1, 1).getValues();
  var latest = null;
  values.forEach(function (line) {
    var parsed = line[0] instanceof Date ? line[0] : new Date(line[0]);
    if (!isNaN(parsed.getTime()) && (!latest || parsed > latest)) {
      latest = parsed;
    }
  });
  if (!latest) {
    return fallback;
  }
  latest.setDate(latest.getDate() - Number(overlapDays || 0));
  return Utilities.formatDate(latest, "Asia/Taipei", "yyyy-MM-dd");
}

function getConfigValue(key, defaultValue) {
  var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = spreadsheet.getSheetByName("config");
  if (!sheet || sheet.getLastRow() < 2) {
    return defaultValue;
  }

  var values = sheet.getRange(2, 1, sheet.getLastRow() - 1, 2).getValues();
  for (var i = 0; i < values.length; i += 1) {
    if (String(values[i][0]) === key) {
      return values[i][1] || defaultValue;
    }
  }
  return defaultValue;
}
