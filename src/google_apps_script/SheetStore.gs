var SheetStore = (function () {
  function upsertRows(sheetName, keyColumns, rows) {
    if (!rows || rows.length === 0) {
      return 0;
    }

    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = spreadsheet.getSheetByName(sheetName) || spreadsheet.insertSheet(sheetName);
    var headers = collectHeaders(rows);

    if (sheet.getLastRow() === 0) {
      sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
    } else {
      headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
    }

    var existing = readExistingIndex(sheet, headers, keyColumns);
    var appends = [];

    rows.forEach(function (row) {
      var key = buildRowKey(row, keyColumns);
      if (existing[key]) {
        return;
      }
      appends.push(headers.map(function (header) {
        return row[header] === undefined ? "" : row[header];
      }));
    });

    if (appends.length > 0) {
      sheet.getRange(sheet.getLastRow() + 1, 1, appends.length, headers.length).setValues(appends);
    }

    return appends.length;
  }

  function readExistingIndex(sheet, headers, keyColumns) {
    var index = {};
    var lastRow = sheet.getLastRow();
    if (lastRow <= 1) {
      return index;
    }

    var values = sheet.getRange(2, 1, lastRow - 1, headers.length).getValues();
    values.forEach(function (line) {
      var row = {};
      headers.forEach(function (header, position) {
        row[header] = line[position];
      });
      index[buildRowKey(row, keyColumns)] = true;
    });
    return index;
  }

  function collectHeaders(rows) {
    var seen = {};
    var headers = [];
    rows.forEach(function (row) {
      Object.keys(row).forEach(function (key) {
        if (!seen[key]) {
          seen[key] = true;
          headers.push(key);
        }
      });
    });
    return headers;
  }

  function buildRowKey(row, keyColumns) {
    return keyColumns.map(function (key) {
      return String(row[key] || "");
    }).join("|");
  }

  return {
    upsertRows: upsertRows
  };
})();

