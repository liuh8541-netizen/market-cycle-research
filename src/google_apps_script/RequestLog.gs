var RequestLog = (function () {
  var SHEET_NAME = "request_log";

  function append(request, requestKey, source, status, responseRows, errorMessage) {
    var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = spreadsheet.getSheetByName(SHEET_NAME) || spreadsheet.insertSheet(SHEET_NAME);

    if (sheet.getLastRow() === 0) {
      sheet.appendRow([
        "timestamp",
        "request_key",
        "dataset",
        "data_id",
        "start_date",
        "end_date",
        "source",
        "status",
        "response_rows",
        "error_message"
      ]);
    }

    sheet.appendRow([
      new Date(),
      requestKey,
      request.dataset,
      request.dataId,
      request.startDate,
      request.endDate,
      source,
      status,
      responseRows,
      errorMessage
    ]);
  }

  return {
    append: append
  };
})();

