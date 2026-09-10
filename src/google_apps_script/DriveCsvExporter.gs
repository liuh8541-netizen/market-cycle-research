var DriveCsvExporter = (function () {
  function exportSheet(sheetName, filename) {
    var folderId = PropertiesService.getScriptProperties().getProperty("FACTOR_CSV_FOLDER_ID");
    if (!folderId) {
      throw new Error("Missing Script Property: FACTOR_CSV_FOLDER_ID");
    }
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(sheetName);
    if (!sheet || sheet.getLastRow() === 0) {
      return null;
    }
    var values = sheet.getDataRange().getDisplayValues();
    var csv = values.map(function (row) {
      return row.map(csvCell).join(",");
    }).join("\r\n") + "\r\n";

    var folder = DriveApp.getFolderById(folderId);
    var files = folder.getFilesByName(filename);
    while (files.hasNext()) {
      files.next().setTrashed(true);
    }
    return folder.createFile(filename, csv, MimeType.CSV).getId();
  }

  function csvCell(value) {
    var text = String(value === null || value === undefined ? "" : value);
    if (/[",\r\n]/.test(text)) {
      return '"' + text.replace(/"/g, '""') + '"';
    }
    return text;
  }

  return { exportSheet: exportSheet };
})();
