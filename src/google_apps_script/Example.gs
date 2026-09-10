function exampleFetchTaiex() {
  var rows = FinMindClient.fetchData({
    dataset: "TaiwanStockPrice",
    dataId: "TAIEX",
    startDate: "2026-01-01",
    endDate: "2026-07-14",
    cacheTtlSeconds: 21600
  });

  Logger.log("Rows: " + rows.length);
}

