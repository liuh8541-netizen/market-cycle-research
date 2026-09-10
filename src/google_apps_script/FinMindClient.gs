var FinMindClient = (function () {
  var API_BASE = "https://api.finmindtrade.com/api/v4/data";
  var HARD_LIMIT_PER_HOUR = 6000;
  var SOFT_LIMIT_PER_HOUR = 5000;

  function fetchData(request) {
    var normalized = normalizeRequest(request);
    var requestKey = buildRequestKey(normalized);
    var cached = CacheStore.get(requestKey);

    if (cached) {
      return cached;
    }

    var lock = LockService.getScriptLock();
    lock.waitLock(30000);

    try {
      cached = CacheStore.get(requestKey);
      if (cached) {
        return cached;
      }

      if (!QuotaStore.canCall("finmind", SOFT_LIMIT_PER_HOUR, HARD_LIMIT_PER_HOUR)) {
        RequestLog.append(normalized, requestKey, "quota", "quota_blocked", 0, "Hourly soft limit reached.");
        throw new Error("Quota blocked for finmind.");
      }

      var token = SecretProvider.getRequired("FINMIND_TOKEN");
      var url = buildUrl(normalized, token);
      var response = UrlFetchApp.fetch(url, {
        method: "get",
        muteHttpExceptions: true
      });

      var code = response.getResponseCode();
      var body = response.getContentText();

      QuotaStore.increment("finmind");

      if (code < 200 || code >= 300) {
        RequestLog.append(normalized, requestKey, "api", "failed", 0, body.slice(0, 300));
        throw new Error("FinMind API failed with status " + code);
      }

      var parsed = JSON.parse(body);
      var rows = parsed.data || [];

      CacheStore.put(requestKey, rows, normalized.cacheTtlSeconds);
      RequestLog.append(normalized, requestKey, "api", "success", rows.length, "");
      return rows;
    } finally {
      lock.releaseLock();
    }
  }

  function normalizeRequest(request) {
    return {
      dataset: String(request.dataset || ""),
      dataId: String(request.dataId || ""),
      startDate: String(request.startDate || ""),
      endDate: String(request.endDate || ""),
      params: request.params || {},
      cacheTtlSeconds: Number(request.cacheTtlSeconds || 21600)
    };
  }

  function buildUrl(request, token) {
    var params = {
      dataset: request.dataset,
      data_id: request.dataId,
      start_date: request.startDate,
      end_date: request.endDate,
      token: token
    };

    Object.keys(request.params).sort().forEach(function (key) {
      params[key] = request.params[key];
    });

    var query = Object.keys(params)
      .filter(function (key) { return params[key] !== ""; })
      .map(function (key) {
        return encodeURIComponent(key) + "=" + encodeURIComponent(params[key]);
      })
      .join("&");

    return API_BASE + "?" + query;
  }

  function buildRequestKey(request) {
    var payload = JSON.stringify({
      dataset: request.dataset,
      dataId: request.dataId,
      startDate: request.startDate,
      endDate: request.endDate,
      params: sortObject(request.params)
    });
    return "finmind:v4:data:" + sha256(payload);
  }

  function sortObject(value) {
    var output = {};
    Object.keys(value).sort().forEach(function (key) {
      output[key] = value[key];
    });
    return output;
  }

  function sha256(text) {
    var digest = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, text);
    return digest.map(function (byte) {
      var value = (byte < 0 ? byte + 256 : byte).toString(16);
      return value.length === 1 ? "0" + value : value;
    }).join("");
  }

  return {
    fetchData: fetchData,
    buildRequestKey: buildRequestKey
  };
})();
