var CacheStore = (function () {
  var PREFIX = "cache:";

  function get(key) {
    var raw = CacheService.getScriptCache().get(PREFIX + key);
    if (raw) {
      return JSON.parse(raw);
    }

    raw = getFromDrive(key);
    if (!raw) {
      return null;
    }

    var parsed = JSON.parse(raw);
    putScriptCache(key, parsed, 21600);
    return parsed;
  }

  function put(key, value, ttlSeconds) {
    putScriptCache(key, value, ttlSeconds);
    putDriveCache(key, value);
  }

  function putScriptCache(key, value, ttlSeconds) {
    var raw = JSON.stringify(value);
    if (raw.length > 90000) {
      return;
    }
    CacheService.getScriptCache().put(PREFIX + key, raw, Math.min(ttlSeconds, 21600));
  }

  function getFromDrive(key) {
    var folder = getCacheFolder();
    if (!folder) {
      return null;
    }

    var files = folder.getFilesByName(fileNameFor(key));
    if (!files.hasNext()) {
      return null;
    }

    return files.next().getBlob().getDataAsString("UTF-8");
  }

  function putDriveCache(key, value) {
    var folder = getCacheFolder();
    if (!folder) {
      return;
    }

    var raw = JSON.stringify(value);
    var fileName = fileNameFor(key);
    var files = folder.getFilesByName(fileName);

    if (files.hasNext()) {
      files.next().setContent(raw);
      return;
    }

    folder.createFile(fileName, raw, MimeType.PLAIN_TEXT);
  }

  function getCacheFolder() {
    var folderId = PropertiesService.getScriptProperties().getProperty("CACHE_FOLDER_ID");
    if (!folderId) {
      return null;
    }
    return DriveApp.getFolderById(folderId);
  }

  function fileNameFor(key) {
    var digest = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, key);
    var hash = digest.map(function (byte) {
      var value = (byte < 0 ? byte + 256 : byte).toString(16);
      return value.length === 1 ? "0" + value : value;
    }).join("");
    return "finmind_cache_" + hash + ".json";
  }

  return {
    get: get,
    put: put
  };
})();
