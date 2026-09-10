var QuotaStore = (function () {
  function hourBucket() {
    return Utilities.formatDate(new Date(), "Asia/Taipei", "yyyy-MM-dd'T'HH");
  }

  function getKey(provider) {
    return "quota:" + provider + ":" + hourBucket();
  }

  function getUsed(provider) {
    return Number(PropertiesService.getScriptProperties().getProperty(getKey(provider)) || 0);
  }

  function canCall(provider, softLimit, hardLimit) {
    var used = getUsed(provider);
    if (used >= hardLimit || used >= softLimit) {
      return false;
    }
    return true;
  }

  function increment(provider) {
    var key = getKey(provider);
    var props = PropertiesService.getScriptProperties();
    var used = Number(props.getProperty(key) || 0);
    props.setProperty(key, String(used + 1));
  }

  return {
    getUsed: getUsed,
    canCall: canCall,
    increment: increment
  };
})();
