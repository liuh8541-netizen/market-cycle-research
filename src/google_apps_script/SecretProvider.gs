var SecretProvider = (function () {
  function getRequired(key) {
    var value = PropertiesService.getScriptProperties().getProperty(key);
    if (!value) {
      throw new Error("Missing secret: " + key);
    }
    return value;
  }

  return {
    getRequired: getRequired
  };
})();

