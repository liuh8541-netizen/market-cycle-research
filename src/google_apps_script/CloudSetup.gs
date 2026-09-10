function installDailyCloudTrigger() {
  removeCloudTriggers_();
  ScriptApp.newTrigger("runDailyCloudUpdate")
    .timeBased()
    .everyDays(1)
    .atHour(15)
    .nearMinute(45)
    .inTimezone("Asia/Taipei")
    .create();
}

function removeCloudTriggers_() {
  ScriptApp.getProjectTriggers().forEach(function (trigger) {
    if (trigger.getHandlerFunction() === "runDailyCloudUpdate") {
      ScriptApp.deleteTrigger(trigger);
    }
  });
}

function verifyCloudConfiguration() {
  var properties = PropertiesService.getScriptProperties();
  var required = ["FINMIND_TOKEN", "FACTOR_CSV_FOLDER_ID"];
  var result = {};
  required.forEach(function (key) {
    result[key] = Boolean(properties.getProperty(key));
  });
  Logger.log(JSON.stringify(result));
  if (!result.FINMIND_TOKEN || !result.FACTOR_CSV_FOLDER_ID) {
    throw new Error("Cloud configuration is incomplete. Check Script Properties.");
  }
  return result;
}
