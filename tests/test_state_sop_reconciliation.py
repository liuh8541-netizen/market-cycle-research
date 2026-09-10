import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from predict_market import reconcile_market_state_sop


def bagua(state_code, gua, status, background="乾"):
    return {
        "roles": {
            "state_gua": {"code": state_code, "gua": gua, "status": status},
            "background_gua": {"gua": background},
        }
    }


def crash(alert_code="observe", health_score=78.6):
    return {
        "enabled": True,
        "state": "合理底部區測試",
        "status_code": "normal_bottom_test",
        "summary": "仍在合理修正底部區附近。",
        "alert_code": alert_code,
        "health_score": health_score,
        "capital_policy": {"summary": "尚未確認崩盤，但仍在坎中。"},
    }


class StateSopReconciliationTest(unittest.TestCase):
    def test_zhen_health_79_routes_to_repair_sop(self):
        result = reconcile_market_state_sop(
            crash(), bagua("ZHEN", "震", "repair_activation")
        )

        self.assertEqual(result["state"], "震啟動修復")
        self.assertEqual(result["health_label"], "修復中")
        self.assertEqual(result["state_sop"]["sop"], "修復確認")
        self.assertEqual(result["state_sop"]["reasonable_bottom_role"], "風控失效防線")
        current_text = " ".join(
            [result["state"], result["summary"], result["capital_policy"]["summary"]]
        )
        for contradiction in ["普通偏弱", "仍在坎中", "合理底部區測試"]:
            self.assertNotIn(contradiction, current_text)
        self.assertEqual(result["state_sop"]["background_gua"], "乾")

    def test_kan_keeps_bottom_as_current_test(self):
        result = reconcile_market_state_sop(
            crash(), bagua("KAN", "坎", "crisis_decline")
        )
        self.assertEqual(result["state"], "坎危機防守")
        self.assertEqual(result["state_sop"]["sop"], "危機防守")
        self.assertEqual(result["state_sop"]["reasonable_bottom_role"], "當前測試目標")

    def test_emergency_alert_is_not_overwritten_by_repair_state(self):
        original = crash(alert_code="confirmed_warning", health_score=25)
        original["state"] = "崩盤警戒"
        result = reconcile_market_state_sop(
            original, bagua("ZHEN", "震", "repair_activation")
        )
        self.assertEqual(result["state"], "崩盤警戒")
        self.assertEqual(result["state_sop"]["sop"], "危機風控")


if __name__ == "__main__":
    unittest.main()
