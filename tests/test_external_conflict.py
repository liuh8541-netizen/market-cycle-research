import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "predict_market.py"
SPEC = importlib.util.spec_from_file_location("predict_market", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_conflicting_external_signals_are_not_calm_neutral():
    row = {
        "nasdaq_return_1d": -0.0100,
        "sp500_return_1d": -0.0087,
        "sox_return_1d": 0.0053,
        "tsm_adr_return_1d": 0.0095,
        "vix_return_1d": 0.0752,
    }
    context = MODULE.analyze_external_conflict(row)
    assert context["is_conflict"] is True
    assert context["regime"] == "taiwan_tech_offset"
    assert context["dispersion"] >= 0.08


def test_quiet_external_session_remains_calm_neutral():
    row = {
        "nasdaq_return_1d": 0.001,
        "sp500_return_1d": -0.001,
        "sox_return_1d": 0.002,
        "tsm_adr_return_1d": 0.001,
        "vix_return_1d": 0.003,
    }
    context = MODULE.analyze_external_conflict(row)
    assert context["is_conflict"] is False
    assert context["regime"] == "calm_neutral"
