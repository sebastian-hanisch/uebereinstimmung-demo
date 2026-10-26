"""Auswertung: Zeilen der Hauptansicht, Zellen des Rasters, Verdikte der Hauptansicht und der Experimente."""

import math

import numpy as np
import pytest

import ueb_algorithm as K
import ueb_constants as C
import ueb_evaluation as ev


def test_defaults_are_the_documented_standard_case():
    s = ev.Settings()
    assert (s.bias, s.slope, s.sigma_e, s.noise, s.sigma_a, s.n, s.tol, s.seed) == (0, 1.0, 4, "additiv", 20, 500, 5, 35) and not s.proportional


def test_settings_are_keyword_only_and_hashable_and_describe_the_world():
    with pytest.raises(TypeError):
        ev.Settings(0, 1.0)
    assert hash(ev.Settings()) == hash(ev.Settings()) and ev.Settings(n=100) != ev.Settings()
    assert ev.Settings(noise="proportional", sigma_a=10, bias=5, slope=0.9, sigma_e=3).world() == (10, 5, 0.9, 3, True)


def test_analysis_rows_match_direct_calls_and_carry_the_exact_values():
    s = ev.Settings(bias=10, slope=0.8, sigma_e=4, n=300, seed=3)
    a = ev.analyse(s)
    f, x = ev.make_data(s)
    assert [r.key for r in a.rows] == ["r", "r2", "ccc", "bias", "rmse", "mae", "within", "ccc_cal"]
    assert a.row("r").value == K.corr(f, x) and a.row("r2").value == K.corr(f, x) ** 2 and a.row("ccc").value == K.ccc(f, x) and a.row("rmse").value == K.rmse(f, x)
    assert a.row("mae").value == K.mae(f, x) and a.row("bias").value == K.mean_error(f, x) and a.row("within").value == K.within_tolerance(f, x, 5.0) and a.row("within").kind == "percent"
    w = s.world()
    assert a.row("r").exact == K.exact_r(*w) and a.row("ccc").exact == K.exact_ccc(*w) and a.row("rmse").exact == K.exact_rmse(*w) and a.row("bias").exact == 10.0
    assert a.row("within").exact == K.exact_within(*w, 5.0) and a.row("mae").exact == K.exact_mae(*w) and a.row("r2").exact == pytest.approx(K.exact_r(*w) ** 2)
    assert a.row("ccc_cal").exact == pytest.approx(K.exact_ccc_after_recalibration(K.exact_r(*w)))
    assert a.row("nichts") is None


def test_the_calibrated_ccc_follows_the_in_sample_identity():
    a = ev.analyse(ev.Settings(bias=30, slope=1.3, n=400))
    assert a.ccc_calibrated == pytest.approx(K.exact_ccc_after_recalibration(a.row("r").value), abs=1e-12) and a.rmse_calibrated < a.row("rmse").value
    slope, icpt = a.calibration
    assert slope == pytest.approx(K.recalibrate(a.f, a.a)[0]) and icpt == pytest.approx(K.recalibrate(a.f, a.a)[1])


def test_a_constant_forecast_gives_nan_for_r_and_the_calibration_but_the_rest_still_works():
    a = ev.analyse(ev.Settings(slope=0.0, sigma_e=0))
    assert math.isnan(a.row("r").value) and math.isnan(a.row("ccc_cal").value) and math.isnan(a.calibration[0]) and math.isfinite(a.row("rmse").value)
    kind, text = ev.verdict_main(a)
    assert kind == "warning" and "konstant" in text and "r ist nicht definiert" in text


def test_de_pct_format_with_decimal_comma_and_dash():
    assert ev.de(0.5) == "0,50" and ev.de(float("nan")) == "–" and ev.de(None) == "–" and ev.de(3, 0) == "3" and ev.pct(0.1234) == "12,3 %" and ev.pct(0.5, 0) == "50 %"


def test_error_cell_structure_and_determinism():
    c = ev.error_cell("skala", 100, 2, seeds=(5, 6))
    assert (c["case"], c["n"], c["runs"]) == ("skala", 100, 4)
    for key in ("r", "r2", "ccc", "rmse", "mae", "bias", "icc", "r_sd", "ccc_sd", "within5", "within10"):
        assert np.isfinite(c[key]), key
    assert set(c["exact"]) == {"r", "r2", "ccc", "rmse", "mae", "bias", "within5", "within10"} and c["exact"]["ccc"] == pytest.approx(1040 / 1992)
    assert ev.error_cell("skala", 100, 2, seeds=(5, 6)) == c


def test_spread_recal_rank_and_ba_cells():
    sp = ev.spread_cell(10, 4, 200, 2, seeds=(5,))
    assert sp["runs"] == 2 and sp["exact"]["r"] == pytest.approx(10 / math.sqrt(116)) and sp["exact"]["rmse"] == pytest.approx(math.sqrt(32))
    rc = ev.recal_cell("flach", 100, 2, seeds=(5,))
    assert rc["runs"] == 2 and rc["ccc_after"] > rc["ccc_before"] and rc["rmse_after"] < rc["rmse_before"]
    rk = ev.rank_cell(100, 3, seeds=(5,))
    assert rk["runs"] == 3 and set(rk["a_wins"]) == {"r", "ccc", "rmse", "mae"} and rk["a_wins"]["r"] == 1.0 and rk["a_wins"]["rmse"] == 0.0
    ba = ev.ba_cell("proportional", 300, 2, seeds=(5,))
    assert ba["runs"] == 2 and ba["outside_high"] > ba["outside_low"] and ba["sd_high"] > ba["sd_low"]


def test_find_and_load_grid(tmp_path):
    cells = [{"case": "a", "n": 1}, {"case": "b", "n": 1}, {"case": "a", "n": 2}]
    assert ev.find(cells, case="a", n=2) is cells[2] and ev.find(cells, case="z") is None
    assert ev.load_grid(tmp_path / "gibt-es-nicht.json") is None
    bad = tmp_path / "kaputt.json"
    bad.write_text("{", encoding="utf-8")
    assert ev.load_grid(bad) is None
    good = tmp_path / "gut.json"
    good.write_text('{"a": 1}', encoding="utf-8")
    assert ev.load_grid(good) == {"a": 1}


def test_the_grid_cases_cover_the_documented_errors():
    assert set(C.CASES) == {"perfekt", "bias", "flach", "skala", "bias_flach", "rauschen"} and set(C.CASE_NAMES) == set(C.CASES) and set(C.E5_CASES) == set(C.E5_NAMES)
    skala = C.CASES["skala"]
    assert 100 + skala["bias"] + skala["slope"] * (0 - 100) == pytest.approx(0.0)
