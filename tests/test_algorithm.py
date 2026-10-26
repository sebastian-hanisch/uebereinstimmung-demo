"""Einheiten des Kerns an Mini-Instanzen von Hand (DEMO-PLAYBOOK §1b): Maße, Bland–Altman, Neukalibrierung, ICC, exakte Formeln, Welt."""

import math

import numpy as np
import pytest

import ueb_algorithm as K
import ueb_constants as C
from prs_scenario import VRng


# --- Maße --------------------------------------------------------------------------------------------------------------------------------

def test_four_points_by_hand(four_points):
    f, a = four_points
    assert K.corr(f, a) == pytest.approx(0.6, abs=1e-12) and K.ccc(f, a) == pytest.approx(0.6, abs=1e-12)
    assert K.rmse(f, a) == 1.0 and K.mae(f, a) == 1.0 and K.mean_error(f, a) == 0.0
    assert K.within_tolerance(f, a, 1.0) == 1.0 and K.within_tolerance(f, a, 0.999) == 0.0


def test_a_perfect_forecast_has_r_and_ccc_one_and_no_error(three_weeks):
    a = three_weeks
    assert K.corr(a, a) == pytest.approx(1.0, abs=1e-12) and K.ccc(a, a) == pytest.approx(1.0, abs=1e-12) and K.rmse(a, a) == 0.0 and K.mean_error(a, a) == 0.0


def test_a_constant_shift_keeps_r_but_lowers_the_ccc(three_weeks):
    a = three_weeks
    f = a + 10.0
    assert K.corr(f, a) == pytest.approx(1.0, abs=1e-12)
    assert K.ccc(f, a) == pytest.approx(533.3333333 / 633.3333333, abs=1e-8) and K.ccc(f, a) == pytest.approx(0.8421052631, abs=1e-8)
    assert K.rmse(f, a) == pytest.approx(10.0, abs=1e-12) and K.mae(f, a) == pytest.approx(10.0, abs=1e-12)


def test_the_wrong_scale_by_hand(three_weeks):
    a = three_weeks
    f = 1.3 * a
    assert K.corr(f, a) == pytest.approx(1.0, abs=1e-12) and K.ccc(f, a) == pytest.approx(693.3333333 / 1617.3333333, abs=1e-8) and K.ccc(f, a) == pytest.approx(0.428689, abs=1e-6)
    assert K.mean_error(f, a) == pytest.approx(30.0, abs=1e-12)


def test_a_constant_forecast_has_no_r_but_a_ccc_of_zero_and_cannot_be_recalibrated(three_weeks):
    a = three_weeks
    f = np.full(3, 100.0)
    assert math.isnan(K.corr(f, a)) and K.ccc(f, a) == 0.0
    with pytest.raises(ValueError):
        K.recalibrate(f, a)


def test_inputs_are_validated(four_points):
    f, a = four_points
    for fn in (K.corr, K.ccc, K.rmse, K.mae, K.mean_error):
        with pytest.raises(ValueError):
            fn(f, a[:3])
        with pytest.raises(ValueError):
            fn(f[:1], a[:1])
    with pytest.raises(ValueError):
        K.bland_altman(f[:2], a[:2])


# --- Bland–Altman und Toleranz -----------------------------------------------------------------------------------------------------------

def test_bland_altman_by_hand(four_points):
    f, a = four_points
    ba = K.bland_altman(f, a)
    assert ba["mean"] == 0.0 and ba["sd"] == pytest.approx(math.sqrt(4 / 3), abs=1e-12)
    assert ba["lower"] == pytest.approx(-1.96 * math.sqrt(4 / 3), abs=1e-12) and ba["upper"] == pytest.approx(1.96 * math.sqrt(4 / 3), abs=1e-12) and ba["lower"] == pytest.approx(-2.2632, abs=1e-4)
    assert ba["outside"] == 0.0 and ba["outside_low"] == 0.0 and ba["outside_high"] == 0.0


def test_bland_altman_outside_shares_and_thirds():
    a = np.arange(1.0, 10.0)                                   # neun Wochen, Drittel zu je drei
    d = np.zeros(9)
    d[8], d[0] = 50.0, -50.0                                    # je ein Ausreißer im oberen und im unteren Drittel
    ba = K.bland_altman(a + d, a, z=1.0)
    assert ba["outside_low"] == pytest.approx(1 / 3) and ba["outside_high"] == pytest.approx(1 / 3) and ba["outside"] == pytest.approx(2 / 9)
    wide = K.bland_altman(a + d, a, z=100.0)
    assert wide["outside"] == 0.0


def test_bland_altman_slope_of_the_differences_against_the_mean():
    a = np.array([10.0, 20.0, 30.0, 40.0])
    f = 2.0 * a                                                 # d = a, m = 1,5 a: Steigung 1 / 1,5
    assert K.bland_altman(f, a)["slope"] == pytest.approx(1 / 1.5, abs=1e-12)
    assert K.bland_altman(a + 3.0, a)["slope"] == pytest.approx(0.0, abs=1e-12)


def test_within_tolerance_is_inclusive():
    a = np.zeros(4)
    f = np.array([0.0, 1.0, 2.0, 3.0])
    assert [K.within_tolerance(f, a, t) for t in (0.0, 1.0, 2.0, 3.0)] == [0.25, 0.5, 0.75, 1.0]


# --- Neukalibrierung und ICC -------------------------------------------------------------------------------------------------------------

def test_recalibration_by_hand():
    slope, icpt = K.recalibrate(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    assert (slope, icpt) == (2.0, 0.0)
    assert list(K.apply_calibration([1.0, 2.0], slope, icpt)) == [2.0, 4.0]
    slope, icpt = K.recalibrate(np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 3.0]))
    assert slope == pytest.approx(0.1, abs=1e-12) and icpt == pytest.approx(0.0, abs=1e-12)


def test_recalibration_in_sample_gives_the_ccc_of_two_r_squared_over_one_plus_r_squared():
    f, a = K.make_world(300, 20, 10, 0.5, 4, False, VRng(5))
    slope, icpt = K.recalibrate(f, a)
    g = K.apply_calibration(f, slope, icpt)
    r = K.corr(f, a)
    assert K.ccc(g, a) == pytest.approx(K.exact_ccc_after_recalibration(r), abs=1e-12)
    assert K.mean_error(g, a) == pytest.approx(0.0, abs=1e-10)


def test_icc_is_one_for_identical_series_and_close_to_the_ccc_otherwise():
    a = np.array([80.0, 100.0, 120.0, 90.0])
    assert K.icc_a1(a, a) == pytest.approx(1.0, abs=1e-12)
    f, x = K.make_world(500, 20, 10, 1.0, 4, False, VRng(2))
    assert K.icc_a1(f, x) == pytest.approx(K.ccc(f, x), abs=0.005)


# --- Welt --------------------------------------------------------------------------------------------------------------------------------

def test_the_world_is_reproducible_and_has_the_documented_moments():
    f1, a1 = K.make_world(50, 20, 10, 1.0, 4, False, VRng(3))
    f2, a2 = K.make_world(50, 20, 10, 1.0, 4, False, VRng(3))
    assert np.array_equal(f1, f2) and np.array_equal(a1, a2) and not np.array_equal(a1, K.make_world(50, 20, 10, 1.0, 4, False, VRng(4))[1])
    f, a = K.make_world(400_000, 20, 10, 0.5, 4, False, VRng(1))
    assert a.mean() == pytest.approx(100.0, abs=0.1) and a.std() == pytest.approx(20.0, abs=0.1)
    assert f.mean() - a.mean() == pytest.approx(10.0, abs=0.1) and np.var(f - a) == pytest.approx(0.25 * 400 + 16, rel=0.01)


def test_proportional_noise_grows_with_the_actual_value():
    f, a = K.make_world(200_000, 20, 0, 1.0, 8, True, VRng(1))
    d = f - a
    high, low = a > 110, a < 90
    assert d[high].std() > 1.2 * d[low].std() and d[low].std() == pytest.approx(8 * 0.85, rel=0.1)


# --- Exakte Formeln ----------------------------------------------------------------------------------------------------------------------

def test_exact_formulas_by_hand():
    assert K.exact_ccc(20, 30, 1.3, 0, False) == pytest.approx(1040 / 1976, abs=1e-12)                    # f = 1,3 · a ohne Rauschen
    assert K.exact_ccc(20, 30, 1.3, 4, False) == pytest.approx(1040 / 1992, abs=1e-12)
    assert K.exact_r(20, 0, 1.0, 4, False) == pytest.approx(20 / math.sqrt(416), abs=1e-12)
    assert K.exact_rmse(20, 10, 1.0, 4, False) == pytest.approx(math.sqrt(116), abs=1e-12)
    assert K.exact_rmse(20, 30, 1.3, 4, False) == pytest.approx(math.sqrt(900 + 36 + 16), abs=1e-12)
    assert K.exact_ccc(10, 10, 1.0, 0, False) == pytest.approx(200 / 300, abs=1e-12)                      # f = a + 10, σ = 10
    assert K.exact_ccc_after_recalibration(0.6) == pytest.approx(0.72 / 1.36, abs=1e-12)


def test_exact_moments_for_proportional_noise():
    m = K.moments(20, 0, 1.0, 8, True)
    assert m["var_f"] == pytest.approx(400 + 64 * (1 + 400 / 10000), abs=1e-12) and m["var_d"] == pytest.approx(64 * 1.04, abs=1e-12)


def test_exact_within_and_mae_for_degenerate_and_simple_cases():
    assert K.exact_within(20, 0, 1.0, 0, False, 1.0) == 1.0 and K.exact_within(20, 5, 1.0, 0, False, 1.0) == 0.0 and K.exact_within(20, 1, 1.0, 0, False, 1.0) == 1.0
    assert K.exact_mae(20, 7, 1.0, 0, False) == pytest.approx(7.0, abs=1e-9)
    assert K.exact_within(20, 0, 1.0, 4, False, 1.96 * 4) == pytest.approx(0.95, abs=1e-3)
    assert K.exact_mae(20, 0, 1.0, 4, False) == pytest.approx(4 * math.sqrt(2 / math.pi), abs=1e-9)
    assert K.exact_within(20, 0, 1.0, 0, True, 1.0) == 1.0                                                  # ohne Rauschen gibt es keinen Unterschied zwischen den Arten
