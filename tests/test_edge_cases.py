"""Randfälle (vor dem ersten Mutationstest): kleinste Stichproben, konstante Reihen, inklusive Grenzen von Toleranz und Dritteln, Reglergrenzen der Welt."""

import math

import numpy as np
import pytest

import ueb_algorithm as K
import ueb_constants as C
from prs_scenario import VRng


def test_two_weeks_are_enough_for_the_measures_but_not_for_bland_altman():
    f, a = np.array([1.0, 3.0]), np.array([2.0, 5.0])
    assert K.corr(f, a) == pytest.approx(1.0) and K.rmse(f, a) == pytest.approx(math.sqrt((1 + 4) / 2)) and np.isfinite(K.ccc(f, a))
    with pytest.raises(ValueError):
        K.bland_altman(f, a)
    assert np.isfinite(K.bland_altman(np.array([1.0, 3.0, 5.0]), np.array([2.0, 5.0, 4.0]))["sd"])                 # drei Wochen genügen


def test_the_smallest_world_runs_at_every_corner_of_the_sliders():
    for sigma_a in (C.SIGMA_A_MIN, C.SIGMA_A_MAX):
        for bias in (C.BIAS_MIN, C.BIAS_MAX):
            for slope in (C.SLOPE_MIN, C.SLOPE_MAX):
                for sigma_e in (C.SIGMA_E_MIN, C.SIGMA_E_MAX):
                    for prop in (False, True):
                        f, a = K.make_world(min(C.N_OPTIONS), sigma_a, bias, slope, sigma_e, prop, VRng(2))
                        assert len(f) == len(a) == 20 and np.isfinite(K.ccc(f, a)) and np.isfinite(K.rmse(f, a))
                        assert 0.0 <= K.within_tolerance(f, a, C.TOL_MIN) <= 1.0 and K.bland_altman(f, a)["sd"] >= 0.0


def test_a_flat_noise_free_forecast_is_constant_and_has_no_r():
    f, a = K.make_world(50, 20, 0, 0.0, 0, False, VRng(1))
    assert np.allclose(f, 100.0) and math.isnan(K.corr(f, a)) and K.ccc(f, a) == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(K.exact_r(20, 0, 0.0, 0, False)) and K.exact_ccc(20, 0, 0.0, 0, False) == 0.0


def test_tolerance_and_limits_are_inclusive_but_the_bland_altman_limit_is_strict():
    a = np.zeros(5)
    f = np.array([0.0, 2.0, -2.0, 4.0, -4.0])
    assert K.within_tolerance(f, a, 2.0) == 0.6 and K.within_tolerance(f, a, 1.999) == 0.2
    ba = K.bland_altman(f, a, z=1.0)
    sd = float(np.std(f, ddof=1))
    assert ba["lower"] == pytest.approx(-sd) and ba["upper"] == pytest.approx(sd)


def test_the_outer_thirds_are_floor_n_over_3_weeks_each():
    for n in (3, 4, 5, 7, 8, 10):
        a = np.arange(float(n))
        f = a + np.where(np.arange(n) % 2 == 0, 1.0, -1.0) * 5.0
        ba = K.bland_altman(f, a, z=0.0)
        assert 0.0 <= ba["outside_low"] <= 1.0 and 0.0 <= ba["outside_high"] <= 1.0
    f, a = np.array([1.0, 2.0, 9.0]), np.array([1.0, 2.0, 3.0])
    ba = K.bland_altman(f, a, z=1.0)
    assert ba["outside_high"] == 1.0 and ba["outside_low"] == 0.0               # n = 3: je ein Drittel = eine Woche


def test_exact_formulas_at_the_slider_corners_are_finite_and_in_range():
    for sigma_a in (C.SIGMA_A_MIN, C.SIGMA_A_MAX):
        for bias in (C.BIAS_MIN, 0, C.BIAS_MAX):
            for slope in (C.SLOPE_MIN, 1.0, C.SLOPE_MAX):
                for sigma_e in (0, 4, C.SIGMA_E_MAX):
                    for prop in (False, True):
                        args = (sigma_a, bias, slope, sigma_e, prop)
                        c = K.exact_ccc(*args)
                        assert -1.0 <= c <= 1.0 and K.exact_rmse(*args) >= 0.0 and 0.0 <= K.exact_within(*args, 5.0) <= 1.0 + 1e-12 and K.exact_mae(*args) >= 0.0


def test_the_calibration_cannot_be_fitted_to_a_constant_actual_series_but_fits_a_constant_error():
    a = np.array([80.0, 100.0, 120.0])
    slope, icpt = K.recalibrate(a + 10.0, a)
    assert slope == pytest.approx(1.0) and icpt == pytest.approx(-10.0)
    assert K.corr(a, np.full(3, 5.0)) != K.corr(a, np.full(3, 5.0))                  # NaN: das Ist ist konstant


def test_world_noise_kinds_agree_without_noise():
    f1, a1 = K.make_world(30, 20, 5, 0.9, 0, False, VRng(4))
    f2, a2 = K.make_world(30, 20, 5, 0.9, 0, True, VRng(4))
    assert np.array_equal(f1, f2) and np.array_equal(a1, a2)


def test_bland_altman_counts_only_strictly_outside_points_so_points_on_the_limit_stay_inside():
    a = np.zeros(6)
    f = np.array([-2.0, 0.0, 2.0, 0.0, -2.0, 2.0])
    ba = K.bland_altman(f, a, z=0.0)                                           # Grenzen = Mittel = 0: die zwei Wochen mit Differenz 0 liegen genau auf der Grenze
    assert ba["lower"] == 0.0 and ba["upper"] == 0.0 and ba["outside"] == pytest.approx(4 / 6)
