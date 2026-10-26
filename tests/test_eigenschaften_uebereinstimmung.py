"""Eigenschaften: Invarianzen, Symmetrien, Schranken und Monotonie der exakten Formeln, Reproduzierbarkeit."""

import numpy as np
import pytest

import ueb_algorithm as K
from prs_scenario import VRng


def _w(seed=3, **kw):
    base = dict(n=300, sigma_a=20, bias=10, slope=0.8, sigma_e=4, proportional=False)
    base.update(kw)
    return K.make_world(base["n"], base["sigma_a"], base["bias"], base["slope"], base["sigma_e"], base["proportional"], VRng(seed))


def test_r_is_invariant_under_shifting_and_scaling_the_forecast_but_the_ccc_and_the_rmse_are_not():
    f, a = _w()
    g = 3.0 * f + 7.0
    assert K.corr(g, a) == pytest.approx(K.corr(f, a), abs=1e-12)
    assert abs(K.ccc(g, a) - K.ccc(f, a)) > 0.1 and abs(K.rmse(g, a) - K.rmse(f, a)) > 10.0


def test_swapping_forecast_and_actual_changes_neither_r_nor_the_ccc_nor_the_rmse():
    f, a = _w()
    assert K.corr(a, f) == pytest.approx(K.corr(f, a), abs=1e-12) and K.ccc(a, f) == pytest.approx(K.ccc(f, a), abs=1e-12)
    assert K.rmse(a, f) == pytest.approx(K.rmse(f, a), abs=1e-12) and K.mean_error(a, f) == pytest.approx(-K.mean_error(f, a), abs=1e-12)


def test_the_ccc_is_between_minus_one_and_one_and_never_larger_than_r():
    rng = VRng(9)
    for _ in range(40):
        sigma_a, bias, slope, sigma_e = (float(5 + 35 * rng.uniform((1,))[0]), float(-30 + 70 * rng.uniform((1,))[0]), float(1.5 * rng.uniform((1,))[0]), float(25 * rng.uniform((1,))[0]))
        f, a = K.make_world(100, sigma_a, bias, slope, sigma_e, False, rng)
        c, r = K.ccc(f, a), K.corr(f, a)
        assert -1.0 <= c <= 1.0 and c <= abs(r) + 1e-12 and (r < 0) == (c < 0)


def test_the_rmse_squared_is_the_bias_squared_plus_the_variance_of_the_differences():
    f, a = _w(bias=12, slope=1.2)
    d = f - a
    assert K.rmse(f, a) ** 2 == pytest.approx(K.mean_error(f, a) ** 2 + np.var(d), abs=1e-9) and K.mae(f, a) <= K.rmse(f, a)


def test_the_exact_ccc_falls_with_the_bias_and_with_the_distance_of_the_slope_from_one():
    by_bias = [K.exact_ccc(20, b, 1.0, 4, False) for b in (0, 5, 10, 20, 40)]
    assert by_bias == sorted(by_bias, reverse=True) and K.exact_ccc(20, -10, 1.0, 4, False) == pytest.approx(K.exact_ccc(20, 10, 1.0, 4, False), abs=1e-12)
    by_slope = [K.exact_ccc(20, 0, s, 4, False) for s in (0.6, 0.8, 1.0, 1.2, 1.4)]
    assert max(by_slope) == by_slope[2]                                                  # größte Übereinstimmung bei Steigung 1 (und dem Rauschen 4)
    assert by_slope[2] > by_slope[1] > by_slope[0] and by_slope[2] > by_slope[3] > by_slope[4]


def test_the_exact_r_does_not_depend_on_the_bias_but_the_exact_rmse_does():
    assert K.exact_r(20, 0, 0.8, 4, False) == K.exact_r(20, 25, 0.8, 4, False)
    assert K.exact_rmse(20, 25, 0.8, 4, False) > K.exact_rmse(20, 0, 0.8, 4, False)


def test_the_same_error_gives_a_larger_r_when_the_actual_values_spread_more():
    rs = [K.exact_r(s, 0, 1.0, 4, False) for s in (5, 10, 20, 40)]
    assert rs == sorted(rs) and rs[0] < 0.8 and rs[-1] > 0.99
    assert len({round(K.exact_rmse(s, 0, 1.0, 4, False), 9) for s in (5, 10, 20, 40)}) == 1 and len({K.exact_within(s, 0, 1.0, 4, False, 5.0) for s in (5, 10, 20, 40)}) == 1


def test_proportional_noise_puts_more_points_outside_the_limits_in_the_upper_third():
    f, a = _w(n=20_000, bias=0, slope=1.0, sigma_e=8, proportional=True)
    ba = K.bland_altman(f, a)
    assert ba["outside_high"] > 2 * ba["outside_low"]
    f, a = _w(n=20_000, bias=0, slope=1.0, sigma_e=8, proportional=False)
    ba = K.bland_altman(f, a)
    assert abs(ba["outside_high"] - ba["outside_low"]) < 0.02


def test_the_same_seed_gives_the_same_world_and_a_different_seed_a_different_one():
    f1, a1 = _w(seed=1)
    f2, a2 = _w(seed=1)
    assert np.array_equal(f1, f2) and not np.array_equal(f1, _w(seed=2)[0])
