"""Orakel (ORACLE-PLAYBOOK): unabhängige Wege zu denselben Zahlen.

Pearson gegen numpy und scipy; die CCC gegen die Schleifenfassung der Definition und gegen die Zerlegung r · C_b (Lin); die exakten Formeln gegen Simulationen mit zwei Millionen
Paaren und gegen numerische Integration (scipy.integrate.quad); die Folded-Normal-Erwartung gegen scipy.stats.foldnorm; der Anteil außerhalb der Bland–Altman-Grenzen gegen
scipy.stats.norm; die Neukalibrierung gegen numpy.polyfit; ICC(A,1) gegen eine Schleifenfassung der ANOVA."""

import math

import numpy as np
import pytest
from scipy import integrate, stats

import ueb_algorithm as K
import ueb_constants as C
from prs_scenario import VRng


def _world(n=400, sigma_a=20, bias=10, slope=0.8, sigma_e=4, prop=False, seed=3):
    return K.make_world(n, sigma_a, bias, slope, sigma_e, prop, VRng(seed))


def test_pearson_matches_numpy_and_scipy():
    f, a = _world()
    assert K.corr(f, a) == pytest.approx(np.corrcoef(f, a)[0, 1], abs=1e-12) and K.corr(f, a) == pytest.approx(stats.pearsonr(f, a)[0], abs=1e-12)


def test_ccc_matches_a_loop_and_the_decomposition_into_r_and_a_bias_factor():
    f, a = _world(300)
    n = len(f)
    mf, ma = sum(f) / n, sum(a) / n
    sf, sa, sfa = sum((x - mf) ** 2 for x in f) / n, sum((y - ma) ** 2 for y in a) / n, sum((x - mf) * (y - ma) for x, y in zip(f, a)) / n
    assert K.ccc(f, a) == pytest.approx(2 * sfa / (sf + sa + (mf - ma) ** 2), abs=1e-12)
    v, u = math.sqrt(sf / sa), (mf - ma) / (sf * sa) ** 0.25
    assert K.ccc(f, a) == pytest.approx(K.corr(f, a) * 2 / (v + 1 / v + u * u), abs=1e-12)                 # Lin: ρ_c = ρ · C_b


def test_rmse_mae_and_tolerance_match_loops():
    f, a = _world(200)
    d = [x - y for x, y in zip(f, a)]
    assert K.rmse(f, a) == pytest.approx(math.sqrt(sum(x * x for x in d) / 200), abs=1e-12) and K.mae(f, a) == pytest.approx(sum(abs(x) for x in d) / 200, abs=1e-12)
    for tol in (2.0, 7.5, 15.0):
        assert K.within_tolerance(f, a, tol) == sum(1 for x in d if abs(x) <= tol) / 200


def test_bland_altman_matches_a_loop_version():
    f, a = _world(300, bias=5, slope=1.2)
    d = [x - y for x, y in zip(f, a)]
    mean = sum(d) / 300
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / 299)
    ba = K.bland_altman(f, a)
    assert ba["mean"] == pytest.approx(mean, abs=1e-12) and ba["sd"] == pytest.approx(sd, abs=1e-12)
    assert ba["outside"] == sum(1 for x in d if x < mean - 1.96 * sd or x > mean + 1.96 * sd) / 300
    mid = np.array([(x + y) / 2 for x, y in zip(f, a)])
    assert ba["slope"] == pytest.approx(np.polyfit(mid, np.array(d), 1)[0], abs=1e-9)


def test_bland_altman_outside_share_is_five_percent_for_normal_differences():
    f, a = _world(300_000, bias=0, slope=1.0, sigma_e=8)
    ba = K.bland_altman(f, a)
    assert ba["outside"] == pytest.approx(2 * stats.norm.sf(1.96), abs=0.003) and ba["outside_low"] == pytest.approx(0.05, abs=0.006) and ba["outside_high"] == pytest.approx(0.05, abs=0.006)


def test_recalibration_matches_polyfit():
    f, a = _world(250)
    slope, icpt = K.recalibrate(f, a)
    ref = np.polyfit(f, a, 1)
    assert slope == pytest.approx(ref[0], abs=1e-10) and icpt == pytest.approx(ref[1], abs=1e-8)


def test_icc_matches_a_loop_version_of_the_anova():
    f, a = _world(120)
    n, k = 120, 2
    rows = [(x, y) for x, y in zip(f, a)]
    gm = sum(x + y for x, y in rows) / (n * k)
    row_means = [(x + y) / 2 for x, y in rows]
    col_means = (sum(f) / n, sum(a) / n)
    ssr = k * sum((m - gm) ** 2 for m in row_means)
    ssc = n * sum((m - gm) ** 2 for m in col_means)
    sst = sum((v - gm) ** 2 for x, y in rows for v in (x, y))
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    assert K.icc_a1(f, a) == pytest.approx((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n), abs=1e-10)


@pytest.mark.parametrize("params", [(20, 10, 1.0, 4, False), (20, 0, 0.5, 4, False), (20, 30, 1.3, 4, False), (20, 0, 1.0, 20, False), (10, 4, 1.0, 4, False), (20, 0, 1.0, 8, True), (40, 5, 0.9, 12, True)])
def test_exact_formulas_match_a_simulation_with_two_million_pairs(params):
    sigma_a, bias, slope, sigma_e, prop = params
    f, a = K.make_world(2_000_000, sigma_a, bias, slope, sigma_e, prop, VRng(7))
    assert K.corr(f, a) == pytest.approx(K.exact_r(*params), abs=0.002)
    assert K.ccc(f, a) == pytest.approx(K.exact_ccc(*params), abs=0.002)
    assert K.rmse(f, a) == pytest.approx(K.exact_rmse(*params), rel=0.003)
    assert K.mae(f, a) == pytest.approx(K.exact_mae(*params), rel=0.004)
    for tol in (5.0, 10.0):
        assert K.within_tolerance(f, a, tol) == pytest.approx(K.exact_within(*params, tol), abs=0.002)


def test_exact_within_matches_scipy_for_additive_noise_and_numerical_integration_for_proportional_noise():
    s = math.sqrt(0.25 * 400 + 16)
    assert K.exact_within(20, 10, 0.5, 4, False, 5.0) == pytest.approx(stats.norm.cdf(5.0, 10, s) - stats.norm.cdf(-5.0, 10, s), abs=1e-12)
    sigma_a, bias, slope, sigma_e = 20, 5, 0.9, 12

    def integrand(x):
        a = C.MU + sigma_a * x
        mu, sd = bias + (slope - 1.0) * (a - C.MU), sigma_e * abs(a) / C.MU
        return stats.norm.pdf(x) * (stats.norm.cdf(7.0, mu, sd) - stats.norm.cdf(-7.0, mu, sd))
    ref = integrate.quad(integrand, -10, 10, points=[-C.MU / sigma_a], limit=200)[0]
    assert K.exact_within(sigma_a, bias, slope, sigma_e, True, 7.0) == pytest.approx(ref, abs=2e-4)


def test_exact_mae_matches_foldnorm_for_additive_noise_and_integration_for_proportional_noise():
    sd = math.sqrt(0.25 * 400 + 16)
    assert K.exact_mae(20, 10, 0.5, 4, False) == pytest.approx(stats.foldnorm.mean(10 / sd, scale=sd), abs=1e-9)
    sigma_a, bias, slope, sigma_e = 20, 5, 0.9, 12

    def integrand(x):
        a = C.MU + sigma_a * x
        mu, s = bias + (slope - 1.0) * (a - C.MU), max(sigma_e * abs(a) / C.MU, 1e-9)
        return stats.norm.pdf(x) * (mu * (2 * stats.norm.cdf(mu / s) - 1) + 2 * s * stats.norm.pdf(mu / s))
    ref = integrate.quad(integrand, -10, 10, points=[-C.MU / sigma_a], limit=200)[0]
    assert K.exact_mae(sigma_a, bias, slope, sigma_e, True) == pytest.approx(ref, abs=2e-4)


def test_the_hook_numbers_of_the_text():
    """Die Zahlen des Aufhängers (Vormessung, hier exakt): f = 1,3 · a gibt das beste r und den schlechtesten RMSE."""
    r = K.exact_r(20, 30, 1.3, 4, False)
    assert r == pytest.approx(0.988, abs=5e-4) and K.exact_ccc(20, 30, 1.3, 4, False) == pytest.approx(0.522, abs=5e-4) and K.exact_rmse(20, 30, 1.3, 4, False) == pytest.approx(30.85, abs=5e-3)
    assert K.exact_r(20, 10, 1.0, 4, False) == pytest.approx(0.981, abs=5e-4) and K.exact_ccc(20, 10, 1.0, 4, False) == pytest.approx(0.873, abs=5e-4)
    assert [round(K.exact_r(s, 0, 1.0, 4, False) ** 2, 2) for s in (5, 10, 20, 40)] == [0.61, 0.86, 0.96, 0.99]
