"""Abbildungen: jede wird gebaut, hat die erwarteten Spuren und Achsen."""

import numpy as np
import pytest

import ueb_constants as C
import ueb_evaluation as ev
import ueb_visualization as V


@pytest.fixture(scope="module")
def grid():
    return ev.load_grid()


def test_every_chart_builds_and_locks_its_axes(grid):
    a = ev.analyse(ev.Settings(bias=10, slope=0.8, n=200, seed=3))
    figs = [V.build_scatter(a), V.build_ba(a), V.build_measures(a), V.build_e1(grid["e1"]), V.build_e2(grid["e2"]), V.build_e3(grid["e3"]), V.build_e4(grid["e4"][0]), V.build_e5(grid["e5"])]
    for fig in figs:
        assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange and len(fig.data) >= 1


def test_scatter_has_the_identity_line_and_a_regression_line_and_thins_large_clouds():
    a = ev.analyse(ev.Settings(n=200, seed=3))
    fig = V.build_scatter(a)
    assert [t.name for t in fig.data][:2] == ["Wochen", "Prognose = Ist"] and fig.data[2].name.startswith("Regressionsgerade (Steigung ")
    assert list(fig.data[1].x) == list(fig.data[1].y)                                    # 45°-Linie
    big = ev.analyse(ev.Settings(n=2000, seed=3))
    assert len(V.build_scatter(big, max_points=500).data[0].x) == 500 and len(V.build_scatter(big).data[0].x) == 2000


def test_scatter_without_variation_in_the_forecast_has_no_regression_line():
    a = ev.analyse(ev.Settings(slope=0.0, sigma_e=0, n=50))
    assert [t.name for t in V.build_scatter(a).data] == ["Wochen", "Prognose = Ist"]


def test_bland_altman_chart_shows_mean_and_both_limits():
    a = ev.analyse(ev.Settings(bias=10, n=200, seed=3))
    fig = V.build_ba(a)
    names = [t.name for t in fig.data]
    assert names[0] == "Wochen" and names[1] == "keine Abweichung" and names[2].startswith("mittlere Abweichung ") and names[3].startswith("untere Grenze ") and names[4].startswith("obere Grenze ")
    assert fig.data[3].y[0] == pytest.approx(a.ba["lower"]) and fig.data[4].y[0] == pytest.approx(a.ba["upper"]) and fig.data[2].y[0] == pytest.approx(a.ba["mean"])
    assert np.allclose(np.array(fig.data[0].y), (a.f - a.a)[:len(fig.data[0].y)])


def test_measures_chart_has_the_five_measures_with_expected_diamonds_and_none_for_nan():
    a = ev.analyse(ev.Settings(n=200, seed=3))
    fig = V.build_measures(a)
    assert list(fig.data[0].y) == ["Pearson r", "R²", "Lins CCC", "CCC nach Neukalibrierung", "innerhalb ±5"] and fig.data[1].name == "erwartet (exakt)"
    flat = V.build_measures(ev.analyse(ev.Settings(slope=0.0, sigma_e=0, n=50)))
    assert flat.data[0].x[0] is None and flat.data[0].text[0] == "–"


def test_e1_has_one_bar_group_per_measure_and_all_cases(grid):
    fig = V.build_e1(grid["e1"])
    assert [t.name for t in fig.data] == ["Pearson r", "R²", "Lins CCC"] and list(fig.data[0].x) == [C.CASE_NAMES[c["case"]] for c in grid["e1"]]


def test_e2_has_r2_and_ccc_for_both_biases(grid):
    fig = V.build_e2(grid["e2"])
    assert [t.name for t in fig.data] == ["R² (Bias 0)", "CCC (Bias 0)", "R² (Bias 4)", "CCC (Bias 4)"] and list(fig.data[0].x) == sorted(C.E2_SIGMAS)


def test_e3_and_e4_and_e5_values(grid):
    e3 = V.build_e3(grid["e3"])
    assert [t.name for t in e3.data] == ["CCC vorher", "CCC nach der Neukalibrierung", "Pearson r"]
    e4 = V.build_e4(grid["e4"][0])
    assert list(e4.data[0].x) == ["Pearson r", "Lins CCC", "RMSE", "MAE"] and list(e4.data[0].y) == [100.0, 0.0, 0.0, 0.0]
    e5 = V.build_e5(grid["e5"])
    assert [t.name for t in e5.data] == ["unteres Drittel des Ist", "oberes Drittel des Ist"] and len(e5.layout.shapes) == 1 and e5.layout.shapes[0].y0 == 5
