"""Das vorgerechnete Raster (data/agreement_grid.json): Aufbau, Plausibilität und Übereinstimmung mit dem Erzeuger (Probe)."""

import pytest

import ueb_algorithm as K
import ueb_constants as C
import ueb_evaluation as ev


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None, "data/agreement_grid.json fehlt: python tools/make_grid.py"
    return g


def _assert_close(a, b, path="probe"):
    """Verschachtelte Wörter und Listen: Zahlen mit relativer Toleranz (Summationsreihenfolge je Plattform, DEMO-PLAYBOOK §4), alles andere exakt."""
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-8, abs=1e-10), path
    else:
        assert a == b, path


def test_probe_still_matches_the_generator(grid):
    """Ändert sich Erzeuger, Statistik oder Seed-Ableitung, passt die gespeicherte Probe nicht mehr: dann tools/make_grid.py neu laufen lassen."""
    _assert_close(ev.grid_probe(), grid["probe"])


def test_grid_has_every_cell_exactly_once(grid):
    assert sorted(c["case"] for c in grid["e1"]) == sorted(C.CASES)
    assert sorted((c["case"], c["n"]) for c in grid["e1_spread"]) == sorted((k, n) for k in ("perfekt", "skala", "bias") for n in C.E1_SPREAD_NS)
    assert sorted((c["sigma_a"], c["bias"]) for c in grid["e2"]) == sorted((s, b) for s in C.E2_SIGMAS for b in C.E2_BIASES)
    assert sorted(c["case"] for c in grid["e3"]) == sorted(["bias", "flach", "skala", "bias_flach", "rauschen"]) and sorted(c["n"] for c in grid["e3_small"]) == [n for n in C.E3_SMALL_NS if n != C.E3_N]
    assert len(grid["e4"]) == 1 and sorted(c["case"] for c in grid["e5"]) == sorted(C.E5_CASES)


def test_grid_meta_and_run_counts(grid):
    k = len(C.GRID_SEEDS)
    assert grid["meta"]["seeds"] == list(C.GRID_SEEDS)
    assert all(c["n"] == C.E1_N and c["runs"] == k * C.E1_RUNS for c in grid["e1"]) and all(c["runs"] == k * C.E1_SPREAD_RUNS for c in grid["e1_spread"])
    assert all(c["n"] == C.E2_N and c["runs"] == k * C.E2_RUNS for c in grid["e2"]) and all(c["runs"] == k * C.E3_RUNS for c in grid["e3"] + grid["e3_small"])
    assert grid["e4"][0]["runs"] == k * C.E4_RUNS and all(c["runs"] == k * C.E5_RUNS and c["n"] == C.E5_N for c in grid["e5"])


def test_measured_values_agree_with_the_exact_formulas(grid):
    for c in grid["e1"]:
        for key in ("r", "ccc", "rmse", "mae"):
            assert c[key] == pytest.approx(c["exact"][key], abs=0.01 * max(1.0, abs(c["exact"][key]))), (c["case"], key)
        for t in C.E1_TOLS:
            assert c[f"within{t}"] == pytest.approx(c["exact"][f"within{t}"], abs=0.01), (c["case"], t)
    for c in grid["e2"]:
        for key in ("r", "r2", "ccc", "rmse", "within"):
            assert c[key] == pytest.approx(c["exact"][key], abs=0.01 * max(1.0, abs(c["exact"][key]))), (c["sigma_a"], c["bias"], key)


def test_stored_exact_values_equal_a_fresh_computation(grid):
    for c in grid["e1"]:
        w = (c["sigma_a"], C.CASES[c["case"]]["bias"], C.CASES[c["case"]]["slope"], C.CASES[c["case"]]["sigma_e"], False)
        assert c["exact"]["ccc"] == pytest.approx(K.exact_ccc(*w), abs=1e-12) and c["exact"]["rmse"] == pytest.approx(K.exact_rmse(*w), abs=1e-12) and c["exact"]["within5"] == pytest.approx(K.exact_within(*w, 5), abs=1e-12)


def test_recalibration_never_lowers_the_ccc_except_for_pure_noise(grid):
    for c in grid["e3"] + grid["e3_small"]:
        if c["case"] != "rauschen":
            assert c["ccc_after"] > c["ccc_before"] + 0.1 and c["rmse_after"] < c["rmse_before"] and c["ccc_after"] <= c["r"] + 0.005, c["case"]
    noise = [c for c in grid["e3"] if c["case"] == "rauschen"][0]
    assert abs(noise["ccc_after"] - noise["ccc_before"]) < 0.01


def test_the_headline_numbers_of_the_experiments(grid):
    scale = [c for c in grid["e1"] if c["case"] == "skala"][0]
    assert scale["r"] > 0.98 and scale["ccc"] < 0.55 and scale["rmse"] > 30 and scale["r"] == max(c["r"] for c in grid["e1"])
    assert grid["e4"][0]["a_wins"] == {"r": 1.0, "ccc": 0.0, "rmse": 0.0, "mae": 0.0}
    r2 = {c["sigma_a"]: c["r2"] for c in grid["e2"] if c["bias"] == 0}
    assert r2[5] < 0.62 and r2[40] > 0.98 and len({round(c["rmse"], 2) for c in grid["e2"] if c["bias"] == 0}) == 1
    prop = [c for c in grid["e5"] if c["case"] == "proportional"][0]
    assert prop["outside_high"] > 5 * prop["outside_low"]
