"""Jede Zahl der README steht hier und wird aus dem vorgerechneten Raster (data/agreement_grid.json) oder über dieselben Funktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Zufall nur aus SplitMix64 mit festen Seeds. Die Tabellen der README werden Zeile für Zeile in der Rundung der App (`_c`, `_p`) gegen die README geprüft. Die Zeilen-Bauer (`e1_rows` usw.) und
`numbers()` dienen auch dazu, die README-Tabellen und -Zahlen zu erzeugen."""

import json
from pathlib import Path

import pytest

import ueb_algorithm as K
import ueb_constants as C
import ueb_evaluation as ev

ROOT = Path(__file__).resolve().parent.parent
README_PATH = ROOT / "README.md"
README = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""


def _c(v, d):
    return f"{round(v, d) + 0.0:.{d}f}".replace(".", ",")                  # kein „-0,000“


def _p(v, d=1):
    return f"{100 * v:.{d}f}".replace(".", ",") + " %"


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None
    return g


def cell(grid, group, **kw):
    c = ev.find(grid[group], **kw)
    assert c is not None, (group, kw)
    return c


# --- Zeilen-Bauer (Rundung der App) ------------------------------------------------------------------------------------------------------------

def e1_rows(grid) -> list[str]:
    rows = []
    for k in C.CASES:
        c = cell(grid, "e1", case=k)
        rows.append(f"| {C.CASE_NAMES[k]} | {_c(c['r'], 3)} ({_c(c['r2'], 3)}) | {_c(c['ccc'], 3)} ({_c(c['exact']['ccc'], 3)}) | {_c(c['rmse'], 2)} ({_c(c['exact']['rmse'], 2)}) | {_c(c['mae'], 2)} | {_c(c['bias'], 2)} |")
    return rows


def tolerance_rows(grid) -> list[str]:
    return [f"| {C.CASE_NAMES[k]} | {_p(cell(grid, 'e1', case=k)['within5'])} ({_p(cell(grid, 'e1', case=k)['exact']['within5'])}) | {_p(cell(grid, 'e1', case=k)['within10'])} ({_p(cell(grid, 'e1', case=k)['exact']['within10'])}) |"
            for k in C.CASES]


def spread_rows(grid) -> list[str]:
    rows = []
    for k, label in (("skala", "Skala 1,3"), ("bias", "Bias +10")):
        cells = [cell(grid, "e1_spread", case=k, n=n) for n in C.E1_SPREAD_NS]
        rows.append(f"| {label}: r (Streuung) | " + " | ".join(f"{_c(c['r'], 3)} ({_c(c['r_sd'], 3)})" for c in cells) + " |")
        rows.append(f"| {label}: CCC (Streuung) | " + " | ".join(f"{_c(c['ccc'], 3)} ({_c(c['ccc_sd'], 3)})" for c in cells) + " |")
        rows.append(f"| {label}: ICC(A,1) − CCC | " + " | ".join(_c(c["icc"] - c["ccc"], 4) for c in cells) + " |")
    return rows


def recal_rows(grid) -> list[str]:
    out = []
    for k in ("bias", "flach", "skala", "bias_flach", "rauschen"):
        c = cell(grid, "e3", case=k)
        out.append(f"| {C.CASE_NAMES[k]} | {_c(c['r'], 3)} | {_c(c['ccc_before'], 3)} → {_c(c['ccc_after'], 3)} | {_c(c['rmse_before'], 2)} → {_c(c['rmse_after'], 2)} |")
    return out


def recal_small_rows(grid) -> list[str]:
    cells = sorted(grid["e3_small"] + [c for c in grid["e3"] if c["case"] == "skala"], key=lambda c: c["n"])
    return [f"| {c['n']} | {c['n'] // 2} | {_c(c['ccc_before'], 3)} → {_c(c['ccc_after'], 3)} | {_c(c['r'], 3)} |" for c in cells]


def rank_rows(grid) -> list[str]:
    c = grid["e4"][0]
    names = {"r": "Pearson r", "ccc": "Lins CCC", "rmse": "RMSE", "mae": "MAE"}
    return [f"| {names[k]} | {_c(c['a'][k], 3 if k in ('r', 'ccc') else 2)} | {_c(c['b'][k], 3 if k in ('r', 'ccc') else 2)} | {_p(c['a_wins'][k], 0)} |" for k in names]


def ba_rows(grid) -> list[str]:
    return [f"| {C.E5_NAMES[k]} | {_p(cell(grid, 'e5', case=k)['outside'])} | {_p(cell(grid, 'e5', case=k)['outside_low'])} / {_p(cell(grid, 'e5', case=k)['outside_high'])} | "
            f"{_c(cell(grid, 'e5', case=k)['sd_low'], 2)} / {_c(cell(grid, 'e5', case=k)['sd_high'], 2)} | {_c(cell(grid, 'e5', case=k)['slope'], 3)} |" for k in C.E5_CASES]


def e2_rows(grid) -> list[str]:
    rows = []
    for s in C.E2_SIGMAS:
        c0, c4 = cell(grid, "e2", sigma_a=s, bias=0), cell(grid, "e2", sigma_a=s, bias=4)
        rows.append(f"| {s} | {_c(c0['r'], 3)} ({_c(c0['r2'], 3)}) | {_c(c0['ccc'], 3)} | {_c(c4['ccc'], 3)} | {_c(c0['rmse'], 2)} / {_c(c4['rmse'], 2)} | {_p(c0['within'])} / {_p(c4['within'])} |")
    return rows


def numbers(grid) -> dict:
    """Die Zahlen im Fließtext der README (Name -> Zeichenkette in der Rundung der App)."""
    e1 = {k: cell(grid, "e1", case=k) for k in C.CASES}
    sp = {(k, n): cell(grid, "e1_spread", case=k, n=n) for k in ("perfekt", "skala", "bias") for n in C.E1_SPREAD_NS}
    e2 = {(s, b): cell(grid, "e2", sigma_a=s, bias=b) for s in C.E2_SIGMAS for b in C.E2_BIASES}
    rc = {k: cell(grid, "e3", case=k) for k in ("bias", "flach", "skala", "bias_flach", "rauschen")}
    sm = {c["n"]: c for c in grid["e3_small"]}
    rk = grid["e4"][0]
    ba = {k: cell(grid, "e5", case=k) for k in C.E5_CASES}
    return {
        "sk_r": _c(e1["skala"]["r"], 3), "sk_r2": _c(e1["skala"]["r2"], 3), "sk_ccc": _c(e1["skala"]["ccc"], 3), "sk_rmse": _c(e1["skala"]["rmse"], 2), "sk_mae": _c(e1["skala"]["mae"], 2),
        "sk_w5": _p(e1["skala"]["within5"]), "sk_w10": _p(e1["skala"]["within10"]), "bias_r": _c(e1["bias"]["r"], 3), "bias_ccc": _c(e1["bias"]["ccc"], 3), "bias_rmse": _c(e1["bias"]["rmse"], 2),
        "perf_ccc": _c(e1["perfekt"]["ccc"], 3), "rau_r": _c(e1["rauschen"]["r"], 3), "rau_ccc": _c(e1["rauschen"]["ccc"], 3), "fl_ccc": _c(e1["flach"]["ccc"], 3),
        "sk_ccc_n20": _c(sp[("skala", 20)]["ccc"], 3), "sk_ccc_sd20": _c(sp[("skala", 20)]["ccc_sd"], 3), "sk_r_sd20": _c(sp[("skala", 20)]["r_sd"], 3), "sk_ccc_sd1000": _c(sp[("skala", 1000)]["ccc_sd"], 3),
        "bias_ccc_sd30": _c(sp[("bias", 30)]["ccc_sd"], 3), "bias_r_sd30": _c(sp[("bias", 30)]["r_sd"], 3), "icc20": _c(abs(sp[("skala", 20)]["icc"] - sp[("skala", 20)]["ccc"]), 4),
        "icc100": _c(abs(sp[("skala", 100)]["icc"] - sp[("skala", 100)]["ccc"]), 4), "icc1000": _c(abs(sp[("skala", 1000)]["icc"] - sp[("skala", 1000)]["ccc"]), 4),
        "e2_r2_5": _c(e2[(5, 0)]["r2"], 3), "e2_r2_40": _c(e2[(40, 0)]["r2"], 3), "e2_rmse": _c(e2[(5, 0)]["rmse"], 2), "e2_rmse4": _c(e2[(5, 4)]["rmse"], 2), "e2_w0": _p(e2[(5, 0)]["within"]), "e2_w4": _p(e2[(5, 4)]["within"]),
        "rc_sk_b": _c(rc["skala"]["ccc_before"], 3), "rc_sk_a": _c(rc["skala"]["ccc_after"], 3), "rc_sk_rb": _c(rc["skala"]["rmse_before"], 2), "rc_sk_ra": _c(rc["skala"]["rmse_after"], 2),
        "rc_rau_b": _c(rc["rauschen"]["ccc_before"], 3), "rc_rau_a": _c(rc["rauschen"]["ccc_after"], 3), "rc_rau_rb": _c(rc["rauschen"]["rmse_before"], 2), "rc_rau_ra": _c(rc["rauschen"]["rmse_after"], 2),
        "rc_sm30": _c(sm[30]["ccc_after"], 3), "rc_sm30_r": _c(sm[30]["r"], 3),
        "rk_ar": _c(rk["a"]["r"], 3), "rk_br": _c(rk["b"]["r"], 3), "rk_accc": _c(rk["a"]["ccc"], 3), "rk_bccc": _c(rk["b"]["ccc"], 3), "rk_armse": _c(rk["a"]["rmse"], 2), "rk_brmse": _c(rk["b"]["rmse"], 2),
        "rk_runs": str(rk["runs"]), "ba_out": _p(ba["proportional"]["outside"]), "ba_low": _p(ba["proportional"]["outside_low"]), "ba_high": _p(ba["proportional"]["outside_high"]),
        "ba_slope": _c(ba["steigung"]["slope"], 3), "ba_st_out": _p(ba["steigung"]["outside_low"]), "grid_seconds": str(grid["meta"]["seconds"]), "e1_runs": str(e1["skala"]["runs"]),
        "sp_runs": str(sp[("skala", 20)]["runs"]), "e2_runs": str(e2[(5, 0)]["runs"]), "e3_runs": str(rc["skala"]["runs"]), "e5_runs": str(ba["additiv"]["runs"]),
    }


# --- Tabellen und Zahlen der README --------------------------------------------------------------------------------------------------------------

def test_claim_readme_exists_and_has_the_required_sections():
    assert README, "README.md fehlt"
    for heading in ("## Warum dieses Problem", "## Modell", "## Methodik", "## Befunde (gemessen, keine Behauptungen)", "## Ehrliche Grenzen", "## Verifikation", "## Dateistruktur", "## Bewusst nicht umgesetzt"):
        assert heading in README, heading
    assert "TODO" not in README.replace("<!-- TODO nach dem Deploy", "")


@pytest.mark.parametrize("builder", [e1_rows, tolerance_rows, spread_rows, recal_rows, recal_small_rows, rank_rows, ba_rows, e2_rows])
def test_claim_readme_tables_match_the_grid_row_by_row(grid, builder):
    for row in builder(grid):
        assert row in README, row


def test_claim_readme_numbers_in_the_text(grid):
    for name, text in numbers(grid).items():
        assert text in README, (name, text)


def test_claim_the_headline_findings_hold_in_the_grid(grid):
    e1 = {c["case"]: c for c in grid["e1"]}
    assert e1["skala"]["r"] == max(c["r"] for c in grid["e1"]) and e1["skala"]["rmse"] == max(c["rmse"] for c in grid["e1"]) and e1["skala"]["ccc"] < 0.55
    assert e1["bias"]["r"] == pytest.approx(e1["perfekt"]["r"], abs=0.002) and e1["bias"]["ccc"] < e1["perfekt"]["ccc"] - 0.1
    sk, bs = cell(grid, "e1_spread", case="skala", n=30), cell(grid, "e1_spread", case="bias", n=30)
    assert sk["ccc_sd"] > 10 * sk["r_sd"] and bs["ccc_sd"] > 4 * bs["r_sd"] and cell(grid, "e1_spread", case="skala", n=20)["ccc"] < cell(grid, "e1_spread", case="skala", n=20)["exact"]["ccc"]
    for c in grid["e1_spread"]:
        assert abs(c["icc"] - c["ccc"]) <= 0.01251 and (c["n"] < 1000 or abs(c["icc"] - c["ccc"]) < 0.001)
    r2 = {s: cell(grid, "e2", sigma_a=s, bias=0)["r2"] for s in C.E2_SIGMAS}
    assert r2[5] < 0.62 and r2[40] > 0.98 and len({round(c["rmse"], 6) for c in grid["e2"] if c["bias"] == 0}) == 1 and len({round(c["within"], 6) for c in grid["e2"] if c["bias"] == 0}) == 1
    assert grid["e4"][0]["a_wins"] == {"r": 1.0, "ccc": 0.0, "rmse": 0.0, "mae": 0.0}
    prop = cell(grid, "e5", case="proportional")
    assert prop["outside_high"] > 5 * prop["outside_low"] and cell(grid, "e5", case="additiv")["outside_low"] == pytest.approx(0.05, abs=0.005)


# --- Texte der Presets ---------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def analyses():
    return {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}


def test_claim_the_preset_help_texts_quote_exact_and_sample_numbers(analyses):
    h = C.PRESET_HELP

    def ex(name):
        a = analyses[name]
        w = a.settings.world()
        return a, {"r": K.exact_r(*w), "ccc": K.exact_ccc(*w), "rmse": K.exact_rmse(*w), "w": K.exact_within(*w, a.settings.tol)}

    a, e = ex("Standardfall (Voreinstellung)")
    assert f"erwartet r = {_c(e['r'], 2)}, CCC = {_c(e['ccc'], 2)} und RMSE = {_c(e['rmse'], 1)}; innerhalb ±5 liegen erwartet {_p(e['w'])} der Wochen" in h["Standardfall (Voreinstellung)"]
    a, e = ex("Falsche Skala")
    cal = K.exact_ccc_after_recalibration(e["r"])
    assert f"erwartet r = {_c(e['r'], 2)} (R² = {_c(e['r'] ** 2, 2)}), aber CCC = {_c(e['ccc'], 2)} und RMSE = {_c(e['rmse'], 1)}; innerhalb ±5 liegen erwartet {_p(e['w'])} der Wochen" in h["Falsche Skala"]
    assert f"auf {_c(cal, 2)}" in h["Falsche Skala"]
    a, e = ex("Konstanter Bias")
    assert f"r bleibt bei erwartet {_c(e['r'], 2)}, die CCC fällt auf {_c(e['ccc'], 2)} und der RMSE steigt auf {_c(e['rmse'], 1)}; innerhalb ±5 liegen erwartet {_p(e['w'])} der Wochen" in h["Konstanter Bias"]
    a, e = ex("Zu flach")
    assert f"erwartet r = {_c(e['r'], 2)}, CCC = {_c(e['ccc'], 2)} und RMSE = {_c(e['rmse'], 1)}" in h["Zu flach"] and f"(Steigung {_c(a.ba['slope'], 2)} in dieser Stichprobe)" in h["Zu flach"]
    a, e = ex("Nur Rauschen")
    assert f"erwartet r = {_c(e['r'], 2)}, CCC = {_c(e['ccc'], 2)} und RMSE = {_c(e['rmse'], 1)}" in h["Nur Rauschen"] and f"(CCC danach in dieser Stichprobe {_c(a.ccc_calibrated, 2)})" in h["Nur Rauschen"]
    a, e = ex("Proportionaler Fehler")
    assert f"im oberen Drittel des Ist {_p(a.ba['outside_high'])} der Wochen außerhalb der Grenzen, im unteren {_p(a.ba['outside_low'])}" in h["Proportionaler Fehler"]
    a, e = ex("Schmale Streuung des Ist")
    std = ex("Standardfall (Voreinstellung)")[1]
    assert f"erwartet r = {_c(e['r'], 2)} (R² = {_c(e['r'] ** 2, 2)}) statt {_c(std['r'], 2)}; RMSE ({_c(e['rmse'], 1)}) und Anteil innerhalb ±5 ({_p(e['w'])}) bleiben gleich" in h["Schmale Streuung des Ist"]
    assert e["rmse"] == std["rmse"] and e["w"] == std["w"]


def test_claim_hand_values_of_the_text():
    import numpy as np
    f, a = np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 1.0, 4.0, 3.0])
    assert K.corr(f, a) == pytest.approx(0.6) and K.ccc(f, a) == pytest.approx(0.6) and K.rmse(f, a) == 1.0
    assert K.exact_ccc(20, 30, 1.3, 0, False) == pytest.approx(1040 / 1976) and K.exact_ccc(20, 30, 1.3, 4, False) == pytest.approx(1040 / 1992)
    assert K.exact_ccc(10, 10, 1.0, 0, False) == pytest.approx(200 / 300)
    assert K.bland_altman(f, a)["lower"] == pytest.approx(-2.2632, abs=1e-4)


def test_claim_mutation_numbers_in_the_readme():
    state = json.loads((ROOT / "tests" / "mutation_state.json").read_text(encoding="utf-8"))
    killed = sum(1 for v in state.values() if v == "killed")
    equivalent = sum(1 for v in state.values() if v == "equivalent")
    assert (killed, equivalent) == (117, 5)
    assert "117 durch eigene Tests gefangen" in README and "5 als gleichwertig festgehalten" in README and "125 Stellen" in README
