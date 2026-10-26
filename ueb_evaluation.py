"""Auswertung: eine Prognose gegen das Ist (Settings -> Analysis mit den Zeilen der Hauptansicht), Zellen des vorgerechneten Rasters (Messreihen), Verdikte als reine Funktionen
und die Probe für den Rastertest."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import ueb_algorithm as K
import ueb_constants as C
from prs_scenario import VRng

GRID_FILE = Path(__file__).resolve().parent / "data" / "agreement_grid.json"


def de(v: float, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma; „–“ für nicht definierte Werte."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.{digits}f}".replace(".", ",")


def pct(v: float, digits: int = 1) -> str:
    return de(100.0 * v, digits) + " %"


@dataclass(frozen=True, kw_only=True)
class Settings:
    bias: float = C.DEFAULT_BIAS
    slope: float = C.DEFAULT_SLOPE
    sigma_e: float = C.DEFAULT_SIGMA_E
    noise: str = "additiv"
    sigma_a: float = C.DEFAULT_SIGMA_A
    n: int = C.DEFAULT_N
    tol: float = C.DEFAULT_TOL
    seed: int = C.DEFAULT_SEED

    @property
    def proportional(self) -> bool:
        return self.noise == "proportional"

    def world(self) -> tuple:
        """Die Parameter der Welt in der Reihenfolge der exakten Formeln: (σ_a, Bias, Steigung, σ_e, proportional)."""
        return (self.sigma_a, self.bias, self.slope, self.sigma_e, self.proportional)


@dataclass(frozen=True)
class Row:
    key: str
    name: str
    value: float
    exact: float | None
    kind: str = "number"                 # „number“ (Dezimalstellen), „percent“ (Anteil)


@dataclass
class Analysis:
    settings: Settings
    f: np.ndarray
    a: np.ndarray
    rows: list
    ba: dict
    calibration: tuple                   # (Steigung, Achsenabschnitt) der Regression des Ist auf die Prognose
    ccc_calibrated: float                # CCC der neukalibrierten Prognose (in der Stichprobe)
    rmse_calibrated: float

    def row(self, key: str):
        return next((r for r in self.rows if r.key == key), None)


def make_data(s: Settings):
    return K.make_world(s.n, s.sigma_a, s.bias, s.slope, s.sigma_e, s.proportional, VRng(s.seed))


def analyse(s: Settings) -> Analysis:
    f, a = make_data(s)
    w = s.world()
    r = K.corr(f, a)
    r_exact = K.exact_r(*w)
    ccc = K.ccc(f, a)
    ba = K.bland_altman(f, a)
    try:
        slope, icpt = K.recalibrate(f, a)
        g = K.apply_calibration(f, slope, icpt)
        ccc_cal, rmse_cal = K.ccc(g, a), K.rmse(g, a)
    except ValueError:
        slope, icpt, ccc_cal, rmse_cal = float("nan"), float("nan"), float("nan"), float("nan")
    rows = [
        Row("r", "Pearson r", r, r_exact),
        Row("r2", "R² (= r²)", r * r, r_exact * r_exact),
        Row("ccc", "Lins CCC", ccc, K.exact_ccc(*w)),
        Row("bias", "mittlere Abweichung (Prognose − Ist)", K.mean_error(f, a), float(s.bias)),
        Row("rmse", "RMSE", K.rmse(f, a), K.exact_rmse(*w)),
        Row("mae", "MAE", K.mae(f, a), K.exact_mae(*w)),
        Row("within", f"Anteil innerhalb ±{s.tol:g}", K.within_tolerance(f, a, s.tol), K.exact_within(*w, s.tol), "percent"),
        Row("ccc_cal", "CCC nach Neukalibrierung", ccc_cal, K.exact_ccc_after_recalibration(r_exact) if not math.isnan(r_exact) else float("nan")),
    ]
    return Analysis(s, f, a, rows, ba, (slope, icpt), ccc_cal, rmse_cal)


# --- Zellen des Rasters (Mittel über viele Läufe) -----------------------------------------------------------------------------------------------

def _worlds(case: dict, sigma_a: float, n: int, runs: int, seeds: tuple):
    for seed in seeds:
        rng = VRng(seed)
        for _ in range(runs):
            yield K.make_world(n, sigma_a, case["bias"], case["slope"], case["sigma_e"], case["noise"] == "proportional", rng)


def _world_args(case: dict, sigma_a: float) -> tuple:
    return (sigma_a, case["bias"], case["slope"], case["sigma_e"], case["noise"] == "proportional")


def error_cell(key: str, n: int, runs: int, seeds: tuple = C.GRID_SEEDS, sigma_a: float = C.DEFAULT_SIGMA_A, tols: tuple = C.E1_TOLS) -> dict:
    """E1: Mittel und Streuung von r und CCC, Mittel von RMSE, MAE, mittlerer Abweichung, Anteilen innerhalb der Toleranzen und ICC(A,1) für eine Fehlerart; dazu die exakten Werte."""
    case = C.CASES[key]
    acc = []
    for f, a in _worlds(case, sigma_a, n, runs, seeds):
        r = K.corr(f, a)
        acc.append([r, r * r, K.ccc(f, a), K.rmse(f, a), K.mae(f, a), K.mean_error(f, a), K.icc_a1(f, a)] + [K.within_tolerance(f, a, t) for t in tols])
    m = np.array(acc, dtype=float)
    w = _world_args(case, sigma_a)
    out = {"case": key, "n": n, "runs": len(acc), "sigma_a": sigma_a}
    for i, name in enumerate(("r", "r2", "ccc", "rmse", "mae", "bias", "icc")):
        out[name] = float(m[:, i].mean())
    out["r_sd"], out["ccc_sd"] = float(m[:, 0].std()), float(m[:, 2].std())
    for j, t in enumerate(tols):
        out[f"within{t}"] = float(m[:, 7 + j].mean())
    out["exact"] = {"r": K.exact_r(*w), "r2": K.exact_r(*w) ** 2, "ccc": K.exact_ccc(*w), "rmse": K.exact_rmse(*w), "mae": K.exact_mae(*w), "bias": float(case["bias"]),
                    **{f"within{t}": K.exact_within(*w, t) for t in tols}}
    return out


def spread_cell(sigma_a: float, bias: float, n: int, runs: int, seeds: tuple = C.GRID_SEEDS, sigma_e: float = C.E2_NOISE, tol: float = C.E2_TOL) -> dict:
    """E2: gleicher Fehler, andere Streuung des Ist: Mittel von r, R², CCC, RMSE und Anteil innerhalb der Toleranz."""
    case = {"bias": bias, "slope": 1.0, "sigma_e": sigma_e, "noise": "additiv"}
    acc = []
    for f, a in _worlds(case, sigma_a, n, runs, seeds):
        r = K.corr(f, a)
        acc.append([r, r * r, K.ccc(f, a), K.rmse(f, a), K.within_tolerance(f, a, tol)])
    m = np.array(acc).mean(axis=0)
    w = _world_args(case, sigma_a)
    return {"sigma_a": sigma_a, "bias": bias, "n": n, "runs": len(acc), "tol": tol, "r": float(m[0]), "r2": float(m[1]), "ccc": float(m[2]), "rmse": float(m[3]), "within": float(m[4]),
            "exact": {"r": K.exact_r(*w), "r2": K.exact_r(*w) ** 2, "ccc": K.exact_ccc(*w), "rmse": K.exact_rmse(*w), "within": K.exact_within(*w, tol)}}


def recal_cell(key: str, n: int, runs: int, seeds: tuple = C.GRID_SEEDS, sigma_a: float = C.DEFAULT_SIGMA_A) -> dict:
    """E3: Neukalibrierung an der ersten Hälfte (Regression des Ist auf die Prognose), Bewertung an der zweiten Hälfte: r, CCC und RMSE vorher und nachher."""
    case = C.CASES[key]
    acc = []
    for f, a in _worlds(case, sigma_a, n, runs, seeds):
        h = n // 2
        slope, icpt = K.recalibrate(f[:h], a[:h])
        ft, at = f[h:], a[h:]
        g = K.apply_calibration(ft, slope, icpt)
        acc.append([K.corr(ft, at), K.ccc(ft, at), K.ccc(g, at), K.rmse(ft, at), K.rmse(g, at)])
    m = np.array(acc).mean(axis=0)
    return {"case": key, "n": n, "runs": len(acc), "r": float(m[0]), "ccc_before": float(m[1]), "ccc_after": float(m[2]), "rmse_before": float(m[3]), "rmse_after": float(m[4])}


def rank_cell(n: int, runs: int, seeds: tuple = C.GRID_SEEDS, sigma_a: float = C.DEFAULT_SIGMA_A) -> dict:
    """E4: zwei Prognosen A (Skala 1,3, Rauschen 4) und B (unverzerrt, Rauschen 10) gegen dieselben Ist-Werte: Mittel der Maße und Anteil der Läufe, in denen A besser abschneidet."""
    wins = {"r": 0, "ccc": 0, "rmse": 0, "mae": 0}
    acc = []
    total = 0
    for seed in seeds:
        rng = VRng(seed)
        for _ in range(runs):
            a = C.MU + sigma_a * rng.normal((n,))                        # dasselbe Ist für beide Prognosen, je eigenes Rauschen
            fa = C.MU + C.RANK_A["bias"] + C.RANK_A["slope"] * (a - C.MU) + C.RANK_A["sigma_e"] * rng.normal((n,))
            fb = C.MU + C.RANK_B["bias"] + C.RANK_B["slope"] * (a - C.MU) + C.RANK_B["sigma_e"] * rng.normal((n,))
            total += 1
            ra, rb, ca, cb = K.corr(fa, a), K.corr(fb, a), K.ccc(fa, a), K.ccc(fb, a)
            ea, eb, ma, mb = K.rmse(fa, a), K.rmse(fb, a), K.mae(fa, a), K.mae(fb, a)
            wins["r"] += ra > rb
            wins["ccc"] += ca > cb
            wins["rmse"] += ea < eb
            wins["mae"] += ma < mb
            acc.append([ra, rb, ca, cb, ea, eb, ma, mb])
    m = np.array(acc).mean(axis=0)
    names = ("r", "ccc", "rmse", "mae")
    return {"n": n, "runs": total, "a": {k: float(m[2 * i]) for i, k in enumerate(names)}, "b": {k: float(m[2 * i + 1]) for i, k in enumerate(names)}, "a_wins": {k: wins[k] / total for k in names}}


def ba_cell(key: str, n: int, runs: int, seeds: tuple = C.GRID_SEEDS, sigma_a: float = C.DEFAULT_SIGMA_A) -> dict:
    """E5: Bland–Altman-Kennzahlen für eine Fehlerart: Anteil außerhalb insgesamt und im unteren und oberen Drittel, Streuung der Differenzen im unteren und oberen Drittel, Steigung, Breite."""
    case = C.E5_CASES[key]
    acc = []
    for f, a in _worlds(case, sigma_a, n, runs, seeds):
        ba = K.bland_altman(f, a)
        d = f - a
        order = np.argsort(a, kind="stable")
        third = n // 3
        acc.append([ba["outside"], ba["outside_low"], ba["outside_high"], float(d[order[:third]].std(ddof=1)), float(d[order[-third:]].std(ddof=1)), ba["slope"], ba["upper"] - ba["lower"]])
    m = np.array(acc).mean(axis=0)
    return {"case": key, "n": n, "runs": len(acc), "outside": float(m[0]), "outside_low": float(m[1]), "outside_high": float(m[2]), "sd_low": float(m[3]), "sd_high": float(m[4]),
            "slope": float(m[5]), "width": float(m[6])}


def grid_probe() -> dict:
    """Kleine feste Probe, die tests/test_grid.py nachrechnet und mit der Datei vergleicht."""
    return {"error": error_cell("skala", 100, 1, seeds=(101, 102)), "spread": spread_cell(10, 4, 100, 1, seeds=(101,)), "recal": recal_cell("flach", 100, 1, seeds=(101,)),
            "rank": rank_cell(50, 2, seeds=(101,)), "ba": ba_cell("proportional", 90, 1, seeds=(101,))}


def load_grid(path: Path = GRID_FILE):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find(cells: list, **kw):
    return next((c for c in cells if all(c.get(k) == v for k, v in kw.items())), None)


# --- Verdikte (reine Funktionen: Art, Text) ----------------------------------------------------------------------------------------------------

def verdict_main(a: Analysis):
    s = a.settings
    r, ccc, rmse, bias = a.row("r").value, a.row("ccc").value, a.row("rmse").value, a.row("bias").value
    within = a.row("within").value
    if math.isnan(r):
        return "warning", (f"Die Prognose ist konstant: r ist nicht definiert. CCC {de(ccc)}, RMSE {de(rmse)}; innerhalb ±{s.tol:g} liegen {pct(within)} der Wochen.")
    if ccc >= 0.95:
        return "success", (f"Prognose und Ist stimmen überein: CCC **{de(ccc)}** (r {de(r)}), RMSE {de(rmse)}, mittlere Abweichung {de(bias)}; innerhalb ±{s.tol:g} liegen {pct(within)} der Wochen.")
    if r >= 0.9 and r - ccc > 0.05 + 1e-9:
        return "warning", (f"**Hoher Zusammenhang, schwache Übereinstimmung:** r = {de(r)} (R² {de(r * r)}), aber CCC = **{de(ccc)}** und RMSE = {de(rmse)} bei einer mittleren Abweichung von {de(bias)}. "
                           f"Ursache ist ein systematischer Fehler (Bias, Steigung); eine lineare Neukalibrierung brächte die CCC auf {de(a.ccc_calibrated)} und den RMSE auf {de(a.rmse_calibrated)}.")
    if r < 0.9:
        return "info", (f"Schon der Zusammenhang ist nicht eng: r = {de(r)} (R² {de(r * r)}), CCC {de(ccc)}, RMSE {de(rmse)}. Streuung lässt sich nicht herauskalibrieren "
                        f"(CCC nach Neukalibrierung {de(a.ccc_calibrated)}).")
    return "info", f"Zusammenhang und Übereinstimmung liegen nah beieinander: r = {de(r)}, CCC = {de(ccc)}, RMSE = {de(rmse)}."


def verdict_ba(a: Analysis):
    ba, s = a.ba, a.settings
    base = (f"Bland–Altman: mittlere Abweichung **{de(ba['mean'])}**, 95-%-Grenzen {de(ba['lower'])} bis {de(ba['upper'])} (Breite {de(ba['upper'] - ba['lower'])}); "
            f"{pct(ba['outside'])} der Wochen außerhalb (unteres Drittel des Ist {pct(ba['outside_low'])}, oberes {pct(ba['outside_high'])}); Steigung der Differenzen gegen das Mittel {de(ba['slope'], 3)}.")
    if s.n < 60:
        return "info", base + f" Bei n = {s.n} sind die Drittel zu klein für eine Aussage über die Form des Fehlers."
    if abs(ba["slope"]) >= 0.15:
        return "warning", base + " Die Differenzen hängen am Mittel: das kann eine Steigung ungleich 1 oder ein proportionaler Fehler sein, bei großem additivem Rauschen aber auch allein die Regression zur Mitte; ein Paar Grenzen passt dann nicht für alle Größen."
    if ba["outside_high"] - ba["outside_low"] >= 0.04 - 1e-9:
        return "warning", base + " Der Fehler wächst mit dem Ist: für große Werte sind die Grenzen zu eng, für kleine zu weit."
    return "info", base


def verdict_tolerance(a: Analysis):
    s = a.settings
    r = a.row("within")
    return "info", f"Innerhalb von ±{s.tol:g} liegen **{pct(r.value)}** der Wochen (erwartet {pct(r.exact)}). Das ist ein absolutes Maß: bei Steigung 1 und additivem Rauschen hängt es nicht an der Streuung des Ist."


# --- Verdikte der Experimente (aus dem Raster) --------------------------------------------------------------------------------------------------

def verdict_e1(cells: list, key: str):
    c = find(cells, case=key)
    return "info", (f"{C.CASE_NAMES[key]} (n = {c['n']}, {c['runs']} Läufe): r **{de(c['r'], 3)}** (R² {de(c['r2'], 3)}), CCC **{de(c['ccc'], 3)}** (erwartet {de(c['exact']['ccc'], 3)}), "
                    f"RMSE {de(c['rmse'])}, MAE {de(c['mae'])}, mittlere Abweichung {de(c['bias'])}; innerhalb ±5 {pct(c['within5'])}, innerhalb ±10 {pct(c['within10'])}.")


def verdict_e1_spread(cells: list, key: str):
    mine = sorted((c for c in cells if c["case"] == key), key=lambda c: c["n"])
    lo, hi = mine[0], mine[-1]
    return "info", (f"{C.CASE_NAMES[key]}: bei n = {lo['n']} streut r mit {de(lo['r_sd'], 3)} und die CCC mit {de(lo['ccc_sd'], 3)} (Mittel {de(lo['ccc'], 3)} gegen exakt {de(lo['exact']['ccc'], 3)}); "
                    f"bei n = {hi['n']} sind es {de(hi['r_sd'], 3)} und {de(hi['ccc_sd'], 3)}. ICC(A,1) liegt bei n = {lo['n']} um {de(abs(lo['ccc'] - lo['icc']), 4)} neben der CCC.")


def verdict_e2(cells: list, bias: float):
    mine = sorted((c for c in cells if c["bias"] == bias), key=lambda c: c["sigma_a"])
    lo, hi = mine[0], mine[-1]
    return "warning", (f"Dieselbe Prognose (RMSE {de(lo['rmse'])}, innerhalb ±{lo['tol']:g} {pct(lo['within'])}) hat bei einer Streuung des Ist von {lo['sigma_a']:g} ein R² von **{de(lo['r2'], 3)}** und "
                       f"bei {hi['sigma_a']:g} eines von **{de(hi['r2'], 3)}** (CCC {de(lo['ccc'], 3)} gegen {de(hi['ccc'], 3)}).")


def verdict_e3(cells: list, key: str):
    c = find(cells, case=key)
    return "info", (f"{C.CASE_NAMES[key]} (n = {c['n']}, Test auf der zweiten Hälfte): CCC {de(c['ccc_before'], 3)} → **{de(c['ccc_after'], 3)}** (r {de(c['r'], 3)}), RMSE {de(c['rmse_before'])} → **{de(c['rmse_after'])}**.")


def verdict_e4(cell: dict):
    a, b, w = cell["a"], cell["b"], cell["a_wins"]
    return "warning", (f"Prognose A (Skala 1,3): r {de(a['r'], 3)}, CCC {de(a['ccc'], 3)}, RMSE {de(a['rmse'])}; Prognose B (unverzerrt, mehr Rauschen): r {de(b['r'], 3)}, CCC {de(b['ccc'], 3)}, RMSE {de(b['rmse'])}. "
                       f"A schneidet besser ab nach r in **{pct(w['r'], 0)}**, nach CCC in **{pct(w['ccc'], 0)}**, nach RMSE in {pct(w['rmse'], 0)} und nach MAE in {pct(w['mae'], 0)} der {cell['runs']} Läufe.")


def verdict_e5(cell: dict):
    return "info", (f"{C.E5_NAMES[cell['case']]}: {pct(cell['outside'])} der Wochen außerhalb der Grenzen, im unteren Drittel {pct(cell['outside_low'])}, im oberen {pct(cell['outside_high'])}; "
                    f"Standardabweichung der Differenzen unten {de(cell['sd_low'])}, oben {de(cell['sd_high'])}; Steigung der Differenzen gegen das Mittel {de(cell['slope'], 3)}.")
