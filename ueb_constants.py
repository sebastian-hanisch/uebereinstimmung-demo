"""Konstanten: Regler-Grenzen, Welt, Raster der Messreihen, Farben und Presets.

Zufallserzeuger und Pearson kommen als wörtliche Kopien aus pearson-demo (prs_rng.py, prs_constants.py, prs_scenario.py, prs_algorithm.py).
Rechenbudget am Regler-Maximum (DEMO-PLAYBOOK §1c): n = 2 000 Wochen; alle Maße, das Bland–Altman-Diagramm und die exakten Formeln brauchen zusammen Millisekunden.
Die Wiederholungsexperimente stehen vorgerechnet im Raster (data/agreement_grid.json)."""

from prs_constants import SEED_MAX  # noqa: F401  (Wiederverwendung der Kopie)

MU = 100.0                                       # Mittelwert des Ist (fest)
SIGMA_A_MIN, SIGMA_A_MAX, DEFAULT_SIGMA_A = 5, 40, 20
BIAS_MIN, BIAS_MAX, DEFAULT_BIAS = -30, 40, 0
SLOPE_MIN, SLOPE_MAX, DEFAULT_SLOPE = 0.0, 1.5, 1.0
SIGMA_E_MIN, SIGMA_E_MAX, DEFAULT_SIGMA_E = 0, 25, 4
NOISES = {"additiv": "additiv (gleich groß)", "proportional": "proportional (wächst mit dem Ist)"}
N_OPTIONS = (20, 30, 50, 100, 200, 500, 1000, 2000)
DEFAULT_N = 500
TOL_MIN, TOL_MAX, DEFAULT_TOL = 1, 20, 5
DEFAULT_SEED = 35
Z_LIMIT = 1.96                                   # Bland–Altman: 95-%-Grenzen der Differenzen

# Raster der Messreihen (tools/make_grid.py)
GRID_SEEDS = (101, 102, 103, 104)
E1_N, E1_RUNS = 2000, 100                        # Fehlerarten: Wochen und Läufe je Seed
E1_TOLS = (5, 10)
E1_SPREAD_NS, E1_SPREAD_RUNS = (20, 30, 100, 1000), 250   # Streuung der Maße gegen n
E2_N, E2_RUNS = 1000, 100                        # Streuung des Ist
E2_SIGMAS, E2_BIASES, E2_NOISE, E2_TOL = (5, 10, 20, 40), (0, 4), 4, 5
E3_N, E3_RUNS = 400, 200                         # Neukalibrierung (Trainingshälfte, Testhälfte)
E3_SMALL_NS = (30, 60, 400)
E4_N, E4_RUNS = 200, 250                         # Rangumkehr
E5_N, E5_RUNS = 2000, 100                        # Bland–Altman bei wachsendem Fehler

# Fehlerarten (Bias, Steigung, Rauschen, proportional) bei Streuung des Ist 20; "Skala": f = 1,3 · a
CASES = {
    "perfekt": {"bias": 0, "slope": 1.0, "sigma_e": 4, "noise": "additiv"},
    "bias": {"bias": 10, "slope": 1.0, "sigma_e": 4, "noise": "additiv"},
    "flach": {"bias": 0, "slope": 0.5, "sigma_e": 4, "noise": "additiv"},
    "skala": {"bias": 30, "slope": 1.3, "sigma_e": 4, "noise": "additiv"},
    "bias_flach": {"bias": 10, "slope": 0.5, "sigma_e": 4, "noise": "additiv"},
    "rauschen": {"bias": 0, "slope": 1.0, "sigma_e": 20, "noise": "additiv"},
}
CASE_NAMES = {"perfekt": "perfekt (Rauschen 4)", "bias": "Bias +10", "flach": "Steigung 0,5", "skala": "Skala: f = 1,3 · a", "bias_flach": "Bias +10 und Steigung 0,5", "rauschen": "nur Rauschen 20"}
E5_CASES = {
    "additiv": {"bias": 0, "slope": 1.0, "sigma_e": 8, "noise": "additiv"},
    "proportional": {"bias": 0, "slope": 1.0, "sigma_e": 8, "noise": "proportional"},
    "steigung": {"bias": 30, "slope": 1.3, "sigma_e": 4, "noise": "additiv"},
}
E5_NAMES = {"additiv": "additiv, Rauschen 8", "proportional": "proportional, 8 % des Ist", "steigung": "Steigung 1,3, additiv 4"}
# Rangumkehr: A = Skala 1,3 mit Rauschen 4, B = unverzerrt mit Rauschen 10
RANK_A = {"bias": 30, "slope": 1.3, "sigma_e": 4, "noise": "additiv"}
RANK_B = {"bias": 0, "slope": 1.0, "sigma_e": 10, "noise": "additiv"}

COLOR_POINT = "#4c78a8"
COLOR_IDENTITY = "#222222"
COLOR_FIT = "#e45756"
COLOR_LIMIT = "#f58518"
COLOR_GOOD = "#2F6B65"
COLOR_GREY = "#9d9d9d"
MEASURE_COLORS = {"r": "#4c78a8", "r2": "#9ecae9", "ccc": "#e45756", "rmse": "#54a24b"}

PRESETS = {
    "Standardfall (Voreinstellung)": {"bias": 0, "slope": 1.0, "sigma_e": 4, "noise": "additiv", "sigma_a": 20, "n": 500, "tol": 5, "seed": DEFAULT_SEED},
    "Falsche Skala": {"bias": 30, "slope": 1.3, "sigma_e": 4, "noise": "additiv", "sigma_a": 20, "n": 500, "tol": 5, "seed": DEFAULT_SEED},
    "Konstanter Bias": {"bias": 10, "slope": 1.0, "sigma_e": 4, "noise": "additiv", "sigma_a": 20, "n": 500, "tol": 5, "seed": DEFAULT_SEED},
    "Zu flach": {"bias": 0, "slope": 0.5, "sigma_e": 4, "noise": "additiv", "sigma_a": 20, "n": 500, "tol": 5, "seed": DEFAULT_SEED},
    "Nur Rauschen": {"bias": 0, "slope": 1.0, "sigma_e": 20, "noise": "additiv", "sigma_a": 20, "n": 500, "tol": 5, "seed": DEFAULT_SEED},
    "Proportionaler Fehler": {"bias": 0, "slope": 1.0, "sigma_e": 8, "noise": "proportional", "sigma_a": 20, "n": 500, "tol": 5, "seed": DEFAULT_SEED},
    "Schmale Streuung des Ist": {"bias": 0, "slope": 1.0, "sigma_e": 4, "noise": "additiv", "sigma_a": 5, "n": 500, "tol": 5, "seed": DEFAULT_SEED},
}
# Die Hilfetexte nennen exakte Werte; tests/test_claims.py prüft jede davon.
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Eine Prognose ohne Bias und mit Rauschen 4 bei einem Ist, das mit 20 streut: erwartet r = 0,98, CCC = 0,98 und RMSE = 4,0; innerhalb ±5 liegen erwartet 78,9 % der Wochen.",
    "Falsche Skala": "Die Prognose ist 1,3 · Ist: erwartet r = 0,99 (R² = 0,98), aber CCC = 0,52 und RMSE = 30,9; innerhalb ±5 liegen erwartet 0,0 % der Wochen. Die Neukalibrierung brächte die CCC auf 0,99.",
    "Konstanter Bias": "Bias +10: r bleibt bei erwartet 0,98, die CCC fällt auf 0,87 und der RMSE steigt auf 10,8; innerhalb ±5 liegen erwartet 10,6 % der Wochen.",
    "Zu flach": "Steigung 0,5: erwartet r = 0,93, CCC = 0,78 und RMSE = 10,8; im Bland–Altman-Diagramm hängen die Differenzen am Mittel (Steigung -0,64 in dieser Stichprobe).",
    "Nur Rauschen": "Rauschen 20 ohne Bias: erwartet r = 0,71, CCC = 0,67 und RMSE = 20,0; die Neukalibrierung hilft nicht (CCC danach in dieser Stichprobe 0,62).",
    "Proportionaler Fehler": "Rauschen von 8 % des Ist: im Bland–Altman-Diagramm liegen in dieser Stichprobe im oberen Drittel des Ist 9,0 % der Wochen außerhalb der Grenzen, im unteren 3,0 % (bei gleichbleibendem Fehler wären es je 5 %).",
    "Schmale Streuung des Ist": "Dieselbe Prognose wie im Standardfall (Rauschen 4), aber das Ist streut nur mit 5: erwartet r = 0,78 (R² = 0,61) statt 0,98; RMSE (4,0) und Anteil innerhalb ±5 (78,9 %) bleiben gleich.",
}
