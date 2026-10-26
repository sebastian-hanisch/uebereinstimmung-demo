"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import numpy as np

import ueb_evaluation as ev


def _analysis(r=0.98, ccc=0.96, rmse=4.0, bias=0.0, within=0.8, n=500, tol=5.0, ccc_cal=0.97, rmse_cal=3.5, ba=None):
    s = ev.Settings(n=n, tol=tol)
    rows = [ev.Row("r", "Pearson r", r, 0.9), ev.Row("r2", "R²", r * r, 0.8), ev.Row("ccc", "Lins CCC", ccc, 0.9), ev.Row("bias", "mittlere Abweichung", bias, 0.0), ev.Row("rmse", "RMSE", rmse, 4.0),
            ev.Row("mae", "MAE", 3.0, 3.0), ev.Row("within", "innerhalb", within, 0.75, "percent"), ev.Row("ccc_cal", "CCC nach Neukalibrierung", ccc_cal, 0.9)]
    ba = ba or {"mean": 1.0, "sd": 4.0, "lower": -6.84, "upper": 8.84, "slope": 0.05, "outside": 0.05, "outside_low": 0.05, "outside_high": 0.05}
    return ev.Analysis(s, np.zeros(3), np.zeros(3), rows, ba, (1.0, 0.0), ccc_cal, rmse_cal)


def test_main_verdict_agreement_is_success_at_ccc_0_95_inclusive():
    kind, text = ev.verdict_main(_analysis(r=0.99, ccc=0.95, rmse=4.0, bias=0.2, within=0.789))
    assert kind == "success" and "CCC **0,95** (r 0,99), RMSE 4,00, mittlere Abweichung 0,20; innerhalb ±5 liegen 78,9 % der Wochen" in text
    assert ev.verdict_main(_analysis(r=0.99, ccc=0.9499))[0] != "success"


def test_main_verdict_high_r_but_low_ccc_is_a_warning_with_the_calibration_numbers():
    kind, text = ev.verdict_main(_analysis(r=0.99, ccc=0.52, rmse=31.08, bias=30.34, ccc_cal=0.99, rmse_cal=3.0))
    assert kind == "warning" and "r = 0,99 (R² 0,98), aber CCC = **0,52** und RMSE = 31,08 bei einer mittleren Abweichung von 30,34" in text
    assert "CCC auf 0,99 und den RMSE auf 3,00" in text


def test_main_verdict_boundaries_of_the_gap_and_of_r():
    assert ev.verdict_main(_analysis(r=0.9, ccc=0.84))[0] == "warning"                  # r ≥ 0,9 und r − CCC > 0,05
    assert ev.verdict_main(_analysis(r=0.9, ccc=0.85))[0] == "info"                     # Abstand genau 0,05 (Gleitkomma: nicht darüber)
    assert ev.verdict_main(_analysis(r=0.8999, ccc=0.5))[0] == "info" and "Schon der Zusammenhang ist nicht eng: r = 0,90" in ev.verdict_main(_analysis(r=0.8999, ccc=0.5))[1]


def test_main_verdict_for_weak_relationship_names_the_calibration():
    kind, text = ev.verdict_main(_analysis(r=0.7, ccc=0.66, rmse=20.0, ccc_cal=0.62))
    assert kind == "info" and "r = 0,70 (R² 0,49), CCC 0,66, RMSE 20,00" in text and "(CCC nach Neukalibrierung 0,62)" in text


def test_main_verdict_for_close_agreement_below_the_success_threshold():
    kind, text = ev.verdict_main(_analysis(r=0.93, ccc=0.9))
    assert kind == "info" and text == "Zusammenhang und Übereinstimmung liegen nah beieinander: r = 0,93, CCC = 0,90, RMSE = 4,00."


def test_main_verdict_constant_forecast():
    kind, text = ev.verdict_main(_analysis(r=float("nan"), ccc=0.0, rmse=20.0, within=0.2, tol=5.0))
    assert kind == "warning" and text.startswith("Die Prognose ist konstant: r ist nicht definiert. CCC 0,00, RMSE 20,00; innerhalb ±5 liegen 20,0 % der Wochen.")


def test_bland_altman_verdict_text_and_flags():
    kind, text = ev.verdict_ba(_analysis())
    assert kind == "info" and "mittlere Abweichung **1,00**, 95-%-Grenzen -6,84 bis 8,84 (Breite 15,68)" in text and "5,0 % der Wochen außerhalb (unteres Drittel des Ist 5,0 %, oberes 5,0 %)" in text
    assert "Steigung der Differenzen gegen das Mittel 0,050." in text
    base = {"mean": 1.0, "sd": 4.0, "lower": -6.84, "upper": 8.84, "outside": 0.05, "outside_low": 0.05, "outside_high": 0.05}
    kind, text = ev.verdict_ba(_analysis(ba={**base, "slope": 0.15}))
    assert kind == "warning" and "Die Differenzen hängen am Mittel" in text
    kind, text = ev.verdict_ba(_analysis(ba={**base, "slope": 0.1499}))
    assert kind == "info" and "hängen am Mittel" not in text
    kind, text = ev.verdict_ba(_analysis(ba={**base, "slope": 0.08, "outside_low": 0.02, "outside_high": 0.06}))
    assert kind == "warning" and "Der Fehler wächst mit dem Ist" in text
    assert ev.verdict_ba(_analysis(ba={**base, "slope": 0.08, "outside_low": 0.02, "outside_high": 0.0599}))[0] == "info"
    kind, text = ev.verdict_ba(_analysis(n=59, ba={**base, "slope": 0.9}))
    assert kind == "info" and "Bei n = 59 sind die Drittel zu klein" in text
    assert ev.verdict_ba(_analysis(n=60, ba={**base, "slope": 0.9}))[0] == "warning"


def test_tolerance_verdict():
    kind, text = ev.verdict_tolerance(_analysis(within=0.789, tol=5.0))
    assert kind == "info" and text.startswith("Innerhalb von ±5 liegen **78,9 %** der Wochen (erwartet 75,0 %).")


def _e1(case="skala"):
    return {"case": case, "n": 2000, "runs": 400, "r": 0.9883, "r2": 0.9767, "ccc": 0.5219, "rmse": 30.85, "mae": 29.99, "bias": 29.99, "within5": 0.0, "within10": 0.003, "exact": {"ccc": 0.5221}}


def test_experiment_verdicts_quote_their_cells():
    kind, text = ev.verdict_e1([_e1()], "skala")
    assert "Skala: f = 1,3 · a (n = 2000, 400 Läufe): r **0,988** (R² 0,977), CCC **0,522** (erwartet 0,522), RMSE 30,85, MAE 29,99, mittlere Abweichung 29,99; innerhalb ±5 0,0 %, innerhalb ±10 0,3 %." in text
    cells = [{"case": "skala", "n": 20, "r_sd": 0.005, "ccc_sd": 0.079, "ccc": 0.5, "icc": 0.5125, "exact": {"ccc": 0.522}}, {"case": "skala", "n": 1000, "r_sd": 0.001, "ccc_sd": 0.012, "ccc": 0.522, "icc": 0.5222, "exact": {"ccc": 0.522}},
             {"case": "bias", "n": 20, "r_sd": 0.0, "ccc_sd": 0.0, "ccc": 0.0, "icc": 0.0, "exact": {"ccc": 0.0}}]
    kind, text = ev.verdict_e1_spread(cells[::-1], "skala")
    assert "bei n = 20 streut r mit 0,005 und die CCC mit 0,079 (Mittel 0,500 gegen exakt 0,522)" in text and "bei n = 1000 sind es 0,001 und 0,012" in text and "um 0,0125 neben der CCC" in text


def test_experiment_verdicts_for_spread_recalibration_rank_and_bland_altman():
    cells = [{"sigma_a": 5, "bias": 0, "tol": 5.0, "rmse": 4.0, "within": 0.788, "r2": 0.611, "ccc": 0.757}, {"sigma_a": 40, "bias": 0, "tol": 5.0, "rmse": 4.0, "within": 0.788, "r2": 0.99, "ccc": 0.995},
             {"sigma_a": 10, "bias": 0, "tol": 5.0, "rmse": 4.0, "within": 0.788, "r2": 0.86, "ccc": 0.93}, {"sigma_a": 5, "bias": 4, "tol": 5.0, "rmse": 5.7, "within": 0.585, "r2": 0.611, "ccc": 0.61}]
    kind, text = ev.verdict_e2(cells, 0)
    assert kind == "warning" and "(RMSE 4,00, innerhalb ±5 78,8 %) hat bei einer Streuung des Ist von 5 ein R² von **0,611** und bei 40 eines von **0,990** (CCC 0,757 gegen 0,995)" in text
    kind, text = ev.verdict_e3([{"case": "skala", "n": 400, "r": 0.988, "ccc_before": 0.518, "ccc_after": 0.988, "rmse_before": 30.86, "rmse_after": 3.06}], "skala")
    assert "CCC 0,518 → **0,988** (r 0,988), RMSE 30,86 → **3,06**" in text
    cell = {"n": 200, "runs": 1000, "a": {"r": 0.988, "ccc": 0.52, "rmse": 30.86}, "b": {"r": 0.894, "ccc": 0.888, "rmse": 9.99}, "a_wins": {"r": 1.0, "ccc": 0.0, "rmse": 0.0, "mae": 0.0}}
    kind, text = ev.verdict_e4(cell)
    assert "A schneidet besser ab nach r in **100 %**, nach CCC in **0 %**, nach RMSE in 0 % und nach MAE in 0 % der 1000 Läufe." in text and "Prognose B (unverzerrt, mehr Rauschen): r 0,894, CCC 0,888, RMSE 9,99" in text
    kind, text = ev.verdict_e5({"case": "proportional", "outside": 0.0537, "outside_low": 0.0134, "outside_high": 0.1014, "sd_low": 6.31, "sd_high": 9.76, "slope": 0.0808})
    assert "proportional, 8 % des Ist: 5,4 % der Wochen außerhalb der Grenzen, im unteren Drittel 1,3 %, im oberen 10,1 %" in text and "unten 6,31, oben 9,76; Steigung der Differenzen gegen das Mittel 0,081." in text
