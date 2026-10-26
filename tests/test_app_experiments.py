"""App: jedes Preset, der Permalink, jedes Experiment-Auswahlfeld einzeln und die Eckfall-Kombinationen (DEMO-PLAYBOOK §1d)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ueb_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.success] + [i.value for i in at.info] + [w.value for w in at.warning])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_shows_its_data(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    p = C.PRESETS[name]
    assert at.session_state["bias_slider"] == p["bias"] and at.session_state["slope_slider"] == p["slope"] and at.session_state["sigma_e_slider"] == p["sigma_e"]
    assert at.session_state["noise_select"] == p["noise"] and at.session_state["sigma_a_slider"] == p["sigma_a"] and at.session_state["n_select"] == p["n"]
    assert len(at.get("plotly_chart")) == 3 + 5          # Hauptansicht: Maße, Streudiagramm, Bland–Altman; E1 bis E5: je eine Abbildung


def test_permalink_is_loaded_and_clamped():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["bias"] = "999"
    at.query_params["slope"] = "-3"
    at.query_params["noise"] = "proportional"
    at.query_params["n"] = "100"
    at.query_params["sigmaa"] = "abc"                 # ungültig: wird ignoriert
    at.query_params["tol"] = "0"
    at.run()
    assert not at.exception
    assert at.session_state["bias_slider"] == C.BIAS_MAX and at.session_state["slope_slider"] == C.SLOPE_MIN and at.session_state["noise_select"] == "proportional"
    assert at.session_state["n_select"] == 100 and at.session_state["sigma_a_slider"] == C.DEFAULT_SIGMA_A and at.session_state["tol_slider"] == C.TOL_MIN


def test_permalink_ignores_unknown_choices():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["noise"] = "zufall"
    at.query_params["n"] = "77"
    at.run()
    assert not at.exception and at.session_state["noise_select"] == "additiv" and at.session_state["n_select"] == C.DEFAULT_N


def test_four_metrics_show_r_ccc_rmse_and_the_tolerance_share():
    at = _app()
    assert [m.label for m in at.metric] == ["Pearson r", "Lins CCC", "RMSE", "Innerhalb ±5"] and all(m.value for m in at.metric)
    at.sidebar.slider(key="tol_slider").set_value(10).run()
    assert not at.exception and at.metric[3].label == "Innerhalb ±10"


def test_the_headline_verdict_depends_on_the_forecast():
    at = _app()
    assert "Prognose und Ist stimmen überein" in _texts(at) and "Hoher Zusammenhang, schwache Übereinstimmung" not in _texts(at)
    at.sidebar.button(key="preset_Falsche Skala").click().run()
    assert not at.exception and "Hoher Zusammenhang, schwache Übereinstimmung" in _texts(at)


def test_the_table_lists_all_measures_with_expected_values():
    at = _app()
    rows = at.dataframe[0].value
    assert list(rows["Maß"])[:3] == ["Pearson r", "R² (= r²)", "Lins CCC"] and len(rows) == 8 and "–" not in list(rows["erwartet (exakt)"])


def test_every_sidebar_input_runs_at_its_corners():
    at = _app()
    for key, value in (("noise_select", "proportional"), ("noise_select", "additiv")):
        at.sidebar.selectbox(key=key).set_value(value).run()
        assert not at.exception, (key, value, [str(e) for e in at.exception])
    for key, value in (("n_select", 20), ("n_select", 2000), ("n_select", 500)):
        at.sidebar.select_slider(key=key).set_value(value).run()
        assert not at.exception, (key, value, [str(e) for e in at.exception])
    at.sidebar.slider(key="slope_slider").set_value(0.0).run()
    at.sidebar.slider(key="sigma_e_slider").set_value(0).run()
    assert not at.exception and "Die Prognose ist konstant" in _texts(at)                  # r nicht definiert, aber keine Ausnahme


def test_pdf_button_then_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception and len(at.get("download_button")) == 1


def test_experiment_selectors_run():
    at = _app()
    for key, value in (("e1_case", "perfekt"), ("e1_case", "rauschen"), ("e1s_case", "bias"), ("e1s_case", "perfekt"), ("e2_bias", 4), ("e3_case", "rauschen"), ("e3_case", "bias"),
                       ("e5_case", "additiv"), ("e5_case", "steigung")):
        at.selectbox(key=key).set_value(value).run()
        assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_the_experiment_sections_show_their_verdicts():
    text = _texts(at := _app())
    for needle in ("Skala: f = 1,3 · a (n = 2000, 400 Läufe): r **0,988**", "streut r mit", "Dieselbe Prognose (RMSE 4,00", "CCC 0,518 → **0,988**", "A schneidet besser ab nach r in **100 %**", "proportional, 8 % des Ist:"):
        assert needle in text, needle
    assert at is not None
