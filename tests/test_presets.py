"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Begrenzung."""

import pytest

import ueb_constants as C
import ueb_evaluation as ev
import ueb_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert C.BIAS_MIN <= p["bias"] <= C.BIAS_MAX and C.SLOPE_MIN <= p["slope"] <= C.SLOPE_MAX and C.SIGMA_E_MIN <= p["sigma_e"] <= C.SIGMA_E_MAX, name
        assert p["noise"] in C.NOISES and C.SIGMA_A_MIN <= p["sigma_a"] <= C.SIGMA_A_MAX and p["n"] in C.N_OPTIONS and C.TOL_MIN <= p["tol"] <= C.TOL_MAX and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()
    assert next(iter(C.PRESETS)) == "Standardfall (Voreinstellung)"


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_differ_and_the_scale_preset_is_the_factor_one_point_three():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)
    p = C.PRESETS["Falsche Skala"]
    assert 100 + p["bias"] + p["slope"] * (0 - 100) == pytest.approx(0.0, abs=1e-9)                 # Prognose = 1,3 · Ist: bei Ist 0 ist die Prognose 0


@pytest.mark.parametrize("key, raw, expected", [
    ("bias_slider", "99", C.BIAS_MAX), ("bias_slider", "-99", C.BIAS_MIN), ("bias_slider", "abc", None),
    ("slope_slider", "3", C.SLOPE_MAX), ("slope_slider", "0.333", 0.33), ("sigma_e_slider", "40", C.SIGMA_E_MAX), ("sigma_a_slider", "1", C.SIGMA_A_MIN),
    ("n_select", "500", 500), ("n_select", "501", None), ("noise_select", "proportional", "proportional"), ("noise_select", "x", None),
    ("tol_slider", "0", C.TOL_MIN), ("tol_slider", "99", C.TOL_MAX), ("seed_input", "99999999", C.SEED_MAX), ("seed_input", "-5", 0),
])
def test_permalink_values_are_cleaned(key, raw, expected):
    assert P.clean_value(P.SETTING_SPECS[key], raw) == expected
