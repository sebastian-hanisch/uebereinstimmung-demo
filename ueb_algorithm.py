"""Kern: eine Prognose f gegen das Ist a (Bias, Steigung, Rauschen), die Maße des Vergleichs (Pearson r, Lins CCC, RMSE, MAE, mittlere Abweichung, Anteil innerhalb einer Toleranz,
Bland–Altman mit Grenzen), die Neukalibrierung, ICC(A,1) als Gegenprobe und die exakten Formeln (Erwartung) für alle Maße.

Welt: a ~ N(100, σ_a²); f = 100 + Bias + Steigung · (a − 100) + Rauschen, das Rauschen additiv (Standardabweichung σ_e) oder proportional zum Ist (σ_e · a / 100).
Zufall und Pearson kommen aus den wörtlichen Kopien von pearson-demo (prs_*.py). Ohne scipy zur Laufzeit."""

import math

import numpy as np

import ueb_constants as C
from prs_algorithm import pearson_r


# --- Welt ------------------------------------------------------------------------------------------------------------------------------------

def make_world(n: int, sigma_a: float, bias: float, slope: float, sigma_e: float, proportional: bool, rng) -> tuple[np.ndarray, np.ndarray]:
    """n Wochen: Ist a und Prognose f. Erst alle Ist-Werte, dann alle Rauschwerte aus dem Strom."""
    a = C.MU + sigma_a * rng.normal((n,))
    z = rng.normal((n,))
    noise = sigma_e * (a / C.MU if proportional else 1.0) * z
    f = C.MU + bias + slope * (a - C.MU) + noise
    return f, a


# --- Maße ------------------------------------------------------------------------------------------------------------------------------------

def _pair(f, a) -> tuple[np.ndarray, np.ndarray]:
    f, a = np.asarray(f, dtype=float), np.asarray(a, dtype=float)
    if len(f) != len(a):
        raise ValueError("Prognose und Ist müssen gleich lang sein")
    if len(f) < 2:
        raise ValueError("mindestens zwei Beobachtungen nötig")
    return f, a


def corr(f, a) -> float:
    """Pearson r; NaN, wenn die Prognose (oder das Ist) konstant ist (dann ist r nicht definiert)."""
    f, a = _pair(f, a)
    try:
        return pearson_r(f, a)
    except ValueError:
        return float("nan")


def ccc(f, a) -> float:
    """Lins Übereinstimmungskorrelationskoeffizient: 2 Cov(f, a) / (Var f + Var a + (f̄ − ā)²), Varianzen und Kovarianz durch n."""
    f, a = _pair(f, a)
    mf, ma = float(f.mean()), float(a.mean())
    cov = float(((f - mf) * (a - ma)).mean())
    denom = float(f.var()) + float(a.var()) + (mf - ma) ** 2
    if denom == 0.0:
        return float("nan")
    return 2.0 * cov / denom


def mean_error(f, a) -> float:
    f, a = _pair(f, a)
    return float((f - a).mean())


def rmse(f, a) -> float:
    f, a = _pair(f, a)
    return float(np.sqrt(((f - a) ** 2).mean()))


def mae(f, a) -> float:
    f, a = _pair(f, a)
    return float(np.abs(f - a).mean())


def within_tolerance(f, a, tol: float) -> float:
    """Anteil der Wochen mit |f − a| ≤ tol (inklusive Grenze)."""
    f, a = _pair(f, a)
    return float((np.abs(f - a) <= tol).mean())


def bland_altman(f, a, z: float = C.Z_LIMIT) -> dict:
    """Differenzen d = f − a: Mittel, Standardabweichung (n − 1), Grenzen Mittel ± z · Standardabweichung, Steigung von d gegen das Mittel (f + a)/2, Anteil außerhalb der Grenzen
    insgesamt und im unteren und oberen Drittel des Ist."""
    f, a = _pair(f, a)
    if len(f) < 3:
        raise ValueError("mindestens drei Beobachtungen nötig")
    d = f - a
    m = (f + a) / 2.0
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    lower, upper = mean - z * sd, mean + z * sd
    out = (d < lower) | (d > upper)
    order = np.argsort(a, kind="stable")
    third = len(a) // 3
    mc = m - m.mean()
    slope = float((mc * (d - mean)).sum() / (mc * mc).sum())
    return {"mean": mean, "sd": sd, "lower": lower, "upper": upper, "slope": slope, "outside": float(out.mean()),
            "outside_low": float(out[order[:third]].mean()), "outside_high": float(out[order[-third:]].mean())}


def recalibrate(f, a) -> tuple[float, float]:
    """Lineare Neukalibrierung: Regression des Ist auf die Prognose, a ≈ intercept + slope · f. Gibt (slope, intercept) zurück."""
    f, a = _pair(f, a)
    fc = f - f.mean()
    ss = float((fc * fc).sum())
    if ss == 0.0:
        raise ValueError("die Prognose ist konstant: keine Neukalibrierung möglich")
    slope = float((fc * (a - a.mean())).sum() / ss)
    return slope, float(a.mean() - slope * f.mean())


def apply_calibration(f, slope: float, intercept: float) -> np.ndarray:
    return intercept + slope * np.asarray(f, dtype=float)


def icc_a1(f, a) -> float:
    """Intraklassenkorrelation für absolute Übereinstimmung bei zwei Messreihen (zweifaktoriell, einzelne Messung; ANOVA-Quadratsummen): die Gegenprobe zur CCC."""
    f, a = _pair(f, a)
    n = len(f)
    data = np.column_stack([f, a])
    gm = float(data.mean())
    msr = 2.0 * float(((data.mean(axis=1) - gm) ** 2).sum()) / (n - 1)
    msc = n * float(((data.mean(axis=0) - gm) ** 2).sum())
    resid = data - data.mean(axis=1, keepdims=True) - data.mean(axis=0, keepdims=True) + gm
    mse = float((resid**2).sum()) / (n - 1)
    return (msr - mse) / (msr + mse + 2.0 * (msc - mse) / n)


# --- Exakte Formeln (Erwartung) -----------------------------------------------------------------------------------------------------------------

def moments(sigma_a: float, bias: float, slope: float, sigma_e: float, proportional: bool) -> dict:
    """Varianzen und Kovarianz der Welt: Var a, Var f, Cov(f, a), mittlere Differenz und Varianz der Differenz."""
    var_a = sigma_a**2
    var_noise = sigma_e**2 * (1.0 + var_a / C.MU**2 if proportional else 1.0)
    return {"var_a": var_a, "var_f": slope**2 * var_a + var_noise, "cov": slope * var_a, "mean_d": float(bias), "var_d": (slope - 1.0) ** 2 * var_a + var_noise}


def exact_r(sigma_a: float, bias: float, slope: float, sigma_e: float, proportional: bool) -> float:
    m = moments(sigma_a, bias, slope, sigma_e, proportional)
    return float("nan") if m["var_f"] == 0.0 else m["cov"] / math.sqrt(m["var_f"] * m["var_a"])


def exact_ccc(sigma_a: float, bias: float, slope: float, sigma_e: float, proportional: bool) -> float:
    m = moments(sigma_a, bias, slope, sigma_e, proportional)
    return 2.0 * m["cov"] / (m["var_f"] + m["var_a"] + m["mean_d"] ** 2)


def exact_rmse(sigma_a: float, bias: float, slope: float, sigma_e: float, proportional: bool) -> float:
    m = moments(sigma_a, bias, slope, sigma_e, proportional)
    return math.sqrt(m["mean_d"] ** 2 + m["var_d"])


def exact_ccc_after_recalibration(r: float) -> float:
    """Nach der Regression des Ist auf die Prognose (in der Stichprobe): CCC = 2 r² / (1 + r²)."""
    return 2.0 * r * r / (1.0 + r * r)


_NODES, _WEIGHTS = np.polynomial.hermite.hermgauss(200)


def _conditional(sigma_a: float, bias: float, slope: float, sigma_e: float):
    """Stützstellen für den Ist-Wert (Gauß–Hermite) mit Gewicht, bedingtem Mittel und bedingter Standardabweichung der Differenz bei proportionalem Rauschen."""
    a = C.MU + sigma_a * math.sqrt(2.0) * _NODES
    return _WEIGHTS / math.sqrt(math.pi), bias + (slope - 1.0) * (a - C.MU), sigma_e * np.abs(a) / C.MU


def _phi(x):
    return 0.5 * np.vectorize(math.erfc)(-np.asarray(x, dtype=float) / math.sqrt(2.0))


def exact_within(sigma_a: float, bias: float, slope: float, sigma_e: float, proportional: bool, tol: float) -> float:
    """Erwarteter Anteil mit |f − a| ≤ tol. Additiv: d ist normal mit Mittel Bias und Varianz Var d. Proportional: gemischte Normalverteilung, Integration über das Ist."""
    m = moments(sigma_a, bias, slope, sigma_e, proportional)
    if not proportional or sigma_e == 0.0:
        sd = math.sqrt(m["var_d"])
        if sd == 0.0:
            return 1.0 if abs(bias) <= tol else 0.0
        return float(_phi((tol - bias) / sd) - _phi((-tol - bias) / sd))
    w, mu, s = _conditional(sigma_a, bias, slope, sigma_e)
    s = np.maximum(s, 1e-12)
    return float((w * (_phi((tol - mu) / s) - _phi((-tol - mu) / s))).sum())


def exact_mae(sigma_a: float, bias: float, slope: float, sigma_e: float, proportional: bool) -> float:
    """Erwartete mittlere absolute Abweichung (gefaltete Normalverteilung; proportional: Integration über das Ist)."""
    m = moments(sigma_a, bias, slope, sigma_e, proportional)

    def folded(mu, sd):
        sd = np.maximum(sd, 1e-12)
        return sd * math.sqrt(2.0 / math.pi) * np.exp(-0.5 * (mu / sd) ** 2) + mu * (1.0 - 2.0 * _phi(-mu / sd))

    if not proportional or sigma_e == 0.0:
        return float(folded(np.array(bias, dtype=float), np.array(math.sqrt(m["var_d"]))))
    w, mu, s = _conditional(sigma_a, bias, slope, sigma_e)
    return float((w * folded(mu, s)).sum())
