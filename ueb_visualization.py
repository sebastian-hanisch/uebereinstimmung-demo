"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import ueb_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="top", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _fmt(v: float, d: int = 3) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def build_scatter(a, max_points: int = 3000, height: int = 380):
    """Prognose gegen Ist: die 45°-Linie (perfekte Übereinstimmung) und die Regressionsgerade der Prognose auf das Ist. Mehr als `max_points` Punkte werden ausgedünnt (nur die Anzeige)."""
    f, x = np.asarray(a.f, dtype=float), np.asarray(a.a, dtype=float)
    step = max(1, len(f) // max_points)
    keep = np.arange(len(f)) % step == 0
    lo, hi = float(min(f.min(), x.min())), float(max(f.max(), x.max()))
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[keep].tolist(), y=f[keep].tolist(), mode="markers", name="Wochen", marker=dict(size=5, color=C.COLOR_POINT, opacity=0.4), hovertemplate="Ist %{x:.1f}, Prognose %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Prognose = Ist", line=dict(color=C.COLOR_IDENTITY, width=2, dash="dash"), hoverinfo="skip"))
    if float(np.std(f)) > 0:
        slope = float(np.polyfit(x, f, 1)[0])
        xs = np.array([float(x.min()), float(x.max())])
        fig.add_trace(go.Scatter(x=xs.tolist(), y=(float(f.mean()) + slope * (xs - float(x.mean()))).tolist(), mode="lines", name=f"Regressionsgerade (Steigung {_fmt(slope, 2)})", line=dict(color=C.COLOR_FIT, width=2), hoverinfo="skip"))
    fig.update_xaxes(title_text="Ist", range=[lo, hi])
    fig.update_yaxes(title_text="Prognose", range=[lo, hi])
    return _base(fig, height)


def build_ba(a, max_points: int = 3000, height: int = 340):
    """Bland–Altman-Diagramm: Differenz (Prognose − Ist) gegen das Mittel beider, mit mittlerer Abweichung und 95-%-Grenzen."""
    f, x = np.asarray(a.f, dtype=float), np.asarray(a.a, dtype=float)
    d, m = f - x, (f + x) / 2.0
    step = max(1, len(f) // max_points)
    keep = np.arange(len(f)) % step == 0
    ba = a.ba
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m[keep].tolist(), y=d[keep].tolist(), mode="markers", name="Wochen", marker=dict(size=5, color=C.COLOR_POINT, opacity=0.4), hovertemplate="Mittel %{x:.1f}, Differenz %{y:.1f}<extra></extra>"))
    xs = [float(m.min()), float(m.max())]
    fig.add_trace(go.Scatter(x=xs, y=[0.0, 0.0], mode="lines", name="keine Abweichung", line=dict(color=C.COLOR_IDENTITY, width=1, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[ba["mean"]] * 2, mode="lines", name=f"mittlere Abweichung {_fmt(ba['mean'], 2)}", line=dict(color=C.COLOR_FIT, width=2), hoverinfo="skip"))
    for key, label in (("lower", "untere Grenze"), ("upper", "obere Grenze")):
        fig.add_trace(go.Scatter(x=xs, y=[ba[key]] * 2, mode="lines", name=f"{label} {_fmt(ba[key], 2)}", line=dict(color=C.COLOR_LIMIT, width=2, dash="dash"), hoverinfo="skip"))
    fig.update_xaxes(title_text="Mittel aus Prognose und Ist")
    fig.update_yaxes(title_text="Differenz (Prognose − Ist)")
    return _base(fig, height)


def build_measures(a, height: int = 320):
    """Hauptansicht: Pearson r, R², Lins CCC, Anteil innerhalb der Toleranz und die CCC nach Neukalibrierung als Balken (gemessen) mit den erwarteten Werten als Rauten."""
    keys = ("r", "r2", "ccc", "ccc_cal", "within")
    names = {"r": "Pearson r", "r2": "R²", "ccc": "Lins CCC", "ccc_cal": "CCC nach Neukalibrierung", "within": f"innerhalb ±{a.settings.tol:g}"}
    rows = [a.row(k) for k in keys]
    vals = [None if np.isnan(r.value) else r.value for r in rows]
    ex = [None if np.isnan(r.exact) else r.exact for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[names[k] for k in keys], x=vals, orientation="h", name="gemessen", marker_color=[C.MEASURE_COLORS.get(k, C.COLOR_GREY) for k in keys], text=[_fmt(v) if v is not None else "–" for v in vals],
                         textposition="outside"))
    fig.add_trace(go.Scatter(y=[names[k] for k in keys], x=ex, mode="markers", name="erwartet (exakt)", marker=dict(size=12, symbol="diamond-open", color=C.COLOR_IDENTITY, line=dict(width=2)),
                             hovertemplate="%{y}: %{x:.3f}<extra>erwartet</extra>"))
    fig.update_xaxes(title_text="Wert (1 = perfekt)", range=[-0.1, 1.2])
    fig.update_yaxes(autorange="reversed", automargin=True)
    return _base(fig, height)


def build_e1(cells: list, height: int = 380):
    """E1: r, R² und CCC je Fehlerart (gruppierte Balken)."""
    names = [C.CASE_NAMES[c["case"]] for c in cells]
    fig = go.Figure()
    for key, label in (("r", "Pearson r"), ("r2", "R²"), ("ccc", "Lins CCC")):
        fig.add_trace(go.Bar(x=names, y=[c[key] for c in cells], name=label, marker_color=C.MEASURE_COLORS[key], text=[_fmt(c[key]) for c in cells], textposition="outside"))
    fig.update_yaxes(title_text="Wert (1 = perfekt)", range=[0, 1.15])
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-20)
    return _base(fig, height)


def build_e2(cells: list, height: int = 360):
    """E2: R² und CCC gegen die Streuung des Ist bei gleichem Fehler (RMSE konstant), für Bias 0 und Bias 4."""
    fig = go.Figure()
    for bias, dash in ((0, "solid"), (4, "dot")):
        mine = sorted((c for c in cells if c["bias"] == bias), key=lambda c: c["sigma_a"])
        xs = [c["sigma_a"] for c in mine]
        fig.add_trace(go.Scatter(x=xs, y=[c["r2"] for c in mine], mode="lines+markers", name=f"R² (Bias {bias})", line=dict(color=C.MEASURE_COLORS["r2"], dash=dash)))
        fig.add_trace(go.Scatter(x=xs, y=[c["ccc"] for c in mine], mode="lines+markers", name=f"CCC (Bias {bias})", line=dict(color=C.MEASURE_COLORS["ccc"], dash=dash)))
    fig.update_xaxes(title_text="Streuung des Ist (Standardabweichung)", type="log", tickvals=[c["sigma_a"] for c in cells if c["bias"] == 0], ticktext=[str(c["sigma_a"]) for c in cells if c["bias"] == 0])
    fig.update_yaxes(title_text="Wert (1 = perfekt)", range=[0, 1.05])
    return _base(fig, height)


def build_e3(cells: list, height: int = 380):
    """E3: CCC vorher und nachher der Neukalibrierung je Fehlerart, mit r als Raute (die Obergrenze, die Kalibrierung erreichen kann)."""
    names = [C.CASE_NAMES[c["case"]] for c in cells]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[c["ccc_before"] for c in cells], name="CCC vorher", marker_color=C.COLOR_GREY, text=[_fmt(c["ccc_before"]) for c in cells], textposition="outside"))
    fig.add_trace(go.Bar(x=names, y=[c["ccc_after"] for c in cells], name="CCC nach der Neukalibrierung", marker_color=C.MEASURE_COLORS["ccc"], text=[_fmt(c["ccc_after"]) for c in cells], textposition="outside"))
    fig.add_trace(go.Scatter(x=names, y=[c["r"] for c in cells], mode="markers", name="Pearson r", marker=dict(size=12, symbol="diamond-open", color=C.COLOR_IDENTITY, line=dict(width=2))))
    fig.update_yaxes(title_text="CCC (1 = perfekt)", range=[0, 1.15])
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-20)
    return _base(fig, height)


def build_e4(cell: dict, height: int = 320):
    """E4: Anteil der Läufe, in denen Prognose A besser abschneidet, je Maß."""
    names = {"r": "Pearson r", "ccc": "Lins CCC", "rmse": "RMSE", "mae": "MAE"}
    keys = list(names)
    vals = [100.0 * cell["a_wins"][k] for k in keys]
    fig = go.Figure(go.Bar(x=[names[k] for k in keys], y=vals, marker_color=[C.MEASURE_COLORS.get(k, C.COLOR_GREY) for k in keys], text=[f"{v:.0f} %" for v in vals], textposition="outside"))
    fig.update_yaxes(title_text="Läufe, in denen A besser ist (%)", range=[0, 115])
    return _base(fig, height)


def build_e5(cells: list, height: int = 340):
    """E5: Anteil außerhalb der Bland–Altman-Grenzen im unteren und oberen Drittel des Ist je Fehlerart; die Linie bei 5 % ist, was bei gleichbleibendem Fehler zu erwarten ist."""
    names = [C.E5_NAMES[c["case"]] for c in cells]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[100 * c["outside_low"] for c in cells], name="unteres Drittel des Ist", marker_color=C.COLOR_POINT, text=[f"{100 * c['outside_low']:.1f} %" for c in cells], textposition="outside"))
    fig.add_trace(go.Bar(x=names, y=[100 * c["outside_high"] for c in cells], name="oberes Drittel des Ist", marker_color=C.COLOR_FIT, text=[f"{100 * c['outside_high']:.1f} %" for c in cells], textposition="outside"))
    fig.add_hline(y=5, line=dict(color=C.COLOR_IDENTITY, dash="dot"), annotation_text="5 % erwartet", annotation_position="top left")
    fig.update_yaxes(title_text="außerhalb der Grenzen (%)", range=[0, 14])
    fig.update_layout(barmode="group")
    return _base(fig, height)
