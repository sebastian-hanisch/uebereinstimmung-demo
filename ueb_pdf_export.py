"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Tabelle der Maße mit gemessenem und erwartetem Wert, Verdikte der Hauptansicht. Kein Gedankenstrich und kein Euro-Zeichen (Kernschriften)."""

import math

from fpdf import FPDF

import ueb_constants as C
import ueb_evaluation as ev


def _text(value: str) -> str:
    """Zeichen, die die Kernschriften nicht kennen, ersetzen (Gedankenstrich, Pfeile, Fettdruck-Markierung, griechische und mathematische Zeichen)."""
    table = {"–": "-", "—": "-", "→": "->", "**": "", "€": "EUR", "·": "*", "≤": "<=", "≥": ">=", "−": "-", "×": "x", "„": '"', "“": '"', "σ": "sigma", "²": "^2", "±": "+/-", "…": "..."}
    for old, new in table.items():
        value = value.replace(old, new)
    return value.encode("latin-1", "replace").decode("latin-1")


def _para(pdf: FPDF, height: float, text: str) -> None:
    """Absatz über die volle Breite; danach steht der Cursor wieder am linken Rand."""
    pdf.multi_cell(0, height, _text(text), new_x="LMARGIN", new_y="NEXT")


def _num(v: float, kind: str) -> str:
    if v is None or math.isnan(v):
        return "-"
    return f"{100 * v:.1f} %" if kind == "percent" else f"{v:.3f}"


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _text("Übereinstimmung statt Korrelation: stimmt die Prognose auch?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5.5, f"Bias: {s.bias:g}   Steigung: {s.slope:g}   Rauschen sigma_e: {s.sigma_e:g} ({C.NOISES[s.noise]})   Streuung des Ist: {s.sigma_a:g}   Wochen n: {s.n}   Toleranz: +/-{s.tol:g}   Zufalls-Seed: {s.seed}")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Maße", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for r in a.rows:
        _para(pdf, 5.5, f"{r.name}: {_num(r.value, r.kind)} (erwartet {_num(r.exact, r.kind)})")
    for title, verdict in (("Ergebnis", ev.verdict_main), ("Bland-Altman", ev.verdict_ba), ("Toleranz", ev.verdict_tolerance)):
        v = verdict(a)
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, _text(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        _para(pdf, 5.5, v[1])
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    _para(pdf, 4.5, "Simulierte Daten mit bekannter Welt. Sebastian Hanisch - Operations Research und Machine Learning.")
    return bytes(pdf.output())
