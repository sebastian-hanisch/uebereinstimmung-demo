"""Übereinstimmung statt Korrelation: stimmt die Prognose auch? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Achtes Stück der Korrelations-Reihe. Eine Prognose gegen das Ist über n Wochen, mit Bias, Steigung und Rauschen als Regler. Gezeigt werden Pearson r und R² (sehen nur die Gerade), Lins CCC (bestraft
Bias und Steigung), RMSE, MAE und der Anteil innerhalb einer Toleranz, das Bland-Altman-Diagramm, die Neukalibrierung und die Rangumkehr zweier Prognosen, jeweils mit dem exakt erwarteten Wert daneben.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import ueb_constants as C
import ueb_evaluation as ev
from ueb_layout import dunkles_design, kopf_css
from ueb_pdf_export import build_pdf
from ueb_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from ueb_visualization import (
    build_ba,
    build_e1,
    build_e2,
    build_e3,
    build_e4,
    build_e5,
    build_measures,
    build_scatter,
)

st.set_page_config(page_title="Übereinstimmung statt Korrelation – Sebastian Hanisch", layout="wide")

de = ev.de
pct = ev.pct


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _grid():
    return ev.load_grid()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in ueb_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📏 Übereinstimmung statt Korrelation: stimmt die Prognose auch?")
st.markdown(
    """
**Achtes Stück der Korrelations-Reihe** und **die erste Demo des Portfolios über Übereinstimmung statt Zusammenhang**. Eine **Prognose** wird über **n Wochen** gegen das **Ist** gelegt. Pearson r und R²
sehen nur, wie gut eine **Gerade** passt, nicht, ob die Werte **gleich** sind: eine Prognose mit r = 0,99 kann um 30 % zu hoch liegen. Die Fragen: **Wann verrät r nichts über Bias und Skala? Was sagt Lins
Übereinstimmungskorrelation (CCC)? Was lässt sich durch Neukalibrieren beheben und was nicht? Und kehrt sich die Rangfolge zweier Prognosen je nach Maß um?**

Weiter unten: fünf Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Prognose-Verfahren und ihr Vergleich (MAE, RMSE, MASE) stehen in der Prognose-Linie
(`naive-forecast-demo`, `forecast-combination-demo`); hier geht es um das **Verhältnis von Zusammenhang und Übereinstimmung** zweier Reihen.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        r"""
| Begriff | Was er hier bedeutet |
|---|---|
| **Welt** | Das Ist streut um 100 mit der Standardabweichung σ_a; die Prognose ist 100 + **Bias** + **Steigung** · (Ist − 100) + **Rauschen**. Die Prognose „1,3 · Ist“ hat Bias 30 und Steigung 1,3. |
| **Pearson r, R²** | Wie eng die Punkte an einer Geraden liegen. Unverändert, wenn man die Prognose verschiebt oder streckt. |
| **Lins CCC** | Wie eng die Punkte an der **45°-Linie** liegen (Prognose = Ist): r, vermindert um Bias und Steigung. Gleich 1 nur bei völliger Übereinstimmung. |
| **RMSE, MAE, mittlere Abweichung** | Absolute Fehlermaße in den Einheiten des Ist; sie hängen nicht an der Streuung des Ist. |
| **Anteil innerhalb ±T** | Anteil der Wochen, in denen die Prognose höchstens T daneben liegt: die Toleranz ist eine Setzung. |
| **Bland–Altman** | Differenz (Prognose − Ist) gegen das Mittel beider; mittlere Abweichung und 95-%-Grenzen (± 1,96 Standardabweichungen). |
| **Neukalibrierung** | Lineare Anpassung des Ist an die Prognose; sie behebt Bias und Skala, aber kein Rauschen. |
| **Erwartet (exakt)** | Zu jedem Maß steht der Wert, den die Formel für diese Welt erwartet; der gemessene Wert schwankt um ihn. |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Prognose**")
    bias = st.slider("Bias (Prognose im Mittel zu hoch)", *bounds("bias_slider"), step=1, key="bias_slider", help="Mittlere Abweichung der Prognose vom Ist; negativ heißt zu niedrig.")
    slope = st.slider("Steigung", *bounds("slope_slider"), step=0.1, key="slope_slider", format="%.1f", help="1 = die Prognose folgt dem Ist eins zu eins; 0,5 = sie reagiert nur halb so stark.")
    sigma_e = st.slider("Rauschen σ_e", *bounds("sigma_e_slider"), step=1, key="sigma_e_slider", help="Streuung des Prognosefehlers, der nicht am Ist hängt.")
    noise = st.selectbox("Art des Rauschens", list(C.NOISES), format_func=C.NOISES.get, key="noise_select", help="Additiv: gleich groß. Proportional: wächst mit dem Ist (σ_e · Ist / 100).")
    st.markdown("**Ist und Stichprobe**")
    sigma_a = st.slider("Streuung des Ist", *bounds("sigma_a_slider"), step=1, key="sigma_a_slider", help="Standardabweichung des Ist (Mittel 100). Sie bestimmt, wie groß r bei gleichem Fehler ausfällt.")
    n = st.select_slider("Wochen n", options=C.N_OPTIONS, key="n_select", help="Zahl der Beobachtungen.")
    tol = st.slider("Toleranz ±T", *bounds("tol_slider"), step=1, key="tol_slider", help="Eine Woche zählt als getroffen, wenn die Prognose höchstens T vom Ist abweicht.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Daten ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"bias_slider": int(bias), "slope_slider": float(slope), "sigma_e_slider": int(sigma_e), "noise_select": noise, "sigma_a_slider": int(sigma_a), "n_select": int(n),
                   "tol_slider": int(tol), "seed_input": int(seed)})

settings = ev.Settings(bias=float(bias), slope=float(slope), sigma_e=float(sigma_e), noise=noise, sigma_a=float(sigma_a), n=int(n), tol=float(tol), seed=int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
cols = st.columns(2)
cols[0].metric("Pearson r", de(a.row("r").value), help=f"R² = {de(a.row('r2').value)}; erwartet r = {de(a.row('r').exact)}.")
cols[1].metric("Lins CCC", de(a.row("ccc").value), help=f"Erwartet {de(a.row('ccc').exact)}; CCC nach Neukalibrierung {de(a.row('ccc_cal').value)}.")
cols = st.columns(2)
cols[0].metric("RMSE", de(a.row("rmse").value), help=f"Erwartet {de(a.row('rmse').exact)}; MAE {de(a.row('mae').value)}; mittlere Abweichung {de(a.row('bias').value)}.")
cols[1].metric(f"Innerhalb ±{settings.tol:g}", pct(a.row("within").value), help=f"Erwartet {pct(a.row('within').exact)}.")
st.plotly_chart(build_measures(a), width="stretch", key="main_measures")
rows = [{"Maß": r.name, "gemessen": pct(r.value) if r.kind == "percent" else de(r.value, 3), "erwartet (exakt)": pct(r.exact) if r.kind == "percent" else de(r.exact, 3)} for r in a.rows]
st.dataframe(rows, hide_index=True, width="stretch")
c1, c2 = st.columns(2)
c1.plotly_chart(build_scatter(a), width="stretch", key="main_scatter")
c2.plotly_chart(build_ba(a), width="stretch", key="main_ba")
st.caption("Links: Prognose gegen Ist mit der 45°-Linie (Übereinstimmung) und der Regressionsgeraden. Rechts: Bland–Altman-Diagramm der Differenzen.")
for verdict in (ev.verdict_main, ev.verdict_ba, ev.verdict_tolerance):
    kind, text = verdict(a)
    getattr(st, kind)(text)
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = (settings, build_pdf(a))
pdf = st.session_state.get("pdf_bytes")
if pdf and pdf[0] == settings:
    st.download_button("⬇️ PDF herunterladen", pdf[1], file_name="uebereinstimmung-demo.pdf", mime="application/pdf", key="pdf_download")

st.markdown("## ❓ Kernfrage: Eine Prognose korreliert zu 0,99, stimmt sie auch?")
st.markdown("**Nicht unbedingt:** r sieht nur die Gerade, nicht Bias und Skala, und es misst den Fehler relativ zur Streuung des Ist. Übereinstimmung zeigen CCC, absolute Fehlermaße und der Anteil innerhalb einer "
            "Toleranz (Experimente unten).")

st.markdown("---")

grid = _grid()
if grid is None:
    st.warning("Das vorgerechnete Raster (data/agreement_grid.json) fehlt; es entsteht mit `python tools/make_grid.py`. Die Experimente zeigen es.")

# --- E1 Fehlerarten ----------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Sechs Fehlerarten, ein Blick auf r")
if grid is not None:
    c0 = grid["e1"][0]
    st.caption(f"Ist ~ N(100, {c0['sigma_a']:g}²), n = {c0['n']:,} Wochen, {c0['runs']} Läufe je Fehlerart; Mittelwerte.".replace(",", " "))
    st.plotly_chart(build_e1(grid["e1"]), width="stretch", key="e1_chart")
    table = [{"Fehlerart": C.CASE_NAMES[c["case"]], "r": de(c["r"], 3), "R²": de(c["r2"], 3), "CCC": de(c["ccc"], 3), "CCC erwartet": de(c["exact"]["ccc"], 3), "RMSE": de(c["rmse"]),
              "MAE": de(c["mae"]), "mittlere Abweichung": de(c["bias"]), "innerhalb ±5": pct(c["within5"]), "innerhalb ±10": pct(c["within10"])} for c in grid["e1"]]
    st.dataframe(table, hide_index=True, width="stretch")
    e1_case = st.selectbox("Fehlerart für das Verdikt", list(C.CASES), format_func=C.CASE_NAMES.get, index=3, key="e1_case")
    kind, text = ev.verdict_e1(grid["e1"], e1_case)
    getattr(st, kind)(text)
    st.markdown("**Wie sicher ist die Zahl?** Streuung von r und CCC über viele Läufe gegen die Zahl der Wochen:")
    e1s_case = st.selectbox("Fehlerart", ["skala", "bias", "perfekt"], format_func=C.CASE_NAMES.get, key="e1s_case")
    mine = sorted((c for c in grid["e1_spread"] if c["case"] == e1s_case), key=lambda c: c["n"])
    st.dataframe([{"Wochen n": c["n"], "r (Streuung)": f"{de(c['r'], 3)} ({de(c['r_sd'], 3)})", "CCC (Streuung)": f"{de(c['ccc'], 3)} ({de(c['ccc_sd'], 3)})", "CCC erwartet": de(c["exact"]["ccc"], 3),
                   "ICC(A,1) − CCC": de(c["icc"] - c["ccc"], 4)} for c in mine], hide_index=True, width="stretch")
    kind, text = ev.verdict_e1_spread(grid["e1_spread"], e1s_case)
    getattr(st, kind)(text)
    st.caption("r und R² sehen weder Bias noch Skala: die falsche Skala hat das beste r und den schlechtesten RMSE. Die CCC streut bei Bias oder falscher Skala deutlich stärker als r und ist bei kleinem n "
               "nach unten verzerrt. ICC(A,1) (absolute Übereinstimmung) liegt bis auf einen Abstand von der Ordnung 1/n bei der CCC.")

st.markdown("---")

# --- E2 Streuung des Ist -------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Gleicher Fehler, andere Streuung des Ist")
if grid is not None:
    c0 = grid["e2"][0]
    st.caption(f"Rauschen {C.E2_NOISE}, Bias 0 oder 4, Streuung des Ist {', '.join(str(s) for s in C.E2_SIGMAS)}; n = {c0['n']:,} Wochen, {c0['runs']} Läufe je Zelle; Toleranz ±{c0['tol']:g}.".replace(",", " "))
    st.plotly_chart(build_e2(grid["e2"]), width="stretch", key="e2_chart")
    st.dataframe([{"Streuung des Ist": c["sigma_a"], "Bias": c["bias"], "r": de(c["r"], 3), "R²": de(c["r2"], 3), "CCC": de(c["ccc"], 3), "RMSE": de(c["rmse"]), f"innerhalb ±{c['tol']:g}": pct(c["within"])}
                  for c in sorted(grid["e2"], key=lambda c: (c["bias"], c["sigma_a"]))], hide_index=True, width="stretch")
    e2_bias = st.selectbox("Bias für das Verdikt", list(C.E2_BIASES), key="e2_bias")
    kind, text = ev.verdict_e2(grid["e2"], e2_bias)
    getattr(st, kind)(text)
    st.caption("r, R² und CCC messen den Fehler **relativ** zur Streuung des Ist; RMSE und Toleranzanteil messen ihn **absolut**. Wer eine Prognose an R² misst, misst auch, wie bunt das Ist ist "
               "(vgl. die Bereichseinschränkung in `aggregationsebene-demo`).")

st.markdown("---")

# --- E3 Neukalibrierung --------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Was lässt sich kalibrieren?")
if grid is not None:
    c0 = grid["e3"][0]
    st.caption(f"Lineare Anpassung des Ist an die Prognose an der ersten Hälfte von n = {c0['n']} Wochen, Bewertung an der zweiten; {c0['runs']} Läufe je Fehlerart.")
    st.plotly_chart(build_e3(grid["e3"]), width="stretch", key="e3_chart")
    e3_case = st.selectbox("Fehlerart für das Verdikt", [c["case"] for c in grid["e3"]], format_func=C.CASE_NAMES.get, index=2, key="e3_case")
    kind, text = ev.verdict_e3(grid["e3"], e3_case)
    getattr(st, kind)(text)
    small = sorted(grid["e3_small"] + [c for c in grid["e3"] if c["case"] == "skala"], key=lambda c: c["n"])
    st.markdown("**Wie viele Wochen braucht die Kalibrierung?** Skala 1,3, Training an der ersten Hälfte:")
    st.dataframe([{"Wochen insgesamt": c["n"], "Trainingswochen": c["n"] // 2, "CCC vorher": de(c["ccc_before"], 3), "CCC nachher": de(c["ccc_after"], 3), "r": de(c["r"], 3)} for c in small], hide_index=True, width="stretch")
    st.caption("Bias und Skala lassen sich wegrechnen (CCC nähert sich r, der RMSE fällt auf das Rauschen); Rauschen nicht: die CCC bleibt bei „nur Rauschen“ fast gleich. Wer r = 0,99 hat, hat eine gut "
               "kalibrierbare Prognose, aber noch keine gute.")

st.markdown("---")

# --- E4 Rangumkehr -------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Zwei Prognosen, zwei Rangfolgen")
if grid is not None:
    cell = grid["e4"][0]
    st.caption(f"A: Skala 1,3 (Bias 30, Steigung 1,3) mit Rauschen 4; B: unverzerrt mit Rauschen 10. Dieselben Ist-Werte, n = {cell['n']} Wochen, {cell['runs']} Läufe.")
    st.plotly_chart(build_e4(cell), width="stretch", key="e4_chart")
    st.dataframe([{"Maß": name, "Prognose A": de(cell["a"][k], 3), "Prognose B": de(cell["b"][k], 3)} for k, name in (("r", "Pearson r"), ("ccc", "Lins CCC"), ("rmse", "RMSE"), ("mae", "MAE"))],
                 hide_index=True, width="stretch")
    kind, text = ev.verdict_e4(cell)
    getattr(st, kind)(text)
    st.caption("Wer Prognosen nach r wählt, nimmt hier die mit dem dreifachen Fehler: die Rangfolge hängt vollständig am Maß, nicht an den Daten.")

st.markdown("---")

# --- E5 Bland-Altman -------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Bland–Altman bei wachsendem Fehler")
if grid is not None:
    c0 = grid["e5"][0]
    st.caption(f"n = {c0['n']:,} Wochen, {c0['runs']} Läufe je Fehlerart; 95-%-Grenzen aus mittlerer Abweichung ± 1,96 Standardabweichungen.".replace(",", " "))
    st.plotly_chart(build_e5(grid["e5"]), width="stretch", key="e5_chart")
    e5_case = st.selectbox("Fehlerart für das Verdikt", [c["case"] for c in grid["e5"]], format_func=C.E5_NAMES.get, index=1, key="e5_case")
    kind, text = ev.verdict_e5(ev.find(grid["e5"], case=e5_case))
    getattr(st, kind)(text)
    st.caption("Ein Paar Grenzen passt nur bei gleichbleibendem Fehler: wächst der Fehler mit dem Ist, liegen im oberen Drittel doppelt so viele Wochen außerhalb. Eine Steigung ungleich 1 zeigt sich als Steigung der "
               "Differenzen gegen das Mittel. Ob ein Fehler tolerierbar ist, sagt das Diagramm nicht: das tut der Anteil innerhalb ±T.")

st.markdown("---")

# --- Grenzen ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    r"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Normalverteiltes Ist, lineare Fehlerstruktur** | die exakten Formeln gelten für diese Welt; echte Reihen haben Schiefe, Ausreißer und Stufen, die Richtung der Effekte bleibt | – |
| **Unabhängige Wochen** | Reihen mit Trend und Gedächtnis machen r und CCC trügerisch hoch (Scheinkorrelation) | `scheinkorrelation-demo` |
| **Die Toleranz ist eine Setzung** | ob ±5 genügt, hängt am Zweck der Prognose, nicht an den Daten | – |
| **Punktprognosen** | Intervall- und Wahrscheinlichkeitsprognosen und ihre Kalibrierung sind ein anderes Thema | `forecast-interval-demo` |
| **Zwei Reihen, eine Messung je Woche** | Mehrfachmessungen und Zufallseffekte-Modelle der Intraklassenkorrelation sind nicht gebaut; ICC(A,1) steht nur als Gegenprobe | – |
| **Lineare Neukalibrierung** | nichtlineare Fehler (Sättigung, Stufen) lassen sich so nicht beheben; Deming- und Passing–Bablok-Regression sind nicht gebaut | – |
| **Prognose-Verfahren** | wie man eine bessere Prognose baut und Verfahren vergleicht, steht in der Prognose-Linie | `naive-forecast-demo`, `forecast-combination-demo` |
"""
)
st.caption("Alle Daten sind simuliert; die Welt ist eine Modellwahl, kein Befund über echte Prognosen.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Das Ist $a$ ist normalverteilt mit Mittel $\mu = 100$ und Varianz $\sigma_a^2$; die Prognose ist

$$ f = \mu + b + s\,(a-\mu) + \varepsilon, \qquad \varepsilon \sim N(0,\sigma_e^2)\ \text{(additiv)} \ \text{oder}\ \varepsilon \sim N\big(0,\sigma_e^2 a^2/\mu^2\big)\ \text{(proportional)} $$

mit dem Bias $b$ und der Steigung $s$.

| Symbol | Bedeutung |
|---|---|
| $\sigma_f^2, \sigma_{fa}$ | Varianz der Prognose und Kovarianz von Prognose und Ist |
| $d = f - a$ | Differenz (Fehler) |
| $T$ | Toleranz |

**Pearson r** und **Lins Übereinstimmungskorrelation** (Varianzen durch $n$):

$$ r = \frac{\sigma_{fa}}{\sigma_f\,\sigma_a}, \qquad \rho_c = \frac{2\,\sigma_{fa}}{\sigma_f^2 + \sigma_a^2 + (\bar f - \bar a)^2} $$

**Erwartung** bei additivem Rauschen ($\sigma_f^2 = s^2\sigma_a^2 + \sigma_e^2$, $\sigma_{fa} = s\,\sigma_a^2$):

$$ \rho_c = \frac{2\,s\,\sigma_a^2}{s^2\sigma_a^2 + \sigma_e^2 + \sigma_a^2 + b^2} $$

**Fehlermaße**, mit $\operatorname{Var}(d) = (s-1)^2\sigma_a^2 + \sigma_e^2$:

$$ \mathrm{RMSE}^2 = b^2 + \operatorname{Var}(d), \qquad \Pr(|d|\le T) = \Phi\!\Big(\frac{T-b}{\sqrt{\operatorname{Var}(d)}}\Big) - \Phi\!\Big(\frac{-T-b}{\sqrt{\operatorname{Var}(d)}}\Big) $$

**Bland–Altman:** die Grenzen sind $\bar d \pm 1{,}96\, s_d$ mit der Standardabweichung $s_d$ der Differenzen (Teiler $n-1$); bei normalen Differenzen liegen 5 % außerhalb.

**Neukalibrierung** durch Regression des Ist auf die Prognose, $\hat a = \alpha + \beta f$ mit $\beta = \sigma_{fa}/\sigma_f^2$; in der Stichprobe gilt für die neukalibrierte Prognose

$$ \rho_c = \frac{2\,r^2}{1 + r^2} $$

**Quellen.** In dieser Fassung sind keine Literaturangaben ausgewiesen; die Formeln sind durch unabhängige Rechenwege (Tests gegen scipy, numpy, Schleifenfassungen und Simulationen) und die Messungen der Experimente gestützt.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
