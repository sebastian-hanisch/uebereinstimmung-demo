# Übereinstimmung statt Korrelation: stimmt die Prognose auch? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-uebereinstimmung-demo.streamlit.app/)** -->

Achtes Stück der Korrelations-Reihe in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Nachfolger von [pearson-demo](https://github.com/sebastian-hanisch/pearson-demo) (Kante `fix`: Pearson misst, wie gut eine Gerade passt, nicht ob die Werte gleich sind), neben [rangkorrelation-demo](https://github.com/sebastian-hanisch/rangkorrelation-demo), [abhaengigkeitsmasse-demo](https://github.com/sebastian-hanisch/abhaengigkeitsmasse-demo), [scheinkorrelation-demo](https://github.com/sebastian-hanisch/scheinkorrelation-demo), [korrelationsmatrix-demo](https://github.com/sebastian-hanisch/korrelationsmatrix-demo), [randabhaengigkeit-demo](https://github.com/sebastian-hanisch/randabhaengigkeit-demo), [korrelierte-merkmale-demo](https://github.com/sebastian-hanisch/korrelierte-merkmale-demo), [aggregationsebene-demo](https://github.com/sebastian-hanisch/aggregationsebene-demo) und [korrelationsnetz-demo](https://github.com/sebastian-hanisch/korrelationsnetz-demo). Es ist die **erste Demo des Portfolios über Übereinstimmung statt Zusammenhang**: eine **Prognose** gegen das **Ist** über **n Wochen**, mit Bias, Steigung und Rauschen als Regler. Die Fragen: **Wann verrät r nichts über Bias und Skala? Was sagt Lins Übereinstimmungskorrelation (CCC)? Was lässt sich durch Neukalibrieren beheben und was nicht? Kehrt sich die Rangfolge zweier Prognosen je nach Maß um? Und was hat die Streuung des Ist mit r zu tun?**

## Warum dieses Problem

Eine Prognose mit r = 0,99 kann um 30 % zu hoch liegen. Befunde (Ist ~ N(100, 20²)): die Prognose **1,3 · Ist** hat das **beste r** der Tabelle (**0,988**, R² 0,977) und den **schlechtesten RMSE** (**30,85**), Lins CCC ist **0,522**, und praktisch keine Woche liegt innerhalb ±5 (0,03 %; innerhalb ±10 0,3 %); ein konstanter Bias von +10 lässt r bei 0,981 und senkt die CCC auf 0,873; die lineare Neukalibrierung macht aus RMSE 30,86 einen RMSE von 3,06, nur Rauschen lässt sich nicht herauskalibrieren (CCC 0,661 → 0,658); **dieselbe Prognose** (RMSE 4,00, innerhalb ±5 78,8 %) hat ein R² von 0,611 oder 0,990, je nachdem, wie stark das Ist streut; und von zwei Prognosen wählt r in **100 %** der Läufe die mit gut dreifachem RMSE (beim MAE 3,8-fach), während CCC, RMSE und MAE in 100 % die andere wählen.

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (`korrelation-planung/STUECK-8-uebereinstimmung.md`, nicht in diesem Repo) stützte sich auf Wegwerfskripte mit gleichem Erzeuger; die gebauten Zahlen stimmen mit der Vormessung überein, mit diesen Ausnahmen:

- **Die Zahlen des ursprünglichen PLAN** (Faktor 1,3: CCC 0,384 und RMSE 30,5; Bias +10: CCC 0,803) hingen an einer anderen Streuung des Ist (rund 15 statt 20). Gebaut sind die Zahlen für 20 (0,522 / 30,85 und 0,873); der Befund bleibt, die Zahlen hängen an der Streuung des Ist, die deshalb ein Regler ist (Experiment 2).
- **Neu:** die Neukalibrierung (Experiment 3, mit der Identität CCC = 2r² / (1 + r²) in der Stichprobe), die Rangumkehr (Experiment 4), das Bland–Altman-Diagramm bei wachsendem Fehler (Experiment 5), der Anteil innerhalb einer Toleranz und die Streuung der Maße gegen die Zahl der Wochen.
- **Als Tab in Stück 1 oder 2** (Vorschlag des PLAN) wurde es nicht gebaut: beide sind fertige Demos mit eigener Welt.
- **Verdikte vergleichen mit Toleranz:** die Schwellen „Abstand r − CCC > 0,05“ und „Anteil außerhalb im oberen minus unteren Drittel ≥ 0,04“ rechnen mit 10⁻⁹ Spielraum, weil 0,90 − 0,85 in Gleitkommazahlen knapp über 0,05 liegt.

## Modell

Das Ist a ist normalverteilt mit Mittel 100 und Streuung σ_a. Die Prognose ist f = 100 + Bias + Steigung · (a − 100) + Rauschen, das Rauschen additiv (Standardabweichung σ_e) oder **proportional** zum Ist (σ_e · a / 100). Die Prognose „1,3 · Ist“ hat daher Bias 30 und Steigung 1,3. Erzeuger `ueb_algorithm.py`, Zufall aus SplitMix64 in wörtlichen Kopien von `pearson-demo` (`prs_*.py`).

## Methodik

- **Maße der Hauptansicht** (`ueb_evaluation.py`): Pearson r und R², **Lins CCC** = 2 Cov(f, a) / (Var f + Var a + (f̄ − ā)²) (Varianzen durch n), mittlere Abweichung, RMSE, MAE, Anteil innerhalb ±T (inklusive Grenze), die CCC nach Neukalibrierung; dazu das **Bland–Altman-Diagramm** mit mittlerer Abweichung, 95-%-Grenzen (± 1,96 Standardabweichungen mit Teiler n − 1), Steigung der Differenzen gegen das Mittel und Anteil außerhalb insgesamt sowie im unteren und oberen Drittel des Ist. **Neukalibrierung:** Regression des Ist auf die Prognose. **ICC(A,1)** (zweifaktoriell, absolute Übereinstimmung) nur als Gegenprobe.
- **Exakte Formeln** (Erwartung als Verhältnis der Erwartungswerte): Cov = Steigung · σ_a², Var f = Steigung² σ_a² + Rauschvarianz, CCC = 2 Cov / (Var f + σ_a² + Bias²), RMSE² = Bias² + (Steigung − 1)² σ_a² + Rauschvarianz; der Anteil innerhalb ±T und das MAE exakt aus der Normalverteilung der Differenzen, bei proportionalem Rauschen durch Gauß–Hermite-Integration über das Ist.
- **Experimente:** (1) sechs Fehlerarten (n = 2 000) und die Streuung der Maße gegen n, (2) gleicher Fehler bei verschiedener Streuung des Ist, (3) Neukalibrierung an der ersten und Bewertung an der zweiten Hälfte, (4) zwei Prognosen gegen dieselben Ist-Werte, (5) Bland–Altman bei additivem, proportionalem und steigungsbedingtem Fehler.
- **Vorgerechnetes Raster** (`tools/make_grid.py`, `data/agreement_grid.json`, 1 s auf 10 Prozessen bei der Erzeugung, rechnerabhängig): E1 6 Fehlerarten · 400 Läufe, Streuung 3 Fehlerarten · 4 Wochenzahlen · 1000 Läufe, E2 4 Streuungen · 2 Bias · 400 Läufe, E3 5 Fehlerarten (und 2 Wochenzahlen) · 800 Läufe, E4 1000 Läufe, E5 3 Fehlerarten · 400 Läufe; feste Seeds.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (Raster-Werte gegen die JSON-Datei, die Tabellen Zeile für Zeile in der Rundung der App, die Hilfetexte der Presets über dieselben Funktionen wie in der App). Mittelwerte über alle Läufe einer Zelle; in Klammern der **exakt erwartete** Wert.

**Sechs Fehlerarten** (Experiment 1; Ist ~ N(100, 20²), n = 2 000, Rauschen 4 außer „nur Rauschen“; r (R²), CCC, RMSE, MAE, mittlere Abweichung):

| Fehlerart | r (R²) | CCC (erwartet) | RMSE (erwartet) | MAE | mittlere Abweichung |
|---|---|---|---|---|---|
| perfekt (Rauschen 4) | 0,981 (0,961) | 0,980 (0,980) | 4,00 (4,00) | 3,19 | 0,00 |
| Bias +10 | 0,981 (0,961) | 0,873 (0,873) | 10,77 (10,77) | 10,02 | 10,00 |
| Steigung 0,5 | 0,928 (0,862) | 0,775 (0,775) | 10,75 (10,77) | 8,58 | 0,01 |
| Skala: f = 1,3 · a | 0,988 (0,977) | 0,522 (0,522) | 30,85 (30,85) | 29,99 | 29,99 |
| Bias +10 und Steigung 0,5 | 0,928 (0,862) | 0,649 (0,649) | 14,69 (14,70) | 12,05 | 10,01 |
| nur Rauschen 20 | 0,707 (0,500) | 0,666 (0,667) | 20,01 (20,00) | 15,97 | 0,00 |

**Anteil innerhalb der Toleranz** (gleiche Zellen; ±5 und ±10):

| Fehlerart | innerhalb ±5 (erwartet) | innerhalb ±10 (erwartet) |
|---|---|---|
| perfekt (Rauschen 4) | 78,9 % (78,9 %) | 98,8 % (98,8 %) |
| Bias +10 | 10,5 % (10,6 %) | 50,0 % (50,0 %) |
| Steigung 0,5 | 35,8 % (35,8 %) | 64,7 % (64,7 %) |
| Skala: f = 1,3 · a | 0,0 % (0,0 %) | 0,3 % (0,3 %) |
| Bias +10 und Steigung 0,5 | 23,9 % (23,9 %) | 46,8 % (46,8 %) |
| nur Rauschen 20 | 19,7 % (19,7 %) | 38,2 % (38,3 %) |

**Wie sicher ist die Zahl?** (Streuung von r und CCC über 4 · 250 Läufe je Zelle; Zellen: Mittel (Streuung)):

| Fall | n = 20 | 30 | 100 | 1 000 |
|---|---|---|---|---|
| Skala 1,3: r (Streuung) | 0,988 (0,005) | 0,988 (0,005) | 0,988 (0,002) | 0,988 (0,001) |
| Skala 1,3: CCC (Streuung) | 0,500 (0,079) | 0,507 (0,067) | 0,518 (0,038) | 0,522 (0,012) |
| Skala 1,3: ICC(A,1) − CCC | 0,0125 | 0,0083 | 0,0025 | 0,0002 |
| Bias +10: r (Streuung) | 0,980 (0,009) | 0,980 (0,008) | 0,980 (0,004) | 0,981 (0,001) |
| Bias +10: CCC (Streuung) | 0,859 (0,045) | 0,863 (0,036) | 0,870 (0,019) | 0,873 (0,006) |
| Bias +10: ICC(A,1) − CCC | 0,0060 | 0,0039 | 0,0011 | 0,0001 |

**Gleicher Fehler, andere Streuung des Ist** (Experiment 2; Rauschen 4, Bias 0 oder +4, n = 1 000; CCC, RMSE und Anteil innerhalb ±5 jeweils für Bias 0 / +4):

| Streuung des Ist | r (R²) bei Bias 0 | CCC (Bias 0) | CCC (Bias +4) | RMSE (0 / +4) | innerhalb ±5 (0 / +4) |
|---|---|---|---|---|---|
| 5 | 0,781 (0,611) | 0,757 | 0,610 | 4,00 / 5,66 | 78,8 % / 58,5 % |
| 10 | 0,929 (0,862) | 0,926 | 0,862 | 4,00 / 5,66 | 78,8 % / 58,5 % |
| 20 | 0,981 (0,962) | 0,980 | 0,961 | 4,00 / 5,66 | 78,8 % / 58,5 % |
| 40 | 0,995 (0,990) | 0,995 | 0,990 | 4,00 / 5,66 | 78,8 % / 58,5 % |

**Neukalibrierung** (Experiment 3; n = 400, angepasst an der ersten, bewertet an der zweiten Hälfte):

| Fehlerart | r | CCC vorher → nachher | RMSE vorher → nachher |
|---|---|---|---|
| Bias +10 | 0,980 | 0,871 → 0,980 | 10,77 → 3,95 |
| Steigung 0,5 | 0,927 | 0,773 → 0,923 | 10,74 → 7,47 |
| Skala: f = 1,3 · a | 0,988 | 0,518 → 0,988 | 30,86 → 3,06 |
| Bias +10 und Steigung 0,5 | 0,927 | 0,646 → 0,923 | 14,67 → 7,47 |
| nur Rauschen 20 | 0,703 | 0,661 → 0,658 | 20,02 → 14,20 |

**Wie viele Wochen braucht die Kalibrierung?** (Skala 1,3, Training an der ersten Hälfte):

| Wochen insgesamt | Trainingswochen | CCC vorher → nachher | r |
|---|---|---|---|
| 30 | 15 | 0,488 → 0,983 | 0,987 |
| 60 | 30 | 0,508 → 0,986 | 0,988 |
| 400 | 200 | 0,518 → 0,988 | 0,988 |

**Zwei Prognosen** (Experiment 4; A: Skala 1,3 mit Rauschen 4, B: unverzerrt mit Rauschen 10, dieselben Ist-Werte, n = 200, 1000 Läufe; Zellen: A, B, Anteil der Läufe, in denen A besser ist):

| Maß | A | B | A besser in |
|---|---|---|---|
| Pearson r | 0,988 | 0,894 | 100 % |
| Lins CCC | 0,520 | 0,888 | 0 % |
| RMSE | 30,86 | 9,99 | 0 % |
| MAE | 30,01 | 7,98 | 0 % |

**Bland–Altman bei wachsendem Fehler** (Experiment 5; n = 2 000; Grenzen aus mittlerer Abweichung ± 1,96 Standardabweichungen; Zellen: Anteil außerhalb insgesamt, im unteren / oberen Drittel des Ist, Standardabweichung der Differenzen unten / oben, Steigung der Differenzen gegen das Mittel):

| Fehlerart | außerhalb | unteres / oberes Drittel | sd(d) unten / oben | Steigung |
|---|---|---|---|---|
| additiv, Rauschen 8 | 5,0 % | 5,0 % / 5,0 % | 8,00 / 7,99 | 0,078 |
| proportional, 8 % des Ist | 5,4 % | 1,3 % / 10,1 % | 6,31 / 9,76 | 0,081 |
| Steigung 1,3, additiv 4 | 5,0 % | 7,4 % / 7,4 % | 5,11 / 5,11 | 0,274 |

Befunde in Worten:

- **r und R² sehen weder Bias noch Skala.** Die falsche Skala hat das beste r (0,988) und den schlechtesten RMSE (30,85); der Bias +10 lässt r bei 0,981 und senkt die CCC von 0,980 (perfekt) auf 0,873. Nur Rauschen wirkt auf r und CCC fast gleich (0,707 gegen 0,666): dort ist r nicht blind. Alle Mittelwerte sind exakt vorhersagbar.
- **Bias und Skala lassen sich wegrechnen, Rauschen nicht.** Nach der Neukalibrierung steigt die CCC der falschen Skala von 0,518 auf 0,988 (r 0,988) und der RMSE fällt von 30,86 auf 3,06; bei „nur Rauschen“ bleibt die CCC bei 0,661 → 0,658 (der RMSE fällt trotzdem von 20,02 auf 14,20, weil die Regression zur Mitte schrumpft). Schon mit 15 Trainingswochen erreicht die falsche Skala im Mittel eine CCC von 0,983 (r 0,987; schlechtester von 1 500 Läufen 0,90), solange die Beziehung gleich bleibt.
- **Die CCC streut bei Bias oder falscher Skala deutlich stärker als r** (falsche Skala bei n = 20: 0,079 gegen 0,005, bei n = 1 000 0,012; Bias +10 bei n = 30: 0,036 gegen 0,008) und ist bei kleinem n nach unten verzerrt (Mittel 0,500 bei n = 20). **ICC(A,1)** liegt bis auf 0,0125 (n = 20), 0,0025 (n = 100) und 0,0002 (n = 1 000) bei der CCC: für zwei Messreihen ist es dasselbe Maß bis auf einen Abstand der Größenordnung 1/n.
- **Dieselbe Prognose, ein ganz anderes r:** bei gleichem RMSE (4,00) und gleichem Anteil innerhalb ±5 (78,8 %) ist R² 0,611 bei einer Streuung des Ist von 5 und 0,990 bei 40. **r und CCC messen den Fehler relativ zur Streuung des Ist; RMSE und Toleranzanteil messen ihn absolut, wenn der Fehler nicht von der Streuung des Ist abhängt** (Steigung 1, additives Rauschen); bei einem Steigungsfehler hängen auch sie daran (Skala 1,3: RMSE 30,30 / 30,85 / 32,56 bei einer Streuung des Ist von 5 / 20 / 40). Mit Bias +4 sind es RMSE 5,66 und 58,5 %.
- **Die Rangfolge hängt am Maß:** Prognose A (r 0,988, CCC 0,520, RMSE 30,86) gegen B (r 0,894, CCC 0,888, RMSE 9,99): r wählt A in 100 %, CCC, RMSE und MAE wählen B in 100 % der 1000 Läufe.
- **Ein Paar Bland–Altman-Grenzen passt nur bei gleichbleibendem Fehler:** bei proportionalem Rauschen liegen insgesamt 5,4 % außerhalb, im unteren Drittel 1,3 %, im oberen 10,1 %; eine Steigung ungleich 1 zeigt sich als Steigung der Differenzen gegen das Mittel (0,274, Ausreißer in beiden Dritteln, unten 7,4 %). Eine Steigung der Differenzen entsteht auch bei Steigung 1 allein durch großes additives Rauschen (Regression zur Mitte: 0,077 bei Rauschen 8, 0,40 bei Rauschen 20). Ob ein Fehler tolerierbar ist, sagt das Diagramm nicht: das tut der Anteil innerhalb ±T.

## Ehrliche Grenzen

- **Normalverteiltes Ist und lineare Fehlerstruktur.** Die exakten Formeln gelten für diese Welt; echte Reihen haben Schiefe, Ausreißer und Stufen; ob die Richtung der Effekte bleibt, zeigt die Demo nicht.
- **Unabhängige Wochen.** Reihen mit Trend und Gedächtnis machen r und CCC trügerisch hoch (Scheinkorrelation, `scheinkorrelation-demo`).
- **Die Toleranz ist eine Setzung.** Ob ±5 genügt, hängt am Zweck der Prognose, nicht an den Daten.
- **Punktprognosen.** Intervall- und Wahrscheinlichkeitsprognosen und ihre Kalibrierung sind ein anderes Thema (`forecast-interval-demo`).
- **Zwei Reihen, eine Messung je Woche.** Mehrfachmessungen und Zufallseffekte-Modelle der Intraklassenkorrelation fehlen; ICC(A,1) steht nur als Gegenprobe.
- **Lineare Neukalibrierung.** Nichtlineare Fehler (Sättigung, Stufen) lassen sich so nicht beheben; Deming- und Passing–Bablok-Regression sind nicht gebaut.
- **Kein Konfidenzintervall der CCC.** Die Streuung kommt aus dem Raster (Experiment 1), nicht aus einer Formel.
- **Quellen:** keine Literaturangaben; die Formeln sind durch unabhängige Rechenwege (unten) und die Messungen gestützt.

## Verifikation

- **Orakel** (`tests/test_oracle_uebereinstimmung.py`): Pearson gegen **`numpy.corrcoef`** und **`scipy.stats.pearsonr`**; die CCC gegen die **Schleifenfassung** der Definition und gegen die Zerlegung r · C_b nach Lin; RMSE, MAE und Toleranzanteil gegen Schleifen; Bland–Altman gegen eine Schleifenfassung und der Anteil außerhalb gegen **`scipy.stats.norm`** (5 %); die Neukalibrierung gegen **`numpy.polyfit`**; ICC(A,1) gegen eine Schleifenfassung der ANOVA; die exakten Formeln (r, CCC, RMSE, MAE, Anteil innerhalb) gegen **Simulationen mit zwei Millionen Paaren**, der Anteil innerhalb und das MAE gegen **`scipy.stats`** (gefaltete Normalverteilung) und bei proportionalem Rauschen gegen **numerische Integration** (`scipy.integrate.quad`).
- **Kern** (`tests/test_algorithm.py`, `tests/test_edge_cases.py`): Handrechnungen (Mini-Instanzen in `conftest.py`: vier Punkte mit r = CCC = 0,6, f = a + 10 mit CCC 0,8421, f = 1,3 · a mit CCC 0,4287, Bland–Altman mit ±2,2632), **inklusive Grenzen** (Toleranz ≤ T, die Grenzen des Bland–Altman-Diagramms strikt, drei Wochen für Bland–Altman), konstante Prognose (r nicht definiert), kleinste Welt an allen Reglerecken.
- **Eigenschaften** (`tests/test_eigenschaften_uebereinstimmung.py`): r invariant gegen Verschieben und Strecken der Prognose, CCC und RMSE nicht; Vertauschen von Prognose und Ist; CCC höchstens |r|; RMSE² = Bias² + Var(d); Monotonie der exakten Formeln; derselbe Fehler gibt bei größerer Streuung ein größeres r.
- **Raster und Aussagen** (`tests/test_grid.py`, `tests/test_claims.py`): Probe des Erzeugers gegen die gespeicherte, jede Zahl dieser README gegen das Raster oder die Live-Funktion, die Tabellen in der Rundung der App.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an seinen Grenzen), alle Presets, Adresszeile, jedes Auswahlfeld der Experimente, Layout, Footer, Laufzeit-Importe.
- **Mutationsprüfung:** `tools/mutate_demo.py` des Website-Repos mutiert den Kern `ueb_algorithm` (die wörtlichen Kopien `prs_*` sind ausgenommen, sie werden in `pearson-demo` geprüft; Auswertung, Verdikte und Abbildungen prüfen die übrigen Tests): 125 Stellen; 117 durch eigene Tests gefangen, 3 durch wortgleiche Zeilen in Nachbar-Demos gesichert und 5 als gleichwertig festgehalten: die Summe Σ(x − x̄) ist null (deshalb ändert ein Vorzeichen im zweiten Faktor von Kovarianz, Regressionssteigung und Steigung der Differenzen nichts) und die Stützstellen der Gauß–Hermite-Integration liegen symmetrisch (das Vorzeichen der Stützstelle ist gleichgültig). Der erste Lauf fand 7 Überlebende; eine Lücke schloss ein neuer Test: die Grenzen des Bland–Altman-Diagramms zählen nur strikt außerhalb liegende Wochen.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `ueb_algorithm.py` | Kern: Welt, Maße, Bland–Altman, Neukalibrierung, ICC, exakte Formeln |
| `ueb_constants.py` | Regler-Grenzen, Welt, Raster der Messreihen, Farben, Presets |
| `ueb_evaluation.py` | Einzelwelt mit ihren Zeilen, Zellen des Rasters, Verdikte |
| `ueb_layout.py` | Fester Kopf (Stylesheet) |
| `ueb_pdf_export.py` | PDF-Export der Hauptansicht |
| `ueb_presets.py` | Permalink, Presets (Schnellstart mit Hervorhebung), Zufalls-Seed-Button |
| `ueb_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `app.py` | Streamlit-App |
| `data/agreement_grid.json` | vorgerechnetes Raster (erzeugt von `tools/make_grid.py`) |
| `prs_*.py` | wörtliche Kopien aus `pearson-demo` (SplitMix64, Szenario, Pearson-Kern, Konstanten) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy für die Orakel |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Orakel, Kern, Eigenschaften, Raster, Aussagen, Verdikte, Oberfläche, Layout, Footer, Laufzeit-Importe, Kopie-Test, Mutationszustand |
| `tools/make_grid.py` | rechnet das Raster vor |

## Bewusst nicht umgesetzt

- **Prognose-Verfahren und ihr Vergleich** (MAE, RMSE, MASE auf einer Reihe): Prognose-Linie, `naive-forecast-demo`, `forecast-combination-demo`.
- **Intervall- und Wahrscheinlichkeitsprognosen** und ihre Kalibrierung: `forecast-interval-demo`.
- **Deming- und Passing–Bablok-Regression**, **Konfidenzintervall der CCC** (Fisher-Transformation nach Lin) und **gewichtete Kappa-Maße** für Kategorien.
- **Mehrfachmessungen und Zufallseffekte-Modelle** der Intraklassenkorrelation.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Das Raster entsteht neu mit `python tools/make_grid.py` (bei der letzten Erzeugung 1 s auf zehn Prozessen; je nach Rechner anders); `tests/test_grid.py` rechnet eine feste Probe nach und schlägt an, wenn Erzeuger oder Statistik das Raster überholt haben.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
