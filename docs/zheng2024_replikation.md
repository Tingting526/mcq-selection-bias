# PriDe: Originalvergleich und zusätzliche Benchmarks

Geprüft am 2026-09-28. Quellen:
- [Paper, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/54dd9e0cff6d9214e20d97eb2a3bae49-Paper-Conference.pdf), Abschnitt 3, Algorithmus 1, Anhang E.
- [Offizieller Code, fixierter Commit](https://github.com/chujiezheng/LLM-MCQ-Bias/tree/2ae2f40c77006f7a00a4675bdfb30151c2406691/code).

## Was übereinstimmt

Die Vier-Optionen-Implementation verwendet dieselbe Rechenstruktur: zyklische
Permutationen, normalisiertes geometrisches Mittel pro Frage als Prior,
arithmetischer Mittelwert der Frage-Priors, Division der beobachteten Verteilung
durch den globalen Prior und Renormalisierung. D_e erhält die über Positionen
zusammengeführte zyklische Inhaltsvorhersage; D_r die globale Korrektur.
RStd ist die Populationsstandardabweichung der Recalls. First-Token-Scoring auf
Option-IDs entspricht dem Ansatz für offene Modelle.

`tests/test_zheng_reference.py` vergleicht die lokale Rechnung mit der gelesenen
Originalfunktion `debias_utils.simple`: 100 synthetische positive 4×4-Matrizen,
Prior, zyklische Inhaltsverteilung, globaler Prior und 100 Korrekturen.
Dies prüft Numerik, nicht die Gleichheit ganzer Modell-Experimente.

## Unterschiede: noch keine exakte Replikation

- Unser Code ist auf vier Optionen beschränkt; das Original unterstützt auch
  fünf (CSQA). Kein Entfernen der fünften Antwort zum Erzwingen der Kompatibilität.
- Original: fünf Durchläufe mit wiederholtem Shuffle und einem vom Ergebnis-Pfad
  abgeleiteten Zufallszustand. Aktuelle Pipeline: sortierte IDs, Seed 42 und ein
  Durchlauf. Sensitivitäts-Seeds sind konfiguriert, aber nicht automatisch Teil
  der Campaign. Für eine Replikation müssen Wiederholungen ausgewertet werden.
- Numerik: Original addiert 1e-10; lokal werden Werte bei 1e-12 abgeschnitten.
  Nahe null können Ergebnisse abweichen. Positive Testverteilungen stimmen bis
  auf numerische Toleranz überein; keine Behauptung bitgenauer Gleichheit.
- Original nutzt eigene Prompts (bei MMLU mit Fachbezeichnung), 0-/5-shot und
  damalige Modellformate. Lokal: eingefrorener eigener Zero-Shot-Prompt und
  modellabhängiges Chat-Template. Promptunterschiede müssen ausgewiesen werden.
- Bisherige MMLU-Vorverarbeitung hat eigene Filter. Für RQ1 die separate
  Referenzpopulation verwenden; bisherige Dateien bleiben unverändert.
- Lokale Pipeline berechnet Schritte teilweise erneut. Ihre reale Laufzeit ist
  nicht mit dem theoretischen PriDe-Mehrkostenfaktor gleichzusetzen.

## Heruntergeladene Daten

Alle Dateien liegen unter `data/reference/zheng2024/` (lokal, noch nicht auf LRZ).
`upstream/code/data_*` enthält unveränderte Originaldateien, `normalized/` enthält
JSONL mit allen Fragen, Optionen, Labels, stabilen IDs und dem Ausschlussindikator
`excluded_by_upstream_filter`. Kein Originaldatensatz wird überschrieben.

| Benchmark | Repository-Evaluationszeilen | Nach Originalfilter | Optionen |
|---|---:|---:|---:|
| MMLU, 57 Fächer | 14.042 | 13.592 | 4 |
| ARC-Challenge | 1.165 | 1.165 | 4 |
| CommonsenseQA | 1.216 | 1.216 | 5 |

Die Zahlen nach Originalfilter entsprechen Tabelle 7 des Papers. Die Filterlisten
werden als Literale aus `utils.py` gelesen und lediglich annotiert. Die
Normalisierung ist **kein** fertiges Artefakt unserer Vier-Optionen-Pipeline.
Dev-Dateien sind ebenfalls vorhanden (MMLU 285, ARC 5, CSQA 5), werden aber nicht
als zusätzliche Testfragen genutzt.

CSQA: `test` im Repository ist **nicht** der versteckte offizielle Testsplit.
Die Autoren mischen den gelabelten Dev-/Validation-Split mit Seed 23, reservieren
fünf Fragen als Few-Shot-Beispiele und evaluieren auf den übrigen 1.216 Fragen.
ARC: aus dem ursprünglichen Challenge-Testsplit werden ausschließlich
Vier-Optionen-Fragen verwendet. Bei MMLU stehen die Ausschlüsse im Auswertungscode.

`manifest.json` dokumentiert Commit, Datei-Hashes, Zeilenzahlen und ausgeschlossene
IDs. Reproduzierbarer Download/Audit: `python scripts/download_zheng2024.py`.
Ein erneuter Aufruf akzeptiert vorhandene unveränderte Dateien und bricht bei
abweichenden Inhalten ab.

## Vorgeschlagene Forschungsfragen (noch keine bestätigte Thesis-Nummerierung)

- **RQ1 – Replikation mit neueren Modellen:** Wie stark sind Selection Bias und
  PriDe-Wirkung auf den Originalbenchmarks mit neueren Modellen? Vergleich anhand
  von Accuracy, RStd und PriDe-Deltas unter dokumentiertem Protokoll.
- **RQ2 – Domainerweiterung:** Zeigen sich diese Effekte auch auf MedQA und
  MedMCQA, und wie verändert sich die Wirksamkeit von PriDe?

MMLU enthält bereits medizinische Fächer und ist damit kein rein nichtmedizinischer
Kontrolldatensatz. Ein historischer Vergleich mit Paperzahlen allein isoliert
keinen Effekt des Modellalters: Prompts, Modellgröße, Instruction-Tuning und
Population müssen berücksichtigt werden. Ideal ist zusätzlich ein altes
Referenzmodell unter demselben neuen Protokoll. Eine Verbesserung ist eine zu
prüfende Hypothese, kein vorausgesetztes Ergebnis.

## Vor echten RQ1-Läufen noch implementieren

1. Referenzdaten als separate Dataset-Aliase/Artefakte integrieren, Originalfilter
   und Splits festlegen; keine erneute Anwendung der medizinischen Filter.
2. Anzahl der Optionen durchgehend auf vier/fünf verallgemeinern: Datenmodell,
   Prompts, Token-Diagnostik, Scoring, Permutationen, Prior, Metriken und Plots.
3. Passendes Replikationsprotokoll und wiederholte Kalibrierungsziehungen
   konfigurieren; anschließend Mock- und kleiner GPU-Test.

Keine GPU-Jobs wurden durch diesen Download gestartet. Der bestehende
medizinische Lauf und seine Daten wurden nicht verändert.

## Ergänzung: Originaler Selection-Bias-Runner (2026-09-29)

Notebook 08 und `run_zheng_selection_bias.py` nutzen nun die hashgeprüften
Originalfunktionen direkt. In diesem separaten Pfad sind die Originaldatensätze,
vier/fünf Optionen, 0-/5-shot, 24/fünf Permutationen, Gold-Moving auf MMLU,
Shuffling IDs und Removing IDs ausführbar. Scoring übernimmt auch die originale
Summierung von Leerzeichen-/Nicht-Leerzeichen-Tokenvarianten; dies war eine weitere
Abweichung des bisherigen Scorers. PriDe und dessen Fünf-Optionen-Erweiterung bleiben
separate Arbeit. Die frühere Liste offener Punkte bezieht sich auf die medizinische
Gesamtpipeline; sie ist nicht mit diesem separaten Referenzrunner gleichzusetzen.
