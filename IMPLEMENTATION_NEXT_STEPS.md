# Implementation Next Steps (Phase 2)

**Status:** Dieses Dokument konkretisiert ausschliesslich Phase 2. Methodische
Entscheidungen werden nicht mehr hier getroffen; verbindlich ist
`../DESIGN_FREEZE_2026-09-25.md`. Der aktive Implementierungsordner ist
`analysis/`. Hauptlaeufe beginnen erst nach Abschluss der unten genannten
Auswertungsfunktionen und des echten Piloten.

## 1. LRZ Setup pruefen

Ziel: Sicherstellen, dass Code, Daten und Python-Umgebung auf LRZ laufen.

- `analysis` liegt auf LRZ unter `/dss/dsshome1/0F/ra39dik2/analysis`.
- Preprocessed Daten liegen unter `outputs/preprocessed/{medqa,medmcqa,mmlu}`.
- Tests laufen mit `python3 -m pytest -q`.
- Kein echter Modelllauf auf dem Login-Knoten.

Offen:

- Virtuelle Umgebung auf LRZ sauber aktivieren oder Container-Setup nutzen.
- Slurm-Testjob mit GPU starten.
- Hugging Face Cache auf DSS legen, damit Modelldownloads nicht im Home stoeren.

## 2. Mini-Dry-Run vor echten Analysen

Ziel: Pipeline technisch testen, ohne wissenschaftliche Resultate zu erzeugen.

Reihenfolge:

1. Baseline mit Mock-Scorer und sehr kleinem Limit.
2. Permutationen mit Mock-Scorer und sehr kleinem Limit.
3. Prior-Schaetzung mit Mock-Scorer.
4. PriDe mit Mock-Prior.

Ergebnis: Wir sehen, ob Outputs, Resuming, Summaries und Pfade stimmen.

## 3. Real-Model Sanity Check

Ziel: Ein echtes Modell auf GPU laden und nur wenige Samples testen.

Reihenfolge:

1. `check_hardware.py` oder ein kleiner Baseline-Lauf mit `--limit 2`.
2. Token-Diagnostik pruefen: A, B, C, D muessen eindeutig als erste Antworttokens funktionieren.
3. Output-Struktur pruefen.

Ergebnis: Danach wissen wir, dass LRZ, GPU, Modell, Tokenizer und Prompt zusammenpassen.

## 4. Selection Bias messen

Selection Bias wird vor PriDe gemessen. Das ist die rohe Verzerrung des Modells in der normalen MCQ-Auswertung.

Pro Modell und Datensatz:

1. `run_baseline.py`
   - originale Antwortreihenfolge
   - misst Accuracy und Antwortverteilung A/B/C/D
2. `run_permutations.py`
   - vier zyklische Antwortpermutationen
   - misst, ob die Vorhersage stabil bleibt, wenn nur die Positionen wechseln
   - erzeugt zusaetzlich cyclic-debiased Resultate

Wichtig:

- Baseline = normale Modellantwort.
- Permutationen = eigentliche Bias-Diagnostik.
- Cyclic debiasing ist eine Vergleichskorrektur, aber noch nicht PriDe.

## 5. PriDe vervollstaendigen und pruefen

PriDe kommt nach der Bias-Messung.

Pro Modell und Datensatz:

1. `estimate_prior.py`
   - waehlt `D_e`, aktuell alpha `0.05`
   - laesst zyklische Permutationen nur auf der Prior-Stichprobe laufen
   - schaetzt globalen Prior fuer A/B/C/D
2. `run_pride.py`
   - nutzt den gespeicherten Prior
   - korrigiert die normalen Vorhersagen auf `D_r`
   - nutzt fuer `D_e` die cyclic-debiased Predictions
3. `run_transfer.py`
   - prueft Priors in beide Richtungen zwischen MedQA und MedMCQA
   - berichtet Transfer minus Default und Transfer minus In-Domain direkt

Wichtig:

- PriDe braucht erst einen gespeicherten Prior.
- PriDe-Resultate werden gepaart mit Baseline und In-Domain-PriDe verglichen.
- Hauptmetriken: Accuracy, RStd, Antwortverteilung A/B/C/D, Delta Accuracy, Delta RStd.
- Der eingefrorene Prior wird sekundaer auf alle vier gespeicherten Zyklen angewandt.
- Eine Sensitivitaetsanalyse verwendet fuer beide Quellen 63 Priorfragen.

## 6. Ergebnisaggregation

Nach den Modelllaeufen:

1. Alle Summaries aus `outputs/summaries` sammeln.
2. Pro Modell und Datensatz eine Tabelle bauen:
   - Baseline
   - Permutation Bias
   - Cyclic debiasing
   - PriDe
   - PriDe Transfer
3. Plots fuer Thesis erstellen:
   - Antwortverteilung A/B/C/D
   - RStd vor/nach Korrektur
   - Accuracy vor/nach Korrektur

## 7. Reihenfolge fuer die naechste Implementierung

Als Naechstes werden genau diese Punkte umgesetzt:

1. Direkte Transfer-minus-In-Domain-Kontraste implementieren.
2. Korrigierte zyklische Metriken aus gespeicherten Logits implementieren.
3. Gepaarte Bootstrapintervalle und MMLU-Subject-Stratifizierung implementieren.
4. Gleiches absolutes Priorschaetzbudget und Seed-Zusammenfassungen implementieren.
5. Einen vollstaendigen Mock-Durchlauf fuer alle vorgesehenen Bedingungen ausfuehren.
6. Danach Slurm-Templates und einen echten `--limit 2`-GPU-Lauf pruefen.
7. Erst nach erfolgreichem Piloten die Batch-Jobs der Hauptlaeufe erzeugen.
