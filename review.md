# Vorverarbeitung zur fachlichen Durchsicht

Dieser Stand dokumentiert die Vorverarbeitung von MedQA, MedMCQA und MMLU für die Untersuchung von Positionsbias bei Multiple-Choice-Fragen. Die eigentlichen Modellanalysen wurden noch nicht durchgeführt.

## Empfohlene Reihenfolge

1. `notebooks/01_datensaetze.ipynb` – Datensätze und einheitliches MCQ-Schema
2. `notebooks/02_vorverarbeitung.ipynb` – Filter- und Ausschlussregeln
3. `notebooks/03_vorverarbeitung_audit.ipynb` – Prüfung der erzeugten Artefakte
4. `notebooks/04_prompt_und_permutationen.ipynb` – Prompt und zyklische Antwortreihenfolgen

`notebooks/05_scoring_und_bias_metriken.ipynb` zeigt ergänzend mit synthetischen Werten, welche Metriken später berechnet werden. Es enthält noch keine wissenschaftlichen Modellresultate.

## Aktueller Datenstand

| Datensatz | Split | Verwendbare Fragen | Ausgeschlossen |
|---|---|---:|---:|
| MedQA | test | 1.273 | 0 |
| MedMCQA | validation | 3.941 | 242 |
| MMLU | test | 13.581 | 461 |

Die eingefrorenen Daten, zyklischen Prompts, Ausschlüsse und Manifeste befinden sich unter `outputs/preprocessed/`.

## Punkte für die fachliche Rückmeldung

- Sind die gewählten Datensplits angemessen?
- Sind die Ausschlussregeln für relative Positionsverweise, Antwortkombinationen und doppelte Optionen sinnvoll?
- Soll die Stichprobenauswahl weiterhin erst nach der Filterung erfolgen?
- Sollen identische Inhalte mit unterschiedlichen Sample-IDs erhalten bleiben?
- Sind vier zyklische Permutationen für die geplante Bias-Untersuchung ausreichend?

## Reproduzierbarkeit

Die Installation und erneute Ausführung der Vorverarbeitung ist in `RUN_PREPROCESSING.md` beschrieben. Konfigurationen liegen unter `configs/`, die Implementierung unter `src/` und die zugehörigen Tests unter `tests/`.
