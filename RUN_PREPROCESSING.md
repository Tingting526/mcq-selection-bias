# Vorverarbeitung vollständig ausführen

Alle implementierten Dateien liegen jetzt unter `analysis/`. Der Lauf benötigt
kein Sprachmodell und keine GPU.

## 1. Einmalige Einrichtung

Im Terminal:

```bash
cd analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks]"
```

## 2. Schneller Test mit je 20 akzeptierten Fragen

```bash
python scripts/run_preprocessing_all.py --limit 20
```

Der Befehl verarbeitet MedQA, MedMCQA und MMLU, validiert alle erzeugten
Prüfsummen und zyklischen Prompts und führt danach die Vorverarbeitungstests aus.

## 3. Vollständiger Lauf

```bash
python scripts/run_preprocessing_all.py
```

Die neuen Artefakte liegen danach unter:

```text
outputs/preprocessed/medqa/<Zeitstempel>_test_mcq-preprocessing-v1/
outputs/preprocessed/medmcqa/<Zeitstempel>_validation_mcq-preprocessing-v1/
outputs/preprocessed/mmlu/<Zeitstempel>_test_mcq-preprocessing-v1/
```

Jedes Verzeichnis enthält `samples.parquet`, `cyclic_prompts.parquet`,
`excluded_rows.parquet` und `manifest.json`.

## 4. Überblick im Notebook

Danach in dieser Reihenfolge öffnen:

1. `notebooks/02_vorverarbeitung.ipynb`
2. `notebooks/03_vorverarbeitung_audit.ipynb`

Das Audit-Notebook findet automatisch den neuesten vollständigen Lauf jedes
Datensatzes. Beide Notebooks enden bewusst vor Modellscoring,
Selection-Bias-Analyse und PriDe.
