# Datenpipeline fuer die MCQ-Selection-Bias-Studie

## Ziel

Die Pipeline wertet MedQA, MedMCQA und MMLU mit genau vier angezeigten
Antwort-IDs (A/B/C/D) aus. Sie generiert keinen Antworttext. Stattdessen liest
sie fuer jede Frage die vier Logits des unmittelbar naechsten Antwort-Tokens
aus und normalisiert diese vier Werte zu Wahrscheinlichkeiten.

## Eingefrorene Thesis-Modelle

| Alias | Hugging Face Repository | Besonderheit |
|---|---|---|
| `llama-3.1-8b-instruct` | `meta-llama/Llama-3.1-8B-Instruct` | HF-Lizenzzugang erforderlich |
| `qwen2.5-7b-instruct` | `Qwen/Qwen2.5-7B-Instruct` | nicht zugangsbeschraenkt |
| `gemma-2-9b-it` | `google/gemma-2-9b-it` | HF-Lizenzzugang erforderlich |

Die drei Modellprofile liegen unter `configs/models/` und enthalten die am
25. September 2026 eingefrorenen Modell- und Tokenizer-Commits. Weitere
Profile im Ordner dienen nur der Entwicklung und gehoeren nicht zum
Thesis-Kernexperiment.

## Ablauf

### 1. Datensatz laden

`src/data/registry.py` liest das jeweilige Profil aus `configs/datasets/` und
laedt den Split ueber Hugging Face Datasets.

Standard-Splits:

- MedQA: `test`
- MedMCQA: `validation` (der oeffentliche `test`-Split hat keine Goldlabels)
- MMLU: `test`

Ein explizites `--split` ueberschreibt den Standard.

### 2. Einheitlich normalisieren

Die Adapter in `src/data/` erzeugen fuer jede Zeile ein unveraenderliches
`MCQSample`:

```text
sample_id
dataset
subject
question
options[4]
gold_index (0..3)
gold_label (A..D)
```

Leere Fragen oder Optionen, ungueltige Labels und eine andere Optionszahl als
vier werden als unbrauchbare Quellzeilen in die Quarantaene geschrieben.

### 3. Permutationssicherheit filtern und auditieren

Die versionierte Konfiguration `configs/preprocessing/thesis.yaml` entfernt vor
jeder Modell- oder Biasberechnung drei Arten methodisch ungeeigneter Aufgaben:

- relative Antworttexte wie `none of the above`;
- Kombinationen der angezeigten Antwort-IDs wie `both A and C`;
- doppelte Antworttexte innerhalb derselben Frage.

Die ersten beiden Regeln sind an der Vorfilterung des offiziellen PriDe-Codes
orientiert. Unser Kombinationsfilter prueft den ganzen Antworttext, damit etwa
`Vitamin A and D` nicht faelschlich entfernt wird. Aufgaben, bei denen mindestens
drei Optionen selbst Buchstabenkombinationen oder Reihenfolgen sind, bleiben
ebenfalls erhalten: Dort gehoeren A-D zum Aufgabeninhalt und sind keine
Verweise auf angezeigte Antwortpositionen. Jede ausgeschlossene Zeile
bleibt mit Grund in `excluded_rows.parquet` erhalten. Doppelte Sample-IDs werden
ebenfalls quarantaenisiert; inhaltliche Dubletten mit verschiedenen IDs werden
nur gezaehlt und beibehalten.

Die akzeptierte Population wird zusammen mit Quellrevision, Fingerprint,
Regel-Hash, Sample-Set-Hash und Dateipruefsummen in einem unveraenderlichen
Artefakt unter `outputs/preprocessed/` gespeichert. `start_index` und `limit`
greifen erst nach der Filterung.

### 4. Prompt und zyklische Varianten erstellen

`src/prompts/mcq.py` baut den als `thesis-mcq-v2` versionierten Zero-Shot-Prompt
und wendet danach das originale Chat-Template des jeweiligen Tokenizers an.
Modellspezifische Argumente kommen aus dem Modellprofil.

Das Vorverarbeitungsartefakt enthaelt bereits fuer jede akzeptierte Frage alle
vier zyklischen Ordnungen, beide Indexabbildungen, das verschobene Goldlabel und
den kanonischen User-Prompt. Das modellspezifische Chat-Template wird erst beim
jeweiligen Modelllauf angewandt.

### 5. Antwort-Tokens pruefen

Vor einem echten Modelllauf prueft `src/models/token_scoring.py` am vollstaendig
gerenderten Prompt, ob A, B, C und D jeweils ein einzelnes, verschiedenes
naechstes Token sind. Sowohl die Form `A` als auch ` A` wird getestet. Wenn die
Pruefung fehlschlaegt, wird keine Auswertung gestartet.

### 6. First-Token-Scoring

Das Modell wird genau einmal pro Prompt vorwaerts ausgefuehrt. Aus der letzten
nicht aufgefuellten Position werden nur die vier A/B/C/D-Logits gelesen. Ein
Softmax ueber diese vier Werte erzeugt `prob_A` bis `prob_D`; der groesste Wert
ist die Vorhersage.

### 7. Auswertungsmodi

- `run_baseline.py`: originale Reihenfolge der Antwortoptionen
- `run_permutations.py`: alle vier zyklischen Permutationen
- `estimate_prior.py`: PriDe-Prior auf dem festgelegten Anteil des Datensatzes
- `run_pride.py`: PriDe-Korrektur mit einem gespeicherten Prior
- `run_transfer.py`: Prior-Transfer zwischen MedQA und MedMCQA

Die Zuordnung erfolgt immer ueber den urspruenglichen Optionsindex, nicht ueber
den Antworttext. Dadurch bleiben auch identische Antworttexte korrekt
zugeordnet.

### 8. Speicherung und Metriken

Rohdaten werden fortlaufend unter `outputs/raw/` als Parquet-Teile gespeichert.
Zu jedem Lauf gibt es Metadaten, unter anderem Modell, Revisionen, Split,
Datensatz-Fingerprint, Seed, Prompt-Version, Chat-Template-Argumente, Backend
und Token-IDs. Abgebrochene Laeufe koennen mit `--resume` fortgesetzt werden.

Zusammenfassungen unter `outputs/summaries/` enthalten:

- Accuracy
- Recall fuer A/B/C/D
- RStd als Populationsstandardabweichung der vier Recalls
- Vorhersagehaeufigkeiten
- Accuracy-Varianz und Spannweite ueber Permutationen
- Baseline/PriDe-Differenzen

## Lokal ohne LRZ testen

```bash
cd analysis
source .venv/bin/activate
python -m pytest

python scripts/preprocess_data.py --dataset medqa
python scripts/preprocess_data.py --dataset medmcqa
python scripts/preprocess_data.py --dataset mmlu

python scripts/run_baseline.py \
  --model qwen2.5-7b-instruct \
  --dataset medqa \
  --limit 20 \
  --batch-size 4 \
  --scorer mock

python scripts/run_baseline.py \
  --model qwen2.5-7b-instruct \
  --dataset medmcqa \
  --limit 20 \
  --batch-size 4 \
  --scorer mock
```

`--scorer mock` prueft nur die Daten-, Prompt-, Speicher- und Metrikpipeline.
Die erzeugten Werte sind keine wissenschaftlichen Modellergebnisse.

## Spaeter auf dem LRZ

Die Python-Pipeline bleibt unveraendert. Hinzu kommen nur die LRZ-spezifische
Umgebung, Hugging-Face-Anmeldung beziehungsweise Cache, GPU-Ressourcen und
SLURM-Startskripte. Echte Laeufe verwenden den Standard `--scorer real` und
lassen `--scorer mock` weg.
