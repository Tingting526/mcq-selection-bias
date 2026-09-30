# LRZ: Setup → Mock → GPU-Smoke → Analyse

Notebooks 01–05 bleiben Dokumentation. Alle Modellläufe laufen über Slurm.
Shared DSS `/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000` wird nicht benötigt und nicht verändert.
Cache-Standard ist das eigene DSS-Projekt `outputs/cache`; `MCQ_CACHE_ROOT` kann auf ein eigenes DSS-Verzeichnis zeigen.

## 1. Upload vom lokalen Rechner

Im lokalen `analysis`-Ordner (keine Löschung, keine Übertragung lokaler Umgebungen oder Mock-Ergebnisse):

```bash
rsync -av --exclude='__pycache__/' scripts/ ra39dik2@login.ai.lrz.de:/dss/dsshome1/0F/ra39dik2/analysis/scripts/
rsync -av --exclude='__pycache__/' src/ ra39dik2@login.ai.lrz.de:/dss/dsshome1/0F/ra39dik2/analysis/src/
rsync -av configs/ ra39dik2@login.ai.lrz.de:/dss/dsshome1/0F/ra39dik2/analysis/configs/
rsync -av pyproject.toml requirements.txt README.md ra39dik2@login.ai.lrz.de:/dss/dsshome1/0F/ra39dik2/analysis/
rsync -av outputs/preprocessed/ ra39dik2@login.ai.lrz.de:/dss/dsshome1/0F/ra39dik2/analysis/outputs/preprocessed/
ssh ra39dik2@login.ai.lrz.de
```

Die Projektdateien im Ziel werden dabei aktualisiert. Vorherige Ergebnisse werden nicht übertragen oder gelöscht.

## 2. Einmalige Umgebung auf LRZ

```bash
cd /dss/dsshome1/0F/ra39dik2/analysis
python3 --version  # mindestens 3.10; empfohlen 3.11/3.12
bash scripts/slurm/setup_micromamba_environment.sh
export MCQ_CONDA_ENV=mcq-analysis
```

Alternativ bestehendes Conda aktivieren und `export MCQ_CONDA_ENV=<env-name>` setzen, dann dasselbe Setup-Script verwenden. Für eine andere Python-Installation `PYTHON_BIN=/pfad/python3.12` setzen. Setup installiert Pakete, lädt aber kein Modell. Paketstände werden unter `outputs/environment` gespeichert. Für spätere identische Installationen den eingefrorenen Paketstand derselben LRZ-Plattform verwenden. Angefragte und tatsächlich aufgelöste Modellrevisionen stehen in den Run-Metadaten. Die drei Thesis-Kernmodelle sind in ihren YAML-Dateien bereits auf die im Design Freeze geprüften Commits festgelegt.

## 3. Mini-Dry-Run (ohne Modell/GPU)

```bash
source scripts/slurm/environment.sh
python scripts/run_pipeline.py --scorer mock --model qwen2.5-7b-instruct --dataset medqa --limit 8
```

Die Pipeline meldet ihr Verzeichnis unter `outputs/pipelines/mock_*`. Für Resume denselben Aufruf um `--resume outputs/pipelines/<id>` ergänzen. Unveränderte abgeschlossene Rohdaten werden wiederverwendet; abgeleitete Summaries/Prior-Dateien werden neu versioniert. Pro Pipeline-Verzeichnis nur einen Prozess gleichzeitig starten. Mock-Ergebnisse sind technisch markiert und werden nicht als echte Ergebnisse aggregiert.

## 4. Zuerst ausschließlich GPU-Smoke-Test

```bash
sinfo -o '%P %a %l %G'
export PARTITION=<aktuell-verfügbare-GPU-Partition>
bash scripts/slurm/submit.sh smoke
squeue -u ra39dik2
```

Keine Partitionsnamen aus dem Tutorial fest eingebaut. `submit.sh` prüft den gewählten Namen mit `sinfo` und erstellt Logs vor `sbatch`. Ressourcenangaben bei Bedarf anhand der tatsächlichen Partition anpassen. Kein automatischer Full-Run nach dem Smoke-Test.

Der Smoke-Test führt ausschließlich eine echte Baseline mit `--limit 2 --batch-size 1` aus. Logs enthalten Job/Partition, Host, `nvidia-smi`, CUDA-Version sowie A/B/C/D-Diagnostik. Zusätzlich `outputs/logs/token_diagnostics/*.json` prüfen. Modellzugriffsrechte müssen ggf. vorher eingerichtet sein; keine Zugangstokens in Jobskripte schreiben.

```bash
sacct -j <job-id> --format=JobID,Partition,State,ExitCode,Elapsed,MaxRSS
cat outputs/logs/slurm/mcq-smoke-<job-id>.out
cat outputs/logs/slurm/mcq-smoke-<job-id>.err
```

Erfolg: COMPLETED/ExitCode 0, CUDA verfügbar, vier gültige kontextuelle Option-Token, Summary mit n=2. Bei zwei Fragen ist RStd häufig NaN, weil nicht alle Goldklassen vorkommen; das ist kein Hardwarefehler.

## 5. Erst danach kleine und volle Pipelines

```bash
MODEL=qwen2.5-7b-instruct DATASET=medqa LIMIT=8 bash scripts/slurm/submit.sh analysis
# Nach erfolgreicher Prüfung: ohne LIMIT für volle Population.
env -u LIMIT MODEL=qwen2.5-7b-instruct DATASET=medqa STEP=pipeline bash scripts/slurm/submit.sh analysis
```

Ein Job pro Modell × Datensatz lädt Baseline, vier zyklische Permutationen, Prior auf D_e (alpha=0.05) und PriDe nacheinander. Datensätze: `medqa medmcqa mmlu`; Modelle: Aliase unter `configs/models`. Einzelne Schritte bleiben über `STEP=baseline|permutations|prior|pride` möglich; bei PriDe `PRIOR=/pfad/prior.json` setzen. `RESUME` ist bei `STEP=pipeline` ein Pipeline-Verzeichnis, sonst ein Rohdaten-Run-Verzeichnis. Gleiche Modell-, Datensatz-, Limit- und Batch-Argumente wie beim ursprünglichen Pipeline-Aufruf angeben.

Sind mehrere Preprocessing-Versionen vorhanden, muss `PREPROCESSED=/pfad/zum/konkreten/artifact` gesetzt werden. Es wird nicht automatisch neu vorverarbeitet.

## 6. Ergebnisübersicht

Jede vollständige Pipeline schreibt `outputs/pipelines/<id>/comparison.csv` aus `outputs/summaries`. Enthalten sind Accuracy, Recall-RStd, A/B/C/D-Verteilungen, Permutations-Accuracy/Spread und PriDe-Deltas (nachher minus vorher). PriDe nutzt die vorhandene Policy: zyklische Vorhersagen auf D_e, Prior-Korrektur auf D_r.

```bash
python scripts/aggregate_results.py --pipeline-id <id> --scorer real --output outputs/summaries/final.csv
```

Mehrere `--pipeline-id` für eine gemeinsame Modell×Datensatz-Tabelle angeben. Die Auswahl ist explizit, damit Smoke-, Voll- und alternative Runs nicht stillschweigend vermischt werden. Pipeline-Logs und `pipeline.json` liegen neben der Vergleichstabelle. Die CSVs können später direkt in einem kompakten Ergebnis-Notebook geplottet werden. `run_transfer.py` bleibt optional und wird nicht automatisch gestartet.

## 7. Vollständige Versuchsreihe planen, einreichen und sammeln

Alle folgenden Befehle **auf LRZ** im Projektordner ausführen. Ein Plan allein
startet keine Jobs. Ohne `--limit` umfasst er die vollständigen vorverarbeiteten
Datensätze. Die Standardmodelle kommen aus `configs/experiments/thesis.yaml`:
Llama 3.1, Qwen 2.5 und Gemma 2. Qwen3 bleibt über `--models qwen3-8b` wählbar.
Für Modelle mit Zugangsbeschränkung muss der HF-Zugang bereits freigeschaltet sein.

```bash
export MCQ_CONDA_ENV=mcq-analysis
source scripts/slurm/environment.sh
python scripts/run_campaign.py plan --directory outputs/campaigns/thesis-main
cat outputs/campaigns/thesis-main/campaign.json
```

Erst nach Prüfung des kleinen Pipeline-Laufs einreichen (Partition vorher mit
`sinfo` prüfen):

```bash
python scripts/run_campaign.py submit --directory outputs/campaigns/thesis-main --partition lrz-dgx-a100-80x8
```

Der Plan enthält neun unabhängige GPU-Jobs. Jeder durchläuft Baseline,
Permutationen, Prior und PriDe. Ein geerbtes `LIMIT=8` wird beim Voll-Lauf entfernt.
Wiederholtes `submit` überspringt registrierte Jobnummern; es startet fehlgeschlagene
Jobs nicht automatisch erneut. Für deren Fortsetzung die Konfiguration und den
Pipeline-Pfad aus `campaign.json` mit `RESUME` an `submit.sh analysis` übergeben.
Nur eine Instanz des Campaign-Befehls gleichzeitig ausführen.

Nach Abschluss aller Pipelines:

```bash
python scripts/run_campaign.py collect --directory outputs/campaigns/thesis-main
```

Die Tabelle liegt in `outputs/campaigns/thesis-main/comparison.csv`.
Unvollständige Pipelines führen zu einem Fehler statt zu einer scheinbar fertigen
Tabelle. `pipeline.json` nennt Status und aktiven Schritt; dessen Details stehen
in `baseline.log`, `permutations.log`, `prior.log` oder `pride.log` daneben.
Ein hart abgebrochener Job kann dort noch `running` anzeigen; Slurm-Status mit
`sacct` prüfen. Der Slurm-Jobname lautet vollständig `mcq-analysis`, auch wenn
`squeue` ihn verkürzt anzeigt: Logs heißen `mcq-analysis-<job-id>.out/.err`.

Falls vom LRZ/Supervisor vorgegeben, können `QOS` und `ACCOUNT` gesetzt werden.
Diese Werte werden an Slurm weitergereicht; es wird keine Berechtigung geraten.

## 8. Separater Originalvergleich nach Zheng

Notebook 08 dokumentiert die Originalbedingungen für offene Modelle. Upload
zusätzlich zu Skripten/Code/Konfiguration (auf dem Mac, im analysis-Ordner):

```bash
rsync -av data/reference/zheng2024/ ra39dik2@login.ai.lrz.de:/dss/dsshome1/0F/ra39dik2/analysis/data/reference/zheng2024/
```

Auf LRZ:

```bash
export MCQ_CONDA_ENV=mcq-analysis
source scripts/slurm/environment.sh
python -m pip install -e '.[reference]'
sinfo -o '%P %a %l %G'
export PARTITION=lrz-dgx-a100-80x8
export MODEL=qwen2.5-7b-instruct
export DATASET=arc
export LIMIT=2
unset RESUME
bash scripts/slurm/submit.sh zheng
```

Dieses Beispiel setzt verfügbare A100-Partition voraus. Der Auftrag führt das
0-/5-shot-Originalprotokoll mit insgesamt zwei Fragen pro Bedingung aus.
Ergebnisse stehen in `outputs/zheng_selection_bias/<run-id>`, Logs heißen
`mcq-zheng-<job-id>.out/.err`. Nach geprüftem kleinen Test kann `unset LIMIT`
für den Voll-Lauf verwendet werden. `DATASET=mmlu|arc|csqa` steuert die Original-
Benchmarks; medizinische Aliase sind in diesem Referenzrunner nicht zugelassen.
Keine aktuellen Chat-Templates: der originale rohe CLM-Prompt wird bewusst erhalten.

PriDe ist nicht Teil dieses separaten Auftrags. Fünf-Optionen-Scoring und die fünf
CSQA-Permutationen sind hier unterstützt, aber noch nicht in der PriDe-Pipeline.

## Full MedQA comparison: five models

`start_medqa_campaign.py` submits one independent full MedQA pipeline for each of
Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen3-8B, Gemma-2-9B-IT and
GLM-4-9B-Chat-HF. Model revisions are fixed in `configs/models`. It selects a
complete 1,273-question MedQA artifact matching the configured prompt version,
checks access to all model repositories, and only then submits jobs. Small
configuration files are downloaded during the access check; model weights are
loaded by the GPU jobs. Llama and Gemma require approved Hugging Face access and
an authenticated account in the LRZ environment.

After uploading code, configurations and the matching MedQA artifact, run on LRZ:

```bash
export MCQ_CONDA_ENV=mcq-analysis
source scripts/slurm/environment.sh
python scripts/start_medqa_campaign.py --submit
```

The default campaign directory is `outputs/campaigns/medqa-five-models-20260930`.
Repeating this command skips already submitted jobs. It does not resubmit failed
jobs. A different configuration requires a new `--directory`.

Each model uses all 1,273 questions, batch size 1, seed 42 and alpha 0.05:
63 estimation questions (D_e) and 1,210 remaining questions (D_r). All models
use the same question population and estimation IDs, but estimate their own
prior. The saved first-token baseline is reused for the paired PriDe comparison.
The pipeline includes four cyclic option arrangements, not all 24 permutations.
Summary columns separately report D_e and D_r accuracy, recall, RStd and predicted
answer counts before/after correction. D_e uses cyclic debiasing; D_r uses the
global prior. This is a single-seed experiment, not a multi-seed robustness study.

Once every job has completed:

```bash
python scripts/run_campaign.py collect \
  --directory outputs/campaigns/medqa-five-models-20260930
```

This creates the campaign's `comparison.csv`. GPU compatibility for the four
additional models must still be verified in their actual LRZ runs; passing the
access check does not guarantee inference compatibility.
