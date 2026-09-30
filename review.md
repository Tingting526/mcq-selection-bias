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

## Targeted medical-pipeline review (2026-09-30)

Verified first-token extraction at the final unmasked prompt position and softmax
restricted to the four option-token logits. These are conditional probabilities
among A–D, not their unnormalized full-vocabulary probabilities. Model evaluation
mode is enabled; inference does not sample generated answers.

Reviewed cyclic content/display mappings, updated gold labels, sample-prior
geometric means, averaging across estimation samples, and division by the global
prior followed by normalization. Numerical parity tests against the pinned Zheng
reference pass for positive probabilities; epsilon handling differs near zero.
The prior-estimation computation does not use gold answers. Gold answers are used
for metrics. Estimation questions receive cyclic debiasing and remaining questions
receive the global-prior correction, with separate D_e/D_r summary fields.

Validation: 78 tests passed, one integration test deselected. In addition, real
pinned Qwen3, Mistral-v0.3 and GLM-4 tokenizers passed contextual A–D continuation
checks on three MedQA questions in four cyclic orders each (12 prompts per model).
These tokenizer checks do not validate GPU inference for Mistral or GLM. The user
provided a successful corrected 100-question Qwen3 LRZ run with explicit baseline
reuse; full-dataset results are still pending.

Scope: the medical pipeline uses four cyclic permutations, current chat templates,
one valid whitespace representation per option token, and one calibration seed.
It is not an exact replication of the full original experimental protocol. Fixed
A–D labels and option rotation alone do not isolate positional from token-label
bias. The separate original-protocol runner provides additional controls; their
existence is not evidence that the medical campaign has run them. Accuracy and
RStd are stored on a 0–1 scale; multiply by 100 when comparing to original paper
percentage-scale tables. RStd is the population standard deviation of per-label
recalls, not the standard deviation of predicted-label frequencies.
