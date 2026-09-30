# MCQ Selection Bias and PriDe

This repository is a thesis-oriented experimental framework for measuring option-ID selection bias in four-option multiple-choice evaluation. It implements the cyclic-permutation baseline and PriDe from Zheng et al., ["Large Language Models Are Not Robust Multiple Choice Selectors"](https://proceedings.iclr.cc/paper_files/paper/2024/hash/54dd9e0cff6d9214e20d97eb2a3bae49-Abstract-Conference.html).

It is deliberately not a generic benchmark runner. The supported path is:

- open-weight Hugging Face causal language models;
- MedQA, MedMCQA, and MMLU with exactly four displayed IDs, A/B/C/D;
- first-token option-ID logits, with no generation, reasoning, or sampling;
- standard evaluation, cyclic permutation, cyclic debiasing, PriDe, and prior transfer;
- raw Parquet records before aggregate metrics are calculated.

The old `questions.json` and `results.csv` files are retained only as prototype artifacts. They are not inputs to the thesis pipeline.

## Methodological Contract

Every dataset is normalized to:

```text
sample_id, dataset, subject, question,
options[4], gold_index (0..3), gold_label (A..D)
```

Content identity is always an integer index into the original option list. Displayed labels are derived after permutation. Option text is never used as identity. Samples with repeated option texts are nevertheless excluded during preprocessing because their semantic identity is ambiguous after a permutation.

Before any model or bias computation, every split passes through the versioned
`mcq-preprocessing-v1` stage. It normalizes the source schema, validates the
four-option/gold-label contract, detects duplicate IDs, and quarantines rows
whose answer semantics do not survive a permutation. In particular, it removes
relative choices such as `none of the above`, displayed-ID combinations such as
`both A and C`, and repeated option texts. The first two rules follow the intent
of the filtering in the official PriDe code; the ID-combination detector is
anchored to the whole option so text such as `Vitamin A and D` is not removed.
It also retains tasks where at least three choices are themselves letter
combinations or sequences, because those letters belong to the task rather than
referencing the displayed answer positions.
Nothing is silently discarded: every excluded row and every reason is saved.

The canonical user prompt is versioned as `thesis-mcq-v2`:

```text
Answer the following multiple-choice question.
Respond with only the option letter (A, B, C, or D).

Question: {question}

A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}

Answer:
```

For instruction-tuned models this user content is rendered through the tokenizer's own chat template. Before any scoring run, the framework verifies under that rendered prompt that A/B/C/D are distinct single next tokens. A failed check stops the run.

Artifacts created with the former `thesis-mcq-v1` prompt are development
artifacts. Regenerate preprocessing artifacts with `thesis-mcq-v2` before a
pilot or final run.

PriDe follows Equations 7 and 8 and Algorithm 1 of Zheng et al.:

1. Select the estimation set without reading gold labels.
2. Score all four cyclic permutations for each estimation sample.
3. Average log observed probabilities by displayed ID and apply softmax to obtain each sample prior.
4. Take the arithmetic mean of sample priors to obtain the global prior.
5. Use cyclic debiasing for estimation samples and prior division for the remaining samples.

Gold labels are present in normalized samples because later evaluation needs them. The prior-estimation function only reads `sample_id`, `permutation_id`, and `prob_A` through `prob_D`; this is covered by a test with no gold fields.

## Project Tree

```text
analysis/
├── configs/
│   ├── datasets/{medqa,medmcqa,mmlu}.yaml
│   ├── experiments/thesis.yaml
│   ├── preprocessing/thesis.yaml
│   └── models/{llama-3.1-8b-instruct,qwen2.5-7b-instruct,
│               gemma-2-9b-it,...}.yaml
├── src/
│   ├── data/{base,medqa,medmcqa,mmlu,preprocessing,registry}.py
│   ├── diagnostics/check_option_tokens.py
│   ├── evaluation/{baseline,permutation_eval,pride_eval}.py
│   ├── experimental/identical_content_prior.py
│   ├── metrics/{accuracy,selection_bias}.py
│   ├── models/{loader,token_scoring}.py
│   ├── permutations/cyclic.py
│   ├── pride/{prior_estimation,debias,transfer}.py
│   ├── prompts/mcq.py
│   └── utils/{config,io,logging,metadata,seeds}.py
├── scripts/
│   ├── download_data.py
│   ├── preprocess_data.py
│   ├── run_baseline.py
│   ├── run_permutations.py
│   ├── estimate_prior.py
│   ├── run_pride.py
│   ├── run_transfer.py
│   └── summarize_results.py
├── tests/
├── outputs/{preprocessed,raw,priors,summaries,logs}/
├── pyproject.toml
└── requirements.txt
```

## Modules

| Area | Responsibility |
|---|---|
| `src/data` | Strict dataset adapters, permutation-safety filtering, audit reports, checksummed artifacts, and the common immutable sample schema. |
| `src/prompts` | One versioned zero-shot four-option prompt. |
| `src/permutations` | Exact cyclic orders plus both mapping directions. |
| `src/models` | Config-driven model loading, token diagnostics, batched first-token scoring. |
| `src/metrics` | Accuracy, per-ID recall, population RStd, prediction frequencies, and permutation variation. |
| `src/pride` | Label-free prior estimation, correction, cyclic content averaging, and transfer summaries. |
| `src/experimental` | Disabled-by-default identical-content estimator, kept separate from PriDe. |
| `src/evaluation` | Construction of inspectable raw records and derived predictions. |
| `src/utils` | Deterministic seeds, metadata, non-overwriting output paths, and resumable Parquet parts. |

## Installation

Use Python 3.10 or newer:

```bash
cd analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

To open and execute the guided notebooks locally, install the notebook extras:

```bash
python -m pip install -e ".[dev,notebooks]"
```

Llama 3.1 and Gemma 2 require accepting their Hugging Face licenses before
logging in with `huggingface-cli login`. No closed-model API is used.

## Real Model Backend, Hardware Gate, and Scoring Mode

The frozen thesis model set is:

| Alias | Hugging Face repository |
|---|---|
| `llama-3.1-8b-instruct` | `meta-llama/Llama-3.1-8B-Instruct` |
| `qwen2.5-7b-instruct` | `Qwen/Qwen2.5-7B-Instruct` |
| `gemma-2-9b-it` | `google/gemma-2-9b-it` |

All three use `AutoModelForCausalLM`, their own tokenizer chat templates, and
the exact model/tokenizer commits recorded in their configuration files.
Additional model profiles remain available for development but are outside the
thesis experiment.

Before any download, check whether a machine can hold a model:

```bash
python scripts/check_hardware.py --model qwen2.5-7b-instruct
```

This prints platform/python/torch/transformers, CUDA/MPS availability, VRAM/RAM,
and the selected device/dtype, then runs a **capacity gate**. These 7B-9B models
need roughly 20-24 GB on the accelerator in bf16, including headroom. If the machine
cannot fit it, the gate fails *early* with a clear message instead of
half-downloading tens of GB. The thesis model is never silently placed on CPU.

Every run selects a scoring backend explicitly:

- `--scorer real` (default): loads the model and reads its next-token logits.
  A failed load raises; it never falls back to the mock.
- `--scorer mock`: a deterministic, model-free scorer for CPU pipeline testing.

The backend is recorded as `scorer_backend` in run metadata, every raw record,
and saved prior files, so mock and real results can never be confused.

### Contextual option-token diagnostic

The A/B/C/D token ids are determined **contextually**, not by tokenizing the
letters in isolation. For each label the diagnostic appends the candidate to the
exact rendered prompt and requires that it adds exactly one token while leaving
the prompt tokenization unchanged (handling the `"A"` vs `" A"` whitespace case).
It fails loudly otherwise, and the result is saved under
`outputs/logs/token_diagnostics/`.

## Complete Preprocessing Stage

For a single validated run across all three datasets, follow
`RUN_PREPROCESSING.md` or run:

```bash
python scripts/run_preprocessing_all.py
```

Create one frozen artifact per dataset before starting model runs:

```bash
python scripts/preprocess_data.py --dataset medqa
python scripts/preprocess_data.py --dataset medmcqa
python scripts/preprocess_data.py --dataset mmlu
```

Each timestamped directory under `outputs/preprocessed/<dataset>/` contains:

- `samples.parquet`: the accepted canonical MCQs with source index and content hash;
- `cyclic_prompts.parquet`: all four mappings, displayed options, gold mapping,
  and canonical user prompt for every accepted sample;
- `excluded_rows.parquet`: quarantined source rows with one or more reason codes;
- `manifest.json`: requested and resolved dataset revision, split, Hugging Face
  fingerprint, preprocessing policy and hash, counts, sample-set hash, prompt
  version, and SHA-256 checksum for every artifact file.

`--start-index` and `--limit` are deliberately applied after filtering. A
20-question pilot therefore contains 20 usable questions rather than 20 raw
rows minus exclusions. Exact duplicate question/option content with different
IDs is retained and counted in the manifest, because MMLU intentionally overlaps
some subjects; duplicate sample IDs retain only their first occurrence.

Downstream commands may consume a frozen artifact explicitly:

```bash
python scripts/run_baseline.py \
  --model qwen2.5-7b-instruct \
  --dataset medqa \
  --preprocessed outputs/preprocessed/medqa/YOUR_ARTIFACT \
  --scorer mock
```

If `--preprocessed` is omitted, the exact same preprocessing policy runs inline.
Baseline, permutation, prior estimation, and PriDe commands all use this shared
loader. Transfer runs accept `--source-preprocessed` and
`--target-preprocessed`. Artifact checksums, dataset alias, and split are
verified before a run starts, and the complete preprocessing metadata is copied
into the run metadata.

### Guided Jupyter overview

- `notebooks/02_vorverarbeitung.ipynb` explains every preprocessing
  decision with a small, model-free example.
- `notebooks/03_vorverarbeitung_audit.ipynb` inspects the real MedQA,
  MedMCQA, and MMLU artifacts, verifies their checksums, summarizes exclusions,
  and displays the cyclic mappings.

Both notebooks stop at the handoff to model scoring. They do not calculate
selection-bias metrics or PriDe results.

## Unit Tests

The test suite is CPU-only and does not download datasets or language models:

```bash
cd analysis
source .venv/bin/activate
python -m pytest
```

It tests dataset normalization, permutation-safety filtering, artifact checksums and round trips, exact cyclic orders, duplicate-content handling, gold mapping, recall, RStd, permutation variance, PriDe normalization and correction, deterministic estimation sampling, and a mock end-to-end scorer.

## Token Diagnostic

Run this before renting time for a large experiment. It loads only the selected model and prints the rendered prompt ending, token IDs, decoded tokens, and validity of A/B/C/D:

```bash
python -m src.diagnostics.check_option_tokens --model qwen2.5-7b-instruct
```

The evaluation commands repeat this check automatically.

## Exact 20-Question Pilot

These commands exercise the full MedQA path without launching a full-dataset run:

```bash
cd analysis
source .venv/bin/activate

python scripts/run_baseline.py \
  --model qwen2.5-7b-instruct \
  --dataset medqa \
  --limit 20 \
  --batch-size 4

python scripts/run_permutations.py \
  --model qwen2.5-7b-instruct \
  --dataset medqa \
  --limit 20 \
  --batch-size 4

python scripts/estimate_prior.py \
  --model qwen2.5-7b-instruct \
  --dataset medqa \
  --limit 20 \
  --alpha 0.20 \
  --seed 42 \
  --batch-size 4 \
  --output outputs/priors/pilot_qwen2.5-7b-instruct_medqa_alpha-0.20.json

python scripts/run_pride.py \
  --model qwen2.5-7b-instruct \
  --dataset medqa \
  --limit 20 \
  --batch-size 4 \
  --prior outputs/priors/pilot_qwen2.5-7b-instruct_medqa_alpha-0.20.json
```

The pilot uses `alpha=0.20` so four of the 20 questions contribute to the prior. A full thesis run can use `alpha=0.05` or the other configured fractions.

## Full Run Examples

```bash
python scripts/run_baseline.py \
  --model qwen2.5-7b-instruct --dataset medqa

python scripts/run_permutations.py \
  --model qwen2.5-7b-instruct --dataset medqa

python scripts/estimate_prior.py \
  --model qwen2.5-7b-instruct --dataset medqa --alpha 0.05 --seed 42

python scripts/run_pride.py \
  --model qwen2.5-7b-instruct --dataset medqa \
  --prior outputs/priors/YOUR_PRIOR_FILE.json

python scripts/run_transfer.py \
  --model qwen2.5-7b-instruct --source medqa --target medmcqa --alpha 0.05
```

Run the reverse transfer by swapping `--source` and `--target`. A within-dataset transfer is represented explicitly when both aliases are the same.

Use `--start-index N` for a fixed slice. To resume an interrupted run, pass its raw run directory back with `--resume`:

```bash
python scripts/run_permutations.py \
  --model qwen2.5-7b-instruct --dataset medqa \
  --resume outputs/raw/permutations/EXISTING_RUN_DIRECTORY
```

MedQA and MMLU default to `test`; MedMCQA defaults to its labelled `validation`
split. `--split` remains available as an explicit override. Transfer runs resolve
the source and target defaults independently. Resume metadata is checked against
the model, dataset, split, seed, prompt version, alpha, and estimation IDs.
Completed `(sample_id, permutation_id)` pairs are skipped.

## Outputs

Raw records are flushed as partitioned Parquet files under `outputs/raw`. A small `debug_prompts.json` captures fully rendered prompts. Every run also records requested and resolved model/tokenizer revisions, dataset fingerprint, split, seed, dtype, device map, batch size, prompt version, alpha, estimation IDs, package versions, timestamp, and Git commit when available.

Large outputs are never silently overwritten. A new run receives a timestamped directory; an explicit `--output` path must not already exist.

CSV summaries include:

- accuracy;
- recall A/B/C/D and population RStd;
- prediction counts and frequencies as diagnostics;
- accuracy by cyclic position plus population variance, standard deviation, minimum, maximum, and range;
- in-domain-minus-default, transfer-minus-default, and
  transfer-minus-in-domain deltas for the held-out PriDe comparison once the
  Phase 2 summary implementation is complete.

For pooled and per-subject MMLU summaries:

```bash
python scripts/summarize_results.py \
  --input outputs/raw/baseline/YOUR_MMLU_RUN \
  --by-subject
```

## Frozen Methodological Decisions

The following choices were frozen on 2026-09-25 and are explicit in
`configs/experiments/thesis.yaml`:

1. **Alpha rounding:** `floor` with at least one estimation sample. The paper specifies `K = alpha * |D|` but does not define integer rounding.
2. **Sampling population order:** sample IDs are sorted before seeded sampling. This makes selection stable if dataset iteration order changes while IDs remain fixed.
3. **Numerical floor:** probabilities are clipped at `1e-12` only for logarithms and prior division.
4. **Estimation-sample policy:** `cyclic`, matching Algorithm 1. Estimation samples receive the cyclic-debiased prediction; remaining samples receive global-prior correction.
5. **Revisions:** the three thesis model/tokenizer profiles use exact commits. Dataset manifests store exact resolved revisions and fingerprints.
6. **MedQA question construction:** non-empty `sent1` and `sent2` are joined with a newline. The current dataset revision has the question in `sent1` and an empty `sent2`.
7. **Option token form:** the exact A/B/C/D token representation is determined
   contextually for each model. The diagnostic tests both bare and leading-space
   forms and stops the run unless all four labels are distinct single tokens.
8. **First-token construct validity:** this framework intentionally measures the instructed first-token A/B/C/D distribution. The token diagnostic proves representational comparability, but it does not claim that first-token probabilities always agree with later free-text generations. Generated-answer agreement would be a separate experiment.
9. **Permutation-safety filtering:** relative-position answers, displayed-ID
   combinations, and duplicate option texts are excluded by default. Every
   exclusion remains inspectable in the frozen artifact. This population must
   be kept identical for baseline, permutation, and PriDe comparisons.

The complete scientific specification and change-control rule are recorded in
`../DESIGN_FREEZE_2026-09-25.md`. Direct transfer contrasts, corrected cyclic
PriDe metrics, paired bootstrap intervals, and the equal-budget analysis are
frozen requirements for Phase 2; their presence in the configuration does not
yet imply that all summary functions are implemented.

The identical-content estimator remains disabled and lives in a separate namespace. It must be described as an experimental extension, never as standard PriDe.


## LRZ-Ausführung

Aktueller Workflow: [LRZ Setup, Mock, GPU-Smoke und Batch-Pipelines](scripts/slurm/README.md). Modellläufe erfolgen über Slurm; Notebooks 01–05 bleiben kompakte Methodik-Dokumentation. `scripts/run_pipeline.py` verwendet die fertigen Preprocessing-Artefakte und `scripts/aggregate_results.py` erzeugt Vergleichstabellen.

### Analysis notebooks (no model execution)

- [06 · Selection Bias, PriDe und LRZ](notebooks/06_selection_bias_pride_und_lrz.ipynb): synthetic first-token and PriDe examples, experiment configuration and Slurm workflow.
- [07 · Ergebnisse und Vergleich](notebooks/07_ergebnisse_und_vergleich.ipynb): explicit selection of saved pipeline summaries, metric tables, plots and optional CSV/PNG/PDF export. Empty by default until pipeline IDs are selected. Mock and limited runs require explicit settings.

These notebooks perform no model downloads, inference or job submissions. Install the `notebooks` extra for plotting and notebook execution.

### Zheng et al. reference benchmarks

The pinned original MMLU, ARC-Challenge and CommonsenseQA files are downloaded
separately under `data/reference/zheng2024`. Reproduce the download and audit with
`python scripts/download_zheng2024.py`. See
[comparison and research-question scope](docs/zheng2024_replikation.md).
These reference JSONL files are not yet pipeline artifacts; CSQA needs five-option
support. Existing preprocessing outputs remain unchanged.

### Original Zheng selection-bias notebook and runner

[08 · Selection Bias nach Zheng](notebooks/08_zheng_selection_bias_original.ipynb)
uses the pinned upstream open-model prompt/scoring functions, including both
space/no-space label tokens, length-normalized no-ID scoring, answer-moving,
ID shuffling and 0-/5-shot settings. This separate runner handles CSQA's five
options. It does not extend the existing PriDe pipeline to five options.

Install `python -m pip install -e '.[notebooks,reference]'`. On LRZ,
`MODEL=qwen2.5-7b-instruct DATASET=arc LIMIT=2 bash scripts/slurm/submit.sh zheng`
submits a small reference test after selecting `PARTITION` and `MCQ_CONDA_ENV`.
The original complete condition matrix is expensive (24 permutations for MMLU/ARC).
No new GPU run has been validated yet; start with the small test.
