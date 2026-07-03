# EXECUTION PLAN — Disaster Tweet Classification: Controlled Re-Evaluation

Master roadmap. Each numbered step below becomes one focused Claude Code
prompt (CONTEXT / HARD RULES / TASKS / ACCEPTANCE CRITERIA), issued one at a
time, gated on actual artifacts before advancing. Nothing in the eventual
paper may cite a number that does not trace to a run ID in the results
ledger (see 0.4).

## Contribution statement (draft — every bracketed claim is conditional)

We present a controlled re-evaluation of text-classification
architectures for disaster tweet detection. Starting from a widely
replicated benchmark setup, we show that common evaluation practices —
selection-contaminated splits, duplicate leakage, absent early stopping,
inconsistent per-model preprocessing, and single-seed point estimates —
[NEEDS EXPERIMENT: materially alter / do not materially alter]
architecture rankings. Under a corrected protocol (deduplicated
stratified splits, per-model early stopping, matched inputs, N seeds,
significance testing), we benchmark four model families (bag-of-words,
from-scratch neural, frozen sentence-encoder transfer, fine-tuned
transformer) on two disaster-tweet datasets, measure cross-dataset
generalization, and quantify the effect of label noise in the widely
used Kaggle/CrowdFlower dataset. [NEEDS EXPERIMENT: 1–2 sentence summary
of actual findings.]

This is a rigorous-benchmark/methodology paper, not a novel-method paper.
Do not oversell it anywhere.

## PHASE 0 — Rebuild: repo, data, harness (~1 week)

### 0.1 Fresh repo scaffold.
New repository (do not carry over course-derived code; the old repo stays
up as-is or is archived — decision: keep separate so the paper's code is
100% original). Layout: `src/` package (data, models, train, eval,
analysis), `configs/` (YAML per experiment), `scripts/` (thin CLI
entrypoints), `tests/`, `results/` (gitignored except ledger), `paper/`.
Python 3.11, uv with committed lockfile. No pipeline logic in notebooks;
notebooks allowed only in `notebooks/` for figure generation, reading from
saved results.

Gate: tree listing + lockfile + CI stub (lint + tests) pasted back.

### 0.2 Data pipeline — Kaggle dataset.

Load `train.csv` (7,613 rows).
Duplicate handling (documented policy):
a. Exact-text duplicate rows with consistent labels → keep one.
b. The 18 conflicting-label groups → drop entire groups; log dropped IDs
to `data/dropped_conflicting.csv`. Rationale: no principled
adjudication available; removal is conservative and reported in the
paper.
Stratified 80/10/10 train/val/test split, fixed seed, saved to disk as
three CSVs with SHA256 hashes recorded. Test set is frozen from this
moment: no experiment reads it until final evaluation runs (Phase 3
discipline, enforced by code path separation).
Assertions/tests: no text overlap across splits (exact match), class
balance within tolerance per split, row counts.
Gate: split stats table + leakage-check test output.

### 0.3 Data pipeline — CrisisLex T6.

FIRST: verify availability of tweet text (not just IDs) in the
distributed CSVs, and the license terms. [RISK: if text is unavailable,
hydration via the Twitter/X API is no longer realistic. Fallbacks, in
order: HumAID, CrisisNLP resources, original CrowdFlower "Disasters on
Social Media" release. Resolve before writing any more code.]
Map labels: on-topic/related → 1, off-topic/unrelated → 0. Document the
mapping and any semantic mismatch vs. the Kaggle labels in a
`data/DATASETS.md` (this becomes the paper's Data section seed).
Same dedup + stratified 80/10/10 + frozen-test discipline. If T6 is much
larger (~60k), optionally subsample to a documented size for training
parity experiments, but keep the full test split.
Gate: DATASETS.md + split stats for both datasets.

### 0.4 Run harness + results ledger.

Every run: config snapshot, git commit hash, seed, dataset+split hashes,
wall time, per-epoch logs, final metrics, and per-example test/val
predictions saved to disk (required for McNemar's and error analysis
later).
`results/ledger.jsonl`: one line per run, unique run ID. Paper rule:
every number cites a run ID.
Metrics module (unit-tested): accuracy, macro-F1, positive-class F1,
per-class precision/recall, confusion matrix. Weighted-F1 computed only
for comparison against the original project's reporting.
Floor baselines: majority-class and stratified-random, run and ledgered.
Gate: passing tests + a dummy end-to-end run appearing in the ledger.

## PHASE 1 — Controlled core benchmark (~1.5–2 weeks)

### 1.1 Preprocessing protocol (design decision, implemented once).

All models receive the same cleaned text. Cleaning is minimal and
uniform: define once (e.g., NFC normalize, collapse whitespace; decide
and document URL/mention handling — recommendation: keep tokens,
document the choice; ablate later only if time permits).
Sequence length for token-based models: cover the ~95th percentile of
training-tweet token counts (not the mean — the original's mean-length
truncation cut ~half of tweets). Record the actual value chosen.
Tokenization is necessarily model-family-specific (TF-IDF vocab, Keras
vectorizer, USE raw string, DistilBERT WordPiece) — that difference is
inherent to the architectures and is documented as such, unlike
truncation policy, which is not inherent and must be matched.

### 1.2 Model implementations (own code, from specs not from the old repo).
Families: (a) TF-IDF + MultinomialNB and TF-IDF + LogisticRegression;
(b) mean-pooled embedding dense net; (c) LSTM, GRU, BiLSTM, Conv1D;
(d) frozen USE + dense head; (e) fine-tuned DistilBERT.

Every neural model: early stopping on val loss (patience ~3, restore
best weights), max-epoch cap high enough that stopping actually binds.
Light documented tuning budget per model on val only: learning rate ×
one capacity knob, small grid (≤6 configs). Search space goes in the
paper's appendix.
Gate per model family: training log showing early stopping engaged,
ledger entries, val metrics. Advance family by family.

### 1.3 Protocol contrast experiment (the paper's centerpiece).

Protocol A ("replicated original"): 90/10 non-stratified split, no
dedup, mean-length truncation, no early stopping, fixed 5 epochs,
single seed, weighted F1, evaluation on the selection set.
Protocol B ("controlled"): everything from 0.2 + 1.1 + 1.2, 5 seeds per
neural model (mean ± std), final metrics on the frozen test set.
Deliverable: side-by-side ranking table. The delta (or its absence) IS
the headline finding. [NEEDS EXPERIMENT — do not pre-commit to the
outcome in any draft text.]
Gate: both tables from ledger, plus seed-variance table.

### 1.4 Data-efficiency curves.

Fractions {1%, 5%, 10%, 25%, 50%, 100%} of train, stratified, ×
representative model per family × 3–5 seeds. Val for stopping, test for
reporting.
Gate: curve data in ledger + one draft figure.

## PHASE 2 — Cross-dataset generalization (~1 week)

2.1 Train on Kaggle → evaluate on CrisisLex test; train on CrisisLex →
evaluate on Kaggle test. Per model family, best config from 1.2, 3–5
seeds. In-domain vs. out-of-domain deltas per family.
2.2 Brief qualitative pass: what breaks cross-dataset (vocabulary
shift? event-specific terms? label-definition mismatch?). Feeds error
analysis.

Gate: 2×2 (train×test) results matrix per family, from ledger.

## PHASE 3 — Statistics + error analysis (~1 week)

### 3.1 Significance testing.

McNemar's test between top model pairs on the same frozen test set
(using saved per-example predictions).
Bootstrap 95% CIs (≥1,000 resamples) on test accuracy/macro-F1 for all
models.
State the multiple-comparisons stance explicitly (e.g., Holm correction
over the tested pairs, or report raw p-values with the caveat — decide
and document).

### 3.2 Label-noise quantification.

Train/eval with vs. without the dedup+conflict-drop policy (Protocol B
otherwise identical). Quantifies how much the known noise moves scores.

### 3.3 Error analysis.

Sample ≥100 test errors (FP and FN) from the top 2–3 models.
Categorize manually (you, not the model — this is the human-labeled part
that makes it defensible): figurative/metaphorical disaster language,
news-style vs. personal report, sarcasm/jokes, ambiguous gold label,
URL-only/low-content, other.
Report per-category error rates and whether the top models fail on the
same examples (overlap of error sets — pairs naturally with McNemar's).
Gate: annotated error CSV + category table + 3–5 exemplar tweets per
category for the paper.

### 3.4 Figures + final results freeze.
Main results table (Protocol B, both datasets, mean ± std, CIs),
protocol-contrast table, data-efficiency figure, cross-dataset matrix,
error-taxonomy table. After this step the ledger is frozen; the paper
cites only frozen runs.

## PHASE 4 — Paper

(Step 4 of the overall plan; separate detailed plan
when Phase 3 gates pass). Target: standard *ACL-style LaTeX, sections as
agreed, real Limitations (two datasets, English-only, dataset-specific
tuning budget, small eval sets → wide CIs where true, no SOTA claim).

## Explicitly out of scope (say so in the paper's Limitations)

Novel architectures; LLM few-shot suites; multilingual; >2 datasets;
large hyperparameter searches; absolute-SOTA chasing; human re-annotation
of gold labels beyond the error-analysis sample.

## Standing rules for every Claude Code prompt in this plan

No number is reported outside the ledger; no ledger entry without a
config + commit hash.
The frozen test sets are read by evaluation code paths only, never by
training/selection code. A test asserting this stays green in CI.
Any deviation from this plan gets written into a `DECISIONS.md` with a
one-line rationale (these become paper/interview material).
If a step's acceptance criteria can't be met honestly, the step stops
and the blocker comes back to chat — no silent workarounds.
