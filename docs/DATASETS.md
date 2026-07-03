# DATASETS.md

Seed for the paper's eventual Data section. Plain, precise, no marketing.
Numbers below are taken directly from `data/kaggle/manifest.json` and
`data/crisislex/manifest.json`, produced by `scripts/prepare_kaggle.py` and
`scripts/prepare_crisislex.py` respectively (seed=42 for both).

## 1. Kaggle "NLP Getting Started" ("Disaster Tweets")

- **Source**: Kaggle competition *Natural Language Processing with Disaster
  Tweets* (`train.csv`, 7,613 labeled rows). Originally derived from a
  CrowdFlower/Figure Eight "Disasters on Social Media" crowd-labeling
  exercise.
- **License**: [UNCLEAR — needs clarification]. Kaggle competition data is
  typically subject to that competition's specific rules page and Kaggle's
  general terms of use, neither of which was fetched/verified in this
  session (raw file was obtained locally, not from Kaggle directly — see
  `docs/DECISIONS.md`). Must be confirmed against the competition's actual
  rules page before any public release of derived data or the paper.
- **Columns**: `id`, `keyword` (optional), `location` (optional), `text`,
  `target` (1 = real disaster, 0 = not a disaster).
- **Label semantics**: `target` is a crowd-worker judgment of whether the
  tweet is actually describing/reporting a real disaster event, evaluated
  tweet-by-tweet without reference to any single named crisis — i.e., a
  general "is this tweet about an actual disaster" judgment, not "is this
  tweet relevant to crisis X."
- **Raw file**: `data/kaggle/raw/train.csv`, SHA256
  `61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df`,
  7,613 rows.

## 2. CrisisLex T6

- **Source**: CrisisLexT6 v1.0 (Olteanu, Castillo, Diaz, Vieweg, *"CrisisLex:
  A Lexicon for Collecting and Filtering Microblogged Communications in
  Crises"*, ICWSM 2014). Distributed as one CSV per crisis event, ~10,000
  tweets each, "50% from the geo-based sample, and 50% from the
  keywords-based sample" (per the distributed `README.md`).
- **License**: [UNCLEAR — needs clarification]. The distributed
  `CrisisLexT6/README.md` states no explicit license (no SPDX identifier,
  no "you may use this under license X" text) — it only names the paper to
  cite and gives contact emails "for inquiries." This is a real gap: before
  any public release of this project's derived CrisisLex splits or the
  paper itself, licensing terms must be confirmed directly with the
  CrisisLex maintainers or via crisislex.org's terms, not assumed from the
  README's silence.
- **Events** (6, each a separate raw CSV under `data/crisislex/raw/`,
  preserved as an `event` column through every split for later cross-event
  analysis):
  - `2012_Sandy_Hurricane`
  - `2013_Alberta_Floods`
  - `2013_Boston_Bombings`
  - `2013_Oklahoma_Tornado`
  - `2013_Queensland_Floods`
  - `2013_West_Texas_Explosion`
- **Raw file format** (verified directly from the distributed files, not
  assumed): header `tweet id, tweet, label`, comma-separated with quoted
  fields — tweet text may contain embedded commas and newlines, which
  breaks naive line-splitting; a proper CSV parser (`pandas.read_csv` /
  `csv.reader`) is required. The `tweet id` field's cell values are wrapped
  in literal single quotes as part of the string content (e.g.
  `"'325208201740029952'"`), not CSV quoting — these are stripped on load
  by `dtc.data.crisislex.load_event_csv`.
- **Label semantics**: `label` is `on-topic` or `off-topic`, a judgment of
  whether the tweet is *relevant to that specific named crisis event*
  (e.g., mentions or clearly concerns Hurricane Sandy), independent of
  whether the tweet itself is "about a disaster happening" in the abstract.
  Mapped in this project as `on-topic -> 1`, `off-topic -> 0`
  (`dtc.data.crisislex.LABEL_MAP`).
- **Documented label-semantics mismatch vs. Kaggle**: CrisisLex's
  "on-topic" is event-specific relevance (could include aftermath
  commentary, jokes, or discussion clearly tied to the named event, as long
  as it's topically about that crisis), whereas Kaggle's "real disaster" is
  a general judgment applied to any tweet without anchoring to one specific
  event. A tweet judged "on-topic" for a specific CrisisLex event is not
  automatically equivalent to a tweet a CrowdFlower worker would have
  labeled `target=1` under the Kaggle task's instructions, and vice versa.
  This mismatch is a first-class caveat for any cross-dataset
  generalization claim in Phase 2, not a footnote.
- **Raw files** (`data/crisislex/raw/`, each `{event}-ontopic_offtopic.csv`):
  total 60,082 rows across the 6 events, individual SHA256 hashes recorded
  in `data/crisislex/manifest.json`.

## 3. Dedup policy (identical for both datasets; `dtc.data.common`)

- Exact-text duplicate rows with a single, consistent label -> keep the
  first occurrence only.
- Exact-text duplicate rows with conflicting labels across the group ->
  drop the ENTIRE group (no principled adjudication available; removal is
  conservative and is reported, not hidden — see each dataset's
  `dropped_conflicting.csv`).

## 4. Split procedure (identical for both datasets; `dtc.data.common`)

Stratified 80/10/10 train/val/test split via two sequential
`sklearn.model_selection.train_test_split` calls stratified on the binary
label, fixed `seed=42`. The test split is frozen the moment it is written:
by convention (and enforced by `dtc.eval.frozen_test_loader`, see
`docs/PLAN.md` "Standing rules"), no training or model-selection code may
read it; only `dtc.eval.*` / `scripts/evaluate_*.py` entrypoints may.

## 5. Measured numbers (as run, seed=42 — not assumed, not rounded)

### Kaggle

| stage | rows |
|---|---:|
| raw | 7,613 |
| exact-duplicate rows (any position after first occurrence, pre-dedup) | 110 |
| conflicting-label groups dropped | 18 (55 rows) |
| deduped total | 7,485 |
| train (80%) | 5,988 |
| val (10%) | 748 |
| test (10%, frozen) | 749 |

Class balance (post-dedup): train positive rate 0.4259, val 0.4265, test
0.4259 — within 0.1 percentage points of each other by construction
(stratified).

### CrisisLex T6

| stage | rows |
|---|---:|
| raw (6 events combined) | 60,082 |
| exact-duplicate rows (any position after first occurrence, pre-dedup) | 4,705 |
| conflicting-label groups dropped | 101 (533 rows) |
| deduped total | 55,276 |
| train (80%) | 44,220 |
| val (10%) | 5,528 |
| test (10%, frozen) | 5,528 |

Class balance (post-dedup): train positive rate 0.5032112, val 0.5030753,
test 0.5032562 — full per-event breakdown for every split is recorded in
`data/crisislex/manifest.json`.

The much higher duplicate/conflict rate in CrisisLex relative to Kaggle
(4,705 vs. 110 exact-duplicate rows; 101 vs. 18 conflicting groups) is a
measured fact reported here as-is — a plausible mechanism is that generic,
crisis-unrelated ("off-topic") tweets recur across the keyword/geo sampling
procedure both within and across the 6 events, but this project does not
assert that mechanism as confirmed, only reports the counts.