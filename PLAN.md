# Plan: Model Selection at a Tenth of the Cost

**Written:** 2026-09-06. **Status:** plan only, nothing built.
**Build window:** 4 weeks, 2026-11-02 to 2026-11-29. Standard practitioner version; a
publishable version is a deferred extension (section 12).
**Feeds:** the AI Release Gate (project 03) takes this project's item bank, reliability
estimates and power function as its statistical core from December 2026.

---

## 1. What this produces

"Which AI model should we use, and did the new version get worse?" answered with
statistical confidence using about a tenth of the evaluation calls a full benchmark
needs. Item response theory, the psychometrics behind every standardised test, applied
to language-model benchmarks: calibrate the items once, then test each new model
adaptively on the items that discriminate at its level, and stop when the confidence
interval is tight enough to make the decision.

The numbers a stranger can check:

| Measure | What it shows |
|---|---|
| **Items used vs Kendall's tau against the full-suite ranking**, with bootstrap CI | The headline curve. The claim "same ranking at ~10% of items" is either on this chart or it is not |
| Items needed to separate two models whose full-suite scores differ by 1, 3 and 5 points, at 80% power | The number a platform team actually needs |
| Cost per ranking decision, full suite vs adaptive, in dollars, on the own-run panel | The tenth-of-the-cost claim in money |
| Count and list of items with near-zero discrimination, misfit, or mis-keyed answers, with evidence | What the benchmark was never measuring |
| Test-retest reliability per model (same items, two runs) | How much of a reported score is noise |
| Position-bias and framing effect sizes per model | What a bare accuracy hides |
| Q3 local-dependence statistic and dimensionality diagnostics per benchmark | The honest limitation, measured rather than hand-waved |

Everything is reported with an interval. A bare number is a bug.

## 2. Scope and boundaries

In scope:

- A response matrix (models by items, correct or not) assembled from **public per-item
  results** for the bulk of the panel, plus **own runs** for a validation set of current
  models.
- 2PL and 3PL item response models fitted to the matrix, with item-fit statistics,
  Yen's Q3 for local dependence, a dimensionality check, and differential item
  functioning across model families.
- A broken-item report: near-zero or negative discrimination, misfit, suspected
  mis-keyed answers, and contamination signatures (items far easier for models released
  after the item's publication than their ability predicts).
- Computerised adaptive testing: maximum Fisher information item selection, EAP ability
  estimation, stopping rules for a target standard error and for pairwise separation.
- A leave-one-model-out simulation that produces the headline curve, against random and
  stratified subsampling baselines.
- Three small measurement experiments on the own-run panel: test-retest reliability,
  position bias in multiple choice, prompt-framing effects.
- A typed package with the estimator, the item bank as data, and a power function
  `items_needed(effect, power, ability)` that project 03 imports.
- A practitioner write-up: "your benchmark is measuring fewer things than you think, and
  here is how to run it for a tenth of the price".

Out of scope, on purpose:

- Open-ended generation quality, judges, human preference. Binary-scored items only.
- New benchmark content. This calibrates existing items; it does not write them.
- Multidimensional IRT as the main model. A dimensionality check is in; a full
  multidimensional fit is the deferred publishable extension.
- Frontier-tier models across the full suite. One frontier snapshot per vendor on the
  adaptive subset only (section 7).
- A hosted service. Deliverable is a package, a results table and a write-up.

## 3. Data

### 3.1 Public per-item results (the free two thirds of the panel)

| Source | What it gives | Notes |
|---|---|---|
| Open LLM Leaderboard "details" datasets on Hugging Face | Per-sample outputs and correctness for hundreds of models on MMLU-Pro, GPQA, MATH (levels 5), BBH, IFEval, MuSR | The leaderboard itself was frozen in 2025 but the per-sample datasets remain published. **Verify availability and format in week 1** before anything depends on it |
| HELM per-instance predictions | Per-instance results across many scenarios and models | Downloadable from the HELM release buckets; formats differ by release. Use one release consistently |
| Own runs | Correctness per item for the validation panel | Section 3.3 |

Target response matrix: at least 100 models by 3,000 to 5,000 items across four
benchmarks that are binary-scored and in the public per-item data (MMLU-Pro and GPQA for
multiple choice, MATH for exact-match free response, BBH for mixed). Items are keyed by a
content hash of the prompt so the same item is the same row across sources.

Model metadata per row: family, provider, parameter count where known, release month,
open or API. Release month drives the contamination analysis; family drives the
differential item functioning analysis.

### 3.2 Item bank construction

- Deduplicate items across sources by content hash.
- Keep an item only if at least 40 models have a recorded response to it.
- Record the answer key and its provenance, since mis-keyed answers are one of the
  things to find.
- Freeze the bank as versioned Parquet with a content hash, exactly as the drift suite in
  project 03 is frozen. Item parameters are stored next to the bank with the fit version.

### 3.3 Own-run validation panel

The public matrix calibrates the items. The own-run panel tests the claim on models the
calibration never saw. Eight to ten current models:

| Tier | Models | Purpose |
|---|---|---|
| Anthropic | Haiku 4.5, Sonnet 5 full suite; Opus 5 on the adaptive subset only | Two full-suite anchors, one frontier check |
| OpenAI | One mid-tier full suite, one frontier on the adaptive subset | Same shape |
| Google | One mid-tier full suite, one frontier on the adaptive subset | Same shape |
| Open weights via a provider | Two mid-size models, full suite | Cheap full-suite anchors, provider-served |
| Local | Two or three small models (3B to 8B, 4-bit) on the laptop | Free; they also spread the ability range downward, which item calibration needs |

Run settings: temperature 0, fixed system prompt, answer-only output format (a letter for
multiple choice, a boxed final answer for MATH), `max_tokens` small. Development caching
is on: every response is cached by content hash of the request so a rerun costs nothing.
This is the opposite of the drift-run rule in project 03, and deliberately so; here the
question is about the items, not about whether the vendor changed.

Anthropic calls go through the Message Batches API at half price where latency does not
matter, which is everywhere in this project.

## 4. Methods

### 4.1 Item response models

- **2PL**: probability of a correct response as a logistic function of model ability
  minus item difficulty, scaled by item discrimination. The workhorse.
- **3PL**: adds a lower asymptote for guessing. Fitted for multiple-choice items only;
  for free-response items the guessing parameter is not identified and is fixed at zero.
- **Fitting**: marginal maximum likelihood via `py-irt` (variational, PyTorch) for speed
  on the full matrix; a Bayesian fit in PyMC on a stratified subset of 500 items to get
  posterior intervals on item parameters and to check that the two agree. Priors are
  weakly informative and stated.
- **Item fit**: infit and outfit mean squares, and S-X2 where sample size allows. Items
  flagged when discrimination is below 0.3, when fit statistics fall outside the usual
  bounds, or when the empirical response curve is non-monotone (the mis-keyed signature).
- **Local dependence**: Yen's Q3 on residual correlations, per benchmark and across
  benchmarks. Report the distribution and the pairs above 0.2, do not hide it.
- **Dimensionality**: eigenvalues of the tetrachoric correlation matrix, parallel
  analysis, and a comparison of fit between a single-ability model and separate abilities
  per benchmark. If the benchmarks are not one dimension, the composite ability is
  reported alongside per-benchmark abilities, and the write-up says so.
- **Differential item functioning**: Mantel-Haenszel and a logistic-regression DIF test
  by open versus API family, and by release month before versus after the item's
  publication. Items with large DIF by release month are the contamination candidates.

### 4.2 Adaptive testing

- **Ability estimation**: expected a posteriori with a standard normal prior; maximum
  likelihood as a check once ten or more items are answered.
- **Item selection**: maximum Fisher information at the current ability estimate, with
  content balancing so each benchmark contributes items in proportion to its weight in
  the full suite. Randomesque selection among the top five to avoid over-using the same
  items across models.
- **Stopping rules**: (a) posterior standard error below a threshold set from the
  target precision; (b) pairwise: stop when the two models' ability intervals stop
  overlapping, or when a maximum item count is hit and the pair is declared
  indistinguishable at that budget.
- **Simulation**: leave-one-model-out over the public matrix. For each held-out model,
  refit item parameters without it, run the adaptive test against its recorded
  responses, and record ability and standard error at every item count. Rank all
  held-out models at each item count and compute Kendall's tau against the full-suite
  ranking. Bootstrap over models for the interval. Repeat with random subsampling and
  with stratified subsampling at the same item counts. That comparison is the headline
  chart.
- **Validation on unseen models**: run the adaptive test live against the own-run panel
  using item parameters from the public matrix only, and compare the adaptive ranking to
  the own-run full-suite ranking. This is the honest test of the claim, since these
  models never touched the calibration.

### 4.3 Measurement experiments (own-run panel)

| Experiment | Design | Output |
|---|---|---|
| Test-retest | 500 items, two runs a day apart at temperature 0, six models | Agreement rate and phi coefficient per model; how much of the score is noise |
| Position bias | 300 multiple-choice items, four cyclic permutations of the option order, six models | Accuracy by correct-option position; share of items whose outcome depends on position; per-model bias index |
| Framing | 300 items, three prompt templates (plain, letter-only instruction, brief-reasoning-then-answer), six models | Variance decomposition of correctness into item, template and item-by-template; the reasoning template also measures how much accuracy the answer-only format costs |

Each experiment is small by design and has its own cost line in section 7.

### 4.4 Power function for project 03

`items_needed(effect, power, ability)` returns how many items, selected adaptively at a
given ability level, are needed to detect a drop of `effect` points in full-suite
accuracy at the requested power. Derived from the information function of the calibrated
bank and validated against the simulation. Project 03 calls this to answer "how many eval
items do you actually need to detect a three-point regression" and to set its own suite
sizes. The item bank, parameters, estimator and this function are the deliverable
interface; they ship as a package with a semantic version.

## 5. Package design

```
mselect/
  data/          public per-item loaders (OLL details, HELM), own-run cache, item bank build
  bank/          frozen item bank, parameters, versions, content hashes
  irt/           twopl, threepl, fit (py-irt and PyMC), fit_stats, q3, dimensionality, dif
  cat/           estimate (EAP, MLE), select (Fisher info, content balance), stop, simulate
  experiments/   retest, position, framing
  runner/        vendor calls (raw HTTP per vendor, Anthropic via the batch endpoint), cache
  power.py       items_needed(effect, power, ability)
  report/        tables and charts for the README and write-up
  cli.py         typer CLI: mselect bank build, mselect fit, mselect simulate, mselect run, mselect report
tests/           synthetic-data tests: parameter recovery, CAT convergence, grader fixtures
docs/
  items-that-measure-nothing.md   the broken-item report, with evidence per item
  rejected.md                     Rule C
  writeup.md                      the practitioner post, drafted here
```

Tests that matter: 2PL and 3PL recover known parameters on simulated matrices within
tolerance; the adaptive estimator converges to the true ability on simulated responses
with the expected standard error; Q3 flags planted dependent item pairs; DIF flags planted
group differences; answer parsers survive markdown, whitespace, and "The answer is (B)".

Stack: Python 3.13, `uv`, `polars`, `py-irt`, `pymc`, `numpy`, `scipy`, `httpx`,
`typer`, `matplotlib`. Typed, `ruff` and `mypy --strict` clean.

## 6. Week by week

| Week | Dates | Build | Done when |
|---|---|---|---|
| 1 | Nov 2 - 8 | Repo scaffold; public per-item loaders; response matrix for four benchmarks; item bank v1 frozen; dimensionality check; first 2PL fit; item parameter plots | Matrix of 100+ models by 3,000+ items; 2PL parameters with a sanity check against classical item difficulty |
| 2 | Nov 9 - 15 | 3PL for multiple choice; fit statistics; Q3; DIF by family and release month; broken-item report drafted; own-run harness with cache and batch calls; full-suite runs on the own-run panel started | `docs/items-that-measure-nothing.md` with evidence; own-run matrix filling |
| 3 | Nov 16 - 22 | CAT estimator, selection, stopping; leave-one-model-out simulation; random and stratified baselines; headline curve; live adaptive runs against the own-run panel; cost-per-decision table | Headline chart with intervals; validation on unseen models reported |
| 4 | Nov 23 - 29 | Three experiments; power function and its validation; README to Rule A shape; `docs/rejected.md`; write-up; package tagged and handed to project 03 | Definition of done all checked; 03 can `import mselect` |

Slack: the Bayesian fit in week 2 and the framing experiment in week 4 are the first
things to drop if behind. Neither is in the definition of done.

## 7. Cost

Anthropic list prices as of the plan date (per million tokens): Haiku 4.5 at $1 in and $5
out, Sonnet 5 at $2 in and $10 out, Opus 5 at $5 in and $25 out; the batch endpoint
halves these. Other vendors are assumed to sit in similar tiers and are checked against
their price pages before the first run. Answer-only format keeps outputs near 40 tokens.

| Line | Calls | Tokens (in / out) | Estimate |
|---|---:|---|---:|
| Own-run full suite, 6 mid-tier API models, 3,000 items | 18,000 | 11M / 0.7M | ~US$25 (Anthropic share ~US$4 with batching) |
| Frontier check, 3 models, adaptive subset of 300 items | 900 | 0.6M / 0.04M | ~US$5 |
| Test-retest, 6 models, 500 items, second run | 3,000 | 1.8M / 0.1M | ~US$4 |
| Position bias, 6 models, 300 items, 3 extra permutations | 5,400 | 3.2M / 0.2M | ~US$7 |
| Framing, 6 models, 300 items, 2 extra templates (one with ~200-token reasoning) | 3,600 | 2.2M / 0.4M | ~US$8 |
| Development and re-runs not covered by cache | | | ~US$10 |
| **Total** | | | **~US$60, about CA$80** |

Against the CA$220 line that leaves about CA$140. Spend it, in order, on: (1) widening the
own-run panel to ten full-suite models, which strengthens the validation; (2) a second
test-retest run a week later; (3) nothing. Record the actual invoice in the portfolio's
STATUS next to the estimate. Hard spend caps in every vendor console before the first
call.

Local models cost electricity. The laptop's 4 GB card runs 3B to 4B models at 4-bit;
anything larger runs on CPU overnight or is dropped.

## 8. Handover to project 03

Delivered by Nov 29 as `mselect` v0.1.0:

- The frozen item bank with 2PL and 3PL parameters and fit flags.
- `cat.estimate`, `cat.select`, `cat.stop` as a typed API.
- `power.items_needed`.
- Reliability figures per benchmark from the retest experiment.
- A note on which item blocks are locally dependent, so 03 does not treat them as
  independent evidence in its confidence intervals.

Project 03's drift suite v1 is frozen before this exists and is not changed. Suite v2, if
ever, picks its items from this bank by discrimination.

## 9. Risks

| Risk | Handling |
|---|---|
| Public per-item datasets moved, renamed, or in an awkward format | Week 1 verifies before anything depends on them. Fallback: own runs on a 1,000-item bank across the open-weights and local models, a smaller but still real matrix |
| Local independence violated | Measured with Q3 and reported. Robustness check: refit per benchmark and compare rankings. The write-up treats this as the interesting section, not a footnote |
| Benchmarks are not one dimension | Dimensionality check in week 1; report composite and per-benchmark abilities. Full multidimensional IRT is deferred (section 12) |
| Panel ability range too narrow for stable item parameters | The public matrix includes small and old models; local small models extend the low end |
| Answer parsing errors masquerade as item misfit | Parsers tested against adversarial fixtures; a sample of flagged items is hand-checked before it goes in the broken-item report |
| The tenth-of-the-cost ratio is not reached | Report the ratio actually achieved. A defensible seventh is worth more than an indefensible tenth |
| Scope creep toward a paper | Deferred by decision. Section 12 lists what is kept so the door stays open at no cost |
| November has four weekends and a holiday | The 03 drift job is already running by then and needs nothing. Week 4 is the only tight one |

## 10. Rule C candidates: what is expected not to work

Whichever produces the clearest evidence gets `docs/rejected.md`:

1. **Random 10% subsampling.** Expected: a ranking close to the full suite on average
   but with wide variance across draws and no principled stopping rule, so it cannot say
   when two models are separated. The headline chart shows the gap.
2. **Classical item statistics instead of IRT.** Ranking items by proportion correct and
   choosing "medium difficulty" items. Expected: ignores discrimination, keeps items that
   every model of a given size gets right or wrong for reasons unrelated to ability.
3. **3PL on free-response items.** Expected: the guessing parameter is unidentified and
   the fit degrades; the plan fixes it at zero for those items, and this documents why.

## 11. Definition of done

Mirrors the portfolio's definition for this project:

- [ ] IRT parameters fitted and published for a real item bank (100+ models, 3,000+ items)
- [ ] Adaptive selection reproduces the full-suite ranking at about 10% of items, Kendall's tau with CI reported, against random and stratified baselines
- [ ] Validation on the own-run panel of models the calibration never saw
- [ ] Position bias, framing effects and test-retest reliability each measured with intervals
- [ ] A list of items that measure nothing, with evidence per item
- [ ] Q3 and dimensionality diagnostics reported
- [ ] `items_needed` power function validated against the simulation and handed to project 03
- [ ] Cost per ranking decision reported in dollars
- [ ] README opens with the one-liner and the results table
- [ ] Practitioner write-up published
- [ ] One rejected approach documented with evidence
- [ ] `mselect` v0.1.0 tagged; repository public

## 12. Deferred: the publishable version

Decided 2026-09-06: build the standard version first, revisit publication when time
allows. Kept at no extra cost so the later version starts from here:

- Item-level raw responses for every own run, cached and committed as Parquet.
- The panel list, sampling seeds and bank version recorded in the results table.
- The Q3, dimensionality and DIF outputs saved in full, not just summarised.

What the publishable version would add: a multidimensional or bifactor IRT fit with a
formal treatment of local dependence, a wider own-run panel, uncertainty on every item
parameter from the Bayesian fit across the whole bank, and a comparison against the
existing literature on efficient benchmarking (tinyBenchmarks and IRT-based leaderboard
work). Six to eight weeks, and a venue with a deadline. Not now.
