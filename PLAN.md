# Plan: Model Selection at a Tenth of the Cost

**Written:** 2026-09-06. **Amended:** 2026-09-10, section 13.
**Status:** the public-data half is built, measured and public (2026-09-11, MIT, three tags);
the own-run panel is not started and waits on project 04's `v0.2.0` tag.
**Build window:** planned 4 weeks, 2026-11-02 to 2026-11-29; started early on 2026-09-10
because the work needs no vendor spend until section 3.3. Standard practitioner version; a
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
| ~~Open LLM Leaderboard "details" datasets on Hugging Face~~ **gated, see section 13.1** | Per-sample outputs and correctness for hundreds of models on MMLU-Pro, GPQA, MATH (levels 5), BBH, IFEval, MuSR | The leaderboard itself was frozen in 2025 but the per-sample datasets remain published. **Verify availability and format in week 1** before anything depends on it |
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
multiple choice, a boxed final answer for MATH). **Amended 2026-09-12, twice, because the panel
contains reasoning models and that is deliberate:**

- **Temperature 0 holds for eight of the eleven.** Claude Sonnet 5, Claude Opus 5 and
  gpt-5.6-sol reject the parameter, so they run at their vendor's default sampling and every
  record of them says so. Section 15.11 has what that costs.
- **`max_tokens` is no longer small.** It is 1024 for every model and every template. A cap is a
  ceiling and not a bill, so the small one saved nothing and cost four models their answers.
  Section 15.14.
- **Answer-only describes the output, not the computation.** A model that reasons internally
  does so whatever it is asked. The prompt now requires the answer on a final line, so that the
  run measures ability rather than format compliance. Section 15.14. Development caching
is on: every response is cached by content hash of the request so a rerun costs nothing.
This is the opposite of the drift-run rule in project 03, and deliberately so; here the
question is about the items, not about whether the vendor changed.

Anthropic calls go through the Message Batches API at half price where latency does not
matter, which is everywhere in this project.

All vendor calls go through the portfolio's gateway library (project 04, version 0, built
in September): routing by alias, an OpenTelemetry span and a cost record per call, and the
project's spend cap enforced there. The cache described above sits on top of it.

## 4. Methods

### 4.1 Item response models

- **2PL**: probability of a correct response as a logistic function of model ability
  minus item difficulty, scaled by item discrimination. The workhorse.
- **3PL**: adds a lower asymptote for guessing. Fitted for multiple-choice items only;
  for free-response items the guessing parameter is not identified and is fixed at zero.
- **Fitting** (amended, section 13.2: Bock-Aitkin EM in numpy, no PyTorch): marginal
  maximum likelihood via `py-irt` (variational, PyTorch) for speed
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
  runner/        vendor calls through the portfolio gateway library (project 04 version 0), Anthropic via its batch endpoint, cache
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

**Measured 2026-09-11, and the estimate above was high by a factor of ten.** The table was
written before a single item had been looked at, so its per-call input figure was a guess: it
assumed 611 tokens and the suite actually averages **188**. `mselect suite` prices the run from
the items themselves against the committed price file, so this is now a number rather than an
assumption, and it is regenerated rather than typed.

| Line | Calls | Tokens (in / out) | At the batch rate |
|---|---:|---|---:|
| Full suite, all 11 aliases, 3,000 items each | 33,000 | 6.2M / 0.53M | **US$5.80** |
| Test-retest, 11 aliases, 500 items again | 5,500 | 0.98M / 0.09M | US$0.93 |
| Position bias, 11 aliases, 300 items, 3 further rotations | 9,900 | 1.85M / 0.16M | US$1.73 |
| Framing, 11 aliases, 300 items, letter-only | 3,300 | 0.65M / 0.05M | US$0.60 |
| Framing, 11 aliases, 300 items, brief reasoning | 3,300 | 0.69M / 0.84M | US$3.11 |
| **Total** | **55,000** | **10.4M / 1.7M** | **US$12.17, US$15.22 with a 1.25x margin** |

That is about **CA$21 against a CA$220 budget**, and it already includes the widening this
section held money for. **Decided 2026-09-11: every model runs the full suite**, where the three
frontier models were on the adaptive subset only. That was a cost compromise, not a method, and
it cost something real: section 4.2 validates an adaptive ranking against the own-run full-suite
ranking, and a model with no full-suite run cannot be in the second of those. The panel had
eleven models and the validation had eight, two of them a 3B and a 7B on the laptop. Now the
validation has all eleven and the top of the range is in it. The frontier check is unchanged,
because the adaptive subset is drawn from what was administered.

The first table and this one are both kept. The estimate was high by a factor of ten because it
assumed 611 input tokens a call against a measured 188, and a plan that quietly swaps its guess
for the answer teaches nobody where the guess went wrong.

Two things stay true. The estimate is a planning number and the gateway's cap is what actually
stops a run; and a cheap run is not a free one, so it still waits on Peter's go.

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

Mirrors the portfolio's definition for this project. Ticked against what is measured and
committed, last reviewed 2026-09-13 after the full-suite arm finished; everything unticked is
work not yet run rather than work that is blocked.

- [x] IRT parameters fitted and published for a real item bank. **150 models by 20,365 items**,
      2PL and 3PL, standard errors on every parameter, bank content-hashed at `1d4c357935c70875`
- [x] Adaptive selection against random and stratified baselines, Kendall's tau with bootstrap
      intervals. **Reproduced at a far smaller share than 10%, and with a crossover the plan did
      not anticipate**: 10 adaptive items match 127 random ones (tau 0.778), and above about 200
      items the baselines win. Section 13.6 and `docs/rejected.md`
- [x] Validation on the own-run panel of models the calibration never saw. **Measured
      2026-09-13**, `mselect validate` and section 15.23: eleven models, 3,000 items each,
      parameters read from the bank and nothing refitted. The bank ranks them at tau 0.855
      (0.617 to 1.000), and adaptive selection reaches that same 0.855 from 100 items, 3.5% of
      the 2,830-item block every model answered. Complete: `google-frontier` was topped up the
      same evening after its daily quota reset
- [ ] Position bias, framing effects and test-retest reliability each measured with intervals.
      **Test-retest is done, 2026-09-14** (section 15.28, `mselect retest`): eleven models, the
      same 500 items twice at temperature 0 a day apart, agreement 0.936 to 1.000 with bootstrap
      intervals, and the number that matters, **a score moves by up to 2.0 points with nothing
      changed**. It agrees with the free partial from HELM, where repeated administrations of the
      same model and item agreed 95.3% of the time (n = 8,431). Position bias and framing are
      three arms of about US$8 that have not been run
- [x] A list of items that measure nothing, with evidence per item. `docs/items-that-measure-nothing.md`
- [x] Q3 and dimensionality diagnostics reported. `docs/diagnostics.md`
- [x] `items_needed` power function validated against the simulation. Validation table in
      `out/simulation-2pl.json`; **it is optimistic at large effects**, see section 13.6
- [x] Handed to project 03. `mselect` v0.1.0 tagged 2026-09-11 with the interface 03's plan
      calls (`items_needed(delta, 0.8, ability)` works with no bank argument), plus
      `dependence()` for the local-dependence correction and `reliability()` for the noise floor
- [x] Cost per ranking decision reported in dollars. **Measured 2026-09-13**: **US$0.72 to rank
      eleven models as well as every response can**, against US$16.05 to ask them everything.
      Priced from the recorded cost of the exact cells the selector chose rather than from an
      average, which matters on a panel spanning a factor of ten in price per item
- [x] README opens with the one-liner and the results table
- [x] Practitioner write-up published. `docs/writeup.md`
- [x] One rejected approach documented with evidence. `docs/rejected.md`
- [x] `mselect` v0.1.0 tagged; repository public. **Decided by Peter 2026-09-11**: the
      repository is public, MIT licensed, with `v0.1.0`, `v0.2.0` and `v0.3.0` tagged

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

---

## 13. What reality changed, 2026-09-10

The plan said not to deviate silently. These are the deviations, each with the evidence that
forced it. Everything else in sections 1 to 12 stands.

### 13.1 The primary data source is HELM, not the Open LLM Leaderboard

Section 3.1 named the leaderboard's per-sample "details" datasets first. They are still
published, but Hugging Face now marks them `gated: auto`: every route that returns bytes
answers HTTP 401 without a signed-in token, while the same anonymous request to a public
dataset answers 200. The evidence is tabulated in `docs/data-sources.md`.

HELM's public buckets need no account, and they turned out to be richer than the plan assumed:
`per_instance_stats.json` per run, `instances.json` for the item text and answer key, and
`schema.json` for model metadata including release date and access. Eight scenarios across
three HELM projects are binary-scored and not model-judged, and they are the bank.

Consequence for the plan: the leaderboard details become a follow-up that needs a Hugging Face
read token, adding models rather than changing the method. Bank v2 when that token exists.
(Section 14.1 corrects one thing this concluded: the token is the whole gate, and no per-
repository access request exists or is made.)

### 13.2 The estimator is Bock-Aitkin EM in numpy, not py-irt and PyMC

Section 4.1 named `py-irt` (variational, PyTorch) for the full matrix and PyMC for posterior
intervals on a subset. The matrix is 150 models by 20,365 items, which marginal maximum
likelihood by EM with 61-point quadrature fits exactly in under a minute on the laptop. A 1.5 GB
PyTorch dependency buys nothing at that size, and a variational approximation would make the
item parameters harder to defend, not easier.

What replaces the Bayesian cross-check: the fit is a MAP fit under stated weakly informative
priors, it reports analytic standard errors on every item parameter, and `bootstrap_items`
resamples models to show how much those analytic errors understate. `mselect/irt/fit.py`
carries the priors and the reasoning.

### 13.3 The matrix is not rectangular, so the headline has a named block

Section 4.2 assumed a full-suite ranking over the whole matrix. Models were run on different
HELM projects, so 54 percent of cells are observed and "the full-suite ranking" is not defined
over all of it. `cat.simulate.dense_block` peels the matrix to a nearly complete block, and the
headline claim is stated on that block: 74 models by 18,921 items, 99.1 percent complete. This
is a reporting change, not a weakening: the claim is now about a suite that actually exists.

### 13.4 Two measurements arrived free, and one deliverable is still blocked

Free: HELM re-ran some models across suites, so 8,431 model-item cells appear twice. They agree
95.3 percent of the time, which is a measured floor on benchmark noise without a single vendor
call. It does not replace the test-retest experiment in section 4.3, which controls temperature
and prompt.

Blocked: everything in section 3.3 and 4.3 that needs vendor calls. Those go through the
portfolio gateway (project 04). **Updated 2026-09-11**: two of the three things this paragraph
waited on have arrived. The Message Batches support is written and merged in 04 rather than
pencilled in for October, and this project's spend caps are confirmed by Peter at US$40 a month
and US$30 a run. What is left is 04's `v0.2.0` tag, which its plan holds for one live batch call
and one live local call, and Peter's go for the spend. Until then the own-run rows in the README
say so rather than being quietly dropped.

### 13.5 Item text is not committed

Section 12 said to keep item-level raw responses as Parquet. The responses are committed; the
item text is not. The bank stores the content hash, the HELM instance id and the scenario, which
is enough to find any item in the cache.

> Amended in v0.3.0, and the amendment is the point of section 14.9: this originally said the
> bank also stores "a short preview", a 160-character excerpt of each question, as evidence for
> the broken-item report. It was removed. GPQA never carried one, at its authors' request.

### 13.6 The efficiency claim has a crossover, and the power function is optimistic

Two things the simulation measured that the plan assumed away.

**The crossover.** Adaptive selection beats random and stratified subsampling by a wide margin
up to about a hundred items and loses to them above about two hundred. The cause is measured,
not guessed: the ability fitted on all 18,921 items agrees with the suite's own average ranking
only at tau 0.921, because the average weights dead and backwards items as heavily as good ones.
Section 10 expected random subsampling to be the approach rejected; the evidence rejects it only
in the small-budget regime, and rejects adaptive testing above it. `docs/rejected.md` carries the
argument, and the README's headline is stated as the crossover rather than as the best case.

**The power function.** `items_needed` is derived from the bank's information function, and
against the simulation it is well calibrated for small effects and optimistic for large ones:
for pairs of models 5 to 10 accuracy points apart at 10 items it predicts 88 percent separation
where 42 percent was observed, closing to within a few points by 50 items for small gaps. Two
reasons, both real and both in the direction of optimism: the information function assumes local
independence, which section 13.3's Q3 numbers show is false, and it assumes the ability
difference maps onto the accuracy difference through the test characteristic curve, which the
construct gap above says it does not do exactly. Project 03 should treat the number it returns
as a floor, not a promise, until the own-run panel refines it.

### 13.7 The dependence handover is a correction factor, not a list of blocks

PLAN.md section 8 promised project 03 "a note on which item blocks are locally dependent, so
that 03 does not treat them as independent evidence". Building it showed that a list of blocks
is the wrong shape. Chaining every pair above the Q3 flag of 0.2 swallows each benchmark whole,
because 8 to 31 percent of pairs are above it, and "treat all 13,937 MMLU items as one piece of
evidence" is true in a useless way.

So the handover is two things instead. Tight blocks, at a Q3 above 0.8, name the 474 items that
are effectively the same question asked twice: 115 blocks, the largest of 47 items. The diffuse
part is handed over as a design effect per benchmark, which is what actually changes an
interval: a hundred MATH items carry about six items' worth of independent evidence, a hundred
GSM8K items about ten, a hundred MMLU items about forty-seven. `mselect.dependence()` returns it
and `mselect/bank/v1/dependent-blocks.json` stores it.

## 14. What the token changed, 2026-09-11

Section 13.1 deferred the Open LLM Leaderboard to "bank v2 when that token exists". The token
exists. These are the deviations that followed, in the same form as section 13.

### 14.1 The gate wanted a token, not a request

Section 13.1 read `gated: auto` as "access has to be asked for, per repository". That was
wrong, and the correction matters because it was the reason to be cautious. Measured on
2026-09-11: an unauthenticated range request for a details file answers **401**, the same
request carrying a read token answers **206**, for a repository the account has never touched,
and nothing is recorded against the account.

So `mselect/data/ollm.py` contains no POST and files no request. An early draft called the
`ask-access` endpoint for each model in the panel; it was deleted once the 401/206 pair was
measured, because it was asking for something the token already had. The full evidence is in
`docs/data-sources.md`.

### 14.2 The panel is chosen, not taken

4,485 submissions have a details repository, which is thirty times more than the plan's target
panel and far more than is worth downloading. Section 3.1 did not say how to choose among them,
because it did not anticipate having to.

The choice is two rules, and both exist to make item parameters identifiable rather than to be
representative of the leaderboard:

1. **Ten models from each of forty equal-width bands of the leaderboard average.** A random
   sample of that leaderboard is mostly seven-billion-parameter fine-tunes inside a narrow band.
   An item bank calibrated on one cannot separate a hard item from an impossible one, because it
   never sees a model that can answer the hard item.
2. **At most eight submissions per hub organisation.** Forty merges of one base model are close
   to one model repeated, and every standard error in the bank depends on the effective panel
   size rather than the nominal one.

The result spans 0.7 to 51.2 on the leaderboard's own average, with quartiles at 12.0, 23.9 and
36.1, across 211 organisations. `data/raw/ollm/panel.json` and `mselect/bank/v2/panel.json`
record the selection with its seed, and the six candidates that were probed and could not be
read.

### 14.3 The cache stores the columns, not the source

Section 3.1's rule is that every fetched byte is cached so a rerun costs nothing. Bank v1 obeys
it literally: `data/raw/helm/` holds the source JSON. Bank v2 does not, deliberately.

One model's run of the 36 tasks is 486 MB of source JSON, 102 MB after the hub's own Parquet
conversion, and about 2 MB of the columns this project keeps. Reads are therefore column
projections over HTTP range requests, and what is cached is the projection. Keeping the source
would cost roughly 40 GB to save a download made once, and this repository is not the right
place to mirror forty gigabytes of someone else's benchmark text.

The reproducibility the rule was protecting is kept another way: every cache entry carries a
provenance line with the source URL, the remote file's size in bytes, the exact columns taken
and the row count, so any extract can be re-derived and checked against the source.

### 14.4 TLS had to be verified against the machine, not against certifi

Not a plan deviation so much as a thing that has to be written down, because it will happen
again in projects 03 and 04. Every request from Python to `huggingface.co` on this network fails
with `CERTIFICATE_VERIFY_FAILED` while a browser on the same machine is fine: the network
inspects TLS, and certifi's bundle, which Python uses by default, does not contain the proxy's
root certificate. The Windows certificate store does.

`ollm.ssl_context` uses `truststore`, which verifies against the operating system's trust store.
Certificates are still verified; they are verified against the trust store the machine actually
has. Two consequences: anything fetched is in clear to whatever performs the inspection,
including the token in the `Authorization` header, which is an argument for the token being
read-only and short-lived; and HELM's Google Cloud Storage host is not intercepted, which is why
bank v1 built without ever meeting this.

### 14.5 Two differential-item-functioning results in bank v1 should not have been published

Adding a second panel forced the grouping code to ask whether a contrast exists before running
one, and the answer changed two numbers that were already in `docs/diagnostics.md`.

**A contrast needs models on both sides, and ten is the floor.** The old check only asked
whether each side was non-empty. The MMLU-Pro contamination result was therefore computed on
three models released before MMLU-Pro was published against sixty-three released after, and
reported as though it meant something. It is now recorded as not testable, with the group sizes
named. The same check now also rejects every other contamination split for the same reason,
where before it rejected them for having an empty side.

**A model with no release date is not an old model.** The median split on release date put the
two HELM models with no recorded date into the older half, because an empty string sorts below
every date. They are now excluded, the split is over the 148 dated models, and the generation
drift result moves from 198 flagged items to 189.

Both corrections make bank v1 say less than it did. That is the right direction: the earlier
numbers were not wrong arithmetic, they were arithmetic on a comparison that was not there.

### 14.6 An item bank inherits the drift of the harness it was built from

Section 3.2 says items are keyed by a content hash, and bank v1 computes that hash from HELM's
question text, options and answer key. Bank v2's first draft took a shortcut: lm-eval-harness
writes a `doc_hash` column, which is a content hash of the document, so why compute another?

Because it is a hash of the *harness's serialisation* of the document, not of the document. The
alignment audit, which exists to check that `doc_id` means the same item for every model, caught
it: on `leaderboard_math_num_theory_hard` a quarter of the audited panel disagrees with the
reference model about every one of the 154 `doc_hash` values, while the problem text at each
`doc_id` is character-for-character identical. Keyed that way, each of those items would have
become two items, each answered by part of the panel, and nothing in the build would have
complained.

Rebuilding identity on the question text this project reads and hashes itself exposed the same
problem one level down, in the answer key. Two releases of the MATH-Hard dataset write the same
answer with different spacing inside the LaTeX: on
`leaderboard_math_intermediate_algebra_hard`, 46 of 280 answers differ between
`\frac{1+\sqrt{5}}{4}` and `\frac{1 + \sqrt{5}}{4}` and in nothing else. So the key is compared
with every space removed, which is the right normalisation for a thing that is an answer rather
than a sentence, while the question keeps its word boundaries and only has runs of whitespace
collapsed. A genuinely different key is still a different item.

Two consequences beyond this bank.

**The audit is not optional, and a sample of it is not enough.** One model in six was misaligned
on a MATH task. A forty-model sample measures that rate and does not say which models, so the
published build verifies all 400 against the reference on all 36 tasks. A model whose mapping
differs is dropped from that task and named in the manifest, which is why bank v2 is not quite
the fully dense matrix it was designed to be.

**Project 03 should read this as a caveat about imported item parameters.** A bank built from
someone else's evaluation harness inherits that harness's version drift, and an identifier that
looks like a content hash may be a hash of a serialisation. The defence is to hash the question
itself and to check that every model agrees about what each position means. That is what bank
v2 now does; it is not what its first draft did, and its first draft would have shipped.

## 15. The own-run runner, 2026-09-11

### 15.1 The vendor call is a seam, not an import

Section 5 puts the vendor call inside `runner/`. It is now behind a one-method `Caller`
protocol instead, and `runner/administer.py` holds the part that has to be right: what to
ask, how to score it, and what to record. Nothing in it imports the gateway, holds a
credential or can make a request.

The reason is testability, and it is the same reason `prompts.py` was written before any
runner existed. The scoring rules are where a mistake does real damage: a reply the parser
could not read, recorded as a wrong answer, becomes an item statistic, and
`docs/items-that-measure-nothing.md` would then be a report about this repository rather
than about the benchmark. Those rules are now exercised against a fake caller, with no
network, no key and no dollar, and the adversarial reply fixtures run in CI like any other
test.

Three rules are enforced there rather than assumed:

- An unparseable reply is recorded as unparsed and never as incorrect. The unparsed share is
  a number this project reports.
- An item whose answer key is not among its own options cannot be scored at all, and is
  recorded with the reason. That is a broken item, not eleven wrong models.
- Replies are matched to prompts by position, and a caller that returns a different number of
  them raises rather than scoring one item against another item's answer.

### 15.2 The gateway is wired in, and the local arm runs

`boundary` v0.2.0 was tagged on 2026-09-11, so the dependency is in `pyproject.toml` pinned
to that tag, the lockfile resolves it to one commit, and CI installs it from the private
repository with a fine-grained read-only token held as `BOUNDARY_GITHUB_TOKEN`. That token
and the step that uses it both disappear when the gateway repository goes public.

`runner/gateway.py` is the adapter: prompts in, replies out, roughly a hundred lines and no
scoring. It decides three things the experiment should not have to know about.

Batching is a price, not a method. Anthropic's Message Batches cost half as much and answer
later, which is fine everywhere here, so it is the default and it is a flag. A provider with
no batch endpoint falls back to one call at a time, automatically and once per alias rather
than once per chunk. A batch problem can therefore delay the bill but never the panel.

A spend cap is a stop, not a result. A failed call is recorded and the run carries on,
because a timeout says nothing about an item. `SpendCapExceeded` is re-raised, because
carrying on would spend the rest of the panel's budget recording that there is no budget.

The gateway owns the money and the record. Every call writes a ledger row costed from
returned usage, and this project's cap is enforced before the request leaves. Nothing in the
adapter adds a price or a retry of its own.

**Proven end to end on 2026-09-11, at zero cost.** Three items administered to
`local-small-a`, which resolves to `llama3.2:3b` on a local Ollama server: all three scored,
none unparsed, the free-response item graded from its answer as well as the two
multiple-choice ones, three ledger rows, none uncosted, and a second pass with the recorded
cells marked done made no calls at all.

### 15.3 Resuming is reading your own record file

`records.py` appends one JSON object per line and never rewrites. The file is the resume
state: `done()` returns the cells already recorded and `administer` skips them, so "run it
again" is the answer to an interrupted run rather than a second bill. A half-written last
line is skipped rather than raised on, because that is the normal shape of a file whose
process was killed, and the cell it belongs to is simply asked again.

A cell whose call failed is not done: an error is usually a timeout or a rate limit and
resuming should pick it up. A cell whose reply was unparseable is done, because asking the
same question at temperature zero again will not read any better.

Records live under `out/`, which is gitignored. What this project commits is the 0 or 1 per
cell, not the replies, for the same reason it does not commit item text (section 13.5): a
reply can quote the question back.

### 14.7 Item parameters transfer, but only for the items that measure something

Section 11 asks for a number a stranger can check. This is the one bank v2 exists to produce,
and it was not in the plan because the plan did not expect two banks to overlap.

Every one of the 998 MMLU-Pro items HELM sampled is among the 12,032 the leaderboard runs. So
the same questions are calibrated twice, on panels with no models in common, through harnesses
that score them differently: HELM reads the model's stated final answer after chain-of-thought,
lm-eval-harness takes the highest-likelihood option. The match was verified before the number
was believed: the pairing is on MMLU-Pro's own `question_id`, which both sources keep, and it
was checked against the question text itself before the excerpts were removed from the bank in
v0.3.0. All 998 matched. Anyone re-running the check rebuilds the bank from the cache, which
still holds the text, and compares there.

| Items compared | Difficulty correlation | Hardest tenth recovered |
|---|---|---|
| All 998 shared items | **-0.04** (-0.10 to 0.02) | 14.1% (7.6 to 21.7) |
| The 532 that discriminate above 0.3 in both banks | **+0.71** (0.67 to 0.75) | 49.1% (34.0 to 60.4) |

The first row says item parameters do not transfer at all. It is arithmetic, not psychometrics:
difficulty is `-d/a`, so an item whose slope is indistinguishable from zero has a difficulty
that is a division rather than a measurement, and the unfiltered correlation is dominated by
those. The second row is what a consumer gets who drops the items that measure nothing first.

Consequences, in the order they matter:

1. **`docs/items-that-measure-nothing.md` is a prerequisite for using a bank, not a curiosity.**
   The difference between the two rows is the whole practical value of knowing which items those
   are.
2. **Project 03 should filter before importing.** A discrimination floor is the cheapest filter
   that works, and 0.3 is the one measured here.
3. **Difficulty is still only an ordering.** Even at 0.71, the two banks agree about half the
   time on which items are in the hardest tenth. Item parameters are portable enough to rank
   items, not portable enough to be used as constants.

### 14.8 A wider ability range is not free, and it is still worth paying for

Bank v2's panel spans the leaderboard from 0.7 to 51.2 on purpose: section 4.1's difficulty
parameter is only identified where the panel has models on both sides of an item. The cost shows
up in the item statistics. Against bank v1, the share of items whose fitted slope is negative
rises from 8.6 to 19.7 percent.

Part of that is the benchmarks: BIG-Bench Hard, GPQA and MATH level 5 are harder than MMLU and
GSM8K, and the leaderboard scores them by log-likelihood over options rather than by reading a
stated answer, which is a noisier measurement for a weak model. Part of it might be the panel:
the bottom of the leaderboard answers close to chance, and answers that are noise dilute every
item-total correlation. So `diagnose` now refits on the stronger half and the weaker half of each
panel separately.

| Bank | Fitted on | Items with a negative slope |
|---|---|---:|
| v1 | all 150 models | **8.6%** |
| v1 | the stronger 75 | 9.4% |
| v1 | the weaker 75 | 13.2% |
| v2 | all 400 models | **19.7%** |
| v2 | the stronger 200 | 23.5% |
| v2 | the weaker 200 | 28.0% |

The sign of a slope is the one statistic here that does not move with the scale, and it says the
same thing about both banks: **the whole panel beats either half, and the weaker half is the
worse half.** Weak models do carry less information per model, and removing them still costs
more than it saves, because the range they provide is worth more than the noise they add. That
is the case for the panel design, measured rather than asserted.

Median discrimination is not comparable across these fits and is not used for the argument. Each
fit identifies its own scale against a standard normal prior over whichever models it used, so
halving the spread of ability halves the apparent slope: bank v1's stronger half reports a median
of 1.23 against the full panel's 0.74 while its share of items below 0.3 barely moves, which is a
change of units and not of items. `docs/diagnostics.md` and `docs/diagnostics-v2.md` carry both.

The practical consequence stands either way, and it is the one section 14.7 turns into a rule:
bank v2 has more items that measure nothing than bank v1 does, they are named in
`docs/items-that-measure-nothing-v2.md`, and a consumer should filter on discrimination before
using any of it.

### 14.9 The bank carried 2.3 MB of other people's benchmark text for nothing

Found in the audit before making the repository public, which is the right time to find it and
later than it should have been found.

Section 13.5 recorded that the bank stores a 160-character excerpt of each question, as evidence
for the broken-item report. Three things were wrong with that, and the third is why it was
removed rather than documented:

1. **The report never rendered it.** `build_report` selected the column into a dataframe and
   printed no part of it. The excerpts were serving nothing at all.
2. **`docs/items-that-measure-nothing.md` said item text was not reproduced "in a public
   repository".** A reader takes that as a claim about the repository. The repository held 2.3
   MB of it, across 19,919 items.
3. **The licences are not uniform.** MMLU, GSM8K, MATH, MedQA, MMLU-Pro and OpenBookQA are MIT
   or Apache and posed no question. LegalBench is a collection of 162 tasks with per-task
   licences, some non-commercial and some unstated, and 2,047 of its items were in there.

So the column is gone from both banks and from the builder. Nothing visible changed, because
nothing visible used it. Both banks were rebuilt from the caches and refitted, which reproduced
every published number exactly and changed only the content hashes, since the fit reads the
response matrix and never the item table. `tests/test_no_secrets.py` now allowlists the string
columns a bank may carry, so adding one back is a decision somebody has to make on purpose.

The general lesson is not about licensing. It is that **a claim in a document is not a property
of a repository until something checks it**, and this project had already learned that about
credentials and had not applied it to content.


### 15.5 What a public release will not let you re-ask, 2026-09-11

The bank stores no item text, so an own run rebuilds each question from the cached release.
`runner/items.py` is that lookup, and writing it turned up the thing worth writing down: a
per-instance release is published so that a score can be checked, not so that a question can be
asked again, and three of bank v1's eight benchmarks do not survive the round trip. None of them
fails loudly. Each would have produced a plausible number, at full price, that meant nothing.

| Benchmark | What arrives | What it would have done | What happens instead |
|---|---|---|---|
| GPQA, 446 items | Every question and option is a placeholder, `[encrypted_text_N]` | Asked a model to choose between four placeholders | Excluded, with the reason recorded. The items stay in the bank, which needs only the 0 or 1 |
| LegalBench, 2,047 items | One option, which is the correct answer | Offered a single choice that was the right one, and scored every model 100 percent | Options rebuilt from the task's own label set: 5 tasks, of 2, 5 and 7 labels |
| GSM8K and MATH, 1,437 items | The answer key is the full worked solution | Compared a model's final answer against a paragraph of derivation, and scored every model 0 | The final value is extracted, by the same parser that reads a model's reply |

GPQA's placeholders are deliberate on HELM's part and correct: its authors ask that the
questions not be published in scrapeable form. That is a benchmark being careful, and the cost
of being careful is that nobody else can re-administer it. Worth saying plainly in the write-up,
because it is the same trade-off this repository makes when it commits no item text.

The check that says all of this is now right is one line: **every one of the 19,919 administrable
items grades its own reference answer as correct**, with the reply shaped the way the prompt asks
for it. It runs in CI whenever the cache is present. The first time it ran, 105 items failed, all
of them symbolic MATH answers sent bare rather than boxed, which was the check being wrong and
not the pipeline; the version that is committed sends what a model would send.

### 15.6 The suite is chosen

`mselect/config/own-run-suite-v1.json`: 3,000 of the 19,919 administrable items, seed 20260911,
drawn 2026-09-11. Committed rather than written to `out/`, because which items the panel was
asked is part of the record of the experiment, the same as the routing table.

Proportional to the pool and uniform inside each benchmark, which is two decisions:

- **Proportional**, so the own-run "full suite" is a miniature of the bank the items were
  calibrated on. MMLU is 70 percent of bank v1 and 2,099 of these 3,000. A rebalanced suite
  would be a different measurement wearing the same name.
- **Uniform inside a benchmark, ignoring the fitted parameters.** Choosing informative items
  would tilt the suite toward exactly what adaptive selection is good at, and the headline claim
  is measured against this suite. The draw must not know what the answer is supposed to be.

Every benchmark clears 66 items at this size, which a test asserts, because proportional
selection can round a small benchmark away and the guard against that is suite size.

### 15.7 What confirming the routes actually means

The routing table is Peter's to confirm, and a decision deserves the checkable parts already
checked. `mselect routes` does all of them and costs nothing: it reads files and environment
variables, opens a TLS connection to each vendor and closes it, and asks the local server
whether it is up. No request is sent, no key is printed, nothing can be billed.

As of 2026-09-11 it reports: every one of the eleven aliases has a route; every vendor model has
a rate in `prices/2026-09-10.yaml`; the two local models are price-zero by configuration; Ollama
is answering. **No API key is set in the shell**, which is the blocker, and it is Peter's to
clear because the keys are his.

One inherited assumption turned out to be wrong and worth re-testing rather than believing.
Project 04 moved its live calls to GitHub Actions on 2026-09-10 because the laptop's network
inspected TLS. That is a property of a network, not of a laptop: on 2026-09-11 all four vendor
certificates verify from here against the operating system trust store, issued by Google Trust
Services rather than re-signed by an inspecting proxy. So the panel can run from the laptop on
this network, and `mselect routes` re-answers the question wherever it is next run, because the
answer changes with the network rather than with the code.

What no amount of reading settles is whether each model answers in the answer-only format the
run assumes. Google's Flash spent a small token budget on reasoning and returned no text on
2026-09-10; only a real call finds that. `mselect smoke` is that call: a few items of the
committed suite, spread across benchmarks, reported per alias as parsed, scored, unparsed and
failed. An unparsed reply is the finding, because a model whose answers cannot be read would
have its noise recorded as wrong answers across 3,000 items.

It cannot spend by default. With no arguments it calls the price-zero local models only; a
vendor alias needs `--yes` and is refused before a gateway is opened, which
`tests/test_cli_guards.py` asserts rather than trusting to habit.

### 15.8 The first paid smoke run, 2026-09-12

Peter set the keys and ran it across all nine vendor aliases. **Twenty-seven calls, US$0.011,
and four of eleven routes were wrong.** The whole own-run programme is about US$12, so this
found four defects for a tenth of one percent of the budget, before any of them could be
multiplied by 3,000 items.

| Alias | Result | Cause | Done |
|---|---|---|---|
| anthropic-haiku | 3/3 scored, 3 correct | | kept |
| openai-mid | 3/3 scored, 3 correct | | kept |
| openai-frontier | 3/3 scored, 3 correct | | kept |
| together-open-a | 3/3 scored, 1 correct | | kept |
| local-small-a | 4/4 scored, 2 correct | | kept |
| google-mid | 2/3, one cut off mid-preamble | reasons before answering | thinking disabled |
| google-frontier | **0/3, no text at all**, US$0.0014 | reasons before answering | thinking disabled |
| together-open-b | **0/3, no text at all**, row uncosted | reasons before answering | effort set to low |
| local-small-b | 0/4 | `qwen2.5:7b` was never pulled | changed to `qwen2.5:3b` |
| anthropic-sonnet | 0/3, the batch errored | rejects `temperature`: see 15.11 | temperature omitted |
| anthropic-opus | 0/3, the batch errored | rejects `temperature`: see 15.11 | temperature omitted |

**The run also found three defects in this repository**, each one in what a failure is allowed
to say, and each one worse than the route it was hiding.

- **A failed call said "errored".** `str(response.status)` names the shape of a failure, not the
  failure, and the vendor's own message was sitting parsed on the response and being thrown
  away. Six Anthropic failures reported nothing at all.
- **A call that succeeded and returned no text was invisible.** `unparsed` needs text to parse
  and an error needs a failure, so a 200 with an empty completion counted as neither:
  `google-frontier` reported "0 scored, 0 correct, 0 unparsed, 0 failed" and had spent money.
  That is precisely the failure this check exists to catch and it was the one shape it could not
  see. An empty reply is now an error that names the finish reason and the tokens spent.
- **`finish_reason` never reached the record.** It is the field that separates "this model will
  not do this" from "this request was shaped wrong", which is the difference between changing
  the panel and changing four lines of configuration.

**Three of the four route failures are the same failure.** A model that reasons before answering
spends the whole 16-token answer-only budget on the reasoning and has nothing left to answer
with. That is not a model failing the task; it is a request shaped for a model that answers
immediately. `boundary.ChatRequest` has always had `extra` for this and this adapter never used
it, so the fix is now configuration: `mselect/config/request-extras.yaml`, per alias, each entry
naming the call that proved it. It is deliberately not in `boundary.yaml`, whose route schema
forbids unknown keys and is right to; the gateway drew the same line when its own smoke command
needed one, and called it caller's business rather than library's.

**The two Anthropic identifiers were left exactly as they were**, because they are wrong or they
are not and replacing them with a guess is how a wrong one gets into a routing file.
`mselect models` asks each vendor for its own list and marks every route the vendor does not
carry, ranking the near misses by how much of the name they share, weighted so that a token
common to every identifier counts for less: without that, "claude" scores as highly as "opus"
and every Claude model ties for first. Listing models generates no tokens and so is billed
nowhere, which is why it is a check rather than a cost.

### 15.9 What the vendors' own lists said, 2026-09-12

**Every one of the eleven routes is listed.** Anthropic carries all eleven of its models and
both `claude-sonnet-5` and `claude-opus-5` are among them. So the identifiers were never wrong
and the Anthropic failure is about the request rather than the route, which is the same
conclusion the other three failures reached and was worth reaching by evidence rather than by
pattern-matching. The routes stay as they are.

**The diagnosis needs the batch turned off.** A batched request cannot report why it failed:
Anthropic returns one outcome word per item, the gateway builds its `ChatResponse` with
`raw=None` whenever an item did not succeed, and the per-item error string it holds in
`BatchItemResult.error` is dropped there. So "batch_errored" is the most any batched failure can
ever say, no matter what this project does with it. That is a real gap in project 04 and is
worth reporting there. Meanwhile `mselect smoke --no-batch` sends the same request on its own,
which returns an error body, and a failed call is billed nowhere so the diagnosis is free.

**The OpenAI frontier route was two releases stale.** Peter caught it: `gpt-5.4` has not been
the frontier for some time. OpenAI's list carries `gpt-5.5` and then the GPT-5.6 line, which is
three models rather than a size ladder, and the prices are what separate them.

| Model | Input | Output | Nearest Anthropic model by price |
|---|---:|---:|---|
| gpt-5.6-luna | 0.20 | 1.20 | below Haiku 4.5 |
| gpt-5.6-terra | 2.00 | 12.00 | Sonnet 5, at 2.00/10.00 |
| **gpt-5.6-sol** | **4.00** | **20.00** | **Opus 5, at 5.00/25.00** |

This slot exists to be the OpenAI counterpart of `anthropic-opus`, so it is **`gpt-5.6-sol`**.
Prices read from the vendor's page on 2026-09-12 and written to a new dated file rather than
edited into the old one, which is the rule the gateway's ledger depends on: a row has to be able
to say what it was costed with. Every rate the previous file carried was re-read at the same
time and none of them had moved, which is only a fact once somebody has looked.

The programme goes from US$12.17 to **US$13.26**, US$16.58 with margin. Both caps still clear.

**Closed 2026-09-14: `google-frontier` is `gemini-3.8-flash` and that is correct.** This entry
used to say a Flash model could not be the counterpart of Opus 5 or gpt-5.6-sol. Peter confirmed
that `gemini-3.8-flash` is Google's current frontier model, so the objection was to the name and
not to the model. Section 15.24 has the reasoning; the short version is that a vendor's product
naming is not a tier ranking, and treating it as one is the sort of assumption this project
exists to check.

It is worth noting that `anthropic-haiku` is the one Anthropic route carrying a dated
identifier, and the one that works.

### 15.28 Five models answer the same questions twice, and 5% of the answers move

The first vendor numbers from the test-retest arm, 2026-09-14. The same model, the same 500
items, the same settings, temperature 0, administered a day apart through a cache namespace of
its own so that every call was really made:

**Completed the same evening with all eleven**, once OpenAI had credits and the Anthropic three
were run with `--no-batch`. `mselect retest` reads both administrations back and calls the
analysis that has been written and tested against fixtures since before the runner existed:

| alias | items | agreement | 95% bootstrap | phi | to right | to wrong |
|---|---:|---:|---|---:|---:|---:|
| `local-small-b` | 494 | 1.000 | 1.000 to 1.000 | 1.000 | 0 | 0 |
| `local-small-a` | 497 | 0.998 | 0.994 to 1.000 | 0.996 | 0 | 1 |
| `anthropic-haiku` | 496 | 0.984 | 0.972 to 0.994 | 0.945 | 4 | 4 |
| `google-frontier` | 499 | 0.982 | 0.970 to 0.992 | 0.881 | 7 | 2 |
| `anthropic-opus` | 496 | 0.978 | 0.964 to 0.990 | 0.877 | 6 | 5 |
| `openai-frontier` | 498 | 0.976 | 0.962 to 0.988 | 0.894 | 7 | 5 |
| `anthropic-sonnet` | 493 | 0.970 | 0.953 to 0.984 | 0.871 | 7 | 8 |
| `google-mid` | 497 | 0.950 | 0.930 to 0.968 | 0.799 | 15 | 10 |
| `together-open-a` | 496 | 0.950 | 0.929 to 0.968 | 0.850 | 9 | 16 |
| `together-open-b` | 490 | 0.939 | 0.916 to 0.959 | 0.767 | 17 | 13 |
| `openai-mid` | 500 | **0.936** | 0.912 to 0.956 | 0.826 | 11 | 21 |

**Every hosted model disagrees with itself.** Between 1.6% and 6.4% of answers flip with nothing
changed, and the two laptop models are effectively deterministic at 0.998 and 1.000. The prompts,
the parser and the scoring are identical for all eleven rows, so the gap between 1.000 and 0.939
is the service rather than this code.

**The number a release gate needs is the last two columns, not the agreement.** Flips that cancel
do not move a score; flips that do not, do. `openai-mid` went 11 right and 21 wrong, a net loss of
**2.0 points on the same 500 questions it had already answered**. The median model moved 0.4
points. So:

    score moved by up to 2.0 points with nothing changed (median 0.4).
    A drop smaller than that is noise.

That sentence is the whole deliverable of this arm, and it is what project 03's release gate has
to be built on. A gate that fires on a two-point drop in `gpt-5.4-mini` would have fired here, on
nothing at all.

Agreement and phi disagree about who is worst: `together-open-b` has the lowest phi (0.767)
while `openai-mid` has the lowest agreement, because phi is sensitive to where in the item range
the flips fall.

**Corrected the same evening.** This section first said the flips were not symmetric and that
"something other than a coin is moving", on the strength of `google-frontier` flipping 7 right
and 2 wrong and `together-open-a` flipping 9 right and 16 wrong. McNemar's exact test says
otherwise:

| alias | to right | to wrong | points | symmetry p |
|---|---:|---:|---:|---:|
| `openai-mid` | 11 | 21 | -2.0 | 0.110 |
| `together-open-a` | 9 | 16 | -1.4 | 0.230 |
| `google-frontier` | 7 | 2 | +1.0 | 0.180 |
| `google-mid` | 15 | 10 | +1.0 | 0.424 |
| the other seven | | | -0.2 to +0.8 | 0.585 to 1.000 |

Not one model reaches significance, the smallest p is 0.110, and pooled across the panel it is
**83 flips to right against 85 to wrong, p = 0.94**. The flips are as symmetric as a coin. Seven
against two looks like a pattern and is nine discordant pairs; reading it as one is precisely the
error this project exists to catch, committed in this file, about this project's own output.

**The corrected finding is the stronger one.** Symmetric flips mean the movement is a random walk
rather than drift, and that sharpens the headline instead of softening it: `openai-mid` lost 2.0
points to nothing at all. A release gate cannot distinguish a two-point drop from noise, because
**the noise itself produces two-point drops**. Drift would have been a different and easier
problem, because a systematic shift can be corrected for and a random walk can only be measured
and allowed for.

The test now lives in `analysis.retest` and `mselect retest` prints it, pooled as well as per
model, because no single model here has enough discordant pairs to say much alone. A claim about
symmetry that nobody can regenerate is the kind of assertion this repository treats as a defect,
and it should not have taken a second look to notice that.

The drift column is what a platform team would act on. `together-open-a` fell 1.4 points and
`together-open-b` rose 0.8 points **because they were asked twice**. Any claim of the form "the
new version dropped two points" about these models is inside the noise unless it is measured
against this floor, and this is the floor. Section 4.3 called this "how much of the score is
noise"; it is about a point and a half either way at 500 items.

Compare the free partial from HELM recorded in section 13.4: repeated administrations of the
same model and item agreed 95.3% of the time (n = 8,431). Three hosted models here agree 93.9%
to 95.0%. Two independent harnesses, the same answer, which is the strongest form this result
could have taken.

**What it cost.** US$4.24 against an estimate of US$2.83, and the overrun is explained rather
than mysterious: the Anthropic three were run with `--no-batch` after section 15.27, which is the
US$1 the estimate did not know about, and OpenAI's 626 refusals and 374 dropped connections were
re-asked. The arm finished with zero calls outstanding across all eleven models.

#### The allowlist that never matched anything

`openai-frontier` also lost 370 calls to `getaddrinfo failed`, a DNS lookup failing on this
laptop mid-run, and **every one was recorded as settled**. `records.done` confirmed 374 transport
failures across the run would never be asked again: holes caused by the network being briefly
unavailable, filed as permanent facts about the items.

Section 15.19 named transport failures as the clearest example of a failure worth repeating, and
then matched them against `{"timeout", "connect_error", "read_error"}`. The gateway puts the
exception's class name in the status, so what actually arrives is `ConnectError` and `ReadError`,
and `"ConnectError".lower()` is `"connecterror"`. **The allowlist never matched anything, on any
run, since it was written.** Nothing noticed, because until this arm nothing had failed that way.

Underneath it was a second bug that guaranteed the first could not be caught: the two failure
paths asked different questions. `_retryable` handled a string status and `except ProviderError`
tested membership of a set of integers, so a transport failure arriving by the second path could
never be transient however it was spelled.

Both paths now use one rule, matched on the shape of the name rather than a list of them, because
httpx has a dozen and the first version guessed two of them wrongly. Batch outcome words stay
outside it deliberately: `errored`, `expired` and `canceled` do not end in `error` or `timeout`,
and section 15.9's reason for settling those is unchanged.

The 374 records already written say `retryable: false`, and a stored False of that vintage cannot
be told apart from a correct one. So `records.done` overrides the recorded verdict for this one
case, which nothing else there does. It is narrow on purpose: it applies only where no response
arrived, and no response arriving is never a statement about the item.

### 15.27 A batch that had not finished ended the run, and left a bill behind

The test-retest arm was run properly on 2026-09-14, from a shell with the keys in it, and died
an hour later with seven of eight models untouched:

    BatchNotReady: batch msgbatch_01WpF9fbZCHheD3bHG8bz8W7 is in_progress, not ended
    (canceled 0, errored 0, expired 0, processing 250, succeeded 0)

`anthropic-haiku`'s first batch of 250 was still running when the poll limit expired.
`batch_results` raised, and nothing caught it: `_ask_one_at_a_time` has an `except BatchNotReady`
and `_ask_batched` did not, so the exception went through `administer`, through the command, and
onto the terminal. Everything after `anthropic-haiku` was lost, including five models that do not
use batches at all and could not have been affected.

**Two things were wrong and the crash is the smaller one.**

A batch still running says nothing about the other aliases in the panel. That is the same
reasoning section 15.19 applied to every other failure in this module, and it simply had not been
applied here. Those items are now retryable failures and the run continues.

The other is the bill. A submitted batch belongs to Anthropic: it will finish and it will be
charged for, whether or not this process was still waiting. Recording the items as failures and
walking away leaves a paid batch in flight, and a resume submits a second one for the same items
and pays twice. So the batch id is carried in the error text and listed at the end of the run:

    1 batch(es) were still running when this run stopped waiting:
      msgbatch_01WpF9fbZCHheD3bHG8bz8W7
    The vendor will finish and charge for those whether or not anyone collected them, so
    re-asking those items buys the same answers twice.

About US$0.06 in this case. The point is not the money, it is that a run should not quietly
abandon something it has already bought.

**The hour was never Anthropic's promise.** Their documented ceiling is 24 hours and most batches
are far quicker. 3,600 seconds was this project's patience, chosen when a 3,000-item run cleared
in 25 minutes, and it is now `--batch-wait` because the right value depends on whether anyone is
waiting for the answer.

**What this says about the panel's dependency on one vendor's queue.** Boundary has one batch
adapter, for Anthropic, against six provider kinds it knows about (section 15.24). So three of
the eleven models in the panel are subject to a queue with a 24-hour ceiling and the other eight
are not, and an arm that finishes in an hour or in a day depending on that queue is not a
schedule anyone can plan around. `--no-batch` for the Anthropic three costs about US$1 more on a
500-item arm and removes the dependency entirely. That is the trade worth taking while the
remaining arms are small; it is the wrong trade for a 3,000-item arm, where the discount is
US$3 and nobody is waiting.

### 15.26 The first test-retest numbers, and ninety minutes of a run that could not work

The test-retest arm was started on 2026-09-14 from a shell that had no vendor keys in it. Every
paid call failed instantly:

    1,500  ConfigError: provider 'anthropic' needs ANTHROPIC_API_KEY in the environment
    1,000  ConfigError: provider 'openai' needs OPENAI_API_KEY
    1,000  ConfigError: provider 'openweights' needs OPENWEIGHTS_API_KEY
      500  ConfigError: provider 'google' needs GOOGLE_API_KEY

Nothing was spent and nothing was lost. A `ConfigError` is a `BoundaryError`, section 15.19
classifies it as worth repeating, and `records.done` confirmed all 4,000 would be asked again on
a resume. The project's own `.env` holds only `HF_TOKEN`; the vendor keys live in project 04's
`.env` and in Peter's interactive session, which is where every paid run has been launched from.

**The defect is the ninety minutes, not the missing key.** The run printed its plan, said 5,500
calls outstanding, and then wrote 4,000 identical records while looking exactly like a run that
was working. Per-call error handling is right and stays: the loop should survive one vendor
refusing one call. A missing API key is not that. It cannot be true for one call and false for
the next, so discovering it 4,000 times is discovering it 3,999 times too many.

`gateway.missing_keys` now reports the environment variables the chosen routes need and the
environment does not have, keyed by variable and valued by the aliases that would fail without
it, because naming the variable without naming what it costs you is half an error message. The
run refuses before the first call:

      ANTHROPIC_API_KEY is not set, and anthropic-haiku cannot run without it
      OPENAI_API_KEY is not set, and openai-mid cannot run without it

    nothing was sent. Vendor keys come from the environment; set them and run again.

It refuses even with `--yes`, because `--yes` is permission to spend rather than an instruction
to proceed regardless. A provider with no `api_key_env` never blocks anything, which is the local
server, and the check is in the command rather than in the gateway: the gateway's job is to be
the one thing that talks to vendors, and this is the command's job to ask before it starts.

**What did run, because it needs no key.** The two laptop models answered the same 500 items a
second time, through the cache namespace of section 15.25, and they are the first test-retest
numbers this project has:

| alias | items | agreement | 95% Wilson |
|---|---:|---:|---|
| `local-small-a` | 497 | 0.998 | 0.989 to 1.000 |
| `local-small-b` | 494 | 1.000 | 0.992 to 1.000 |

These are the least interesting rows in the arm and they were always going to be. A local server
at temperature 0 is close to deterministic, so near-perfect agreement is a check that the
machinery works rather than a finding about models. It does establish one thing worth having:
the namespace change of section 15.25 works, because without it these would have been answered
from the first administration's cache and would have read 1.000 by construction. 0.998 is not
1.000, and the item that moved is evidence that the calls were really made.

The number the arm exists for is the vendors', and it is still missing.

### 15.25 The cache would have answered the test-retest arm out of its own records

Found on 2026-09-14, while costing the four remaining arms and before any of them were run.

Test-retest (section 4.3) asks the same model the same 500 items at the same settings a day
apart. Holding the settings identical is the design rather than an accident: anything that
varied the request would measure the variation instead of the model, and what moves between two
identical administrations is the noise floor under every "the new version dropped two points"
claim anyone will ever make about that model.

Identical settings means identical request bytes, and section 3.3 caches every vendor response
by the hash of exactly those bytes. `out/own-run-cache` held **24,053 replies, 46.8 MB**, from
the full-suite arm. The second administration would have been answered from the first one's
replies: free, instant, and in perfect agreement. The arm would have reported a reliability
coefficient of 1.0 and it would have been measuring the cache.

Worse than a clean failure, it would have half worked. Anthropic's three models go through the
Message Batches endpoint and boundary deliberately does not consult the cache inside a batch, so
the arm would have produced three honest rows and eight fabricated ones, in one table, with
nothing to tell them apart.

**Both rules are right.** Section 3.3's "cache every vendor response, a rerun must cost nothing"
is correct and is not weakened here. The mistake was in the word rerun. A second administration
is not a rerun: it sends the same bytes in order to measure a different thing, and no hash of
the request can see the difference, because the difference is not in the request.

So an administration gets a cache namespace of its own. `mselect run --repeat 2` writes to
`own-run-plain-0-r2.jsonl` and reads and writes `out/own-run-cache/r2`. Inside one
administration a rerun still costs nothing, which is what section 3.3 is actually for; across
administrations nothing is shared, so a repeat genuinely calls the vendor. The two have to move
together, and the code comment says why: changing only the record file would write the same
fabricated agreement into a fresh file and look like it had worked.

A flag that simply turned the cache off would have been the wrong fix. Resuming a half-finished
repeat would then pay full price for everything already asked, and resuming is not re-measuring.

**The good news in this.** The first administration is already paid for. The full-suite arm asked
every model all 3,000 items on 2026-09-12 and 2026-09-13, so test-retest needs only its second
administration, which is what the US$2.83 estimate already assumed. The day's gap the design
calls for is satisfied by the calendar rather than by waiting.

**Where this generalises.** Position bias and prompt framing are unaffected, because rotating the
options and changing the template both change the prompt and so change the hash honestly. The
class of defect is the one this project keeps finding in itself: a measurement that reads its own
machinery back and reports it as a result. A truncated reply read as a token count (15.17), an
error string read as a settled verdict (15.22), a model's own item set read as a common frame
(15.23), a published batch rate read as a discount received (15.24). Four of the five were found
by not believing a number that looked reasonable.

### 15.24 A discount that was never earned, and a tier label that was right all along

**`google-frontier` is settled.** Section 15.4 has carried "a Flash model is not the counterpart
of Opus 5 or of gpt-5.6-sol whatever else it is" as an open question since 2026-09-12. Peter
closed it on 2026-09-14: `gemini-3.8-flash` **is** Google's current frontier model. The objection
was about the word rather than about the model, and reading a vendor's product naming as a tier
ranking is exactly the kind of assumption this project is supposed to check rather than carry.
The panel result agrees: it tops the panel at 0.936.

**The cost estimate was wrong twice, and the second was the larger.** Section 15.22 blamed the
22% overrun on the sample the output tokens came from, three items per model, and that was a
real cause. Re-estimating against the 33,000 replies the arm produced closes some of the gap and
leaves a residual with a shape:

| alias | estimated, measured tokens | billed | after the fix |
|---|---:|---:|---:|
| `anthropic-haiku` | 0.68 | 0.69 | 0.68 |
| `openai-frontier` | 2.49 | 4.97 | **4.99** |
| `google-mid` | 0.68 | 1.35 | **1.35** |
| `google-frontier` | 1.47 | 2.60 | **2.93** |
| `together-open-a` | 1.00 | 1.04 | 1.00 |
| whole arm | 12.44 | 18.25 | **17.36** |

Anthropic and Together landed within a few percent; OpenAI and Google came in at about half of
what they cost. The ledger says why: of the 30,000 calls in the arm, the 9,045 Anthropic ones
carried a batch id and every other one carried none. The gateway can only send a batch to
Anthropic today. It tries, and falls back to standard calls where the provider has no batch
endpoint, so nothing about the run was wrong.

What was wrong was the estimate. The vendors do publish batch rates and the price file records
them correctly, and `estimate` applied `batch_multiplier` to everything on the strength of one
`batch=True` flag. It was pricing a discount the run had no way to earn. A price list that is
right can still be read wrongly.

The rule now matches the runner: a batch rate applies where a batch can actually be sent, which
is `suite.BATCHING_PROVIDERS`, today just Anthropic, and the day boundary grows another batch
adapter that is the line that changes. With both fixes the arm estimates at US$17.36 against a
billed US$18.25, within 5%, and the residual is Anthropic's truncated replies: 25 calls that
each spent a full 1,024-token budget producing no scorable answer, so they are billed output
that no measurement of output tokens can see.

**What the four remaining arms actually cost**, on that basis:

| arm | calls | US$ |
|---|---:|---:|
| test-retest, 500 items | 4,500 | 2.83 |
| position bias, 300 items x 3 rotations | 8,100 | 5.19 |
| framing, letter-only | 2,700 | 1.74 |
| framing, brief reasoning | 2,700 | 1.77 |
| **total** | **18,000** | **11.53** |

US$14.41 with the 1.25x margin, against US$18.25 already spent and a US$100 monthly cap. The
largest single arm is US$5.19, well inside the US$30 per-run cap. The old figure for the same
four arms was US$8.91.

### 15.23 The panel is in, and the bank ranks models it never saw

The full-suite arm finished at 15:05 on 2026-09-13, twenty-one hours after it started. Eleven
models, 3,000 items each, 30,936 scored cells, US$16.42. `mselect validate` reads it back and
answers the question section 11 has carried unticked from the beginning.

**Amended 2026-09-13, 23:00.** The quota reset at 21:30, `google-frontier` was topped up for
US$1.83, and the comparable block went from 1,031 items to **2,830**. The numbers below are the
complete-panel ones. The provisional set is kept in the git history rather than here, with one
exception noted at the end, because it moved in an instructive direction.

**The bank transfers.** Item parameters fitted on 150 public HELM models, none of them in this
panel, rank these eleven at **tau 0.855 (0.617 to 1.000)** against their own observed accuracy.
Nothing was refitted; the parameters were read as given. That is the difference between an item
bank and a description of the panel it was fitted on.

**A twentieth of the money, not a tenth.**

| items | share | adaptive tau | US$ | of full |
|---:|---:|---|---:|---:|
| 10 | 0.4% | 0.527 (0.020 to 0.880) | 0.07 | 0.4% |
| 50 | 1.8% | 0.745 (0.489 to 1.000) | 0.36 | 2.2% |
| **100** | **3.5%** | **0.855 (0.617 to 1.000)** | **0.72** | **4.5%** |
| 300 | 10.6% | 0.891 (0.667 to 1.000) | 2.17 | 13.5% |
| 750 | 26.5% | 0.927 (0.750 to 1.000) | 5.29 | 33.0% |

At 100 items adaptive selection reaches 0.855, which is the same tau the whole 2,830-item block
gives. That is the claim in its strongest form: **3.5% of the items and 4.5% of the money buy
everything knowing every response is worth**, US$0.72 against US$16.05. Random selection does
not reach 0.855 until 300 items and stratified not until 150, both erratically. The cost line is
not an item count times an average price, it is the recorded price of the exact cells the
selector chose, which matters when the panel spans a factor of ten in price per item.

Above 100 items adaptive drifts to 0.891 and 0.927, which is **above the full-information
value**. That is not more knowledge, it is sampling noise: with eleven models tau moves in steps
of about 0.036 and a subset can outrank the whole by accident. It is reported rather than
smoothed because pretending the curve is monotone would be the more misleading choice.

**The honest complication, and it is the same one section 13.6 found.** A raw score over enough
random items beats every IRT method at the top end: 0.945 at 500 items against the IRT ceiling
of 0.855. That is not a defect, it is what the target is. The truth being reproduced is observed
accuracy, and a raw score estimates observed accuracy directly while ability estimates a latent
trait that ranks it only nearly. IRT wins where items are scarce and loses where they are not,
and the crossover here is around 300 items, or 11% of the block.

**The instructive move.** On the 1,031-item block the transfer was tau 0.917; on the full 2,830
it is 0.855. Tripling the evidence made the agreement worse, which sounds wrong and is not. The
truth got sharper, not the estimate: with three times the items, each model's observed accuracy
is measured precisely enough that the places where ability and accuracy genuinely disagree stop
being hidden by noise. A ceiling that falls when you look harder was a real ceiling all along.

**Every interval overlaps every other.** Eleven models is eleven models, and a bootstrap over
them is wide. Nothing in the table above is separated from anything else with confidence, and it
is reported that way rather than reported as a win.

#### Two silent defects, found by not believing the first answer

Neither raised anything. Both produced output that looked like a result.

**An item the bank could not place erased the model that answered it.** 68 of the 3,000 suite
items have a non-finite fitted difficulty, which is the bank behaving correctly: every model in
its fit answered them the same way, the likelihood has no maximum in `b`, and NaN is the honest
record of that. Scoring one makes `log p` NaN, and the posterior sums over every answered item,
so one such item makes a model's whole ability NaN. The first run printed a transfer correlation
computed on **zero models** and exited 0. The fix is a `usable` mask on the panel and a
`measurable` frame that everything reading parameters goes through, because "answered" and
"can be measured" are different facts and the count of the difference belongs in the output.

**A model with a short row was ranked against models with long ones.** This one inverted the
conclusion rather than erasing it. `google-frontier` was refused after 1,088 items, so its
accuracy was computed over a third of the suite while every other model's was computed over all
of it, and comparing those ranks the item sets as much as the models. The run reported that
adaptive selection was beaten by random selection at every size and never improved: a flat tau
of 0.745 from 10 items to 1,000, which is the shape of an answer that is not listening to its
input. On the 1,031 items every model answered, adaptive goes 0.587, 0.881, 0.917 and the
conclusion reverses.

That second one is worth stating as a rule rather than a bug. A ranking is a comparison, and a
comparison needs a common frame. `simulate.dense_block` already enforced it for the public bank
and this module had to find it again from the other direction.

**What that cost, and it was not nothing.** One model refused after a third of the suite cost
every model the other two thirds: until the top-up the comparable block was 1,031 items rather
than 2,830. An arm is not finished when the last model stops answering, it is finished when
every model has answered the same items.

### 15.22 A missing verdict is not a verdict, and 1,910 calls nearly went quiet

Found on the morning of 2026-09-13, in the run that was still going.

`google-frontier` finished the night with **1,088 items scored of 3,000**. At 00:23 Google's
free-tier daily quota for `gemini-3.8-flash` ran out and refused every remaining call:

    1,908  google returned 429: RESOURCE_EXHAUSTED, generate_requests_per_model_per_day
        2  google returned 503: UNAVAILABLE

That on its own is an inconvenience with a known remedy, which is to ask again after the quota
resets. The defect is what would have happened next.

**Section 15.19 settled them.** That section, written at 21:29 the previous evening, made a
failure retryable or not at the point where the information is, and defaulted a record with no
verdict to "not retryable". The reasoning was recorded in the code: the only verdict-less
records in existence were the 33 deterministic failures from the first four models, so reading
"no verdict" as "settled" described them correctly.

It stopped describing them correctly three minutes later, because the run was started at 18:39
and Python had already imported the old module. The process went on writing verdict-less
records for another nine hours, and 1,910 of them were the most repeatable failure a vendor
produces. `records.done` read every one as settled. Resuming would have skipped all 1,910 in
silence and left the panel with one model measured on 36% of the suite, with nothing in the
run's own output to say so: the resume prints what it will ask, and it would have said there
was nothing to ask.

Checked rather than reasoned about, against the live file:

    records.done() before the fix: all 1,910 google-frontier failures settled
    records.done() after  the fix: 1,910 asked again, 68 deterministic ones still settled

The fix is to keep three states apart where there were two. A verdict of True is asked again,
a verdict of False is settled, and **no verdict at all is decided from the recorded error**,
which is the only thing those records have left. Reading a status back out of an error string
is what `gateway._retryable` refuses to do and should keep refusing, because the adapter holds
the status and the exception type and a sentence is a lossy copy of both. That refusal is about
records that have a verdict. This is about records that never got one, and for those the
sentence is not a lossy copy of the information, it is the information.

The general lesson is the one about long-running processes rather than about defaults. A
default chosen against the data in front of you is a claim about data that does not exist yet,
and a six-hour run is long enough for the code to change underneath it. Anything that reads a
record file has to treat an absent field as absent, not as its current default.

**Two facts about the quota, for the arms still to come.** The reset is at midnight UTC, so
the 1,910 can be re-asked after about 21:30 local. And the 1,908 refusals cost 4 HTTP requests
each, one call and three retries, because the gateway retries a 429 as though it were a busy
minute rather than an exhausted day: 5,724 requests spent on a wall that was not going to move.
Whether the panel keeps a free-tier route at all is a question for section 15.4, which already
has `google-frontier` open on other grounds.

### 15.21 `mselect rescore`: a scoring fix should not cost a run

Three of the defects found on 2026-09-12 were in scoring rather than in asking: a parser that
could not read "Answer: B", a fallback that invented answers out of the article "a", and a
refusal to read an option named rather than lettered. Every one was found after real calls had
been made.

Section 15.15 said a scoring change "is recoverable after the fact, by design", because
`Administration.reply` keeps the text. That was an intention rather than a fact: nothing could
do it. This is the command that makes the claim true, and the distinction it draws is the one
that governs what future compute this project needs.

- **Changing what is asked** costs a full run. The request hash covers the prompt, the system
  message, the budget, the temperature, the model and the vendor fields, so changing any of them
  re-asks those cells deliberately: US$13.40 and about six hours for the full-suite arm.
- **Changing how a stored reply is scored** should cost nothing, and now does. `mselect rescore`
  reads a record file, scores every stored reply again with the current parser, reports what
  moved and writes only when told to, keeping the previous file beside it.

Run against the live panel records while the run was still going: **15,020 records read, 14,982
rescored, none changed.** The 38 it skipped had no reply to score, which is what a failed or
truncated call leaves behind. So the scoring in the run agrees with the parser as it stands, and
the first use of the tool was as a check rather than a repair.

This matters most for the arms still to come. After tonight the settings are frozen in practice,
because a prompt change is no longer cheap. A parser change is, and the two should not be
confused when the next defect turns up.

### 15.19 Resuming re-asked every failure, and almost none of them were transient

Found in the live run, 2026-09-12, from its first four models:

```
 20  anthropic-sonnet   the model returned no text (max_tokens), 1024 output tokens spent
  5  anthropic-opus     the model returned no text (max_tokens), 1024 output tokens spent
  2  anthropic-sonnet   the model returned no text (refusal)
  2  anthropic-opus     the model returned no text (refusal)
  1  anthropic-opus     the model returned no text (end_turn), 43 output tokens spent
```

Section 15.3 said "a cell whose call failed is not done: an error is usually a timeout or a rate
limit, and resuming should pick it up". **Not one of these 33 is that.** A model that reasons past
its token budget will reason past it again; a model that declines will decline again. Asking
again at identical settings buys the identical failure and pays for it, and those 25 truncations
generated 1024 output tokens each, so one resume repeats about US$0.16 of nothing with five
experiment arms still to run.

A failure is now recorded as retryable or not, and only a retryable one comes back. The decision
is made where the information is rather than by matching error strings afterwards: the adapter
knows the status and the exception type, and `administer` knows that a reply with no text is a
fact about the model rather than about the network.

**Nothing is lost by settling a deterministic failure**, and that follows from section 15.12
rather than from a special case. The request hash covers everything that determines the reply, so
raising the token budget asks all 25 again by itself, because it makes them different requests.
Keying resume on the request rather than on the cell is what makes this safe.

### 15.20 Items that cannot be answered because the question is not all there

`anthropic-haiku` returned eleven unparsed replies, and they are the most interesting thing in
the run so far. Three of them:

| Item | What the model said |
|---|---|
| "The Anglo-American model being considered the best model in light of the recession in the late 2000s." options `1,2,3` / `1,3,4` / `2,3,4` / `1,2,3,4` | "I need to see the numbered statements to evaluate which ones are correct" |
| "Which of these qualities is NOT listed as something to consider when choosing an opinion leader" | "I don't have access to the specific source material or textbook that lists the qualities" |
| "In this chapter's Senior View, Dr. Shealy advises you to" | "I don't have access to the specific chapter" |

The model is right in all three. The first asks which of four numbered statements are correct and
the statements are not in the item. The second and third refer to a textbook the item does not
carry. **These questions cannot be answered by anybody**, and a model that says so is describing
the item rather than failing it.

This is a third category for `docs/items-that-measure-nothing.md`, and it is not the same as the
first two. An item with near-zero discrimination is found by fitting the bank; an item whose
answer key is wrong is found by a model disagreeing with it consistently. **An item that is
incomplete is found by reading what a model says when it refuses**, which is a signal this
project has been throwing away as "unparsed" and should be reporting instead. The unparsed pile
is not all noise, and the share of it that is a well-formed complaint about the question is worth
counting on its own.

### 15.18 `mselect run`, the loop that spends the budget

Built 2026-09-12. Everything it needs already existed and had been exercised separately: the
suite is chosen and committed, `items` rebuilds administrable items from the cache, `administer`
scores them, `records` makes a run resumable, and the gateway owns the money. This is thin on
purpose.

Four things it does that a bare loop would not.

- **It writes as it goes**, appending after every chunk rather than every alias, so a run that
  dies at item 2,900 of 3,000 loses nothing. The record file is the state and the key is the
  request hash, so an edited configuration re-asks what changed and inherits what did not.
- **It shows the plan and stops.** Without `--yes` it prints what would be sent, per alias, and
  sends nothing. The plan names the model each alias resolves to, because a confirmation step
  that hides what is about to be called is not one.
- **It carries the per-model configuration**: vendor fields, omitted parameters and token
  budgets, all keyed by alias, so eleven models that need eleven different things get them
  without the experiment knowing anything about it.
- **It counts what a cost table cannot represent.** Unparsed replies and uncosted calls are
  reported separately from wrong answers and failed calls, because they are four different
  facts and three of them are findings rather than noise.

Proven end to end at zero cost before any vendor call: twenty items to `local-small-b` in two
chunks, all twenty scored; the same command again asks nothing and says so; changing the
template puts all twenty back on the list. Those twenty records are real and stay, which is what
resume is for.

Template and rotation are arguments, so the framing and position-bias arms of section 4.3 are
the same command with different flags rather than separate code.

### 15.17 Eleven of eleven, and a prompt change that nearly doubled the bill

**2026-09-12: every alias in the panel scored every item.** Thirty-three calls, US$0.033. Each
of the eleven routes is now confirmed by a real call rather than by a configuration file, which
is what section 15.4 said confirming meant.

Two things came out of the run that the pass rate hides.

**`together-open-b` wrote three uncosted rows.** Together serves prompt caching and reports
`cached_tokens` in its usage; gpt-oss-120b returned 66 to 68 per call, and the price file had no
`cache_read` rate for it. The gateway refuses to price a call that used a feature it has no rate
for, which is the right refusal and is the only reason this surfaced before an invoice did.
Rates read from the vendor's page and added: neither Together model discounts cached input,
where others in its catalogue do, so cache reads are charged as ordinary input. Not a repricing
and so not a new dated file: no rate changed, a missing one was added, and no row already costed
against that file would compute differently.

An uncosted reply is not a free one and not an error. The call happened and the bill for it is
unknown, which is the one outcome a cost table cannot represent, and it took reading the ledger
to notice three of them. `mselect smoke` now names them.

**Four aliases reported US$0.00000 while scoring.** The development cache had answered, because
their requests were unchanged since the previous run, which is section 3.3 working exactly as
written. But a paid alias reporting no spend is either a cache hit or a costing failure, and
those look identical in a total: it took opening the ledger to tell. A reply now records whether
the cache answered it, and `mselect smoke` says so. A claim that a rerun costs nothing should be
checkable from the run's own output rather than from a database.

**`anthropic-haiku` went from four output tokens an item to 291.** It wrote an essay. So did
Gemini, at 157 and 166, and Together's Llama at 106.

That was self-inflicted, one commit earlier. Rewriting the system prompt for reasoning models
(section 15.14) dropped "do not explain", on the grounds that a model which reasons cannot obey
it. **That conflated two different things.** A reasoning model's reasoning is internal and is not
the text it returns: it happens on a separate channel and the visible output can still be one
line. Dropping the instruction did not accommodate reasoning models, it invited every model to
write an essay, and output tokens are the expensive half of the bill.

| Prompt | Whole programme | With margin | Monthly cap |
|---|---:|---:|---|
| Answer-only, before the rewrite | US$21.27 | US$26.59 | fits |
| **After dropping "do not explain"** | **US$31.90** | **US$39.88** | US$40.00, by twelve cents |
| Answer-only restored, answer-last kept | **US$22.31** | **US$27.89** | fits |

It would have cleared the cap by twelve cents, which is the sort of margin that is indistinguishable
from luck. The instruction is back, with the guarantee that made the rewrite necessary in the
first place: reply with the answer only, and if you write more than the answer, put the answer
last. A model that can answer in one line does; a model that cannot still puts the answer where
the parser will find it.

**And a finding, not just a correction.** Restoring the instruction recovered two thirds of the
difference and no more. `anthropic-haiku` fell from 291 output tokens an item to 95;
`gemini-3.5-flash-lite` writes 318 and `gemini-3.8-flash` 188. Those are not essays invited by a
loose prompt, they are models explaining themselves while being told twice not to, and their
answers are right. Measured per item across the panel, answer-only output ranges from 4 tokens
to 318, a factor of eighty.

So **"answer-only" is an instruction a large part of the current frontier does not follow**, and
that belongs in the write-up rather than in a footnote. It is the same observation as section
15.13 with numbers behind it, and it has a consequence for anyone reading a published benchmark
number: a suite that scores only what it can parse from a short reply is measuring compliance as
well as ability, and the models it penalises are not the weak ones.

The general lesson is worth keeping, because it is the third version of the same mistake this
week. A change made for good reasons to one part of a measurement moved another part nobody was
looking at. It was caught only because the estimate reads real output tokens rather than assuming
them, which was itself a fix made two commits earlier for an unrelated reason.

### 15.16 Nine of eleven, and the two that were left were the same failure twice more

The smoke run of 2026-09-12 under the fixed parser and the new prompt: **nine of eleven aliases
scored every item**, twenty-seven calls for US$0.029. Three things left, and none of them is a
model that cannot do the work.

**`openai-frontier` refuses temperature, exactly as Claude 5 does.**

```
openai returned 400: Unsupported value: 'temperature' does not support 0.0 with this model.
Only the default (1) value is supported.
```

A third model and a second vendor. Whatever reason each gives, the effect on this experiment is
one thing: **three of eleven models run at the vendor's default sampling and cannot be pinned**,
so section 15.11's consequence grows by one. Test-retest now measures the temperature-0 floor
for eight models and default nondeterminism for three, and gpt-5.6-sol at temperature 1 is the
noisiest of them.

**`google-mid` refuses the thinking field that `google-frontier` accepts.** Same vendor, same
family, 400 INVALID_ARGUMENT on `gemini-3.5-flash-lite` and a clean 3 of 3 on
`gemini-3.8-flash`. So a vendor field is a property of the model and not of the provider, which
is the reason `request-extras.yaml` is keyed by alias. The field is removed for that route and
nothing is lost: its original problem was one reply truncated mid-preamble at sixteen tokens,
and the 1024-token budget of section 15.14 solves that without asking the vendor for anything.

**`openai-mid` answered a LegalBench item correctly and lost it.** The reply was `Answer: No`.
The options are `("No", "Yes")` and the key is A. The model named the right option and was
recorded unparsed for not lettering it.

That is the confound of section 15.14 again, in the one place it does the most damage.
LegalBench is 2,047 items of bank v1 and about 308 of the own-run suite, and **not one of them
offers lettered alternatives**: the options are Yes, No, Analysis, Rule. A model answering "No"
there is not failing an instruction in any way that should cost it an item, and MMLU has the
same shape whenever a model replies "Answer: Paris".

So an explicit statement is now matched against the option text as well as the letters, and only
where it is unambiguous: the statement names exactly one option, or the whole reply is one
option's text. **Never inside prose**, and that restraint is the point. "Yes" and "No" are
ordinary English words; `together-open-a` wrote "there is no clear connection" in a reply whose
answer was A, and reading that "no" as a choice would be a coin toss dressed as a measurement.

**The whole-bank check earned its keep here.** Reading option text before letters broke four
items, and they are a nice demonstration of why benchmarks are hard to score: MMLU carries logic
items whose options are `A`, `~A`, `B`, `~B`, and physics items offering `2c`, `c`, `0.8c`,
`0.5c`. On those, a reply of "B" or "c" is ambiguous between the label and the content. The
label is what the model was asked for, so a reply that is nothing but a letter is read as a
letter first, and option text only afterwards. Four items out of 19,919, caught by a test that
runs the whole bank rather than a fixture, before any of it cost a call.

Option text maps to a letter through `prompts.rotate`, the same function that lays the options
out in the prompt. The position-bias experiment permutes the display order, so "No" is not
always A, and two implementations of that rotation would come apart the first time one changed.

### 15.15 The parser could not read the format it asks for, and invented answers instead

The first smoke run under the new prompt, 2026-09-12. `local-small-a` replied with a paragraph
ending **"Answer: B"** and was recorded unparsed. Two defects, both harmless while every reply
was a single letter, both waiting for the moment replies became prose. That moment was section
15.14, one commit earlier.

**The explicit pattern was case sensitive.** It matched `answer: B` and not `Answer: B`, which is
the exact string `TEMPLATES["plain"]` now asks every model to end with. Every compliant reply
from a model that capitalises fell through to the fallback below.

**The fallback then manufactured answers out of English.** A standalone letter A to J counted as
a candidate, and "a" is the indefinite article. So a reply that reasoned and never answered,
containing the word "a", **scored as answering A**: not unparsed, not wrong, scored, with a
letter the model never chose.

That second one is the worst failure available to this code, and it is worth being precise about
why. Everything else that has gone wrong this week produced an absence: a failed call, an empty
reply, a refused parse. Each is visible and each is counted. This produced a **number**. Thirty
thousand cells of a response matrix, some unknown share of them answers no model gave, fitted
into item parameters and published as a calibration. And the document it would have corrupted
first is `docs/items-that-measure-nothing.md`, which would have become a report about this
function rather than about the benchmark. The module's own docstring names that risk in its
opening paragraph. It was still there.

The fallback stays, because "It's B." is a real answer and no template makes every model comply.
It now ignores the two letters that are also English words and refuses when more than one
candidate survives, which is the rule the module was written to follow. A reply that is nothing
but a letter is still read in any case, so "a" alone is still option A.

**Nothing published is affected.** `runner/parse.py` scores this project's own vendor calls and
nothing else; bank v1 and bank v2 carry per-item correctness as HELM and lm-eval-harness scored
it. No own run has happened. The cost of this defect was one smoke reply and an afternoon.

**It is also recoverable after the fact, by design.** `Administration.reply` stores the reply
text, so a scoring change can be applied to records that already exist without asking a vendor
anything. Section 15.3 stores replies under `out/`, which is gitignored, for the same reason it
stores no item text: a reply can quote the question back. That they are kept at all is what
makes a scorer fix free rather than a rerun.

Eight adversarial fixtures now cover prose replies specifically: a capital answer line, prose
that never answers, a lone letter in either case, an explicit A or I surviving the word
exclusion, reasoning that names options before settling, reasoning that never settles, and a
model correcting itself.

### 15.14 The panel keeps its reasoning models, so the run settings are built for them

**Peter's call, 2026-09-12, and it settles section 15.13.** A frontier tier without reasoning
models is not a frontier tier. Four of the eleven models reason before answering, across three
vendors, so this is what the current generation is rather than a quirk to route around. Three
things changed, and the token budget is the least of them.

**The budget is a ceiling, not a bill.** Output is billed on what a model generates, not on the
cap, so a model that replies "B" costs two tokens whether the cap is 16 or 1024. The 16-token
cap therefore saved nothing and cost four models their answers outright. It is now 1024 for
every model and every template. Uniform on purpose: a per-model budget is a per-model tuning
decision inside a comparison between models, and there is no version of that which is not a
thumb on the scale. The `tokens` override in `request-extras.yaml` remains for a model that
needs more than 1024, and any use of it is a fact about that model that gets reported.

The two budgets in `Settings` are now equal for the same reason. They differed when the
answer-only cap was 16, and giving the brief-reasoning template more room than the others would
have measured the room alongside the framing, when the framing is the thing being compared.

**The prompt now guarantees a final answer marker, and this is the real fix.** The parser
refuses to choose between letters when a reply names several without saying which is the answer.
That is correct, and reasoning replies trip it constantly: "rule out A, and C is negative, so D"
is three letters and no statement. Loosening the parser to take the last letter would be
guessing, and would guess wrong on "so it is not D". Instead `plain` now ends with `Give the
answer on a final line as "Answer: X"`, so the explicit branch fires for any model that
complies, and a model that does not is genuinely unparsed rather than ambiguous.

Without that change the run would have confounded ability with format compliance, and the
models penalised would have been exactly the reasoning ones. That is the failure this project
exists to find in other people's benchmarks, and it would have been in this one.

The system prompt changed for the same reason. It used to forbid explanation outright, which a
model that reasons internally cannot obey; it now requires the answer, in the format asked for,
as the last thing written.

**The cost estimate now uses what models actually generate.** It assumed every model fills its
token budget, which was a small overestimate at a cap of 16 and is nonsense at 1024: the worst
case times eleven models times 3,000 items, presented as the number that authorises the spend.
`mselect suite` reads real output-token counts from the smoke records, marks which lines rest on
measurement and which on the cap, and improves every time a smoke run happens.

**One trap inside that, which this nearly walked into.** A reply cut off by the cap is not a
measurement of what a model generates, it is a measurement of what it was allowed. Counting
those would have reported a reasoning model at sixteen tokens an item because sixteen was the
cap: the truncation reading itself back as evidence, understating the bill by an order of
magnitude. Only replies that reached a scorable answer count. The two aliases with no such reply
yet, `google-frontier` and `together-open-b`, are priced at the cap and labelled as the upper
bound they are.

| Estimate | Whole programme | Note |
|---|---:|---|
| Before any measurement, cap of 16 | US$13.26 | assumed every model fills a 16-token budget |
| Measured, cap of 1024 | **US$21.27** | nine aliases measured, two at the cap as a worst case |

Both caps still clear at the worst case, which is the version that matters. The two unmeasured
aliases have had their reasoning disabled since those records were written, so the figure should
fall once they are smoked again; it is not adjusted downward in advance of that.

**What it actually cost, 2026-09-13.** The full-suite arm was estimated at US$13.40 and the
ledger says **US$16.42**, 22% over, with `google-frontier` still owing about US$1.85 of items it
was refused. Which direction the error runs is the useful part: the estimate reads output
tokens from a smoke run of three items per model, and three items is too few to see a reasoning
model's spread. Two models account for the whole overrun, and both are the reasoning ones:

| alias | output tokens per item | share of the arm |
|---|---:|---:|
| `anthropic-opus` | 73.5 | US$4.92 |
| `openai-frontier` | 43.5 | US$4.99 |
| everything else together | - | US$6.51 |

So the estimator is sound and its sample is not. The smoke run exists to settle whether a route
works, and it was read as though it also settled what a route generates. The remaining arms are
estimated on the same three-item basis and should be read as low by something like a fifth;
there is now 3,000 items of measurement to re-estimate from, which is section 15.18's
`observed_output` reading the real record file rather than the smoke file.

**What this costs the framing experiment** is what section 15.13 said and is now settled rather
than open. Section 3.3 reserved reasoning for the framing experiment; that distinction is no
longer observable from outside, because a model that reasons internally does so under all three
templates. "Answer-only" is a claim about the output this project requires and not about the
computation the vendor performs. The three templates still measure exactly what they ask for,
which is a narrower question than the plan intended, and the write-up says so rather than
reporting the old one.

### 15.13 Opus 5 reasons before it answers, and Sonnet 5 does not

With the temperature fixed, `anthropic-sonnet` scored 3 of 3. `anthropic-opus` scored 1 of 3 and
returned **no text at all** on the other two, 16 output tokens spent, finish reason `max_tokens`.
So it reasons before answering and the answer-only budget buys none of the answer. That is the
fourth model to do this and the third distinct vendor, which makes it a property of the current
generation rather than of any one of them.

The same finding also says the diagnostic added on 2026-09-12 works: "the model returned no text
(max_tokens), 16 output tokens spent" is a sentence the previous version of this code could not
have produced, and it took one run to earn itself.

`request-extras.yaml` gains a `tokens` section for a per-alias budget, deliberately **left
empty** until the number is measured rather than guessed:
`mselect smoke --alias anthropic-opus --max-tokens N --yes` finds it, and because a budget is
inside the request hash, raising it really does re-ask the items.

What has to be decided once the number is known, and it is a decision about the experiment
rather than about configuration: section 3.3 says answer-only for the bank runs and reserves
reasoning prompts for the framing experiment. A model that reasons internally whatever it is
asked cannot be held to that condition. The honest reading is that "answer-only" describes the
**output format** this project requires and not the computation the vendor performs, that the
distinction has stopped being observable from outside, and that the framing experiment's
comparison of a plain against a brief-reasoning template is measuring something narrower than it
was designed to. That belongs in the write-up.

### 15.12 Resume was keyed on the wrong thing, and it took a repointed alias to show it

The third smoke run, 2026-09-12, reported **"every cell already recorded; nothing called"** for
four of seven aliases, including `openai-frontier`, which had been repointed from `gpt-5.4` to
`gpt-5.6-sol` an hour earlier. A different model answering the same question is a different
measurement and resume could not see it.

`records.done` keyed on the cell: alias, item, template, rotation. That is the right name for a
result and the wrong key for "have we already asked this", because it says nothing about what
was asked. Meanwhile `request_sha256` sat in every record, claimed in its own docstring to be
"everything that determines the reply", and was read by nothing. It was not true either: it
covered the prompt but not the model the alias resolves to, nor the vendor fields merged into
the body.

Both halves are fixed. A `Prompt` now carries the route it was sent by, which is
`provider/model` plus a fingerprint of its vendor fields, and that is inside the hash; `done`
returns hashes. Change a template, a token budget, a temperature, a vendor field or a model, and
those items are asked again. Change nothing, and an interrupted run resumes for free, which is
what section 15.3 promised.

**This mattered much more for the panel than for the smoke test.** A smoke run that skips four
aliases wastes a minute. A 3,000-item run that is interrupted, edited and resumed would have
produced a column of one model's answers under another model's name, with nothing anywhere
saying so, and the calibration would have been fitted on it.

Two defects in one week have had the same shape: a piece of evidence that described the
configuration rather than the request. The temperature in section 15.11 was the other. A record
that says what was configured instead of what was sent is not a small inaccuracy, because the
whole claim of this repository is that a stranger can check the numbers.

### 15.11 Two models will not take a temperature, which section 3.3 assumed they all would

`mselect smoke --no-batch` returned the same 400 six times on 2026-09-12:

```
anthropic returned 400: invalid_request_error: `temperature` is deprecated for this model.
```

Claude Sonnet 5 and Claude Opus 5 refuse the parameter outright. Not a bad value, the parameter
itself. That closes the last of the four route failures, and all four turned out to be the same
kind of thing: a request shaped for how models behaved when the plan was written.

**This one costs the experiment something, and the cost is not recoverable.** Section 3.3 says
every run is at temperature 0, fixed system prompt, answer-only, so that a rerun is the same
experiment. For these two models that is now impossible: they run at Anthropic's default
sampling because there is no longer any way to ask for anything else.

What follows, and what has to be said wherever these numbers are reported:

- **The test-retest experiment measures two different things.** For eight models it is the
  temperature-0 floor the plan intended. For Sonnet 5, Opus 5 and gpt-5.6-sol it is the
  vendor's default nondeterminism, which is a different and probably larger quantity.
  **Updated 2026-09-12**: gpt-5.6-sol refuses temperature too, in a second vendor's words
  ("does not support 0.0 with this model. Only the default (1) value is supported"), so this
  is three of eleven rather than two. Reporting one mean across
  all eleven would be averaging two measurements that are not the same measurement. Section 4.3
  gets a per-model column rather than a single figure, which it should have had anyway.
- **Their item responses are noisier than the rest of the panel's**, by an amount the
  test-retest will measure rather than assume. A 0 or a 1 from those three carries more sampling
  noise than a 0 or a 1 from the other eight, and the calibration should be read knowing that.
- **It is not a reason to drop them.** A panel of current models that excluded the two most
  capable ones because their vendor removed a parameter would be a worse panel and a less
  honest one. The answer is to measure the difference and report it.

**Where the fix lives matters.** Sending no temperature is done in `Prompt`, not in the adapter.
The request hash and the settings block in every record are both built from the prompt, so a
prompt carrying `None` produces a record that says `None` and a hash of the request that was
actually sent. Fixing it one layer lower would have left 3,000 records per model claiming a
temperature of 0 that was never sent, and a content hash of a request nobody made. That is the
difference between a limitation and a lie, and it is four lines of code.

`mselect/config/request-extras.yaml` carries an `omit` section for this, kept separate from the
`aliases` section that adds fields, because removing a field and adding one are different
operations and `temperature: 0` is not the same request as no temperature at all.

### 15.10 Local models

Run free on 2026-09-11, `local-small-a` answered four items across MMLU, MedQA, LegalBench and
MATH: four scored, two correct, none unparsed. That is the first end-to-end evidence that the
rebuilt LegalBench options and the extracted MATH keys work against a real model rather than
only against their own references.

`local-small-b` is `qwen2.5:3b` as of 2026-09-12, Peter's choice, and it is pulled: 1.93 GB at
3.1B parameters, beside `llama3.2:3b` at 2.02 GB and 3.2B. It was `qwen2.5:7b`, which was never
pulled and which is about 4.7 GB at 4-bit against the 4 GB card section 7 describes, so it would
have run on the CPU; the constraint for this slot is 3B to 4B.

### 15.4 The panel is configured but not yet chosen

`mselect/config/boundary.yaml` carries one route per alias in `prompts.PANEL`, so the code is
wired end to end. The local routes are settled and have answered. **The vendor routes are
provisional and are Peter's to confirm before any vendor spending.** They hold the current
identifiers from the 2026-09-10 price list and from the live calls the gateway made that day,
which found that Google's Flash thinks by default and returns no text inside a small token
budget while Flash-Lite answers.

Choosing the panel is a decision about the experiment rather than a detail of the plumbing,
and project 03 treated its own panel the same way: identifiers chosen, dated and recorded
before the first run rather than inherited from a configuration file nobody reread. The
`describe` helper exists so a run prints what each alias resolves to before it spends
anything.

The project's own caps live in `mselect/config/caps.yaml`, confirmed by Peter on 2026-09-11
at US$40 a month and US$30 a run, and the same figures appear in the gateway repository's
caps file so the portfolio total stays honest. The development cache is on, as section 3.3
requires, so a rerun costs nothing; that is the opposite of the drift runner's rule and
deliberately so.
