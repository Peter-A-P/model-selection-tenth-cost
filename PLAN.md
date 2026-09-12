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
multiple choice, a boxed final answer for MATH), `max_tokens` small. Development caching
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

Mirrors the portfolio's definition for this project. Ticked 2026-09-11 against what is
measured and committed; everything unticked is blocked on vendor calls or on a decision.

- [x] IRT parameters fitted and published for a real item bank. **150 models by 20,365 items**,
      2PL and 3PL, standard errors on every parameter, bank content-hashed at `1d4c357935c70875`
- [x] Adaptive selection against random and stratified baselines, Kendall's tau with bootstrap
      intervals. **Reproduced at a far smaller share than 10%, and with a crossover the plan did
      not anticipate**: 10 adaptive items match 127 random ones (tau 0.778), and above about 200
      items the baselines win. Section 13.6 and `docs/rejected.md`
- [ ] Validation on the own-run panel of models the calibration never saw. **Blocked on one
      thing as of 2026-09-11**: project 04's `v0.2.0` tag, which its own plan says waits on one
      live batch call and one live local call. The spend caps this line used to wait on were
      confirmed by Peter the same day (US$40 a month, US$30 a run, in 04's `config/caps.yaml`),
      and the batch support was written rather than deferred to October. Held-out validation
      within the public panel is done instead (leave-one-model-out, item parameters refitted
      without the held-out model)
- [ ] Position bias, framing effects and test-retest reliability each measured with intervals.
      **Blocked on the same one thing.** A free partial arrived anyway: repeated HELM administrations
      of the same model and item agree 95.3% of the time (n = 8,431)
- [x] A list of items that measure nothing, with evidence per item. `docs/items-that-measure-nothing.md`
- [x] Q3 and dimensionality diagnostics reported. `docs/diagnostics.md`
- [x] `items_needed` power function validated against the simulation. Validation table in
      `out/simulation-2pl.json`; **it is optimistic at large effects**, see section 13.6
- [x] Handed to project 03. `mselect` v0.1.0 tagged 2026-09-11 with the interface 03's plan
      calls (`items_needed(delta, 0.8, ability)` works with no bank argument), plus
      `dependence()` for the local-dependence correction and `reliability()` for the noise floor
- [ ] Cost per ranking decision reported in dollars. **Blocked**: needs the own-run panel
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
| anthropic-sonnet | 0/3, the batch errored | identifier, probably | **unresolved** |
| anthropic-opus | 0/3, the batch errored | identifier, probably | **unresolved** |

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

**The two Anthropic identifiers are left exactly as they were**, because they are wrong or they
are not and replacing them with a guess is how a wrong one gets into a routing file.
`mselect models` asks each vendor for its own list and marks every route the vendor does not
carry, ranking the near misses by how much of the name they share, weighted so that a token
common to every identifier counts for less: without that, "claude" scores as highly as "opus"
and every Claude model ties for first. Listing models generates no tokens and so is billed
nowhere, which is why it is a check rather than a cost.

It is worth noting that `anthropic-haiku` is the one Anthropic route carrying a dated
identifier, and the one that works.

### 15.9 Local models

Run free on 2026-09-11, `local-small-a` answered four items across MMLU, MedQA, LegalBench and
MATH: four scored, two correct, none unparsed. That is the first end-to-end evidence that the
rebuilt LegalBench options and the extracted MATH keys work against a real model rather than
only against their own references.

`local-small-b` is `qwen2.5:3b` as of 2026-09-12, Peter's choice. It was `qwen2.5:7b`, which was
never pulled and which is about 4.7 GB at 4-bit against the 4 GB card section 7 describes, so it
would have run on the CPU; the constraint for this slot is 3B to 4B. It needs
`ollama pull qwen2.5:3b` before the panel runs.

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
