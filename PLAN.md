# Plan: Model Selection at a Tenth of the Cost

**Written:** 2026-09-06. **Amended:** 2026-09-10, section 13.
**Status:** the public-data half is built and measured; the own-run panel is not started.
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
- [ ] Validation on the own-run panel of models the calibration never saw. **Blocked**: needs the
      gateway's batch support and this project's spend caps. Held-out validation within the
      public panel is done instead (leave-one-model-out, item parameters refitted without the
      held-out model)
- [ ] Position bias, framing effects and test-retest reliability each measured with intervals.
      **Blocked on the same thing.** A free partial arrived anyway: repeated HELM administrations
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
- [ ] `mselect` v0.1.0 tagged; repository public. **Peter's decision**, not the build's

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
portfolio gateway (project 04), whose Message Batches support lands in its v0.2 in October, and
they need this project's own spend caps and Peter's go. Until then the own-run rows in the
README say so rather than being quietly dropped.

### 13.5 Item text is not committed

Section 12 said to keep item-level raw responses as Parquet. The responses are committed; the
item text is not. The bank stores the content hash, the HELM instance id, the scenario and a
short preview, which is enough to find any item in the cache and enough evidence for the
broken-item report, without republishing benchmark questions in a public repository. GPQA items
carry no preview at all, at its authors' request.

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
