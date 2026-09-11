# Changelog

Versions follow semantic versioning on a 0.x line: the interface re-exported from `mselect`
itself is stable within a minor version, and everything else in the package is internal.

## v0.1.0 - 2026-09-11

First tagged version, handed to project 03 (AI Release Gate) as its statistical core. The
calibration half of PLAN.md is built and measured; the own-run half is not, and this entry says
exactly where the line falls.

### What is in it

**The frozen bank.** `mselect/bank/v1/`, content hash `1d4c357935c70875`: 150 models by 20,365
binary-scored items from the public HELM per-item releases, across MMLU, MMLU-Pro, GPQA, MATH,
GSM8K, MedQA, LegalBench and OpenBookQA. Items keyed by a content hash of the question, its
options and its answer key. Item text is not committed; the hash, the HELM instance id and a
short preview are.

**Item parameters.** 2PL and 3PL fitted by marginal maximum likelihood (Bock-Aitkin EM, MAP with
stated weakly informative priors), with standard errors on every parameter, stored next to the
bank with the bank hash and the fit version that produced them. Fit flags per item: negative or
low discrimination, infit and outfit, non-monotone empirical curve, perfect separation, no
information.

**The interface.**

| Call | What it gives |
|---|---|
| `mselect.items_needed(effect, power, ability)` | items per model to detect a drop of `effect` accuracy points |
| `mselect.detectable_effect(n_items, power, ability)` | the smallest drop `n_items` can see |
| `mselect.power_at(n_items, effect, ability)` | power of a given budget |
| `mselect.dependent_blocks()` | item blocks that are not independent evidence |
| `mselect.dependent_block_index()` | item id to block number |
| `mselect.reliability()` | agreement across repeated administrations, with its caveat |
| `mselect.load_bank()`, `default_bank()`, `default_items()` | the bank and its fitted parameters |
| `mselect.Ability`, `Selector`, `precision_reached`, `separated` | the adaptive test |

All three power functions default to the packaged bank, so `items_needed(3, 0.8, 0.0)` works
with no setup and returns 118 items.

**Diagnostics, published rather than summarised.** Yen's Q3 per benchmark, tetrachoric
dimensionality with parallel analysis, Mantel-Haenszel and logistic differential item
functioning by access and by model generation. `docs/diagnostics.md`.

**The measured result.** Ten adaptively chosen items rank a 74-model panel as well as 127
randomly chosen ones, and the advantage ends above about 200 items. `docs/rejected.md` has the
argument and the crossover.

### What is not in it, and should not be assumed

- **No own-run reliability.** PLAN.md section 4.3 specifies test-retest at temperature 0 on a
  panel this project runs itself. That needs vendor calls and is blocked on the gateway's batch
  support and on spend caps. `reliability()` returns what does exist, which is the agreement
  between repeated HELM administrations of the same model and item, 95.3 percent over 8,431
  repeated cells, and it labels itself as a floor rather than the planned experiment. Expected
  in v0.2.0.
- **No position-bias or framing figures.** Same reason. The analyses are written and tested
  against fixtures (`mselect.experiments.analysis`); only the runs are missing.
- **No cost per ranking decision in dollars.** Same reason.
- **`items_needed` is optimistic for large effects.** Against the simulation it is well
  calibrated for small gaps and over-promises for large ones: for model pairs 5 to 10 accuracy
  points apart at 10 items it predicts 88 percent separation where 42 percent was observed. It
  assumes local independence, and `dependent_blocks()` is the evidence that the bank does not
  have it. Treat the number as a floor on the items needed, not a promise.
- **`dependent_blocks()` is a lower bound.** Q3 is computed on a reproducible sample of each
  benchmark's items, so dependence between two items that were not both sampled cannot appear.
  The file records the sample size and seed.
- **The ability scale belongs to this panel.** Item parameters are identified against a standard
  normal prior over the 150 models in the bank. A bank calibrated on a different panel is a
  different bank, and abilities are not comparable across the two.
