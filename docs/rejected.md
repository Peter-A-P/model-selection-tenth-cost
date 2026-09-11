# What did not work

Rule C: name an approach tried and rejected, with the evidence. This one was the project's own
headline assumption, which makes it worth writing down properly.

## Rejected: adaptive ability testing as a drop-in replacement for the benchmark score

**The idea.** Fit item response theory to the bank, then rank models by their adaptively
estimated ability instead of by their benchmark score. Since ability is estimated from the most
informative items, the ranking should match the full suite at a fraction of the calls, at any
budget. The plan (section 10) expected the opposite approach, random subsampling, to be the one
that failed.

**What the measurement says.** The leave-one-model-out simulation over 74 models and 18,921
items, with the item parameters refitted without each held-out model, produces this (Kendall's
tau against the full-suite ranking, 95 percent bootstrap intervals over models):

| Items asked | Adaptive (IRT) | Stratified sample | Random sample, IRT scored | Random sample, raw score |
|---:|---|---|---|---|
| 10 | **0.778** (0.716 to 0.832) | 0.559 (0.457 to 0.647) | 0.536 (0.427 to 0.646) | 0.477 (0.349 to 0.593) |
| 25 | **0.816** (0.758 to 0.869) | 0.725 (0.649 to 0.789) | 0.687 (0.610 to 0.759) | 0.472 (0.354 to 0.578) |
| 50 | **0.845** (0.790 to 0.892) | 0.750 (0.672 to 0.819) | 0.770 (0.694 to 0.831) | 0.700 (0.611 to 0.778) |
| 100 | **0.858** (0.801 to 0.903) | 0.825 (0.771 to 0.870) | 0.814 (0.760 to 0.864) | 0.754 (0.687 to 0.814) |
| 200 | 0.851 (0.797 to 0.897) | **0.892** (0.851 to 0.925) | 0.878 (0.836 to 0.915) | 0.856 (0.814 to 0.890) |
| 800 | 0.870 (0.820 to 0.909) | 0.905 (0.870 to 0.936) | 0.910 (0.875 to 0.940) | **0.916** (0.881 to 0.946) |

Adaptive selection is far ahead up to about a hundred items and then stops improving, while
every baseline walks past it. Ten adaptive items rank the panel as well as 127 randomly chosen
ones, a 12.7 times saving; a hundred adaptive items are worth only about 216, and beyond two
hundred the saving is negative.

**Why, measured rather than guessed.** It is not a broken estimator. The adaptive ability
estimate converges on the ability the full fit assigns:

| Items asked | tau vs the ability fitted on all 18,921 items | tau vs the full-suite score |
|---:|---:|---:|
| 10 | 0.811 | 0.778 |
| 50 | 0.884 | 0.845 |
| 200 | 0.898 | 0.851 |
| 800 | 0.922 | 0.870 |

The left column is doing what a computerised adaptive test is supposed to do. The ceiling is in
the right column, and it is set before any adaptive test runs: the ability fitted on **every**
item in the suite agrees with the suite's own average ranking only at tau 0.921. Ability and
benchmark score are different constructs. The suite average weights all 18,921 items equally,
including the 4,221 that carry no information and the 1,757 whose slope is negative; the ability
estimate weights them by how much they discriminate, and per-benchmark abilities correlate
between 0.42 and 0.96, so there is no single ability for a composite to be a clean summary of.
A random sample scored as a proportion is an unbiased estimator of the suite average and
converges to it by construction. An ability estimate cannot, however many items it is given.

**What survives the rejection.** The method is kept, with its claim narrowed to where the
evidence supports it:

- **Screening and pairwise decisions, tens of items.** "Is this model better than that one" and
  "which three of these twenty are worth a full evaluation" are ability questions, and ten to a
  hundred adaptive items answer them at a tenth of the calls or better.
- **Power and item quality.** The fitted bank answers "how many items do I need to detect a
  three-point regression" (118 at mid-panel ability) and names the items that measure nothing.
  Neither of those has a subsampling equivalent at all.
- **Reproducing the published leaderboard number.** Use a random sample and score it directly.
  It is simpler, it is unbiased, and above two hundred items it is also more accurate.

The README states the crossover rather than the best-case number alone, and the headline chart
is titled after it.

## Also tried, also abandoned

**`py-irt` and PyMC for the fit.** Planned in section 4.1. Dropped after the matrix turned out
to be small enough that Bock-Aitkin EM in numpy fits it exactly in under a minute; a 1.5 GB
PyTorch dependency and a variational approximation would both have cost more than they bought.
The reasoning and the replacement cross-check are in PLAN.md section 13.2.

**The Open LLM Leaderboard per-sample datasets as the primary source.** Planned in section 3.1.
They are gated behind a signed-in Hugging Face token now, verified request by request in
[data-sources.md](data-sources.md). HELM's open buckets replaced them and turned out to be
larger.

## What a reader should not conclude

None of this says item response theory is the wrong tool for benchmarks. It says the thing a
leaderboard publishes is not the thing item response theory estimates, and any claim of the form
"same ranking, fewer calls" has to name which ranking it means. The 12.7 times saving is real
and measured; so is the crossover that ends it.
