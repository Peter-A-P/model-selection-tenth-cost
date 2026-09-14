# What did not work

Rule C: name an approach tried and rejected, with the evidence. The first was the project's own
headline assumption. The second would have corrupted an item bank silently rather than loudly,
which is the more instructive failure.

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

## Rejected: the evaluation harness's own content hash as item identity

**The idea.** lm-eval-harness writes a `doc_hash` column beside every per-item result: a hash of
the document, computed by the thing that ran the evaluation. Bank v2 needs exactly that, to
answer "is this the same question the other model was asked". Recomputing it looked like
duplicated work.

**What the measurement says.** It is not a hash of the document. It is a hash of the harness's
serialisation of the document, so it moves when the harness moves. On
`leaderboard_math_num_theory_hard`, a quarter of the panel disagrees with the reference model
about **every one of the 154 values**, while the problem text at each position is
character-for-character identical:

```
doc_id 0, reference model: 'How many perfect square factors does the number 46,656 have?'
doc_id 0, disagreeing model: 'How many perfect square factors does the number 46,656 have?'
doc_hash: different
```

Keyed that way, each of those items would have become two items, each answered by part of the
panel, each with half the evidence behind its parameters. Nothing in the build would have
complained: the item count would have risen, which looks like more data.

**What replaced it, in two corrections.** The question text is read out of the Parquet document
column and hashed here, and then discarded without being stored. That exposed the same problem
one level down, in the answer key, which the first replacement hashed alongside the question the
way bank v1 does: two releases of MATH-Hard write the same answer as `\infty` and `\iny`, and as
`-\frac{1}{{}2x}` and `-\frac1{2x}`. That split 33 of 307 algebra items and dropped 74 of the
400 models from that task, for a difference that is typographic.

So the identity is the question, and the key is used only where it has to be: two of
`leaderboard_bbh_causal_judgement`'s 187 questions are asked twice with the opposite key, and
those are genuinely two measurements. The drift itself is reported rather than absorbed, per
task, in the bank manifest.

**The general form, which matters more than this bank.** An identifier published by an
evaluation harness may be a hash of a serialisation rather than of the thing. A bank built from
someone else's harness inherits that harness's version drift, and the only defence is to hash
the question yourself and check that every model agrees about what each position means. Bank v2
checks all 400 models on all 36 tasks for exactly this, which is not a precaution that was in
the plan; it is one the plan needed.

## Also tried, also abandoned

**`py-irt` and PyMC for the fit.** Planned in section 4.1. Dropped after the matrix turned out
to be small enough that Bock-Aitkin EM in numpy fits it exactly in under a minute; a 1.5 GB
PyTorch dependency and a variational approximation would both have cost more than they bought.
The reasoning and the replacement cross-check are in PLAN.md section 13.2.

**The Open LLM Leaderboard per-sample datasets as the primary source.** Planned in section 3.1.
They are gated behind a signed-in Hugging Face token, verified request by request in
[data-sources.md](data-sources.md). HELM's open buckets replaced them for bank v1 and turned out
to be larger. With a token they are readable and are bank v2, so this one was deferred rather
than rejected.

**Asking Hugging Face for access to each gated dataset.** An early draft of the v2 loader
called the `ask-access` endpoint for every model in the panel. Deleted once the gate was
measured properly: the same range request answers 401 without a token and 206 with one, for a
repository the account has never touched. The loader now contains no POST at all.

**Diagrams delivered as source code as a fourth broken-item category.** MATH carries 44 items
whose figure arrives as Asymptote drawing code rather than as a picture, under questions that
say "the graph of $y = f(x)$ is shown below". The expectation was that these are unanswerable in
the same way as the incomplete items in
[items-that-measure-nothing.md](items-that-measure-nothing.md), and that models would score at
or below chance on them.

They do not. Over the own-run panel of eleven models, accuracy on those items is 0.733 (0.631 to
0.815) against 0.808 (0.776 to 0.837) on the other MATH items in the suite: intervals that
overlap across most of their width, on 86 scored replies. Whatever penalty exists is smaller
than this panel can resolve, and the honest reading is that the models simply read the drawing
code.

It is kept here rather than dropped because the rule that replaced it was found the same way and
could have gone the same way. An item that looks broken to a person reading it is a hypothesis,
not a finding, and the difference between the two is a measurement with an interval on it.

## What a reader should not conclude

None of this says item response theory is the wrong tool for benchmarks. It says the thing a
leaderboard publishes is not the thing item response theory estimates, and any claim of the form
"same ranking, fewer calls" has to name which ranking it means. The 12.7 times saving is real
and measured; so is the crossover that ends it.
