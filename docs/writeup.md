# Your benchmark is measuring fewer things than you think

A practitioner's write-up of what fell out of fitting item response theory to 150 language
models by 20,365 public benchmark items. Every number here comes from a file `mselect report`
regenerates from the repository, and the run that produced it is named in
[items-that-measure-nothing.md](items-that-measure-nothing.md) and
[diagnostics.md](diagnostics.md).

## The setup, in one paragraph

Benchmarks are tests, and tests have a century of statistics behind them. Item response theory
models the probability that a test-taker answers an item correctly as a function of one number
for the test-taker (ability) and two or three for the item (difficulty, discrimination, and for
multiple choice a guessing floor). Fit it to a matrix of models by items and you get something a
leaderboard cannot give you: a per-item measure of how much each question actually contributes
to telling models apart. HELM publishes per-instance results for hundreds of models across its
releases, which is exactly that matrix, free and needing no account.

## Finding 1: a fifth of your benchmark is not measuring anything

Of 20,365 items, 4,262 (20.9 percent) have a fitted discrimination below 0.3, and 4,221 carry
essentially no information at the middle of the panel's ability range. Those items still cost a
call, still cost tokens, still take wall-clock time, and still move your headline average
around by sampling noise.

It is not spread evenly:

| Benchmark | Items | Discrimination below 0.3 | Negative slope |
|---|---:|---:|---:|
| MMLU | 13,937 | 18.6% | 7.2% |
| LegalBench | 2,047 | 53.0% | 27.2% |
| MedQA | 1,000 | 20.1% | 7.1% |
| GSM8K | 1,000 | 4.9% | 0.9% |
| MMLU-Pro | 998 | 14.8% | 5.0% |
| OpenBookQA | 500 | 4.4% | 0.6% |
| GPQA | 446 | 32.3% | 12.1% |
| MATH | 437 | 5.9% | 0.7% |

GSM8K, MATH and OpenBookQA are in good health by this measure. Over half of LegalBench's items,
as administered in HELM Lite, do not separate strong models from weak ones, and more than a
quarter run backwards.

## Finding 2: 1,757 items run backwards

An item with a negative slope is one that stronger models get wrong more often. There are 1,757
of them, 8.6 percent of the bank. The worst have empirical curves like this, reading left to
right from the weakest fifth of the panel to the strongest:

```
item 67577ba4d1b7385e (MMLU)        0.85  0.33  0.10  0.00  0.00
item 7c55045b69d786a1 (MMLU)        0.54  0.14  0.00  0.00  0.00
item 7053ad01d780d54c (LegalBench)  0.17  0.00  0.00  0.00  0.00
```

Four things produce that shape, and they are not equally interesting: a mis-keyed answer, an
ambiguous question where the better model sees the ambiguity, a grader that marks a correct
answer wrong, and a genuine inverse-scaling item. Separating them needs the item text and a
human, which is why the report names no item as mis-keyed. But if you own a benchmark, this
list is where to look first, and it takes one fit to produce.

## Finding 3: the items are not independent, and nobody's error bars say so

Yen's Q3 measures the correlation between two items' residuals once ability is accounted for.
Under the assumption every benchmark confidence interval quietly makes, it should sit slightly
below zero. Measured on the bank, the share of item pairs above 0.2 runs from 8 percent
(MMLU-Pro) to 31 percent (MATH). Items share passages, templates, subjects and answer formats,
and they move together.

The practical consequence: a benchmark of 500 items does not carry 500 items' worth of
independent evidence, so the standard error on your headline accuracy is smaller than the truth,
and two models whose intervals just barely separate may not be separated at all. Report the
statistic; it takes one pass over the residuals.

## Finding 4: the benchmarks do not measure one thing

Fit each benchmark separately and correlate the abilities across the models that took both. The
correlations run from 0.42 (GPQA against LegalBench) to 0.96. A composite score over these is a
weighted average of several different abilities with weights nobody chose deliberately, and it
is set by how many items each benchmark happens to contribute.

## Finding 5: adaptive testing buys a lot, and then stops

Choosing each next item by maximum Fisher information at the model's current ability estimate,
with content balancing across benchmarks, gives this against the full 18,921-item ranking
(Kendall's tau, 95 percent bootstrap interval over 74 held-out models):

| Items asked | Adaptive | Best baseline at the same budget |
|---:|---|---|
| 10 | **0.778** (0.716 to 0.832) | 0.559 |
| 50 | **0.845** (0.790 to 0.892) | 0.770 |
| 200 | 0.851 (0.797 to 0.897) | **0.892** |
| 800 | 0.870 (0.820 to 0.909) | **0.916** |

Ten adaptively chosen items rank the panel as well as 127 randomly chosen ones: a 12.7 times
saving on the screening decision. Above about two hundred items the advantage is gone and a
plain random sample, scored the ordinary way, does better.

The reason is worth internalising because it is not a bug in anyone's code. Ability and
benchmark average are different quantities. The ability fitted on **every** item in the suite
agrees with the suite's own ranking only at tau 0.921, because the suite average weights the
dead items and the backwards items exactly as heavily as the good ones. A random sample is an
unbiased estimator of that average and converges to it; an ability estimate converges to
something else. [rejected.md](rejected.md) has the full argument and the numbers behind it.

## Finding 6: your model disagrees with itself, and by more than you would gate on

Everything above is measured on other people's published results. This one needed our own money:
eleven current models from four vendors, plus two small ones on a laptop, asked the same 500
questions twice at temperature 0, a day apart.

| model tier | agreement with itself | score moved |
|---|---:|---:|
| two small models on a laptop | 0.998 and 1.000 | 0.0 and -0.2 points |
| nine hosted models | 0.936 to 0.984 | -2.0 to +1.0 points |

Temperature 0 is not determinism. Between 1.6 and 6.4 percent of answers changed on hosted
models with nothing changed at all: same prompt, same settings, same questions, a day apart. The
laptop models are effectively deterministic under the identical harness, so this is the service
rather than the measurement.

**The number to take away is the second column.** One model came back 2.0 points lower on the
same 500 questions it had already answered. If your release gate fires on a two-point drop, it
would have fired here, on nothing.

And it really is nothing rather than a trend: the flips are symmetric, 83 answers moving to right
against 85 moving to wrong across the whole panel, p = 0.94 on McNemar's exact test. That is a
random walk. Drift would have been the easier problem, because a systematic shift can be
corrected for and a random walk can only be measured and allowed for.

### The part that was a surprise

Item response theory treats an answer as a coin weighted by p, so a model re-asked an item it has
a 50-50 chance on should change its answer half the time. **It changes 3 percent of the time.**
Across the panel, answers move five times less often than the response model says they should,
and that holds for every model separately.

The reason is that p is not what it looks like. It describes how models *at the same ability*
differ from each other, not how *one model* differs from itself between administrations. A given
model's answer to a given question is close to fixed; what the model calls chance is largely a
persistent model-by-question quirk.

That points the friendly way for anyone building a gate. A drift test compares a model against
its own earlier self on the same questions, so it sits in the small within-model variance rather
than the large across-model one:

| | on 100 items | on 500 items |
|---|---:|---:|
| noise the response model predicts | 4.0 points | 1.8 points |
| noise actually observed | 1.7 points | 0.8 points |

So a paired re-run of the same questions is about twice as sensitive as the information function
suggests. `mselect.reliability()` returns the measured figure, and `points_sd(n)` gives the
column on the right, so a gate can size itself from the measurement rather than from the theory.

## What to do on Monday

1. **If you are choosing between models**, ask ten to fifty well-chosen questions per model
   rather than running everything. The ranking you get is as good as a hundred-plus random
   items, and the interval tells you when to stop.
2. **If you are watching for regressions**, measure your own noise floor before you set a
   threshold. Run the same 500 questions twice a day apart and see how far the score moves with
   nothing changed. On this panel that was up to two points. A threshold below your own floor is
   an alarm that fires on the weather.
3. **Then ask how many items you actually need.** On this bank, detecting a three-point drop at
   80 percent power takes 118 items per model at mid-panel ability; detecting a one-point drop
   takes 3,559. If your release gate runs 200 items and claims to catch one-point regressions,
   it does not. Those figures come from the information function, which is conservative for a
   paired re-run of the same questions by about a factor of two: see finding 6.
4. **If you own a benchmark**, fit a 2PL to whatever per-item results you already have and read
   the bottom of the discrimination list. The dead items and the backwards items are free to
   find and cost you money every run.
5. **If you publish a number**, publish Q3 and a dimensionality check beside it. Both are cheap,
   and both change how the interval should be read.

## What this write-up does not cover

**Position bias** across cyclic option permutations, and **prompt-framing effects**. Both need
vendor calls that have not been made. The analyses are written and tested; what is missing is
the money, about eight dollars, and the README says so rather than estimating them.

Everything else in the own-run half is now measured and is above: test-retest at temperature 0
(finding 6), and the cost per ranking decision, which is **US$0.72 to rank eleven models as well
as asking them all 2,830 questions does, against US$16.05 to ask everything.**


## Postscript: it replicates, and the transfer number is the one to take away

Everything above is one item bank: 150 models scored by HELM. The same method was then run on a
second bank with almost nothing in common with the first, 400 open-weight Open LLM Leaderboard
submissions by 20,323 items, scored by a different harness that reads the highest-likelihood
option instead of a stated answer.

The shape survives the change of everything. Ten adaptive items rank the second panel as well as
143 random ones, against 127 on the first. The crossover where random sampling overtakes adaptive
selection is in the same place. And the ceiling is nearly identical: ability fitted on every item
agrees with the suite average at tau 0.928 on the second bank and 0.921 on the first, which means
the construct gap in section 4 is a property of the two measurements and not of one dataset.

The two banks also happen to contain the same 998 MMLU-Pro questions, which makes the caveat every
item bank carries into a measurement. Over all 998, the two calibrations of difficulty correlate
**-0.04**. Over the 532 that separate models in both banks, they correlate **+0.71**.

The first number looks like a refutation of the whole idea and is a division problem. Difficulty
is `-d/a`: when an item's discrimination is indistinguishable from zero, its difficulty is a
quotient by nothing, and the unfiltered correlation is dominated by those. So the practical rule
is short, and it is the one thing to take from this whole write-up if you take nothing else:

**Item parameters travel between panels. The items that measure nothing do not, and they will
swamp the ones that do unless you drop them first.**

Which is why a report of which questions measure nothing is not an appendix to an item bank. It
is the part you have to read before you are allowed to use the rest.
