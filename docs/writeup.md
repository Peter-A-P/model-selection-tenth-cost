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
nine current models from four vendors, plus three on a laptop, asked the same 500 questions
twice at temperature 0, a day apart.

| model tier | agreement with itself | score moved |
|---|---:|---:|
| three models on a laptop | 0.998 to 1.000 | -0.2 to +0.2 points |
| nine hosted models | 0.936 to 0.984 | -2.0 to +1.0 points |

Temperature 0 is not determinism. Between 1.6 and 6.4 percent of answers changed on hosted
models with nothing changed at all: same prompt, same settings, same questions, a day apart. The
laptop models are effectively deterministic under the identical harness, so this is the service
rather than the measurement.

**The number to take away is the second column.** One model came back 2.0 points lower on the
same 500 questions it had already answered. If your release gate fires on a two-point drop, it
would have fired here, on nothing.

And it really is nothing rather than a trend: the flips are symmetric, 84 answers moving to right
against 85 moving to wrong across the whole panel, p = 1.00 on McNemar's exact test. That is a
random walk. Drift would have been the easier problem, because a systematic shift can be
corrected for and a random walk can only be measured and allowed for.

### The part that was a surprise

Item response theory treats an answer as a coin weighted by p, so a model re-asked an item it has
a 50-50 chance on should change its answer half the time. **It changes 2.8 percent of the time.**
Across the panel, answers move six times less often than the response model says they should,
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
| noise the response model predicts | 4.2 points | 1.9 points |
| noise actually observed | 1.7 points | 0.8 points |

So a paired re-run of the same questions is about two and a half times as sensitive as the
information function suggests. `mselect.reliability()` returns the measured figure, so a gate can
size itself from the measurement rather than from the theory.

The row above is the panel average, and **a gate should not use it.** `points_sd(n)` returns the
worst hosted model instead, 2.6 points on 100 items and 1.2 on 500, because a gate has to hold
for the model it is watching rather than for the average one. The three models on a laptop agree
with themselves almost perfectly at temperature 0, and averaging them in makes a gate look 1.5
times more sensitive than it can actually be for the hosted models it exists to watch. That is
the direction that passes a release it should have caught. `points_sd(n, pooled=True)` gives the
panel figure back for describing the panel.

## Finding 7: eleven of twelve models are worse when the answer is A

Same money, same twelve models. 300 multiple-choice questions, each asked four times with the
correct answer moved to a different option position and nothing else changed.

Accuracy goes **up** as the right answer moves down the list, for eleven of twelve:

| model | at A | at C | at D | spread |
|---|---:|---:|---:|---:|
| `local-small-a` | 0.387 | 0.542 | 0.490 | 0.220 |
| `local-small-b` | 0.455 | 0.609 | 0.615 | 0.160 |
| `together-open-a` | 0.730 | 0.803 | 0.822 | 0.092 |
| `local-mid-a` | 0.659 | 0.687 | **0.622** | 0.067 |
| `anthropic-opus` | 0.877 | 0.924 | 0.927 | 0.051 |
| `google-frontier` | 0.894 | 0.924 | 0.924 | 0.038 |

Eleven out of twelve, across four vendors, two open-weight models and three models running on a
laptop. That is not the direction the folklore predicts. The usual claim is that models prefer
the first option, and on this suite eleven of them are worst there.

**The twelfth is the model added last, and it is worth being exact about what it does and does
not overturn.** `local-mid-a` is still worse at A than at C, and it is the D column that breaks
the pattern: 0.622 (0.565 to 0.679) against 0.659 (0.607 to 0.708) at A. Those intervals overlap
almost entirely, so this is not a model that prefers the first option; it is a model whose
position effect is not ordered, measured on 262 items in that cell. The honest reading is that
the monotone version of this claim was always the weaker one, and that it took a twelfth model to
show it. The claim that survives unchanged is A against C, which holds twelve times out of twelve.

The spread is small for the good models and enormous for the small ones, which is worth saying
plainly: **a 22-point swing on `local-small-a` from moving the answer down the list** is larger
than the gap between most adjacent pairs of models in the ranking.

### The correction that changed the ranking

The obvious second number is how many individual questions change outcome when only the order
moves. The obvious way to count it is also wrong, and finding 6 is why.

Across four administrations, a model that disagrees with itself 3 percent of the time will answer
about 6 percent of questions inconsistently with no letter involved at all. The panel's flip rates
run from 0.0 to 6.4 percent. So the raw count is partly a measurement of instability, and charging
each model for its own reorders the table:

| model | raw | its own flip rate | net |
|---|---:|---:|---:|
| `local-small-a` | 43.3% | 0.2% | **42.9%** (37.6 to 48.6) |
| `local-small-b` | 36.3% | 0.0% | **36.3%** (31.3 to 42.0) |
| `local-mid-a` | 24.0% | 0.2% | **23.6%** (18.9 to 28.6) |
| `anthropic-haiku` | 17.3% | 1.6% | **14.1%** (9.8 to 18.5) |
| `together-open-a` | 19.3% | 5.0% | **9.4%** (5.0 to 14.0) |
| `openai-mid` | 20.3% | 6.4% | **7.7%** (3.4 to 12.4) |
| `together-open-b` | 14.0% | 6.1% | **1.9%** (0.0 to 5.9) |
| `openai-frontier` | 5.7% | 2.4% | **0.9%** (0.0 to 3.5) |

`openai-mid` reads third worst on the raw column and is mid-panel once charged. `together-open-b`
reads sixth and is ninth. `anthropic-haiku` goes the other way: it has the second lowest flip rate
on the panel, so almost all of its 17.3% really is the option order, and it belongs third rather
than fifth. Publishing the raw column would have put three models in the wrong place.

The spread column needs the same treatment in the opposite direction. Noise has no preferred
letter, so it does not push the spread up or down, but a maximum minus a minimum over four noisy
estimates is positive even when the truth is flat. That floor runs from 0.000 to 0.022 here, and
every one of the twelve spreads is above its own.

## Finding 8: the prompt format does not move the score, and for small models it moves everything else

This project runs its whole item bank answer-only, no reasoning, because reasoning tokens are
what make a 3,000-item bank expensive. The honest worry is that the bank is therefore measuring
something cheaper than the benchmark it claims to reproduce. So: 300 questions, twelve models,
three prompt templates, the same questions each time.

**For ten of the twelve, answer-only costs nothing measurable.** The interval on the difference
between the answer-only prompt and letting the model reason briefly first spans zero:

| model | answer only | reason briefly | difference |
|---|---:|---:|---|
| `anthropic-opus` | 0.902 | 0.903 | +0.0% (-2.0 to +2.0) |
| `google-frontier` | 0.910 | 0.910 | +0.0% (-1.7 to +2.0) |
| `openai-frontier` | 0.870 | 0.863 | -0.7% (-2.7 to +1.3) |
| `local-mid-a` | 0.632 | 0.674 | **+4.4% (+0.7 to +7.8)** |
| `anthropic-haiku` | 0.803 | 0.850 | **+5.0% (+2.0 to +8.4)** |

The two exceptions are the cheapest model from one vendor and the 7B on the laptop, and four to
five points is not nothing. What they do not share is a place in the ranking: `openai-mid` scores
lower than `anthropic-haiku` and pays nothing, and `local-small-a` and `local-small-b` score
lower than `local-mid-a` and their intervals span zero. So "small models need the reasoning
prompt" is not what this shows, and the two exceptions are so far a pair of facts about two
models rather than a pattern. Across the whole panel the template explains at most **0.3
percent** of the variance in whether an answer is right. The item explains 73 to 91 percent.
Which question you ask matters two hundred times more than how you dress it up.

### The part worth the extra work

A variance decomposition with one observation per cell cannot separate item-by-template
interaction from noise: they are the same term. That term ran from 8.7 to 26.6 percent and was
unreadable, because a model that disagrees with itself produces interaction without any template
doing anything.

Finding 6 measured the missing quantity. Two administrations of the same cell disagree with
probability 2q(1-q), so a measured flip rate gives the per-cell noise variance directly, and it
can be charged against the residual:

| model | interaction | its own noise | what is left |
|---|---:|---:|---:|
| `local-small-a` | 26.6% | 0.1% | **26.5%** |
| `anthropic-haiku` | 17.6% | 4.2% | **13.5%** |
| `local-small-b` | 12.1% | 0.1% | **11.9%** |
| `together-open-a` | 18.2% | 6.3% | **11.9%** |
| `local-mid-a` | 11.8% | 2.2% | **9.6%** |
| `google-mid` | 12.9% | 12.6% | **0.4%** |
| `anthropic-sonnet` | 8.8% | 9.5% | **0.0%** |

So the two halves of this finding point opposite ways and are both true. The template does not
move the score. For the weakest models it moves **which questions they get right**, by a quarter
of all the variance there is, and those changes cancel almost exactly in the total.

That is the same shape as finding 7, and probably the same underlying fact. A small model's
answer is decided by surface features of how a question is presented, and a frontier model's
answer is decided by the question. Every one of these prompts is a rephrasing that a human reader
would call irrelevant, and they are worth 26 percent of the variance to a 3B model and between
1 and 5 percent to the three frontier ones. Benchmark scores are usually compared as though the prompt were a neutral
container. It is neutral for the models that need the least help.

### The instruction is obeyed by some families and ignored by others

Everything above measures what the template does to **correctness**, and finds almost nothing:
0.3% of the variance at most. It says nothing about what the template does to **how much a model
writes**, and there the same instruction produces a sixty-fold spread.

Every model in the panel is asked to give the answer on a final line and nothing else. Measured
on identical items:

| family | tokens spent answering |
|---|---:|
| Qwen 7B, Llama 3B, and the nine hosted models | 4 to 15 |
| Gemma 3 4B | 268 |
| Gemma 3n 8B | 114 |
| Gemma 4, on-device and hosted alike | 780 to 1,021 |

That first row was a smoke-test figure when it was written and is now measured over 3,599 calls
for the 7B, which sharpens it in a way worth keeping. `local-mid-a` writes a median of 4 tokens
under the answer-only template and a median of 5 when it is explicitly asked to reason first, so
the middle of the distribution really does ignore the instruction. Its **mean** under that
template is 15.5, its 90th percentile 36 and its longest reply 251. It is not that the family
cannot reason on request; it is that it does so on a minority of items, and that minority is
where its 4.4-point gain in the table above comes from. A median is the right number for sizing a
run and the wrong one for asking whether an instruction was followed.

Four Gemma builds were tested locally and a fifth through Google's API, and all five write an
explanation whatever the prompt asks for. It is not a reasoning mode that can be switched off:
Gemma 4 on the Gemini API rejects a thinking budget outright, and the local builds move the words
from the thinking block into the prose when you suppress it.

**This is why Gemma is not in the panel.** Not the cost, which was zero on Google's API, but
because Findings 7 and 8 are experiments whose independent variable is the prompt format. A model
that does not follow the format instruction has not received the treatment, so including it would
mean reporting a format experiment run on a model the format never reached.

The practical form of this, for anyone sizing a benchmark run: **"answer only" is a request, not a
setting.** If you budget tokens on the assumption it is honoured, you will be wrong by a factor of
sixty on some model families, and on a 1,024-token cap Gemma 4 returns an empty reply rather than
a short one, which reads as a failed call rather than a verbose one.

### The caveat this used to carry, and what happened when it was paid off

Until 2026-09-18 the paragraph here said that the flip rate subtracted above was measured under
the answer-only template alone, that a reasoning prompt has more room to wander and is probably
less stable, and that the net figures were therefore upper bounds. Closing it took a second
administration of both other templates, 7,200 calls and US$3.38. The guess was half right and the
half it got wrong is the more useful half.

| template | flip rate, pooled | worst hosted |
|---|---:|---:|
| `plain` | 0.0280 | 0.0658 |
| `letter_only` | **0.0241** | 0.0546 |
| `brief_reasoning` | **0.0303** | 0.0584 |

`brief_reasoning` is the least stable and `letter_only` the most, which is the predicted
ordering, but the pooled spread is 0.006 and nothing in the table above moves much on it.

**Per model it is not one ordering at all**, and that is what the arm bought:

| model | `plain` | `letter_only` | `brief_reasoning` |
|---|---:|---:|---:|
| `local-mid-a` | 0.002 | 0.013 | **0.030** |
| `together-open-a` | 0.050 | **0.010** | 0.040 |
| `openai-mid` | **0.064** | 0.053 | 0.043 |
| `google-frontier` | **0.018** | 0.033 | 0.027 |

The 7B on the laptop is **fifteen times** less consistent when asked to reason than when asked
for an answer, which is the caveat's fear in its strongest form. `openai-mid` runs the other way
and is at its *least* consistent under answer-only. `together-open-a` is five times steadier
under `letter_only` than under `plain`. So "charge each template its own noise" was the right
correction, and "the net figures are upper bounds" was the wrong prediction: they moved in both
directions, `together-open-a` from 8.7% up to 11.9% and `local-mid-a` from 11.5% down to 9.6%,
and the model whose interaction disappears into its own noise changed from `together-open-b` to
`anthropic-sonnet`.

The item term is still not noise-free, and that is not fixed by anything here.

**Finding 7 never had this problem**, which the old caveat obscured by naming both findings. All
four option rotations are administered under `plain`, so the flip rate charged there was always
measured under the template being analysed.

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
6. **If you compare multiple-choice scores**, rotate the answer position and report the spread.
   It costs one extra administration and on this panel it was worth up to 22 points on a small
   model. If you cannot afford to rotate, at least do not compare two models whose option orders
   were generated differently.
7. **Subtract your own noise before you believe any of it.** Both numbers in findings 7 and 8
   are partly a measurement of the model disagreeing with itself, and on the raw figures three
   models sit in the wrong place. You need the test-retest number from finding 6 first; it is
   the cheapest measurement here and it is what makes the rest readable.

## What this write-up does not cover

**A second administration of the `letter_only` and `brief_reasoning` templates.** The flip rate
that findings 7 and 8 both subtract as noise was measured under the answer-only template alone,
so a template with more room to wander is charged too little and every net figure in them is an
upper bound. That is 7,200 calls and about US$5 on items already chosen, and it is on the list
rather than done.

Everything else in the own-run half is now measured and is above: position bias (finding 7),
prompt framing (finding 8), test-retest at temperature 0
(finding 6), and the cost per ranking decision, which is **US$0.73 to rank twelve models as well
as asking them all 2,815 questions does, against US$15.97 to ask everything.**


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
