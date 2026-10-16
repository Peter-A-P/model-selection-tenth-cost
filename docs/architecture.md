# Architecture

Three figures: the system as a whole, the adaptive test as a loop, and how each published
number is produced. Each names the module that implements what it shows; [PLAN.md](../PLAN.md)
says why, and [writeup.md](writeup.md) makes the argument for a practitioner.

## The system

![Public per-item results are frozen into two banks and fitted by item response theory; the diagnostics name the items that measure nothing; the adaptive test runs against twelve current models' recorded answers and against simulation; the report and the demo are written from the same artefacts, and the power function is what project 03 imports.](architecture/system.svg)

The left column is the bank. Public per-item results from HELM and the Open LLM Leaderboard
(`mselect.data.helm`, `mselect.data.ollm`) are fetched, frozen and hashed into two independent
banks under `mselect/bank`, fitted as 2PL and 3PL models by marginal maximum likelihood
(`mselect.irt.fit`) with standard errors on every parameter, and diagnosed for item fit,
differential item functioning, local dependence and dimensionality (`mselect.irt`); the items
that measure nothing are listed with their evidence. The middle column is the test. Twelve
current models were administered the frozen 3,000-item suite through the portfolio gateway,
which enforced the spend caps and recorded every call (`mselect.runner`,
`mselect.experiments.ownrun`); the adaptive test (`mselect.cat`) selects by Fisher information,
estimates by expected a posteriori and stops at a precision target or when two models have come
apart, and runs against those recorded answers, which is models the fit never saw; the three
experiments (`mselect.experiments`) measure test-retest noise, option order and framing on the
same panel; and `mselect.power` with `mselect.handover` is what project 03 imports. The right
column is what is published: the simulation of adaptive against random subsampling
(`mselect.cat.simulate`), the report that writes the README tables and figures
(`mselect.report`), and the page at adaptive.peterparker.ca, five static files and a folder of
JSON written by `mselect demo build` (`mselect.demo`).

## The adaptive test

![From a prior ability the test picks the item carrying the most Fisher information at that ability, balanced across benchmarks, administers it from the recorded answers or a live call, updates the ability by expected a posteriori with its standard error, and stops at a target precision or when two models have come apart; random subsampling at the same budget usually fails to settle it.](architecture/adaptive-loop.svg)

The loop starts from a prior ability for the model under test. `mselect.cat.select` picks the
item in the calibrated bank with the most Fisher information at the current ability, balanced
across benchmarks; the item is administered from the recorded answers, or live through the
gateway when a caller is handed in (`mselect.runner`); `mselect.cat.estimate` updates the
ability by expected a posteriori with its standard error; and `mselect.cat.stop` ends the test
at a target precision or when two models' intervals have come apart, otherwise the loop picks
the next item. `items_needed` says in advance how many items a decision can need, so a budget
too small to find the drop you care about is known before the first call. The result is an
ability with its interval, the items used and the dollars spent; random subsampling at the
same budget is the comparison, and at ten items it usually has not settled the ranking.

## How the published numbers are produced

![Each number comes from a stated input through one command into a stored artefact; the report writes the README tables and the demo's data from the same artefacts, and every number carries a 95% interval.](architecture/measurement.svg)

The fit (`mselect.irt.run_fit`) writes the parameters and the broken-item report from the two
public banks. The simulation (`mselect.cat.run_sim`) writes the curve of Kendall's tau against
items used, adaptive against random, on models held out of the fit. `mselect run` administered
the 3,000-item suite to twelve current models through the gateway for US$18.25 and `mselect
validate` reads the records back for nothing, which is where the transfer result and the cost
per ranking decision come from (`mselect.experiments.ownrun`). The three experiments
(`mselect.experiments.analysis`) re-asked the same models and added two prompt templates for
US$3.38, so each template is charged noise measured under that template. The cross-bank check
(`mselect.experiments.crossbank`) asks whether difficulty calibrated on one panel means
anything on the other across the 998 questions they share. `mselect report` writes every
number in the README from those artefacts, and `mselect demo build` writes `demo/data`, so
nothing on the page can say anything the repository did not measure. Every number carries a
95% interval.
