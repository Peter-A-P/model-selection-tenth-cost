"""Model selection at a tenth of the cost.

Item response theory fitted to public per-item benchmark results, then computerised adaptive
testing over the calibrated bank. PLAN.md is the specification and the README carries the
measured results.

This module is the supported interface. Everything re-exported here is what project 03 and any
other consumer import, and it does not change without a version bump; everything else in the
package is internal and free to move.

There are two banks. `v1` is the default and is HELM; `v2` is the Open LLM Leaderboard, and every
call below that names a version takes it. They are not interchangeable: each identifies its own
ability scale against its own panel, and over the 998 questions they share, difficulty correlates
0.71 for the items that discriminate in both and not at all for the rest. Filter on
discrimination before importing difficulty from either.

    import mselect

    mselect.items_needed(3, 0.8, ability=0.0)   # how many items to see a three point drop
    mselect.dependence()["math"].effective_items(100)   # 100 MATH items are worth how many?
    mselect.dependent_blocks()                  # items that are the same question asked twice
    mselect.reliability()                       # how much of a score is noise
    mselect.load_bank()                         # the frozen bank itself
    mselect.banks()                             # which bank versions ship in this release

The adaptive test is `Ability` (the running estimate), `Selector` (what to ask next) and the two
stopping rules, `precision_reached` and `separated`.
"""

from mselect import power
from mselect.cat.estimate import Ability, score
from mselect.cat.select import Selector, benchmark_strata
from mselect.cat.stop import Decision, precision_reached, separated
from mselect.data.bank import Bank, default_bank, default_items
from mselect.data.bank import available as banks
from mselect.data.bank import load as load_bank
from mselect.handover import (
    Dependence,
    DependentBlock,
    Reliability,
    dependence,
    dependent_block_index,
    dependent_blocks,
    reliability,
)
from mselect.irt.model import Items
from mselect.power import ItemsNeeded, detectable_effect, items_needed, power_at

__version__ = "0.3.0"

__all__ = [
    "Ability",
    "Bank",
    "Decision",
    "Dependence",
    "DependentBlock",
    "Items",
    "ItemsNeeded",
    "Reliability",
    "Selector",
    "__version__",
    "banks",
    "benchmark_strata",
    "default_bank",
    "default_items",
    "dependence",
    "dependent_block_index",
    "dependent_blocks",
    "detectable_effect",
    "items_needed",
    "load_bank",
    "power",
    "power_at",
    "precision_reached",
    "reliability",
    "score",
    "separated",
]
