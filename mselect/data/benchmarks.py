"""What each benchmark in the bank is, and when it was published.

The publication month is not decoration: it is the cut that the contamination analysis in
PLAN.md section 4.1 uses. An item that is much easier for models released after its benchmark
was published than their ability predicts is a contamination candidate.

Dates are the month of first public release of the benchmark (the arXiv posting, which is when
the items became scrapeable), with the identifier recorded so a reader can check.

`scoring` says how the item was turned into a 0 or a 1. Two benchmarks appear in both banks and
are scored differently in each, because the harnesses differ: bank v1 takes HELM's stated final
answer, bank v2 takes lm-eval-harness's highest-likelihood option. That difference is not a
detail, it is the thing `mselect crossbank` measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Benchmark:
    name: str
    title: str
    published: str  # ISO month
    source: str
    scoring: str


BENCHMARKS: Final[dict[str, Benchmark]] = {
    "mmlu": Benchmark(
        "mmlu",
        "Massive Multitask Language Understanding",
        "2020-09",
        "arXiv:2009.03300",
        "exact match on the chosen letter",
    ),
    "mmlu_pro": Benchmark(
        "mmlu_pro",
        "MMLU-Pro",
        "2024-06",
        "arXiv:2406.01574",
        "v1: chain-of-thought answer correctness; v2: highest-likelihood option",
    ),
    "gpqa": Benchmark(
        "gpqa",
        "GPQA (graduate-level Q&A)",
        "2023-11",
        "arXiv:2311.12022",
        "v1: chain-of-thought answer correctness; v2: highest-likelihood option",
    ),
    "math": Benchmark(
        "math",
        "MATH (competition mathematics)",
        "2021-03",
        "arXiv:2103.03874",
        "equivalence of the final answer",
    ),
    "gsm8k": Benchmark(
        "gsm8k",
        "GSM8K (grade-school word problems)",
        "2021-10",
        "arXiv:2110.14168",
        "exact match on the final number",
    ),
    "med_qa": Benchmark(
        "med_qa",
        "MedQA (US medical licensing questions)",
        "2020-09",
        "arXiv:2009.13081",
        "exact match on the chosen option",
    ),
    "legalbench": Benchmark(
        "legalbench", "LegalBench", "2023-08", "arXiv:2308.11462", "quasi-exact match on the label"
    ),
    "bbh": Benchmark(
        "bbh",
        "BIG-Bench Hard",
        "2022-10",
        "arXiv:2210.09261",
        "highest-likelihood option, length-normalised",
    ),
    "musr": Benchmark(
        "musr",
        "MuSR (multistep soft reasoning)",
        "2023-10",
        "arXiv:2310.16049",
        "highest-likelihood option, length-normalised",
    ),
    "math_hard": Benchmark(
        "math_hard",
        "MATH level 5 (the hardest competition problems)",
        "2021-03",
        "arXiv:2103.03874",
        "exact match on the final answer",
    ),
    "commonsense": Benchmark(
        "commonsense",
        "OpenBookQA",
        "2018-09",
        "arXiv:1809.02789",
        "exact match on the chosen option",
    ),
}


def published(name: str) -> str:
    benchmark = BENCHMARKS.get(name)
    return benchmark.published if benchmark else ""


def title(name: str) -> str:
    benchmark = BENCHMARKS.get(name)
    return benchmark.title if benchmark else name
