"""Write the JSON the static page reads.

Every figure on the page comes from an artefact this repository already produces: the frozen
bank, the leave-one-model-out simulation, the own-run validation, and the three experiments.
Nothing is retyped, so the page cannot drift from the README's table without this command
being rerun and the difference showing up in a diff.

One thing on that page is not precomputed, and it is deliberate. The adaptive test itself runs
in the browser, item by item, against the recorded answers of the twelve own-run models: the
selector, the posterior and the stopping rule are a few dozen lines of arithmetic, and a
visitor who can watch the interval narrow as items arrive has understood the project in a way
no chart of a finished result conveys. What the browser must not do is invent a response, so
what ships is the response matrix as measured, one bit per cell, and the price the ledger
recorded for each of those cells. The page can only replay what was actually asked and
actually paid for.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from mselect import paths, power
from mselect.cat.estimate import Ability
from mselect.cat.select import benchmark_strata
from mselect.data import bank as bank_mod
from mselect.data.benchmarks import title as title_of
from mselect.experiments import ownrun

#: Where `mselect demo build` writes, beside the page that reads it.
DEFAULT_DATA_DIR = paths.ROOT / "demo" / "data"

#: The panel administration the page replays: the committed suite, answer-only prompt,
#: no option rotation. The same file `mselect validate` reads.
PANEL_RECORDS = "own-run-plain-0.jsonl"

#: Item parameters for the cloud chart are sampled, because 20,365 points is a picture of a
#: blob rather than of a bank, and every one of them would be sent over the wire to draw it.
#: Seeded, so the same items are drawn every build and a reader comparing two versions of the
#: page is looking at the same sample.
CLOUD_SAMPLE = 500
CLOUD_SEED = 0

#: The grid the page's "how many items do you need" control moves over. Precomputed here
#: rather than in the browser because `items_needed` iterates over the bank's information
#: function, and because a number the repository computed is a number a reader can check.
EFFECT_POINTS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0)
ABILITIES: tuple[float, ...] = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
POWERS: tuple[float, ...] = (0.8, 0.9)


@dataclass(frozen=True, slots=True)
class Written:
    """One file written, and how big it turned out. The page is a file copy, so size is a fact
    worth printing: it is the whole cost of hosting it."""

    path: Path
    bytes: int


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=False)


def _round(values: NDArray[np.float64] | list[float], places: int) -> list[float]:
    """Rounded floats, with a non-finite value carried across as None rather than as NaN.

    `NaN` is not JSON. A bank records an unidentified difficulty as NaN on purpose, so the
    page has to be told about it rather than handed a number that is not one.
    """
    out: list[float] = []
    for value in np.asarray(values, dtype=float).tolist():
        out.append(round(float(value), places) if np.isfinite(value) else float("nan"))
    return out


def _nullable(values: NDArray[np.float64], places: int) -> list[float | None]:
    return [
        round(float(v), places) if np.isfinite(v) else None
        for v in np.asarray(values, dtype=float).tolist()
    ]


def _bits(flags: NDArray[np.bool_]) -> str:
    """A boolean row as base64 packed bits: 2,815 answers in 470 characters rather than 5,630."""
    return base64.b64encode(np.packbits(np.asarray(flags, dtype=bool)).tobytes()).decode("ascii")


def _read_json(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def panel_payload(version: str = "v1", kind: str = "2pl") -> dict[str, Any]:
    """The twelve own-run models, the items they were asked, and what each answer cost.

    Restricted to the common frame, the items every model answered, for the reason
    `Panel.dense` gives: a ranking is a comparison, and a comparison needs one item set.
    """
    panel = ownrun.load(paths.out_for(version) / PANEL_RECORDS, version=version, kind=kind).dense()
    strata, weights, names = benchmark_strata(panel.benchmarks)
    accuracy = panel.accuracy()
    spend = panel.spend()

    models: list[dict[str, Any]] = []
    for row, alias in enumerate(panel.aliases):
        correct = np.nan_to_num(panel.responses[row], nan=0.0) > 0.5
        ability = Ability()
        for column in range(panel.n_items):
            if np.isfinite(panel.responses[row, column]):
                ability.update(column, int(panel.responses[row, column]), panel.items)
        lo, hi = ability.interval()
        cost = np.nan_to_num(panel.cost_usd[row], nan=0.0)
        models.append(
            {
                "alias": alias,
                "accuracy": round(float(accuracy[row]), 4),
                "ability": round(ability.theta, 4),
                "ability_lo": round(lo, 4),
                "ability_hi": round(hi, 4),
                "full_usd": round(float(spend[row]), 6),
                # Micro-dollars, as integers. A tenth of a cent is below the resolution of
                # anything the page says, and an integer array is a third the size of the
                # float one.
                "cost_micro": [round(value * 1e6) for value in cost.tolist()],
                "correct": _bits(correct),
            }
        )

    return {
        "bank_version": version,
        "kind": kind,
        "n_items": panel.n_items,
        "n_models": panel.n_models,
        "benchmarks": [{"name": name, "title": title_of(name)} for name in names],
        "weights": _round(weights, 6),
        "items": {
            "a": _round(panel.items.a, 4),
            "b": _round(panel.items.b, 4),
            "c": _round(panel.items.c, 4),
            "stratum": [int(value) for value in strata.tolist()],
            "usable": _bits(panel.usable),
        },
        "models": models,
    }


def _curve_rows(simulation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in simulation["curve"]:
        rows.append(
            {
                "items": int(point["items"]),
                "method": str(point["method"]),
                "tau": round(float(point["tau"]), 4),
                "lo": round(float(point["tau_lo"]), 4),
                "hi": round(float(point["tau_hi"]), 4),
            }
        )
    return rows


def _own_run_rows(validation: dict[str, Any]) -> list[dict[str, Any]]:
    """The own-run checkpoints in the same shape the simulation curves use."""
    methods = {
        "adaptive": "tau_adaptive",
        "stratified": "tau_stratified",
        "random": "tau_random",
    }
    rows: list[dict[str, Any]] = []
    for point in validation["checkpoints"]:
        for method, key in methods.items():
            estimate = point[key]
            rows.append(
                {
                    "items": int(point["n_items"]),
                    "method": method,
                    "tau": round(float(estimate["point"]), 4),
                    "lo": round(float(estimate["lo"]), 4),
                    "hi": round(float(estimate["hi"]), 4),
                }
            )
    return rows


def curves_payload() -> dict[str, Any]:
    """The headline: rank agreement against number of items, for all three selection methods.

    Three panels, deliberately. Two of them are simulations inside a public leaderboard's own
    results, one on each bank, which is the only way to get a hundred models. The third is the
    twelve models this repository paid to run, which is the only one where the models were
    never in the fit at all.
    """
    v1 = _read_json(paths.out_for("v1") / "simulation-2pl.json")
    v2 = _read_json(paths.out_for("v2") / "simulation-2pl.json")
    own = _read_json(paths.out_for("v1") / "own-run-validation-plain-0.json")

    panels: list[dict[str, Any]] = []
    for key, simulation, label, note in (
        (
            "v1",
            v1,
            "Bank v1: 150 models from HELM",
            "74 models held out of the fit one at a time, ranked against their own full-suite "
            "score over 18,921 items.",
        ),
        (
            "v2",
            v2,
            "Bank v2: 400 open-weight models",
            "150 models held out of the fit, over 20,323 items from the Open LLM Leaderboard. "
            "A different source, a different harness and a different kind of model.",
        ),
    ):
        equivalent = simulation["equivalent_budget"][0]
        panels.append(
            {
                "key": key,
                "label": label,
                "note": note,
                "n_items": int(simulation["full_suite_items"]),
                "n_models": int(simulation["block"]["models"]),
                "tau_full_fit": round(float(simulation["tau_of_the_full_fit"]), 4),
                "equivalent": {
                    "adaptive_items": int(equivalent["adaptive_items"]),
                    "random_items": round(float(equivalent["random_raw_score"]), 1),
                    "tau": round(float(equivalent["adaptive_tau"]), 4),
                },
                "curve": _curve_rows(simulation),
            }
        )

    transfer = own["transfer_tau"]
    panels.append(
        {
            "key": "own",
            "label": "The twelve models this project paid to run",
            "note": (
                "Models the bank never saw, asked through this repository's own prompts and "
                "parser. The item parameters were read as given and nothing was refitted."
            ),
            "n_items": int(own["n_items"]),
            "n_models": len(own["aliases"]),
            "tau_full_fit": None,
            "transfer": {
                "point": round(float(transfer["point"]), 4),
                "lo": round(float(transfer["lo"]), 4),
                "hi": round(float(transfer["hi"]), 4),
            },
            "full_suite_usd": round(float(own["full_suite_usd"]), 4),
            "cost": [
                {
                    "items": int(point["n_items"]),
                    "usd_adaptive": round(float(point["usd_adaptive"]), 4),
                    "usd_random": round(float(point["usd_random"]), 4),
                }
                for point in own["checkpoints"]
            ],
            "curve": _own_run_rows(own),
        }
    )
    return {"panels": panels}


def items_payload(version: str = "v1", kind: str = "2pl") -> dict[str, Any]:
    """A sample of the bank's item parameters, and the share of each benchmark that is dead.

    The sample is what the page draws; the shares are counted over every item, so a reader is
    never shown a percentage computed from five hundred points.
    """
    bank = bank_mod.load(version)
    items, frame, _ = bank_mod.load_params(bank, kind)
    diagnostics = pl.read_parquet(paths.out_for(version) / f"item-diagnostics-{kind}.parquet")
    benchmarks = bank.benchmarks
    difficulty = frame["b"].to_numpy()
    slope = frame["a"].to_numpy()

    rng = np.random.default_rng(CLOUD_SEED)
    groups: list[dict[str, Any]] = []
    for name in sorted(set(benchmarks.tolist())):
        inside = np.flatnonzero(benchmarks == name)
        a_all = slope[inside]
        take = inside if inside.size <= CLOUD_SAMPLE else rng.choice(inside, CLOUD_SAMPLE, False)
        take = np.sort(take)
        groups.append(
            {
                "name": name,
                "title": title_of(name),
                "n_items": int(inside.size),
                "share_low": round(float((a_all < 0.3).mean()), 4),
                "share_negative": round(float((a_all < 0.0).mean()), 4),
                "a": _round(slope[take], 3),
                "b": _nullable(difficulty[take], 3),
                "sampled": int(take.size),
            }
        )

    flags = {
        str(column): int(diagnostics[column].sum())
        for column in ("no_information", "everyone_right", "everyone_wrong")
        if column in diagnostics.columns
    }
    return {
        "bank_version": version,
        "bank_hash": bank.bank_hash,
        "n_items": int(items.n_items),
        "n_models": int(bank.n_models),
        "sample_per_benchmark": CLOUD_SAMPLE,
        "flags": flags,
        "benchmarks": groups,
    }


def _flip_rates(version: str) -> dict[str, dict[str, float]]:
    """The measured flip rate per model under each prompt template, from the retest summaries."""
    out: dict[str, dict[str, float]] = {}
    for template in ("plain", "letter_only", "brief_reasoning"):
        stem = f"own-run-retest-{version}" + ("" if template == "plain" else f"-{template}")
        path = paths.ROOT / "mselect" / "config" / f"{stem}.json"
        if not path.is_file():
            continue
        agreement = _read_json(path).get("agreement", {})
        out[template] = {
            str(alias): round(1.0 - float(value), 4) for alias, value in agreement.items()
        }
    return out


def experiments_payload(version: str = "v1") -> dict[str, Any]:
    """The three things a single accuracy figure cannot say: noise, option order, framing."""
    retest = _read_json(paths.ROOT / "mselect" / "config" / f"own-run-retest-{version}.json")
    position = _read_json(paths.out_for(version) / "position-bias.json")
    framing = _read_json(paths.out_for(version) / "framing.json")

    order: list[dict[str, Any]] = []
    for model in position["models"]:
        net = model["share_order_dependent_net"]
        order.append(
            {
                "alias": str(model["model"]),
                "net": round(float(net["point"]), 4),
                "lo": round(float(net["lo"]), 4),
                "hi": round(float(net["hi"]), 4),
                # Only the four positions every rotation used. The tail letters exist because
                # a handful of MMLU-Pro items carry ten options, and eleven observations on
                # "E" is not a measurement of anything.
                "by_position": {
                    key: round(float(model["accuracy_by_position"][key]["point"]), 4)
                    for key in model["positions_counted"]
                },
            }
        )

    template: list[dict[str, Any]] = []
    for model in framing["models"]:
        cost = model["cost_of_answer_only"]
        template.append(
            {
                "alias": str(model["model"]),
                "accuracy_by_template": {
                    key: round(float(value["point"]), 4)
                    for key, value in model["accuracy_by_template"].items()
                },
                "cost": round(float(cost["point"]), 4),
                "lo": round(float(cost["lo"]), 4),
                "hi": round(float(cost["hi"]), 4),
                "variance_item": round(float(model["variance_item"]), 4),
                "variance_template": round(float(model["variance_template"]), 4),
                "variance_interaction": round(float(model["variance_interaction_net"]), 4),
            }
        )

    return {
        "retest": {
            "measured": retest.get("measured"),
            "n_models": retest.get("n_models"),
            "pooled_flip_rate": retest.get("observed_flip_rate"),
            "worst_hosted_flip_rate": retest.get("worst_hosted_flip_rate"),
            "agreement": {
                str(alias): round(float(value), 4)
                for alias, value in retest.get("agreement", {}).items()
            },
        },
        "flip_rates": _flip_rates(version),
        "n_items": int(position["n_items"]),
        "order": order,
        "framing": {
            "templates": [str(name) for name in framing["templates"]],
            "n_items": int(framing["n_items"]),
            "models": template,
        },
    }


def power_payload(version: str = "v1", kind: str = "2pl") -> dict[str, Any]:
    """`items_needed` over a grid, so the page's slider answers without a network call.

    The grid is the function this repository exports, evaluated here rather than approximated
    there. What the page adds is the sentence the number needs: the same answer read as a
    budget, at the price the own-run panel actually paid per item.
    """
    items = bank_mod.default_items(version, kind)
    grid: list[dict[str, Any]] = []
    for effect in EFFECT_POINTS:
        for ability in ABILITIES:
            for level in POWERS:
                answer = power.items_needed(effect, level, ability, items=items)
                grid.append(
                    {
                        "effect": effect,
                        "ability": ability,
                        "power": level,
                        "items": int(answer.items),
                        "capped": bool(answer.capped),
                    }
                )
    return {
        "bank_version": version,
        "items_available": int(items.n_items),
        "effects": list(EFFECT_POINTS),
        "abilities": list(ABILITIES),
        "powers": list(POWERS),
        "grid": grid,
    }


def build_all(out_dir: Path | None = None) -> list[Written]:
    """Write every data file the page reads, and say how large each one came out."""
    directory = out_dir or DEFAULT_DATA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict[str, Any]] = {
        "panel.json": panel_payload(),
        "curves.json": curves_payload(),
        "items.json": items_payload(),
        "experiments.json": experiments_payload(),
        "power.json": power_payload(),
    }
    written: list[Written] = []
    for name, payload in payloads.items():
        path = directory / name
        text = _json(payload)
        path.write_text(text, encoding="utf-8")
        written.append(Written(path, len(text.encode("utf-8"))))
    index = directory / "index.json"
    manifest = {
        "files": [{"name": item.path.name, "bytes": item.bytes} for item in written],
    }
    index.write_text(_json(manifest), encoding="utf-8")
    written.append(Written(index, len(_json(manifest).encode("utf-8"))))
    return written
