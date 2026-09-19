"""The page at adaptive.peterparker.ca can only say what the repository measured.

The page is the one artefact here that a reader meets before the README, so the things that
would make it lie quietly are what these tests are about: data files that have drifted from the
run they claim to come from, a response matrix that lost a bit in packing, a page that asks for
a file nobody writes, and the hosting policy that keeps the whole thing off-origin.

Nothing here needs a browser. The arithmetic the browser does is a port of `mselect.cat`, which
has its own tests; what cannot be tested there is whether the numbers handed to it are the ones
this repository produced, and that is testable here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mselect import paths
from mselect.demo import build
from mselect.demo.serve import declared_headers

DEMO = paths.ROOT / "demo"
DATA = DEMO / "data"


def _read(name: str) -> Any:
    """The payloads are JSON written by this repository for a browser to read, so they are
    typed here the way a browser sees them: as whatever came out of the file."""
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def unpack(encoded: str, length: int) -> Any:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8))
    return bits[:length]


def test_the_page_asks_for_exactly_the_files_that_are_written() -> None:
    """A data file the page never reads is dead weight; one it reads and nobody writes is a
    blank page. Both are silent, so the two lists are compared rather than trusted."""
    page = (DEMO / "app.js").read_text(encoding="utf-8")
    asked = set(re.findall(r'"(panel|curves|items|experiments|power|index)"', page))
    written = {path.stem for path in DATA.glob("*.json")}
    assert asked == written


def test_the_response_matrix_survives_being_packed_into_bits() -> None:
    """Every answer on the page is one bit of a base64 string. A bit lost in packing would be a
    model silently credited with an answer it did not give."""
    panel = _read("panel.json")
    n_items = int(panel["n_items"])
    models = panel["models"]
    assert isinstance(models, list)
    for model in models:
        correct = unpack(str(model["correct"]), n_items)
        assert correct.size == n_items
        # The proportion correct the bits imply is the accuracy the validation reported, to
        # within the one item in a thousand that rounding the stored figure allows.
        assert float(correct.mean()) == pytest.approx(float(model["accuracy"]), abs=0.001)


def test_the_panel_matches_the_validation_it_claims_to_come_from() -> None:
    """The page and the README's own-run table have to be describing one administration."""
    panel = _read("panel.json")
    validation = json.loads(
        (paths.out_for("v1") / "own-run-validation-plain-0.json").read_text(encoding="utf-8")
    )
    assert panel["n_items"] == validation["n_items"]
    aliases = [model["alias"] for model in panel["models"]]
    assert sorted(aliases) == sorted(validation["aliases"])
    for model in panel["models"]:
        expected = validation["accuracy"][model["alias"]]
        assert model["accuracy"] == pytest.approx(expected, abs=0.001)


def test_every_item_the_page_can_ask_has_parameters_to_ask_it_with() -> None:
    """The selector chooses by Fisher information, which is not a number for an item whose
    difficulty was never identified. Those items are marked unusable rather than left to become
    a NaN in the browser."""
    panel = _read("panel.json")
    items = panel["items"]
    assert isinstance(items, dict)
    n_items = int(panel["n_items"])
    usable = unpack(str(items["usable"]), n_items)
    a = np.array(items["a"], dtype=float)
    b = np.array(items["b"], dtype=float)
    assert np.isfinite(a[usable == 1]).all()
    assert np.isfinite(b[usable == 1]).all()
    assert usable.sum() > 0


def test_the_cost_of_a_subset_never_exceeds_the_cost_of_the_whole_run() -> None:
    """The page prices a short test by adding up what those exact calls cost. If the per-cell
    prices did not add to the run's own total, the saving on the page would be invented."""
    panel = _read("panel.json")
    for model in panel["models"]:
        total = sum(model["cost_micro"]) / 1e6
        assert total == pytest.approx(float(model["full_usd"]), abs=0.002)


def test_every_model_is_named_by_the_identifier_the_vendor_returned() -> None:
    """The page shows the model, not the alias. An alias is a handle the code needs so a vendor
    renaming a model cannot break it; a reader comparing `local-small-a` with `local-small-b`
    learns nothing, and comparing `llama3.2:3b` with `qwen2.5:3b` learns the whole point."""
    panel = _read("panel.json")
    records = paths.out_for("v1") / build.PANEL_RECORDS
    returned: dict[str, set[str]] = {}
    with records.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            name = row.get("model_returned")
            if isinstance(name, str) and name:
                returned.setdefault(str(row["alias"]), set()).add(name)
    for model in panel["models"]:
        # One alias answered by two different models would mean the run spans a vendor's
        # rename, and naming it on the page would then be a claim about the wrong thing.
        assert len(returned[model["alias"]]) == 1, model["alias"]
        assert model["model"] == next(iter(returned[model["alias"]]))
        assert model["hosted"] is not model["alias"].startswith("local-")


def test_the_readme_names_the_same_models_the_page_does() -> None:
    """`mselect report` writes that table and `mselect demo build` writes the page's, from the
    same records. Running one and not the other is the way they drift apart."""
    readme = (paths.ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split("<!-- mselect:panel:start -->", 1)[1].split("<!-- mselect:panel:end -->")[
        0
    ]
    for model in _read("panel.json")["models"]:
        row = [line for line in block.splitlines() if f"`{model['alias']}`" in line]
        assert len(row) == 1, model["alias"]
        assert f"`{model['model']}`" in row[0]


def test_a_display_name_drops_the_host_prefix_and_the_dated_snapshot() -> None:
    """What a reader is shown, against what the record keeps. Both matter: the page is a
    comparison and wants the name, the panel table is a record and wants the identifier."""
    cases = {
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama-3.3-70B-Instruct-Turbo",
        "openai/gpt-oss-120b": "gpt-oss-120b",
        "claude-haiku-4-5-20251001": "claude-haiku-4-5",
        "gpt-5.4-mini-2026-03-17": "gpt-5.4-mini",
        # Left alone: a version is not a date, and a local tag is not a host prefix.
        "claude-opus-5": "claude-opus-5",
        "gemini-3.5-flash-lite": "gemini-3.5-flash-lite",
        "qwen2.5:7b": "qwen2.5:7b",
    }
    for identifier, shown in cases.items():
        assert build.display_name(identifier) == shown


def test_the_page_shows_a_name_and_the_readme_keeps_the_identifier() -> None:
    """The date on `claude-haiku-4-5-20251001` is noise in a comparison and evidence in a run
    record, so it is dropped in one place and kept in the other."""
    panel = _read("panel.json")
    readme = (paths.ROOT / "README.md").read_text(encoding="utf-8")
    for model in panel["models"]:
        assert model["label"] == build.display_name(model["model"])
        assert f"`{model['model']}`" in readme


def test_machine_time_is_measured_or_absent_and_never_zero() -> None:
    """A model with no recorded latency must say so rather than read as instant. The three
    Anthropic models have none: they go through the Message Batches endpoint at half price, and
    a call inside a batch has no latency worth reporting."""
    panel = _read("panel.json")
    n_items = int(panel["n_items"])
    silent = []
    for model in panel["models"]:
        timing = model["latency_ms"]
        if timing is None:
            assert model["full_seconds"] is None
            silent.append(model["alias"])
            continue
        assert len(timing) == n_items
        measured = [ms for ms in timing if ms is not None]
        # A cache hit during the run is recorded at zero latency, because the model was never
        # asked. Those cells carry no time rather than no elapsed time, and there are few of
        # them: every model that was timed at all was timed on at least 97% of its calls.
        assert all(ms > 0 for ms in measured)
        assert len(measured) == model["timed_calls"] >= 0.97 * n_items
        assert model["full_seconds"] == pytest.approx(sum(measured) / 1000.0, abs=0.2)
    assert all(alias.startswith("anthropic-") for alias in silent)


def test_a_laptop_model_costs_nothing_and_still_costs_hours() -> None:
    """The figure the page exists to correct: free is not the same as cheap. If a local model
    ever reported both zero dollars and no time, the page would be saying a full suite on your
    own hardware is free, which is the claim a reader would rightly not believe."""
    panel = _read("panel.json")
    local = [model for model in panel["models"] if not model["hosted"]]
    assert local
    for model in local:
        assert model["full_usd"] == 0.0
        assert model["full_seconds"] and model["full_seconds"] > 3600


def test_the_curves_carry_an_interval_on_every_point() -> None:
    """A bare tau is a defect in this repository. It is a defect on its web page too."""
    curves = _read("curves.json")
    panels = curves["panels"]
    assert isinstance(panels, list)
    assert {panel["key"] for panel in panels} == {"v1", "v2", "own"}
    for panel in panels:
        assert panel["curve"]
        for point in panel["curve"]:
            assert point["lo"] <= point["tau"] <= point["hi"]


def test_more_questions_are_needed_to_catch_a_smaller_drop() -> None:
    """The one relationship in the power grid a reader will check by moving the slider."""
    power = _read("power.json")
    grid = power["grid"]
    assert isinstance(grid, list)
    for ability in power["abilities"]:
        rows = [row for row in grid if row["ability"] == ability and row["power"] == 0.8]
        rows.sort(key=lambda row: float(row["effect"]))
        counts = [int(row["items"]) for row in rows]
        assert counts == sorted(counts, reverse=True)


def test_the_data_files_are_stamped_and_the_page_asks_for_them_by_stamp() -> None:
    """A payload that gains a field is served to browsers still holding the previous one, and
    the page then reads new code against an old file. That shipped once: the fix is that the
    index carries a hash of every payload and the page requests the others at that hash, so the
    two can never be mixed. The index itself is the one file the host is told not to cache."""
    index = _read("index.json")
    assert len(str(index["build"])) >= 8
    page = (DEMO / "app.js").read_text(encoding="utf-8")
    assert "?v=${encodeURIComponent(build)}" in page
    assert 'load("index")' in page

    config = json.loads((DEMO / "staticwebapp.config.json").read_text(encoding="utf-8"))
    routes = {route["route"]: route["headers"]["Cache-Control"] for route in config["routes"]}
    assert routes["/data/index.json"] == "no-cache"


def test_the_stamp_changes_when_a_payload_changes(tmp_path: Path) -> None:
    """A stamp that did not move with the bytes would be worse than none: it would pin a stale
    payload in every cache rather than let it expire."""
    build.build_all(tmp_path)
    first = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))["build"]
    panel = tmp_path / "panel.json"
    edited = json.loads(panel.read_text(encoding="utf-8"))
    edited["n_items"] = int(edited["n_items"]) + 1
    panel.write_text(json.dumps(edited, separators=(",", ":")), encoding="utf-8")
    second = hashlib.sha256()
    for name in ("panel", "curves", "items", "experiments", "power"):
        second.update((tmp_path / f"{name}.json").read_text(encoding="utf-8").encode("utf-8"))
    assert second.hexdigest()[:12] != first


def test_every_payload_the_page_checks_on_arrival_is_one_it_is_sent() -> None:
    """The page refuses a payload missing the fields it needs, which is only a safety net if the
    names it checks are the names the builder writes."""
    page = (DEMO / "app.js").read_text(encoding="utf-8")
    checked = set(re.findall(r"^  (\w+): \(data\)", page, flags=re.MULTILINE))
    assert checked == {"panel", "curves", "items", "experiments", "power"}


def test_the_hosting_policy_lets_the_page_load_itself_and_nothing_else() -> None:
    """The page fetches its own JSON and self-hosts its fonts, so those two are allowed and
    every other source is not. This is the header the local server sends as well, which is the
    only reason a policy violation is visible before it is published."""
    headers = declared_headers(DEMO)
    policy = headers["Content-Security-Policy"]
    assert "default-src 'none'" in policy
    for directive in ("script-src 'self'", "font-src 'self'", "connect-src 'self'"):
        assert directive in policy
    assert "unsafe-inline" not in policy


def test_a_folder_without_a_config_is_served_without_headers_rather_than_refused(
    tmp_path: Path,
) -> None:
    assert declared_headers(tmp_path) == {}


def test_a_malformed_config_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    """Sending no policy because the file is broken is the failure this exists to prevent."""
    (tmp_path / "staticwebapp.config.json").write_text('{"globalHeaders": "none"}', "utf-8")
    with pytest.raises(TypeError):
        declared_headers(tmp_path)


def test_the_page_carries_no_inline_style_attribute() -> None:
    """`style-src 'self'` blocks a style attribute in markup, and the failure is silent: the
    element simply renders unstyled on the live site and correctly in any local check that
    sends no headers. Project 01 shipped exactly that defect for two weeks."""
    for name in ("index.html", "app.js", "charts.js", "irt.js"):
        text = (DEMO / name).read_text(encoding="utf-8")
        assert not re.search(r'\bstyle\s*=\s*["\']', text), name


def test_the_builder_writes_every_file_the_page_needs(tmp_path: Path) -> None:
    """The one test that runs the builder end to end, into a directory of its own."""
    written = build.build_all(tmp_path)
    names = {item.path.name for item in written}
    assert names == {
        "panel.json",
        "curves.json",
        "items.json",
        "experiments.json",
        "power.json",
        "index.json",
    }
    for item in written:
        assert item.bytes > 0
        json.loads(item.path.read_text(encoding="utf-8"))
