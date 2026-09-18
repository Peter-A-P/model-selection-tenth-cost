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
