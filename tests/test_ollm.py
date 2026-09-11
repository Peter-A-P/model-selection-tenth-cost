"""The Open LLM Leaderboard reader, without touching the network.

Three things here can be wrong in ways that would silently corrupt bank v2 rather than fail:
the panel can be drawn from the wrong part of the ability range, the range reader can pull the
wrong bytes, and an item's identity can start depending on something the evaluation harness is
free to change between versions. Each gets a test that fails if it happens.

The reader is exercised against a real Parquet file served by a fake transport that honours
Range headers, so the column projection, the redirect handling and the sort are all covered
without a token and without a request leaving the machine.
"""

from __future__ import annotations

import io
import json

import httpx
import numpy as np
import polars as pl
import pytest

from mselect.data import bank as bank_io
from mselect.data import build, ollm
from mselect.report import build_report


def _submission(name: str, average: float, organisation: str = "org") -> ollm.Submission:
    return ollm.Submission(
        fullname=f"{organisation}/{name}",
        repo=f"{ollm.ORG}/{organisation}__{name}-details",
        average=average,
        model_type="pretrained",
        params_b=7.0,
        precision="bfloat16",
        upload_date="2024-06-01",
    )


def test_the_panel_spans_the_ability_range_instead_of_where_the_models_pile_up() -> None:
    """Almost every submission sits mid-range; a panel that copies that cannot calibrate."""
    pool = [_submission(f"mid-{i}", 25.0 + i * 0.001, f"org{i % 50}") for i in range(2000)]
    pool += [_submission(f"low-{i}", 2.0 + i * 0.5, f"low{i}") for i in range(20)]
    pool += [_submission(f"high-{i}", 45.0 + i * 0.3, f"high{i}") for i in range(20)]

    panel = ollm.stratified_panel(pool, size=60, strata=10, per_organisation=4, seed=0)

    assert len(panel) == 60
    averages = [member.average for member in panel]
    assert min(averages) < 6.0, "the bottom of the range is missing"
    assert max(averages) > 44.0, "the top of the range is missing"
    # A random sample of this pool would be 98 percent mid-range.
    mid = sum(1 for value in averages if 24.9 < value < 25.1)
    assert mid < 45


def test_no_organisation_can_fill_the_panel_with_its_own_merges() -> None:
    pool = [_submission(f"m-{i}", float(i % 40), "oneorg") for i in range(500)]
    pool += [_submission(f"o-{i}", float(i % 40), f"other{i}") for i in range(200)]

    panel = ollm.stratified_panel(pool, size=100, strata=10, per_organisation=5, seed=1)

    from_one = sum(1 for member in panel if member.organisation == "oneorg")
    assert from_one <= 5


def test_an_empty_pool_is_an_empty_panel_rather_than_an_error() -> None:
    assert ollm.stratified_panel([], size=10) == []


def test_item_identity_is_the_question_and_nothing_else() -> None:
    """A different question is a different item; the same one asked twice is not."""
    task = ollm.TASKS[0]
    same = ollm.item_id(task, ["What is 2 + 2?"])
    assert same == ollm.item_id(task, ["What is 2 + 2?"])
    assert same != ollm.item_id(task, ["What is 2 + 3?"])
    assert same != ollm.item_id(ollm.TASKS[1], ["What is 2 + 2?"])


def test_item_identity_ignores_reformatting_but_not_field_boundaries() -> None:
    """Bank v1 normalises the same way: NFC and collapsed whitespace, word boundaries kept."""
    task = ollm.TASKS[0]
    assert ollm.item_id(task, ["a  b\n c"]) == ollm.item_id(task, ["a b c"])
    assert ollm.item_id(task, ["a", "b"]) != ollm.item_id(task, ["ab"])


def test_a_question_a_task_asks_twice_is_told_apart_by_its_key() -> None:
    """bbh_causal_judgement asks two of its questions twice, with the opposite key each time.

    Those are two measurements, and bank v1's rule is right about them. What the rule must not
    do is put the key into every item, because that is what the MATH answer-spelling drift
    punishes.
    """
    task = ollm.TASKS[0]
    ids = ollm.task_identities(task, ["once", "twice", "other", "twice"], ["A", "Yes", "B", "No"])
    assert len(set(ids)) == 4, "the repeated question must not collapse into one item"
    # A question asked once is identified by the question alone, key or no key.
    assert ids[0] == ollm.item_id(task, ["once"])
    assert ids[2] == ollm.item_id(task, ["other"])
    # And a question repeated with the same key too is still split, by order of appearance.
    assert len(set(ollm.task_identities(task, ["same", "same"], ["k", "k"]))) == 2


def test_the_answer_key_is_reported_not_hashed_into_the_item() -> None:
    """Two releases of MATH-Hard spell the same answer differently.

    They write it as `\\infty` and `\\iny`, and as `-\\frac{1}{{}2x}` and `-\\frac1{2x}`.
    Hashing the key with the question split 33 of 307 algebra items and dropped 74 of the 400
    models on that task, for a difference that is typographic. The key is carried beside the
    identity instead, so the drift can be counted and reported.
    """
    unique = ollm.task_identities(ollm.TASKS[0], ["q1", "q2"], [r"\infty", "x"])
    drifted = ollm.task_identities(ollm.TASKS[0], ["q1", "q2"], [r"\iny", "x"])
    assert unique == drifted, "a key spelled differently must not make a different item"
    assert "answer key is not part of the identity" in (ollm.item_id.__doc__ or "")


def test_every_task_declares_which_document_fields_are_the_question() -> None:
    """An identity built on lm-eval's own doc_hash is not stable across harness versions."""
    for task in ollm.TASKS:
        assert task.content, f"{task.suffix} has no content field"
        assert all(field for field in task.content)
    # MMLU-Pro asks 392 of its questions more than once with different options each time, so
    # its options are part of the question. No other task in the set repeats a question.
    mmlu_pro = next(task for task in ollm.TASKS if task.benchmark == "mmlu_pro")
    assert mmlu_pro.content == ("question", "options")


def test_a_list_valued_document_field_keeps_its_order() -> None:
    """Options are a list. Two items with the same options in a different order are not one."""
    assert ollm._flatten(["a", "b"]) != ollm._flatten(["b", "a"])
    assert ollm._flatten(["a", "b"]) == ollm._flatten(["a", "b"])
    assert ollm._flatten("plain") == "plain"


def test_every_task_in_the_set_is_distinct_and_scored_binary() -> None:
    suffixes = [task.suffix for task in ollm.TASKS]
    assert len(set(suffixes)) == len(suffixes)
    assert {task.metric for task in ollm.TASKS} <= {"acc", "acc_norm", "exact_match"}
    assert "leaderboard_ifeval" not in suffixes, "IFEval is not binary per item"
    assert set(ollm.BENCHMARKS) == {"bbh", "gpqa", "musr", "math_hard", "mmlu_pro"}


def test_a_metric_that_is_not_binary_stops_the_build() -> None:
    frame = pl.DataFrame({"doc_id": [0, 1], "score": [0.0, 0.5]})
    with pytest.raises(ollm.FetchError, match="not binary"):
        ollm.binary(frame, "fixture")
    ok = ollm.binary(pl.DataFrame({"doc_id": [0, 1], "score": [0.0, 1.0]}), "fixture")
    assert ok.to_list() == [0, 1]


def test_the_panel_file_survives_a_round_trip() -> None:
    panel = ollm.Panel(
        members=(_submission("a", 10.0), _submission("b", 20.0)),
        size=2,
        seed=0,
        strata=4,
        per_organisation=8,
        rejected=("org/c: unreadable",),
    )
    again = ollm.Panel.from_json(json.loads(json.dumps(panel.to_json())))
    assert again.members == panel.members
    assert again.rejected == panel.rejected
    assert "panel of 2 models" in panel.describe()


def test_merges_are_their_own_class_not_folded_into_tuned() -> None:
    assert build._ollm_family("\U0001f91d base merges and moerges") == "merge"
    assert build._ollm_family("\U0001f7e2 pretrained") == "base"
    assert build._ollm_family("\U0001f4ac chat models (RLHF, DPO, IFT, ...)") == "tuned"
    assert build._ollm_family("\U0001f536 fine-tuned on domain-specific datasets") == "tuned"
    assert build._ollm_family("something new") == "other"


def _parquet_bytes() -> bytes:
    """A file shaped like a leaderboard task file: the wanted columns, plus a fat one."""
    rng = np.random.default_rng(0)
    fat = [rng.bytes(250_000).hex() for _ in range(3)]  # incompressible: a real prompt column
    frame = pl.DataFrame(
        {
            "doc_id": [2, 0, 1],  # deliberately out of order: the harness stores batch order
            "doc": fat,
            "doc_hash": ["hash-two", "hash-zero", "hash-one"],
            "target_hash": ["key-two", "key-zero", "key-one"],
            "acc_norm": [1.0, 0.0, 1.0],
        }
    )
    buffer = io.BytesIO()
    # No column statistics: Parquet stores the minimum and maximum of every column in the
    # footer, and for a 120 KB string column that would put the fat column in the footer too.
    frame.write_parquet(buffer, statistics=False)
    return buffer.getvalue()


def _range_transport(payload: bytes, seen: list[str]) -> httpx.MockTransport:
    """A transport that answers Range requests the way a CDN does, and counts the bytes."""

    def handle(request: httpx.Request) -> httpx.Response:
        header = request.headers.get("Range", "")
        seen.append(header)
        first, _, last = header.removeprefix("bytes=").partition("-")
        start = int(first)
        end = min(int(last), len(payload) - 1)
        chunk = payload[start : end + 1]
        return httpx.Response(
            206,
            content=chunk,
            headers={"content-range": f"bytes {start}-{end}/{len(payload)}"},
        )

    return httpx.MockTransport(handle)


def test_the_reader_takes_three_columns_and_leaves_the_rest_on_the_server(tmp_path) -> None:  # type: ignore[no-untyped-def]
    payload = _parquet_bytes()
    seen: list[str] = []
    cache = ollm.Cache(tmp_path / "cache")
    client = ollm.Client(cache, "token-for-the-fake-transport")
    client._client = httpx.Client(transport=_range_transport(payload, seen))
    task = ollm.Task("leaderboard_fixture", "fixture", "acc_norm", "multiple_choice", ("text",))
    member = _submission("reader", 10.0)

    frame = ollm.read_task(client, member, task)

    assert frame["doc_id"].to_list() == [0, 1, 2], "rows must be sorted into document order"
    assert frame["score"].to_list() == [0.0, 1.0, 1.0]
    assert "doc" not in frame.columns and "doc_hash" not in frame.columns
    pulled = sum(
        int(header.removeprefix("bytes=").split("-")[1])
        - int(header.removeprefix("bytes=").split("-")[0])
        + 1
        for header in seen
    )
    # pyarrow reads the last 64 KB in one go to find the footer, so the floor is a fixed cost,
    # not a share of the file: on a real 71 MB MMLU-Pro run this projection pulls under 3 MB.
    assert pulled < len(payload) / 8, "a projection that reads most of the file is not one"

    # And the second read is free: it comes from the cache, with no further requests.
    before = len(seen)
    again = ollm.read_task(client, member, task)
    assert len(seen) == before
    assert again.equals(frame)
    assert cache.provenance.exists()
    record = json.loads(cache.provenance.read_text(encoding="utf-8").splitlines()[0])
    assert record["rows"] == 3 and record["remote_bytes"] == len(payload)


def test_a_packed_matrix_round_trips_through_the_bank_format(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Bank v2 ships its matrix packed; a byte that comes back wrong is a silent wrong answer."""
    x = np.array([[1.0, 0.0, np.nan], [0.0, 1.0, 1.0]])
    pl.DataFrame({"item_id": ["a", "b", "c"], "benchmark": ["f"] * 3, "kind": ["f"] * 3}).sort(
        "item_id"
    ).write_parquet(tmp_path / "items.parquet")
    pl.DataFrame({"model_id": ["m1", "m2"]}).write_parquet(tmp_path / "models.parquet")
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"bank_hash": "abc"}), encoding="utf-8")
    bank_io.write_matrix(tmp_path, x)

    loaded = bank_io.load(tmp_path.name, root=tmp_path.parent)

    assert np.array_equal(loaded.x, x, equal_nan=True)
    assert loaded.n_models == 2 and loaded.n_items == 3
    size = (tmp_path / bank_io.MATRIX_FILE).stat().st_size
    assert size < 1000


def test_bank_v1_keeps_every_filename_and_marker_it_already_had() -> None:
    """A second bank must not move v1's figures, documents or README block."""
    assert build_report._suffix("v1") == ""
    assert build_report._markers("v1") == (build_report.START, build_report.END)
    assert build_report._suffix("v2") == "-v2"
    start, end = build_report._markers("v2")
    assert start != build_report.START and "v2" in start and "v2" in end
    assert build_report._bench_markers("v1") == (build_report.BENCH_START, build_report.BENCH_END)
    bench_start, bench_end = build_report._bench_markers("v2")
    assert bench_start != build_report.BENCH_START
    assert "v2" in bench_start and "v2" in bench_end
    # Every marker the report writes into must be distinct, or one block overwrites another.
    assert (
        len(
            {
                start,
                end,
                bench_start,
                bench_end,
                build_report.START,
                build_report.END,
                build_report.BENCH_START,
                build_report.BENCH_END,
            }
        )
        == 8
    )


def test_the_report_names_the_source_from_the_bank_rather_than_assuming_helm() -> None:
    bank = bank_io.default_bank("v1")
    assert "HELM" in build_report._source_name(bank)
    for version in bank_io.available():
        assert build_report._source_name(bank_io.default_bank(version))
