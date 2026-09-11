"""Per-item results from the Open LLM Leaderboard v2 "details" datasets.

This is the second source PLAN.md section 3.1 named, and the one that was blocked when bank v1
was built: the details datasets are gated on Hugging Face (`gated: auto`), so they need a
signed-in token. They are now reachable, and this module is what reaches them.

What the leaderboard gives that HELM does not:

* **Open-weight models across the whole ability range.** HELM's panel is mostly frontier APIs
  clustered at the top. These 4,486 submissions run from 0.7 to 52.1 on the leaderboard's own
  average, which is what item difficulty needs in order to be identified at the bottom end.
* **BBH and MuSR**, two reasoning suites HELM does not carry, and MATH at level 5 only.
* **A dense matrix.** Every submission runs every item of every task, so bank v2 has no holes,
  where bank v1 was 54 percent observed and needed a dense block peeled out of it.

Three facts about the format drive the whole design:

1. The details repositories are auto-converted to Parquet at `refs/convert/parquet`, one file
   per task per run, and the conversion keeps a `latest` split. So "the latest run of each
   task" is a URL, not a search.
2. Those files are mostly prompt and response text: one model's MMLU-Pro run is 71 MB of
   Parquet, of which the two columns this project needs are under two. Every read here is a
   column projection over HTTP range requests, so a build transfers about 800 MB rather than
   28 GB. `_RemoteFile` is that range reader.
3. `doc_hash` is lm-eval-harness's own content hash of the item, and it is stable across
   models; `doc_id` is the item's position in the task, and is stable too, but the *row order*
   is not (it follows the batching). So rows are keyed by `doc_id` and identity comes from
   `doc_hash`, and `audit_alignment` checks that the mapping between them is the same for
   every audited model rather than assuming it.

Everything fetched is cached under `data/raw/ollm/` as the projected columns, not the source
file, with a provenance line per fetch recording the remote file's size and the rows taken.
That is a deliberate departure from the HELM cache, which stores the source bytes: here the
source is a hundred times larger than the information taken from it, and storing it would cost
30 GB to save a download this project makes once. `docs/data-sources.md` says so too.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import random
import re
import ssl
import threading
import time
import urllib.parse
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx
import polars as pl
import pyarrow.parquet as pq
import truststore

ORG: Final = "open-llm-leaderboard"
HUB: Final = "https://huggingface.co"
CONTENTS_URL: Final = (
    f"{HUB}/datasets/{ORG}/contents/resolve/main/data/train-00000-of-00001.parquet"
)
PARQUET_REF: Final = urllib.parse.quote("refs/convert/parquet", safe="")
USER_AGENT: Final = "mselect/0.2 (item bank build; github.com/Peter-A-P)"
TIMEOUT_S: Final = 180.0
RETRY_STATUS: Final = frozenset({408, 429, 500, 502, 503, 504})
RETRIES: Final = 6
BACKOFF_CAP_S: Final = 60.0
WORKERS: Final = 8

# How many models per task have their `doc_id` to `doc_hash` mapping checked against the
# reference model's. Every model's row count is checked; this is the stronger check, and it
# costs the `doc_hash` column, which on MMLU-Pro is most of the download.
AUDIT_MODELS: Final = 40

# The identity cache stores a derived frame, not the source bytes, so its key names the
# derivation as well as the columns. Bump this whenever `item_id` or `read_identity` changes
# what it computes, or a rebuild will read yesterday's identities back off disk and mix two
# schemes in one bank without a word. 1: lm-eval doc_hash and target_hash. 2: the question text
# hashed here, with the answer key hashed in. 3: the question text alone, key reported beside it.
IDENTITY_SCHEME: Final = 3


class FetchError(RuntimeError):
    """A source could not be fetched or did not look the way this module requires."""


@dataclass(frozen=True, slots=True)
class Task:
    """One leaderboard task, and the column that scores it 0 or 1 per item."""

    suffix: str  # the config name after the model stub, e.g. "leaderboard_bbh_snarks"
    benchmark: str  # the family it contributes to in the bank
    metric: str  # the Parquet column that is 0 or 1
    kind: str  # "multiple_choice" or "free_response"; 3PL is fitted only for the former
    content: tuple[str, ...]  # the `doc` fields that are the question, and give the item its id

    @property
    def label(self) -> str:
        """The task name without the leaderboard prefix, for item metadata."""
        return self.suffix.removeprefix("leaderboard_")


_BBH: Final = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
)

_MATH: Final = (
    "algebra_hard",
    "counting_and_prob_hard",
    "geometry_hard",
    "intermediate_algebra_hard",
    "num_theory_hard",
    "prealgebra_hard",
    "precalculus_hard",
)

# The six task families the v2 leaderboard averages, minus IFEval. IFEval is excluded for the
# same reason it was excluded from bank v1: its per-item score is the fraction of instructions
# a response satisfied, not a binary outcome, and rounding it would invent an item response.
# `leaderboard_arc_challenge` is present in the repositories but is not part of the v2 average
# and was not re-run for every submission, so it is left out to keep the matrix dense. The
# `leaderboard_bbh_fewshot_*` and non-hard `leaderboard_math_*` configs are an earlier naming
# that only some submissions carry, and are left out for the same reason.
TASKS: Final[tuple[Task, ...]] = (
    *(
        Task(f"leaderboard_bbh_{name}", "bbh", "acc_norm", "multiple_choice", ("input",))
        for name in _BBH
    ),
    Task("leaderboard_gpqa_main", "gpqa", "acc_norm", "multiple_choice", ("Question",)),
    *(
        Task(
            f"leaderboard_musr_{name}",
            "musr",
            "acc_norm",
            "multiple_choice",
            ("narrative", "question"),
        )
        for name in ("murder_mysteries", "object_placements", "team_allocation")
    ),
    *(
        Task(f"leaderboard_math_{name}", "math_hard", "exact_match", "free_response", ("problem",))
        for name in _MATH
    ),
    Task("leaderboard_mmlu_pro", "mmlu_pro", "acc", "multiple_choice", ("question",)),
)

BENCHMARKS: Final = tuple(dict.fromkeys(task.benchmark for task in TASKS))


@dataclass(frozen=True, slots=True)
class Submission:
    """One leaderboard row: a model, its details repository, and its published averages."""

    fullname: str  # "org/name" on the hub
    repo: str  # "open-llm-leaderboard/org__name-details"
    average: float  # the leaderboard's own normalised average, 0 to 100
    model_type: str  # pretrained, fine-tuned, chat, merge, ...
    params_b: float | None
    precision: str
    upload_date: str  # ISO date, "" when the leaderboard does not say

    @property
    def stub(self) -> str:
        """The double-underscore form the config names are built from."""
        return self.fullname.replace("/", "__")

    @property
    def organisation(self) -> str:
        return self.fullname.split("/")[0]

    def config(self, task: Task) -> str:
        return f"{self.stub}__{task.suffix}"

    def parquet_url(self, task: Task, shard: str = "0000") -> str:
        path = f"{self.config(task)}/latest/{shard}.parquet"
        return f"{HUB}/datasets/{self.repo}/resolve/{PARQUET_REF}/{path}"


def _now() -> str:
    return datetime.now(UTC).date().isoformat()


class Cache:
    """Projected columns on disk keyed by URL, plus a provenance line per fetch.

    A cached entry is a gzipped Parquet frame of the columns this project keeps, not the source
    file. The provenance line records the source URL, the remote file's size, the columns taken
    and the row count, which is what a stranger needs to re-derive the same extract.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.blobs = root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.provenance = root / "provenance.jsonl"
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.blobs / digest[:2] / f"{digest}.parquet.gz"

    def get(self, key: str) -> pl.DataFrame | None:
        path = self._path(key)
        if not path.exists():
            return None
        return pl.read_parquet(io.BytesIO(gzip.decompress(path.read_bytes())))

    def put(self, key: str, frame: pl.DataFrame, record: dict[str, object]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        frame.write_parquet(buffer)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_bytes(gzip.compress(buffer.getvalue(), compresslevel=6))
        tmp.replace(path)
        with self._lock, self.provenance.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**record, "key": key, "fetched": _now()}) + "\n")


class _RemoteFile(io.RawIOBase):
    """A seekable read-only file over HTTP range requests.

    Parquet is a footer-first format, so a reader that can seek pulls the schema, then only the
    column chunks it was asked for. That is the difference between two megabytes and seventy on
    an MMLU-Pro run. The first request doubles as the redirect resolution: the hub answers with
    a redirect to a CDN URL, and every later range request goes straight there.
    """

    def __init__(self, url: str, client: httpx.Client) -> None:
        response = _with_retry(client, url, headers={"Range": "bytes=0-0"})
        content_range = response.headers.get("content-range", "")
        if "/" not in content_range:
            raise FetchError(f"{url} did not answer a range request")
        self._url = str(response.url)
        self._client = client
        self.size = int(content_range.rsplit("/", 1)[-1])
        self.transferred = 1
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        wanted = self.size - self._pos if size is None or size < 0 else size
        if wanted <= 0 or self._pos >= self.size:
            return b""
        last = min(self._pos + wanted, self.size) - 1
        response = _with_retry(
            self._client, self._url, headers={"Range": f"bytes={self._pos}-{last}"}
        )
        payload = response.content
        self._pos += len(payload)
        self.transferred += len(payload)
        return payload

    def readinto(self, buffer: Any) -> int:
        payload = self.read(len(buffer))
        buffer[: len(payload)] = payload
        return len(payload)


_RETRY_RNG = random.Random(0)
_RETRY_LOCK = threading.Lock()


def _with_retry(
    client: httpx.Client, url: str, *, headers: dict[str, str] | None = None
) -> httpx.Response:
    """One GET with the backoff a free, shared, rate-limited hub deserves."""
    last = ""
    for attempt in range(RETRIES):
        try:
            response = client.get(url, headers=headers)
        except httpx.HTTPError as exc:  # transport level: retried like a 503
            last = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code in (200, 206):
                return response
            if response.status_code not in RETRY_STATUS:
                raise FetchError(f"{url} returned HTTP {response.status_code}")
            last = f"HTTP {response.status_code}"
        with _RETRY_LOCK:
            jitter = 0.5 + _RETRY_RNG.random()
        time.sleep(min(BACKOFF_CAP_S, 2.0**attempt) * jitter)
    raise FetchError(f"{url} failed after {RETRIES} attempts ({last})")


def ssl_context() -> ssl.SSLContext:
    """Verify against the operating system's trust store rather than certifi's bundle.

    On a network that inspects TLS, the proxy presents its own certificate signed by a root
    that the machine trusts and certifi does not, so every request to huggingface.co fails
    certificate verification while a browser on the same machine is fine. This is not a way of
    skipping verification: it verifies against the trust store the machine actually has.
    `docs/data-sources.md` records where this was found.
    """
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


class Client:
    """A pooled, authenticated HTTP client for the hub."""

    def __init__(self, cache: Cache, token: str) -> None:
        if not token:
            raise FetchError("HF_TOKEN is not set: the details datasets are gated")
        self.cache = cache
        self._client = httpx.Client(
            timeout=TIMEOUT_S,
            follow_redirects=True,
            verify=ssl_context(),
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": USER_AGENT,
            },
            limits=httpx.Limits(max_connections=WORKERS * 2, max_keepalive_connections=WORKERS),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def raw(self) -> httpx.Client:
        return self._client

    def json_at(self, url: str) -> Any:
        return json.loads(_with_retry(self._client, url).content)

def token_from_env(environ: dict[str, str] | None = None) -> str:
    """The hub token, from the environment or from a gitignored .env beside the repository.

    CLAUDE.md: vendor keys come from the environment only, and `.env` is never committed.
    """
    import os

    from mselect import paths

    env = dict(os.environ if environ is None else environ)
    if env.get("HF_TOKEN"):
        return env["HF_TOKEN"]
    dotenv = paths.ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "HF_TOKEN":
                return value.strip().strip("'\"")
    return ""


def leaderboard(client: Client) -> pl.DataFrame:
    """The leaderboard's own results table: one row per submission, with its averages."""
    payload = _with_retry(client.raw, CONTENTS_URL).content
    return pl.read_parquet(io.BytesIO(payload))


def details_repos(client: Client) -> set[str]:
    """Every dataset under the leaderboard organisation whose name ends in `-details`."""
    names: set[str] = set()
    url: str | None = f"{HUB}/api/datasets?author={ORG}&limit=1000"
    while url:
        response = _with_retry(client.raw, url)
        names.update(
            str(row["id"]) for row in response.json() if str(row["id"]).endswith("-details")
        )
        match = re.search(r'<([^>]+)>;\s*rel="next"', response.headers.get("link", ""))
        url = match.group(1) if match else None
    return names


def candidates(frame: pl.DataFrame, repos: set[str]) -> list[Submission]:
    """Leaderboard rows that have a details repository, one per model, flagged rows dropped.

    A model evaluated at more than one precision has one details repository, so the rows are
    collapsed to the highest-scoring one; taking both would put the same model in the panel
    twice and make the ability distribution lie about how many independent models it has.
    """
    average = "Average ⬆️"  # the leaderboard's own column name, emoji and all
    prepared = (
        frame.with_columns(
            (
                pl.lit(f"{ORG}/")
                + pl.col("fullname").str.replace("/", "__", literal=True)
                + pl.lit("-details")
            ).alias("repo")
        )
        .filter(pl.col("repo").is_in(list(repos)))
        .filter(~pl.col("Flagged"))
        .filter(pl.col(average).is_not_null())
        .sort([average, "eval_name"], descending=[True, False])
        .unique(subset=["repo"], keep="first", maintain_order=True)
    )
    return [
        Submission(
            fullname=str(row["fullname"]),
            repo=str(row["repo"]),
            average=float(row[average]),
            model_type=str(row["Type"] or ""),
            params_b=(None if row["#Params (B)"] is None else float(row["#Params (B)"])),
            precision=str(row["Precision"] or ""),
            upload_date=str(row["Upload To Hub Date"] or ""),
        )
        for row in prepared.iter_rows(named=True)
    ]


def stratified_panel(
    pool: Sequence[Submission],
    *,
    size: int = 400,
    strata: int = 40,
    per_organisation: int = 8,
    seed: int = 0,
) -> list[Submission]:
    """A panel that spans the ability range instead of piling up where the submissions do.

    Two rules, both there to make item parameters identifiable rather than to be fair to
    anyone. Equal-width strata over the leaderboard average, because a panel drawn at random
    from 4,486 submissions is 80 percent seven-billion-parameter fine-tunes in a narrow band,
    and an item bank calibrated on it cannot tell a hard item from an impossible one. A cap per
    organisation, because forty merges of one base model are close to one model repeated, and
    the effective panel size is what the standard errors depend on.
    """
    if not pool:
        return []
    rng = random.Random(seed)
    low = min(s.average for s in pool)
    high = max(s.average for s in pool)
    width = (high - low) / strata if high > low else 1.0

    bins: list[list[Submission]] = [[] for _ in range(strata)]
    for submission in pool:
        index = min(strata - 1, int((submission.average - low) / width))
        bins[index].append(submission)
    for bucket in bins:
        rng.shuffle(bucket)

    chosen: list[Submission] = []
    by_organisation: dict[str, int] = {}

    def take(bucket: list[Submission], quota: int) -> None:
        taken = 0
        for submission in list(bucket):
            if taken >= quota or len(chosen) >= size:
                return
            if by_organisation.get(submission.organisation, 0) >= per_organisation:
                continue
            chosen.append(submission)
            bucket.remove(submission)
            by_organisation[submission.organisation] = (
                by_organisation.get(submission.organisation, 0) + 1
            )
            taken += 1

    quota = max(1, size // strata)
    for bucket in bins:
        take(bucket, quota)
    # Strata that ran out give their places back to the ones that did not, widest first, so a
    # short tail at the top of the range costs coverage in the middle rather than panel size.
    while len(chosen) < size and any(bins):
        before = len(chosen)
        for bucket in sorted(bins, key=len, reverse=True):
            take(bucket, 1)
        if len(chosen) == before:
            break
    return sorted(chosen, key=lambda s: s.fullname)


# There is no access request. The details datasets are `gated: auto`, which on this hub means
# any authenticated account may read them: an unauthenticated range request answers 401, the
# same request with a token answers 206, and nothing is recorded against the account. Bank v1's
# notes said otherwise, and `docs/data-sources.md` records the correction. It matters beyond
# tidiness: this module now writes nothing anywhere, to Hugging Face or to anyone else.


def read_task(client: Client, submission: Submission, task: Task) -> pl.DataFrame:
    """One model's responses to one task: `doc_id` and the 0/1 metric, and nothing else.

    The frame comes back sorted by `doc_id` because the stored row order follows the evaluation
    batching and differs between models, which would otherwise make two models' answers to the
    same item look like answers to different ones.
    """
    columns = ["doc_id", task.metric]
    key = f"{submission.parquet_url(task)}#{','.join(columns)}"
    cached = client.cache.get(key)
    if cached is not None:
        return cached

    handle = _RemoteFile(submission.parquet_url(task), client.raw)
    reader = pq.ParquetFile(handle, pre_buffer=True)
    available = set(reader.schema_arrow.names)
    missing = [name for name in columns if name not in available]
    if missing:
        raise FetchError(f"{submission.repo}/{task.suffix} has no column {missing}")
    frame = pl.from_arrow(reader.read(columns=columns))
    if not isinstance(frame, pl.DataFrame):  # pragma: no cover - a one-row-group file is a frame
        raise FetchError(f"{submission.repo}/{task.suffix} did not read as a table")
    frame = frame.rename({task.metric: "score"}).sort("doc_id")
    client.cache.put(
        key,
        frame,
        {
            "url": submission.parquet_url(task),
            "remote_bytes": handle.size,
            "transferred_bytes": handle.transferred,
            "columns": columns,
            "rows": frame.height,
        },
    )
    return frame


def read_doc_field(client: Client, submission: Submission, task: Task, field: str) -> pl.DataFrame:
    """One field of the document column, by `doc_id`, without downloading the document.

    Parquet stores a struct column field by field, so asking for `doc.question_id` reads that
    field's chunks and nothing else: 1.9 MB of a 72 MB MMLU-Pro file. This is how the two banks
    are bridged without fetching, or committing, any benchmark text.
    """
    columns = ["doc_id", f"doc.{field}"]
    key = f"{submission.parquet_url(task)}#{','.join(columns)}"
    cached = client.cache.get(key)
    if cached is not None:
        return cached

    handle = _RemoteFile(submission.parquet_url(task), client.raw)
    reader = pq.ParquetFile(handle, pre_buffer=True)
    frame = pl.from_arrow(reader.read(columns=columns))
    if not isinstance(frame, pl.DataFrame):  # pragma: no cover - a projection is always a table
        raise FetchError(f"{submission.repo}/{task.suffix} did not read as a table")
    frame = (
        frame.with_columns(pl.col("doc").struct.field(field).cast(pl.Utf8).alias(field))
        .drop("doc")
        .sort("doc_id")
    )
    client.cache.put(
        key,
        frame,
        {
            "url": submission.parquet_url(task),
            "remote_bytes": handle.size,
            "transferred_bytes": handle.transferred,
            "columns": columns,
            "rows": frame.height,
        },
    )
    return frame


def binary(frame: pl.DataFrame, where: str) -> pl.Series:
    """The score column as 0 or 1, refusing anything else.

    A metric that is not binary is a mistake in the task table, not a number to round: bank v1
    made the same refusal for the same reason (PLAN.md section 2).
    """
    values = frame["score"]
    distinct = set(values.drop_nulls().unique().to_list())
    if not distinct <= {0.0, 1.0}:
        raise FetchError(f"{where}: metric values {sorted(distinct)[:4]} are not binary")
    return values.cast(pl.Int8)


@dataclass(frozen=True, slots=True)
class TaskItems:
    """The item identities of one task, taken from the reference model and then audited."""

    task: Task
    reference: str  # the model the doc_id to item mapping came from
    doc_ids: list[int]
    item_ids: list[str]
    audited: int  # how many other models were checked against it
    mismatched: list[str]  # models whose doc_id to item mapping differs; dropped from this task
    key_drift_items: int  # items whose answer key some model spells differently
    key_drift_models: int  # how many models spell at least one key differently


def normalise(text: str) -> str:
    """Runs of whitespace collapsed, so reformatting is not a different question."""
    return " ".join(text.split())


def item_id(task: Task, content: Sequence[str], disambiguator: str = "") -> str:
    """The item's identity: the task and the question's own text. Nothing else.

    Two decisions here, both forced by measurement rather than taste.

    **The question text is hashed by this project, not taken from the harness.**
    lm-eval-harness writes a `doc_hash` column, which looks like exactly this and is not: it
    hashes the harness's serialisation of the document, so it moves when the harness moves. On
    `leaderboard_math_num_theory_hard` a quarter of the panel disagrees with the reference model
    about every one of the 154 `doc_hash` values while the problem text at each `doc_id` is
    character-for-character identical.

    **The answer key is not part of the identity, which is a departure from bank v1.** Bank v1
    hashes the key with the question because the same question can arrive from two HELM
    scenarios with different keys, and telling those apart is what the mis-keyed-item check
    needs. Here each task is one dataset, and what varies between models is not the key but its
    spelling: two releases of MATH-Hard write the same answer as `\\infty` and `\\iny`, and as
    `-\\frac{1}{{}2x}` and `-\\frac1{2x}`. Keeping the key in the identity split 33 of 307
    algebra items and cost 74 of the 400 models on that task alone, for a difference that is
    typographic. The drift is not hidden: `audit_alignment` counts the items whose key any model
    spells differently, and the manifest reports it per task.

    `disambiguator` is how the few genuine exceptions are handled, and `task_identities` is what
    fills it in. `leaderboard_bbh_causal_judgement` asks two of its questions twice, each time
    with the opposite answer key, and those are two measurements rather than one. So where a task
    asks the same question more than once, and only there, the key tells the occurrences apart.
    """
    digest = hashlib.sha256()
    digest.update(task.suffix.encode())
    for field in content:
        digest.update(b"\x01")
        digest.update(normalise(field).encode())
    if disambiguator:
        digest.update(b"\x02")
        digest.update(disambiguator.encode())
    return digest.hexdigest()[:16]


def task_identities(task: Task, questions: Sequence[str], keys: Sequence[str]) -> list[str]:
    """Item ids for one task's documents, in the order given, with repeats told apart.

    A question a task asks once is its own identity. A question it asks more than once is
    identified by the question and its answer key, and if the same question and key appear twice
    as well, by their order of appearance, which is stable because `doc_id` order is.
    """
    repeated = Counter(questions)
    seen: Counter[tuple[str, str]] = Counter()
    out: list[str] = []
    for question, key in zip(questions, keys, strict=True):
        if repeated[question] == 1:
            out.append(item_id(task, [question]))
            continue
        seen[(question, key)] += 1
        occurrence = seen[(question, key)]
        suffix = key if occurrence == 1 else f"{key}#{occurrence}"
        out.append(item_id(task, [question], suffix))
    return out


def read_identity(client: Client, submission: Submission, task: Task) -> pl.DataFrame:
    """What each `doc_id` in a task is, for one model: `doc_id`, `item_id` and the answer key.

    Reads the question fields out of the `doc` struct and the answer beside it, which is a few
    hundred kilobytes of a file that can be seventy megabytes. The question text is hashed here
    and then discarded: it is never written to the cache, the bank or the repository. The key is
    kept, with its whitespace removed, only so that a disagreement about it can be counted.
    """
    columns = ["doc_id", "target", *(f"doc.{field}" for field in task.content)]
    key = f"{submission.parquet_url(task)}#identity{IDENTITY_SCHEME}:{','.join(columns)}"
    cached = client.cache.get(key)
    if cached is not None:
        return cached

    handle = _RemoteFile(submission.parquet_url(task), client.raw)
    reader = pq.ParquetFile(handle, pre_buffer=True)
    frame = pl.from_arrow(reader.read(columns=columns))
    if not isinstance(frame, pl.DataFrame):  # pragma: no cover - a projection is always a table
        raise FetchError(f"{submission.repo}/{task.suffix} did not read as a table")
    frame = frame.sort("doc_id")
    questions = [
        "\x01".join(str(row["doc"][field]) for field in task.content)
        for row in frame.iter_rows(named=True)
    ]
    keys = ["".join(str(value).split()) for value in frame["target"].to_list()]
    identities = task_identities(task, questions, keys)
    out = pl.DataFrame(
        {
            "doc_id": frame["doc_id"],
            "item_id": pl.Series(identities),
            "answer_key": pl.Series(keys),
        }
    )
    if out["item_id"].n_unique() != out.height:  # pragma: no cover - the tie-break is exhaustive
        raise FetchError(
            f"{submission.repo}/{task.suffix}: two documents still hash to the same item after "
            f"the answer key and the order of appearance; {task.content} cannot identify an item"
        )
    client.cache.put(
        key,
        out,
        {
            "url": submission.parquet_url(task),
            "remote_bytes": handle.size,
            "transferred_bytes": handle.transferred,
            "columns": columns,
            "rows": out.height,
            "note": "question text hashed into item_id and discarded; not stored",
        },
    )
    return out


def audit_alignment(
    client: Client,
    task: Task,
    panel: Sequence[Submission],
    *,
    audit: int = AUDIT_MODELS,
    seed: int = 0,
) -> TaskItems:
    """Take the item identities from one model, then check other models agree about them.

    The cheap column set that every model is read with (`doc_id` and the score) is only safe if
    `doc_id` means the same item for every model. It should: the harness walks a fixed dataset
    in order. This checks it on a seeded sample rather than trusting it, by reading the question
    text itself, and any model whose mapping differs is named and dropped from that task.
    """
    if not panel:
        raise FetchError(f"{task.suffix}: an empty panel has nothing to align")
    reference = panel[0]
    frame = read_identity(client, reference, task)
    doc_ids = [int(value) for value in frame["doc_id"].to_list()]
    identities = [str(value) for value in frame["item_id"].to_list()]
    expected = dict(zip(doc_ids, identities, strict=True))
    expected_keys = dict(zip(identities, frame["answer_key"].to_list(), strict=True))
    drifted_items: set[str] = set()
    drifted_models: set[str] = set()

    rng = random.Random(f"{seed}:{task.suffix}")
    others = list(panel[1:])
    rng.shuffle(others)
    sample = others[:audit]
    mismatched: list[str] = []
    checked = 0
    lock = threading.Lock()

    def check(submission: Submission) -> None:
        nonlocal checked
        try:
            other = read_identity(client, submission, task)
        except FetchError:  # no run of this task to align; the row-count check drops it later
            return
        mapping = {
            int(doc): str(item)
            for doc, item in zip(other["doc_id"], other["item_id"], strict=True)
        }
        different = {
            str(item)
            for item, key in zip(other["item_id"], other["answer_key"], strict=True)
            if str(item) in expected_keys and expected_keys[str(item)] != key
        }
        with lock:
            checked += 1
            if mapping != expected:
                mismatched.append(submission.fullname)
            if different:
                drifted_items.update(different)
                drifted_models.add(submission.fullname)

    _each(check, sample)
    mismatched.sort()
    return TaskItems(
        task=task,
        reference=reference.fullname,
        doc_ids=doc_ids,
        item_ids=identities,
        audited=checked,
        mismatched=mismatched,
        key_drift_items=len(drifted_items),
        key_drift_models=len(drifted_models),
    )


def _each[T](fn: Callable[[T], object], items: Sequence[T], *, workers: int = WORKERS) -> int:
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(fn, items):
            done += 1
    return done


def fetch_task(
    client: Client,
    task: Task,
    panel: Sequence[Submission],
    *,
    progress: Callable[[str], None] = lambda _: None,
) -> int:
    """Warm the cache for one task across the whole panel. Resumable; already-cached is free."""
    failures: list[str] = []
    lock = threading.Lock()

    def one(submission: Submission) -> None:
        try:
            read_task(client, submission, task)
        except FetchError as exc:
            with lock:
                failures.append(f"{submission.fullname}: {exc}")

    _each(one, list(panel))
    if failures:
        progress(f"  {task.suffix}: {len(failures)} models failed, first: {failures[0]}")
    return len(panel) - len(failures)


def fetch_all(
    client: Client,
    panel: Sequence[Submission],
    *,
    tasks: Iterable[Task] = TASKS,
    progress: Callable[[str], None] = lambda _: None,
) -> None:
    """Warm the cache for every task in the panel, smallest tasks first so failures show early."""
    ordered = list(tasks)
    for index, task in enumerate(ordered, start=1):
        started = time.monotonic()
        kept = fetch_task(client, task, panel, progress=progress)
        progress(
            f"[{index}/{len(ordered)}] {task.suffix}: {kept}/{len(panel)} models "
            f"in {time.monotonic() - started:.0f}s"
        )


PROBE: Final = TASKS[0]  # the smallest file in the set: 80 KB, and every submission has it


@dataclass(frozen=True, slots=True)
class Panel:
    """The agreed slice of the leaderboard, and what it cost to get at it."""

    members: tuple[Submission, ...]
    size: int
    seed: int
    strata: int
    per_organisation: int
    rejected: tuple[str, ...]

    def describe(self) -> str:
        averages = [m.average for m in self.members]
        return (
            f"panel of {len(self.members)} models, leaderboard average "
            f"{min(averages):.1f} to {max(averages):.1f}; "
            f"{len(self.rejected)} candidates dropped for missing task files"
        )

    def to_json(self) -> dict[str, object]:
        return {
            "size": self.size,
            "seed": self.seed,
            "strata": self.strata,
            "per_organisation": self.per_organisation,
            "selected": _now(),
            "rejected": list(self.rejected),
            "members": [
                {
                    "fullname": m.fullname,
                    "repo": m.repo,
                    "average": m.average,
                    "model_type": m.model_type,
                    "params_b": m.params_b,
                    "precision": m.precision,
                    "upload_date": m.upload_date,
                }
                for m in self.members
            ],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Panel:
        return cls(
            members=tuple(
                Submission(
                    fullname=str(row["fullname"]),
                    repo=str(row["repo"]),
                    average=float(row["average"]),
                    model_type=str(row["model_type"]),
                    params_b=(None if row["params_b"] is None else float(row["params_b"])),
                    precision=str(row["precision"]),
                    upload_date=str(row["upload_date"]),
                )
                for row in payload["members"]
            ),
            size=int(payload["size"]),
            seed=int(payload["seed"]),
            strata=int(payload["strata"]),
            per_organisation=int(payload["per_organisation"]),
            rejected=tuple(str(name) for name in payload.get("rejected", [])),
        )


def panel_path(root: Path | None = None) -> Path:
    from mselect import paths

    return (root or paths.OLLM_CACHE) / "panel.json"


def load_panel(root: Path | None = None) -> Panel:
    path = panel_path(root)
    if not path.exists():
        raise FetchError(f"no panel at {path}: run `mselect bank panel` first")
    return Panel.from_json(json.loads(path.read_text(encoding="utf-8")))


def _probe(client: Client, chosen: Sequence[Submission]) -> tuple[list[str], list[str]]:
    """Read the smallest file in the set for each candidate, and name the ones that cannot."""
    failures: list[str] = []
    reasons: list[str] = []
    lock = threading.Lock()

    def one(submission: Submission) -> None:
        try:
            read_task(client, submission, PROBE)
        except (FetchError, OSError) as exc:
            with lock:
                failures.append(submission.fullname)
                reasons.append(f"{submission.fullname}: {exc}")

    _each(one, list(chosen))
    return failures, reasons


def assemble_panel(
    client: Client,
    *,
    size: int = 400,
    strata: int = 40,
    per_organisation: int = 8,
    seed: int = 0,
    rounds: int = 3,
    progress: Callable[[str], None] = lambda _: None,
) -> Panel:
    """Choose the panel, ask for access to it, and prove every member can actually be read.

    The proof is the point. A leaderboard row does not guarantee that the run's files survived
    the Parquet conversion, and a model that turns up missing halfway through a build leaves a
    hole in a matrix whose whole advantage over bank v1 is that it has none. So each candidate
    is probed on the smallest file in the set, and one that cannot answer is replaced before
    anything large is downloaded.
    """
    frame = leaderboard(client)
    repos = details_repos(client)
    pool = candidates(frame, repos)
    progress(f"{len(pool)} leaderboard submissions have a details repository")

    rejected: list[str] = []
    keep: list[Submission] = []
    blocked: set[str] = set()
    for attempt in range(1, rounds + 1):
        available = [s for s in pool if s.fullname not in blocked and s not in keep]
        wanted = size - len(keep)
        if wanted <= 0:
            break
        chosen = stratified_panel(
            available,
            size=wanted,
            strata=strata,
            per_organisation=per_organisation,
            seed=seed + attempt - 1,
        )
        if not chosen:
            break
        failures, why = _probe(client, chosen)
        rejected.extend(why)
        blocked.update(failures)
        keep.extend(s for s in chosen if s.fullname not in blocked)
        progress(
            f"round {attempt}: {len(chosen)} probed, {len(failures)} unreadable, "
            f"{len(keep)}/{size} in the panel"
        )

    members = tuple(sorted(keep, key=lambda s: s.fullname))
    return Panel(
        members=members,
        size=size,
        seed=seed,
        strata=strata,
        per_organisation=per_organisation,
        rejected=tuple(rejected),
    )
