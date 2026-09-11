"""Per-item results from the public HELM buckets.

PLAN.md section 3.1 named two public sources. The Open LLM Leaderboard "details" datasets are
gated on Hugging Face (`gated: auto`, verified 2026-09-10), so they need a signed-in token;
HELM's public bucket needs nothing. HELM is therefore the primary source and bank v1 is built
from it alone. The leaderboard is now reachable and is bank v2: see `mselect/data/ollm.py`,
`docs/data-sources.md` for both verifications, and PLAN.md sections 13.1 and 14.1.

What the bucket gives, per run (one model on one scenario):

* `per_instance_stats.json`: one record per instance with a list of named statistics. One of
  those names is the binary correctness metric for that scenario.
* `instances.json`: the instance text, its references, and which reference is the answer key.
* `<release>/schema.json`: model metadata (organisation, access, release date, parameters).
* `<release>/runs_to_run_suites.json`: which suite directory holds each run.

Everything fetched is cached under `data/raw/helm/` by URL hash and never fetched twice, and
every fetch appends a provenance line (URL, SHA-256 of the bytes, date) so a stranger can say
exactly which bytes a bank version was built from.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx

BUCKET: Final = "https://storage.googleapis.com/crfm-helm-public"
USER_AGENT: Final = "mselect/0.1 (item bank build; github.com/Peter-A-P)"
TIMEOUT_S: Final = 120.0
RETRY_STATUS: Final = frozenset({408, 429, 500, 502, 503, 504})
RETRIES: Final = 6
BACKOFF_CAP_S: Final = 60.0
WORKERS: Final = 12


class FetchError(RuntimeError):
    """A source could not be fetched. The build stops rather than freeze a short bank."""


@dataclass(frozen=True, slots=True)
class Scenario:
    """One HELM scenario to take into the bank, and the metric that scores it 0 or 1."""

    project: str  # HELM project: "mmlu", "lite", "capabilities"
    release: str  # release directory under the project, e.g. "v1.13.0"
    prefix: str  # run-name prefix, e.g. "gsm"
    metric: str  # the per-instance statistic that is 0 or 1
    benchmark: str  # the label this scenario contributes to in the bank
    kind: str  # "multiple_choice" or "free_response"; 3PL is fitted only for the former


# Every scenario in the public HELM releases that is scored 0 or 1 per instance and is not
# judged by a model. `ifeval` is excluded because its per-instance score is the fraction of
# instructions satisfied, not a binary outcome; `omni_math` and `wildbench` are excluded
# because they are model-judged; `narrative_qa`, `natural_qa` and `wmt_14` are excluded
# because they are scored with F1 or BLEU against references. PLAN.md section 2: binary-scored
# items only.
SCENARIOS: Final[tuple[Scenario, ...]] = (
    Scenario("mmlu", "v1.13.0", "mmlu", "exact_match", "mmlu", "multiple_choice"),
    Scenario(
        "capabilities",
        "v1.15.0",
        "mmlu_pro",
        "chain_of_thought_correctness",
        "mmlu_pro",
        "multiple_choice",
    ),
    Scenario(
        "capabilities",
        "v1.15.0",
        "gpqa",
        "chain_of_thought_correctness",
        "gpqa",
        "multiple_choice",
    ),
    Scenario("lite", "v1.13.0", "math", "math_equiv_chain_of_thought", "math", "free_response"),
    Scenario("lite", "v1.13.0", "gsm", "final_number_exact_match", "gsm8k", "free_response"),
    Scenario("lite", "v1.13.0", "med_qa", "exact_match", "med_qa", "multiple_choice"),
    Scenario("lite", "v1.13.0", "legalbench", "quasi_exact_match", "legalbench", "multiple_choice"),
    Scenario("lite", "v1.13.0", "commonsense", "exact_match", "commonsense", "multiple_choice"),
)


@dataclass(frozen=True, slots=True)
class Run:
    """One model's run of one scenario, and the suite directory that holds it."""

    scenario: Scenario
    name: str  # the full HELM run name
    suite: str
    model: str  # HELM model id, "org/name"
    scenario_key: str  # the run name with the model dropped: the same items for every model

    @property
    def url(self) -> str:
        quoted = urllib.parse.quote(self.name, safe="")
        return f"{BUCKET}/{self.scenario.project}/benchmark_output/runs/{self.suite}/{quoted}"


@dataclass(frozen=True, slots=True)
class ModelMeta:
    """Row metadata from the release schema. Release month drives the contamination check."""

    model: str
    display_name: str
    organisation: str
    access: str  # "open", "limited", "closed"
    release_date: str  # ISO date, "" when the release does not say
    num_parameters: int | None


@dataclass(frozen=True, slots=True)
class Instance:
    """One benchmark item as HELM stores it, before it is hashed into the bank."""

    instance_id: str
    text: str
    options: tuple[str, ...]
    answer: str  # the reference tagged correct, or "" when HELM does not tag one


def _now() -> str:
    return datetime.now(UTC).date().isoformat()


class Cache:
    """Bytes on disk keyed by URL, plus a provenance line per fetch.

    The cache is what makes a bank build reproducible and free to repeat; PLAN.md section 3.3
    applies the same rule to vendor calls. Payloads are stored gzipped because HELM's JSON is
    repetitive and the raw form runs to several gigabytes.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.blobs = root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.provenance = root / "provenance.jsonl"
        self._lock = threading.Lock()

    def _path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()
        return self.blobs / digest[:2] / f"{digest}.json.gz"

    def get(self, url: str) -> bytes | None:
        path = self._path(url)
        if not path.exists():
            return None
        return gzip.decompress(path.read_bytes())

    def put(self, url: str, payload: bytes) -> None:
        path = self._path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_bytes(gzip.compress(payload, compresslevel=6))
        tmp.replace(path)
        record = {
            "url": url,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "fetched": _now(),
        }
        with self._lock, self.provenance.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


class Client:
    """A pooled HTTP client with the backoff a free public bucket deserves."""

    def __init__(self, cache: Cache, *, seed: int = 0) -> None:
        self.cache = cache
        self._client = httpx.Client(
            timeout=TIMEOUT_S,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"},
            limits=httpx.Limits(max_connections=WORKERS, max_keepalive_connections=WORKERS),
        )
        self._rng = random.Random(seed)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def bytes_at(self, url: str, *, missing_ok: bool = False) -> bytes | None:
        cached = self.cache.get(url)
        if cached is not None:
            return cached
        last = ""
        for attempt in range(RETRIES):
            try:
                response = self._client.get(url)
            except httpx.HTTPError as exc:  # transport level: retried like a 503
                last = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    payload = response.content
                    self.cache.put(url, payload)
                    return payload
                if response.status_code == 404 and missing_ok:
                    return None
                if response.status_code not in RETRY_STATUS:
                    raise FetchError(f"{url} returned HTTP {response.status_code}")
                last = f"HTTP {response.status_code}"
            time.sleep(min(BACKOFF_CAP_S, 2.0**attempt) * (0.5 + self._rng.random()))
        raise FetchError(f"{url} failed after {RETRIES} attempts ({last})")

    def json_at(self, url: str, *, missing_ok: bool = False) -> Any:
        payload = self.bytes_at(url, missing_ok=missing_ok)
        if payload is None:
            return None
        return json.loads(payload)


def _release_url(project: str, release: str, name: str) -> str:
    return f"{BUCKET}/{project}/benchmark_output/releases/{release}/{name}"


def parse_run_name(name: str) -> tuple[str, str, str]:
    """Split a HELM run name into (scenario prefix, model id, scenario key).

    A run name is `scenario:k=v,k=v,model=org_name,k=v`. The model id uses an underscore where
    the HELM model id uses a slash, and can itself contain a colon (`amazon_nova-lite-v1:0`),
    so only the first colon separates the scenario from its parameters.
    """
    prefix, _, rest = name.partition(":")
    kept: list[str] = []
    model = ""
    for part in rest.split(","):
        key, _, value = part.partition("=")
        if key == "model":
            model = value.replace("_", "/", 1)
        else:
            kept.append(part)
    return prefix, model, f"{prefix}:{','.join(kept)}"


def runs_for(client: Client, scenario: Scenario) -> list[Run]:
    """Every run of one scenario in a release, from the release's run-to-suite map."""
    url = _release_url(scenario.project, scenario.release, "runs_to_run_suites.json")
    mapping: Mapping[str, str] = client.json_at(url)
    runs: list[Run] = []
    for name, suite in sorted(mapping.items()):
        prefix, model, key = parse_run_name(name)
        if prefix != scenario.prefix or not model:
            continue
        runs.append(Run(scenario, name, suite, model, key))
    return runs


def models_for(client: Client, project: str, release: str) -> dict[str, ModelMeta]:
    """Model metadata from a release schema, keyed by HELM model id."""
    schema = client.json_at(_release_url(project, release, "schema.json"))
    out: dict[str, ModelMeta] = {}
    for row in schema.get("models", []):
        params = row.get("num_parameters")
        out[row["name"]] = ModelMeta(
            model=row["name"],
            display_name=row.get("display_name") or row["name"],
            organisation=row.get("creator_organization") or "",
            access=row.get("access") or "",
            release_date=row.get("release_date") or "",
            num_parameters=int(params) if isinstance(params, int) else None,
        )
    return out


def responses(client: Client, run: Run) -> dict[str, int]:
    """Instance id to 0 or 1 for this run's scoring metric.

    A metric that is not 0 or 1 is a bug in the scenario table, not a number to round, so it
    raises rather than quietly becoming an item response.
    """
    payload = client.json_at(f"{run.url}/per_instance_stats.json", missing_ok=True)
    if payload is None:
        return {}
    out: dict[str, int] = {}
    for record in payload:
        for stat in record["stats"]:
            if stat["name"]["name"] != run.scenario.metric or stat.get("count", 0) < 1:
                continue
            value = float(stat["mean"])
            if value not in (0.0, 1.0):
                raise FetchError(f"{run.name}: metric {run.scenario.metric} is {value}, not binary")
            out[record["instance_id"]] = int(value)
    return out


def instances(client: Client, run: Run) -> dict[str, Instance]:
    """The items of one scenario run: text, options, and the reference tagged correct."""
    payload = client.json_at(f"{run.url}/instances.json", missing_ok=True)
    if payload is None:
        return {}
    out: dict[str, Instance] = {}
    for record in payload:
        options: list[str] = []
        answer = ""
        for reference in record.get("references", []):
            text = str(reference.get("output", {}).get("text", ""))
            options.append(text)
            if "correct" in reference.get("tags", []):
                answer = text
        out[record["id"]] = Instance(
            instance_id=record["id"],
            text=str(record.get("input", {}).get("text", "")),
            options=tuple(options),
            answer=answer,
        )
    return out


def _each[T](fn: Callable[[T], object], items: Sequence[T]) -> int:
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for _ in pool.map(fn, items):
            done += 1
    return done


def instance_runs(runs: Sequence[Run]) -> list[Run]:
    """One run per (scenario key, suite): the items are the same for every model there."""
    first: dict[tuple[str, str], Run] = {}
    for run in runs:
        first.setdefault((run.scenario_key, run.suite), run)
    return list(first.values())


def fetch_all(
    client: Client,
    scenarios: Iterable[Scenario] = SCENARIOS,
    *,
    progress: Callable[[str], None] = lambda _: None,
) -> None:
    """Warm the cache for every run of every scenario, and one instance file per scenario key."""
    for scenario in scenarios:
        runs = runs_for(client, scenario)
        label = f"{scenario.project}/{scenario.prefix}"
        progress(f"{label}: {len(runs)} runs")
        client.json_at(_release_url(scenario.project, scenario.release, "schema.json"))

        def instances_of(run: Run) -> object:
            return client.bytes_at(f"{run.url}/instances.json", missing_ok=True)

        def stats_of(run: Run) -> object:
            return client.bytes_at(f"{run.url}/per_instance_stats.json", missing_ok=True)

        done = _each(instances_of, instance_runs(runs))
        progress(f"{label}: {done} instance files")
        done = _each(stats_of, runs)
        progress(f"{label}: {done} run files, done")
