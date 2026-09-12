"""What each vendor says it will serve today, so a route is checked rather than believed.

A routing file is a set of claims about identifiers that someone else owns and renames without
telling you. Four of the eleven routes were wrong on 2026-09-12 and the smoke run found them one
paid call at a time. This asks each vendor for its own list instead.

Listing models costs nothing anywhere: no tokens are generated, so no tokens are billed. That is
why this does not go through the gateway, which exists to meter and price chat calls. Nothing
here can spend, and nothing here writes a ledger row.

A model the vendor does not list is not always broken, because some vendors omit models an
account can still call. So a missing identifier is reported as "not listed" and never as
"wrong": the thing that proves a route is `mselect smoke`, and this is what tells you which
routes are worth smoking.
"""

from __future__ import annotations

import os
import re
import ssl
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import httpx
import truststore

TIMEOUT: Final = 20.0

# How each provider kind answers "what do you serve". The fourth element is where the ids live
# in the response; the fifth is the key under each entry.
LISTINGS: Final[dict[str, tuple[str, str, str]]] = {
    "anthropic": ("/v1/models", "data", "id"),
    "openai_compat": ("/models", "data", "id"),
    "google": ("/models", "models", "name"),
}


@dataclass(frozen=True, slots=True)
class Catalogue:
    """One provider's answer, or the reason it did not give one."""

    provider: str
    models: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True)
class RouteCheck:
    """One panel route against its provider's list."""

    alias: str
    provider: str
    model: str
    listed: bool
    known: bool  # whether the provider answered at all
    nearest: tuple[str, ...] = field(default=())


def _client() -> httpx.Client:
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return httpx.Client(timeout=TIMEOUT, verify=context, follow_redirects=False)


def fetch(name: str, spec: dict[str, Any], *, client: httpx.Client | None = None) -> Catalogue:
    """Ask one provider what it serves. Never raises: a failure is a result here."""
    kind = str(spec.get("kind", ""))
    listing = LISTINGS.get(kind)
    if listing is None:
        return Catalogue(name, error=f"no model listing known for kind {kind!r}")
    # A local server is asked exactly the way a hosted OpenAI-compatible one is: Ollama serves
    # /v1/models too, and a route that names a model nobody pulled is the same mistake as one
    # that names a model the vendor retired.
    path, container, id_key = listing

    base = str(spec.get("base_url", "")).rstrip("/")
    key_env = spec.get("api_key_env")
    key = os.environ.get(str(key_env), "") if key_env else ""
    if key_env and not key:
        return Catalogue(name, error=f"{key_env} is not set")

    headers: dict[str, str] = {}
    params: dict[str, str] = {}
    if kind == "anthropic":
        headers["x-api-key"] = key
        headers["anthropic-version"] = str(spec.get("api_version", "2023-06-01"))
    elif kind == "google":
        params["key"] = key
        base = f"{base}/{spec.get('api_version', 'v1beta')}"
        params["pageSize"] = "200"
    elif key:
        headers["Authorization"] = f"Bearer {key}"

    owned = client is None
    session = client or _client()
    try:
        response = session.get(f"{base}{path}", headers=headers, params=params)
        if response.status_code != httpx.codes.OK:
            return Catalogue(name, error=f"HTTP {response.status_code}: {response.text[:160]}")
        payload = response.json()
    except (httpx.HTTPError, ValueError) as e:
        return Catalogue(name, error=f"{type(e).__name__}: {e}")
    finally:
        if owned:
            session.close()

    entries = payload.get(container, payload) if isinstance(payload, dict) else payload
    models: list[str] = []
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and entry.get(id_key):
                # Google returns "models/gemini-3.8-flash"; the routing file names the tail.
                models.append(str(entry[id_key]).removeprefix("models/"))
            elif isinstance(entry, str):
                models.append(entry)
    return Catalogue(name, models=tuple(sorted(set(models))))


def _tokens(model: str) -> set[str]:
    """An identifier as the words and numbers in it, so `claude-sonnet-5` is three things."""
    return {part for part in re.split(r"[^a-z0-9.]+", model.split("/")[-1].lower()) if part}


def _nearest(model: str, models: Sequence[str], limit: int = 4) -> tuple[str, ...]:
    """Listed identifiers ranked by how much of the name they share with the one asked for.

    Deliberately crude, and ranked rather than filtered: the point is to put the family's real
    names in front of a reader, not to guess which one they meant. Choosing a model is a
    decision about the experiment.

    Ranking is the whole trick. Every Anthropic identifier begins "claude", so anything keyed
    on the first word matches all of them and tells a reader nothing.
    """
    wanted = _tokens(model)
    if not wanted:
        return ()
    # A token common to every identifier a vendor lists distinguishes nothing, so it is worth
    # less. Without this, "claude" counts as much as "opus" and every Claude model ties.
    seen: Counter[str] = Counter()
    parsed = {m: _tokens(m) for m in models}
    for tokens in parsed.values():
        seen.update(tokens)
    scored = [
        (sum(1.0 / seen[token] for token in wanted & tokens), name)
        for name, tokens in parsed.items()
    ]
    ranked = sorted(((-score, name) for score, name in scored if score > 0), key=lambda x: x)
    return tuple(name for _, name in ranked[:limit])


def check_routes(
    providers: dict[str, Any],
    routes: dict[str, dict[str, str]],
    aliases: Sequence[str],
    *,
    catalogues: dict[str, Catalogue] | None = None,
) -> tuple[list[RouteCheck], dict[str, Catalogue]]:
    """Every named alias against its provider's own list."""
    if catalogues is None:
        wanted = {routes[a]["provider"] for a in aliases if a in routes}
        with _client() as client:
            catalogues = {
                name: fetch(name, spec, client=client)
                for name, spec in providers.items()
                if name in wanted and isinstance(spec, dict)
            }

    checks: list[RouteCheck] = []
    for alias in aliases:
        route = routes.get(alias)
        if route is None:
            continue
        catalogue = catalogues.get(route["provider"])
        known = catalogue is not None and catalogue.ok
        models = catalogue.models if catalogue else ()
        listed = known and route["model"] in models
        checks.append(
            RouteCheck(
                alias=alias,
                provider=route["provider"],
                model=route["model"],
                listed=listed,
                known=known,
                nearest=() if listed or not known else _nearest(route["model"], models),
            )
        )
    return checks, catalogues
