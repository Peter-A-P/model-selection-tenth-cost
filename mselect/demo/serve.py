"""Serve the demo directory under the headers the hosting config declares.

A static page is only as checked as the headers it was checked under. `python -m http.server`
sends none, so an inline style attribute, an off-origin font or a fetch the policy forbids all
work locally and fail on the live site. This reads `staticwebapp.config.json` from the folder
being served and sends its `globalHeaders` on every response, which turns that class of defect
back into something a person can see before it is published.

It is a development server and says so: it binds the loopback interface only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

#: The hosting config, committed beside the page, that the live site is deployed with.
CONFIG_NAME = "staticwebapp.config.json"


def declared_headers(directory: Path) -> dict[str, str]:
    """The ``globalHeaders`` block of the hosting config, or nothing if there is no config.

    A missing config is not an error: the folder may simply be served by something else. A
    malformed one is, because the whole point is to send what the live site sends.
    """
    config = directory / CONFIG_NAME
    if not config.is_file():
        return {}
    parsed = json.loads(config.read_text(encoding="utf-8"))
    headers = parsed.get("globalHeaders", {})
    if not isinstance(headers, Mapping):
        raise TypeError(f"{config}: globalHeaders is not an object")
    return {str(key): str(value) for key, value in headers.items()}


def build_server(directory: Path, port: int) -> ThreadingHTTPServer:
    """A server for ``directory`` on the loopback interface, not yet running."""
    declared = declared_headers(directory)

    class Handler(SimpleHTTPRequestHandler):
        """A file handler that adds the declared headers to every response."""

        def end_headers(self) -> None:
            for key, value in declared.items():
                self.send_header(key, value)
            super().end_headers()

        def log_message(self, format: str, *args: object) -> None:
            """Quiet. The page loads a handful of files and the log says nothing useful."""

    return ThreadingHTTPServer(
        ("127.0.0.1", port),
        partial(Handler, directory=str(directory.resolve())),
    )
