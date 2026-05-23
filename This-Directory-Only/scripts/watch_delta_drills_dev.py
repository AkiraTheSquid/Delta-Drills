#!/usr/bin/env python3
"""watch_delta_drills_dev.py

Keep Delta Drills Chrome isolated.

This watcher only talks to the Chrome instance launched by `delta_drills_dev`
via its remote-debugging port. It prunes any tab that is not Delta Drills, so
an old profile or a stale Chrome session cannot bleed another app into this
debug session.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

DEFAULT_ALLOWED_PREFIXES = (
    "http://localhost:5174/",
    "http://127.0.0.1:5174/",
    "about:blank",
    "chrome://newtab/",
)

running = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument(
        "--frontend-url",
        default="http://localhost:5174",
        help="Primary URL allowed in the Chrome session.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.5,
        help="How often to inspect the remote-debugging tab list.",
    )
    return parser.parse_args()


def on_signal(_signum, _frame):
    global running
    running = False


def remote_json(port: int, path: str):
    url = f"http://127.0.0.1:{port}{path}"
    with urlopen(url, timeout=2) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def close_tab(port: int, tab_id: str, url: str) -> None:
    close_path = f"/json/close/{tab_id}"
    try:
        with urlopen(f"http://127.0.0.1:{port}{close_path}", timeout=2) as response:
            _ = response.read()
        print(f"closed stray tab: {url}", flush=True)
    except HTTPError as exc:
        print(f"close failed for {url}: {exc}", flush=True)
    except URLError as exc:
        print(f"close error for {url}: {exc}", flush=True)


def allowed(url: str, prefixes: tuple[str, ...]) -> bool:
    return any(url.startswith(prefix) for prefix in prefixes)


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    prefixes = DEFAULT_ALLOWED_PREFIXES + (args.frontend_url.rstrip("/") + "/",)

    print(
        f"watching Chrome debug port {args.port} for non-Delta-Drills tabs",
        flush=True,
    )

    while running:
        try:
            tabs = remote_json(args.port, "/json/list")
        except Exception:
            time.sleep(args.poll_seconds)
            continue

        for tab in tabs:
            url = tab.get("url", "")
            tab_id = tab.get("id", "")
            if not tab_id:
                continue
            if allowed(url, prefixes):
                continue
            # Keep the session scoped to Delta Drills only.
            close_tab(args.port, tab_id, url)

        time.sleep(args.poll_seconds)

    return 0


if __name__ == "__main__":
    sys.exit(main())
