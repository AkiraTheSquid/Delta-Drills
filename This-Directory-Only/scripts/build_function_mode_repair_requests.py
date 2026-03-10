#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
THIS_DIR_ONLY = REPO_DIR / "This-Directory-Only"
CHATGPT_RUNTIME_DIR = Path(
    os.environ.get("DELTA_CHATGPT_RUNTIME_DIR", str(THIS_DIR_ONLY / "chatgpt"))
).resolve()
FAILURES_PATH = CHATGPT_RUNTIME_DIR / "function_mode_validation_failures.jsonl"
OUT_PATH = CHATGPT_RUNTIME_DIR / "function_mode_repair_requests.jsonl"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not FAILURES_PATH.exists():
        OUT_PATH.write_text("", encoding="utf-8")
        print(f"No failures file at {FAILURES_PATH}")
        return

    rows = [json.loads(line) for line in FAILURES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} repair requests to {OUT_PATH}")


if __name__ == "__main__":
    main()
