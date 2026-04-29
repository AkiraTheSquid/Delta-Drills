#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
# ── pipeline bootstrap ──
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from delta_paths import get_chatgpt_runtime_dir

CHATGPT_RUNTIME_DIR = get_chatgpt_runtime_dir()
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
