#!/usr/bin/env python3

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
THIS_DIR_ONLY = REPO_DIR / "This-Directory-Only"
BACKEND_PYTHON = THIS_DIR_ONLY / "backend" / ".venv" / "bin" / "python3"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(THIS_DIR_ONLY), check=True)


def main() -> None:
    run(["python3", "scripts/export_questions_json.py"])
    run(["python3", "scripts/test_function_validator.py"])
    run(["python3", "scripts/validate_function_bank.py"])
    run(["python3", "scripts/export_questions_json.py"])
    max_rounds = 3

    for round_idx in range(1, max_rounds + 1):
        failures_path = THIS_DIR_ONLY / "chatgpt" / "function_mode_validation_failures.jsonl"
        if not failures_path.exists() or not failures_path.read_text(encoding="utf-8").strip():
            print(f"Function-mode question bank validated in {round_idx - 1} repair rounds.")
            return

        print(f"Starting repair round {round_idx}...")
        run(["python3", "scripts/build_function_mode_repair_requests.py"])
        run(
            [
                str(BACKEND_PYTHON if BACKEND_PYTHON.exists() else "python3"),
                "chatgpt/function_mode_batch.py",
                "--requests",
                "chatgpt/function_mode_repair_requests.jsonl",
                "--outputs",
                "chatgpt/function_mode_overrides.jsonl",
                "--system-prompt-file",
                "chatgpt/function_mode_repair_system.txt",
                "--max-attempts",
                "3",
            ]
        )
        run(["python3", "scripts/export_questions_json.py"])
        run(["python3", "scripts/test_function_validator.py"])
        run(["python3", "scripts/validate_function_bank.py"])
        run(["python3", "scripts/export_questions_json.py"])

    print("Reached max repair rounds; inspect chatgpt/function_mode_validation_failures.jsonl.")


if __name__ == "__main__":
    main()
