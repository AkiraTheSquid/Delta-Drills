"""
Reads 'Export of numpy problems.csv', runs each Answer code snippet,
captures stdout, and writes 'Export of numpy problems with outputs.csv'
with an added 'Output' column.
"""

import csv
import subprocess
import sys
import os

INPUT_CSV = os.path.join(os.path.dirname(__file__), "Export of numpy problems.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "Export of numpy problems with outputs.csv")

PYTHON = sys.executable  # use the same python that runs this script


def run_code(code: str, timeout: int = 10) -> str:
    """Run a code snippet and return its stdout. Returns error string on failure."""
    # Prepend numpy import
    full_code = "import numpy as np\n" + code
    try:
        result = subprocess.run(
            [PYTHON, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"[ERROR] {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "[ERROR] Timeout"
    except Exception as e:
        return f"[ERROR] {e}"


def main():
    # Read original CSV (skip 2 empty rows, row 3 is header)
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    # Parse with csv reader starting from row 3 (0-indexed: line 2)
    reader = csv.reader(raw_lines[2:])
    header = next(reader)  # Topic, Subtopic, Question, Answer, Problem difficulty

    rows = []
    total = 0
    success = 0
    no_output = 0
    errors = 0

    for row in reader:
        if len(row) < 5 or not row[0].strip():
            continue
        total += 1
        answer_code = row[3]
        output = run_code(answer_code)

        if output.startswith("[ERROR]"):
            errors += 1
            print(f"  [{total}] ERROR: {row[2][:60]}...")
            print(f"         {output}")
        elif not output:
            no_output += 1
            print(f"  [{total}] NO OUTPUT: {row[2][:60]}...")
        else:
            success += 1

        rows.append(row + [output])

    # Write new CSV with Output column
    new_header = header + ["Output"]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        # Write the 2 empty rows to match original format
        f.write(",,,,\n")
        f.write(",,,,\n")
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(rows)

    print(f"\nDone! {total} problems processed:")
    print(f"  {success} with output")
    print(f"  {no_output} with no output (no print statement)")
    print(f"  {errors} errors")
    print(f"Written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
