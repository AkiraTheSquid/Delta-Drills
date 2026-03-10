"""
Reads einops_problems.csv, runs each Answer code snippet with a
standard einops + numpy preamble, captures stdout, and writes
einops_problems_with_outputs.csv with an added 'Output' column.

NOTE: Most einops answers are code snippets that reference pre-defined
variables from a notebook context and do not contain print() statements.
For those problems the Output column will be empty, and the practice
system falls back to running the answer code at evaluation time (with
the AI judge providing semantic grading when outputs differ).

For outputs to be captured the answer code must produce stdout — either
by having a print() statement or by the preamble's exec wrapper printing
the last expression value.

Run this script from within the backend venv (where einops is installed):
    python "csv files of problems/generate_einops_outputs.py"
"""

import csv
import subprocess
import sys
import os

INPUT_CSV = os.path.join(os.path.dirname(__file__), "einops_problems.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "einops_problems_with_outputs.csv")

PYTHON = sys.executable  # use the same python that runs this script

# ---------------------------------------------------------------------------
# Preamble: defines all variables referenced by einops answer snippets.
# Shapes chosen to be divisible by 2, 3, 4, and 8 (common pool strides).
# ---------------------------------------------------------------------------
PREAMBLE = """\
import numpy as np
import einops
try:
    from einops.layers.torch import Reduce
except ImportError:
    pass
np.random.seed(42)

# --- dimensions ---
b, c, h, w = 8, 16, 24, 24   # batch, channels, height, width
t, d       = 6, 16            # sequence length, feature dim
hs, ws     = 2, 2             # subgrid strides

# --- standard BCHW batch (channels-first) ---
arr       = np.random.rand(b, c, h, w).astype(np.float32)
arr_even  = arr                                         # even batch alias
x         = np.random.rand(b, c, h, w).astype(np.float32)

# --- single image (CHW, channels-first) ---
img = np.random.rand(c, h, w).astype(np.float32)

# --- single image in HWC (channels-last) ---
img_hwc = np.random.rand(h, w, c).astype(np.float32)

# --- batch of HWC images as a numpy array (b h w c) ---
hwcs = np.random.rand(b, h, w, c).astype(np.float32)

# --- list of BCHW tensors (represented as a single array here) ---
list_of_tensors = np.random.rand(b, c, h, w).astype(np.float32)

# --- two separate HWC images for interleave problems ---
img_a = np.random.rand(h, w, c).astype(np.float32)
img_b = np.random.rand(h, w, c).astype(np.float32)
img_c = np.random.rand(h, w, c).astype(np.float32)
img_d = np.random.rand(h, w, c).astype(np.float32)

# --- sequences ---
seq        = np.random.rand(b, t, d).astype(np.float32)
seq_chunks = einops.rearrange(seq, 'b (n p) d -> b n p d', p=2)

# --- flat feature map (c, h*w) ---
flat = einops.rearrange(img, 'c h w -> c (h w)')

# --- BCHW patches (n_patches, c, p, p) ---
patches = einops.rearrange(img, 'c (h p1) (w p2) -> (h w) c p1 p2', p1=4, p2=4)

# --- class embeddings and per-token weights ---
cls     = np.random.rand(b, d).astype(np.float32)
weights = np.random.rand(b, t).astype(np.float32)

# --- attention head inputs ---
x_heads = np.random.rand(b, 8, t, d).astype(np.float32)   # already split heads
x_seq   = np.random.rand(b, c, t * 3).astype(np.float32)  # 1-D sequence (b c T)
x_seq2  = np.random.rand(b, t, 8 * d).astype(np.float32)  # pre-split-heads

# --- 3-D volume for 3-D pooling ---
x_3d = np.random.rand(b, c, 8, 8, 8).astype(np.float32)

# --- 6-image batch for grid problems ---
arr6x = np.random.rand(6, h, w, c).astype(np.float32)

# --- pre-computed subgrid-in-batch (used by unpack problems) ---
y = einops.rearrange(x, 'b c (hh hs) (ww ws) -> (hs ws b) c hh ww', hs=2, ws=2)

# --- mock display helper (prints shape instead of showing image) ---
def display_array_as_img(a):
    print(a.shape)

"""


def run_code(code: str, timeout: int = 15) -> str:
    """Run a code snippet and return its stdout. Returns error string on failure."""
    full_code = PREAMBLE + code
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
            return f"[ERROR] {result.stderr.strip()[:120]}"
    except subprocess.TimeoutExpired:
        return "[ERROR] Timeout"
    except Exception as e:
        return f"[ERROR] {e}"


def main():
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # Ensure Output column exists in fieldnames
    out_fields = list(fieldnames)
    if "Output" not in out_fields:
        out_fields.append("Output")

    total   = 0
    success = 0
    no_output = 0
    errors  = 0
    updated = []

    for row in rows:
        if not (row.get("Question") or "").strip():
            updated.append(row)
            continue

        total += 1
        answer_code = (row.get("Answer") or "").strip()

        # Only recompute if Output is not already populated
        existing = (row.get("Output") or "").strip()
        if existing and not existing.startswith("[ERROR]"):
            success += 1
            updated.append(row)
            continue

        output = run_code(answer_code)

        if output.startswith("[ERROR]"):
            errors += 1
            print(f"  [{total}] ERROR   : {(row.get('Question') or '')[:60]}...")
            print(f"             {output}")
        elif not output:
            no_output += 1
            # Uncomment to see which problems produce no output:
            # print(f"  [{total}] NO OUTPUT: {(row.get('Question') or '')[:60]}...")
        else:
            success += 1
            print(f"  [{total}] OK      : {(row.get('Question') or '')[:60]}")

        row["Output"] = output
        updated.append(row)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(updated)

    print(f"\nDone! {total} problems processed:")
    print(f"  {success}    with output")
    print(f"  {no_output}  with no output (answer has no print statement)")
    print(f"  {errors}  errors")
    print(f"Written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
