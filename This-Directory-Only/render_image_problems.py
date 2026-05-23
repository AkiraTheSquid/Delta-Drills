#!/usr/bin/env python3
"""
Render every image-output problem's expected_expr to PNG for visual auditing.

Output: /tmp/dd_renders/<topic>_<id>_<slug>.png
        /tmp/dd_renders/INDEX.md (question text + answer per id)
        /tmp/dd_renders/FAILURES.md (errors)
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
QUESTIONS = REPO / "This-Directory-Only" / "questions_full.json"
DELTA_NUMBERS = REPO / "Local_Deployed_Shared" / "delta_numbers.npy"
OUT = Path("/tmp/dd_renders")


def _normalize_3d(a: np.ndarray) -> np.ndarray:
    """Normalize a 3D array to (H, W, C) where C in {1, 3}, else collapse to (H, W*C)."""
    if a.shape[-1] in (1, 3):
        return a
    if a.shape[0] in (1, 3):
        return np.moveaxis(a, 0, -1)
    # No clear channel axis — collapse last dim horizontally
    return a.reshape(a.shape[0], -1)


def _flatten_to_2d_or_3d(a: np.ndarray) -> tuple[np.ndarray, str]:
    """Reduce arbitrary-rank array to something PIL can render. Returns (array, note)."""
    note = ""
    while a.ndim > 4:
        # Collapse the leading axis into a tile-grid along width
        a = np.concatenate([a[i] for i in range(a.shape[0])], axis=-1)
        note += f" collapse->{a.shape}"
    if a.ndim == 4:
        # Treat axis 0 as batch; normalize each (...,) frame, tile along width
        frames = [_normalize_3d(a[i]) if a[i].ndim == 3 else a[i] for i in range(a.shape[0])]
        # Pad to common height
        h = max(f.shape[0] for f in frames)
        padded = []
        for f in frames:
            if f.shape[0] < h:
                pad = np.zeros((h - f.shape[0], *f.shape[1:]), dtype=f.dtype)
                f = np.concatenate([f, pad], axis=0)
            padded.append(f)
        a = np.concatenate(padded, axis=1)
        note += f" tiled->{a.shape}"
    if a.ndim == 3:
        a = _normalize_3d(a)
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim == 0:
        a = np.array([[float(a)]])
    return a, note


def to_uint8_image(arr: np.ndarray) -> tuple[Image.Image, str]:
    """Convert any numeric array into a viewable PIL image. Returns (img, note)."""
    a = np.asarray(arr)
    base_note = f"shape={a.shape} dtype={a.dtype}"
    a, extra = _flatten_to_2d_or_3d(a)
    note = base_note + extra

    a = a.astype(np.float64)
    if a.size == 0:
        raise ValueError("empty array")
    mn, mx = float(a.min()), float(a.max())
    if mx > mn:
        a = (a - mn) / (mx - mn) * 255.0
    else:
        a = np.zeros_like(a)
    a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        return Image.fromarray(a, mode="L"), note
    return Image.fromarray(a, mode="RGB"), note


def slugify(text: str, n: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text)[:n]
    text = re.sub(r"\s+", "_", text).strip("_")
    return text.lower() or "untitled"


def evaluate(q: dict) -> tuple[np.ndarray | None, str | None]:
    """Execute expected_expr with its setup. Returns (array, error)."""
    tcs = q.get("test_cases") or []
    if not tcs:
        return None, "no test_cases"
    tc = tcs[0]
    starter = (q.get("starter_code") or "").replace(
        "/delta_numbers.npy", str(DELTA_NUMBERS)
    )
    setup = (tc.get("setup_code") or "").replace(
        "/delta_numbers.npy", str(DELTA_NUMBERS)
    )
    expr = (tc.get("expected_expr") or "").strip()
    expected_setup = (tc.get("expected_setup_code") or "").replace(
        "/delta_numbers.npy", str(DELTA_NUMBERS)
    )
    if not expr or expr == "None":
        return None, "no expected_expr"
    env: dict = {"np": np}
    try:
        import einops
        env["einops"] = einops
        env["rearrange"] = einops.rearrange
        env["reduce"] = einops.reduce
        env["repeat"] = einops.repeat
        env["einsum"] = einops.einsum
    except ImportError:
        pass
    try:
        if starter.strip():
            exec(starter, env)
        if setup.strip():
            exec(setup, env)
        if expected_setup.strip():
            exec(expected_setup, env)
        result = eval(expr, env)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    return result, None


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    qs = json.loads(QUESTIONS.read_text())
    visual = [
        q for q in qs
        if q.get("expected_artifact_type") == "image"
        or q.get("supports_visual_output") is True
    ]
    print(f"Total image-output problems: {len(visual)}")

    index_lines: list[str] = ["# Render index", ""]
    failures: list[str] = []
    rendered = 0
    for q in sorted(visual, key=lambda x: x["id"]):
        qid = q["id"]
        topic = (q.get("subtopic_key") or q.get("topic") or "misc").replace(" ", "_").replace(":", "")
        slug = slugify(q.get("question_text", "")[:60])
        name = f"{topic}_{qid:03d}_{slug}.png"
        arr, err = evaluate(q)
        if err is not None:
            failures.append(f"- id={qid} [{topic}] {q['question_text'][:60]!r}\n  {err}")
            continue
        try:
            img, note = to_uint8_image(arr)
            img.save(OUT / name)
            rendered += 1
            index_lines.append(
                f"- **{qid}** [{topic}] `{name}` — {note}\n"
                f"  - Q: {q['question_text'][:100]}\n"
                f"  - A: `{(q.get('answer_code') or '')[:120]}`"
            )
        except Exception as e:
            failures.append(
                f"- id={qid} [{topic}] {q['question_text'][:60]!r}\n"
                f"  render-error: {type(e).__name__}: {e}\n"
                f"  shape={getattr(arr, 'shape', '?')} dtype={getattr(arr, 'dtype', '?')}"
            )

    (OUT / "INDEX.md").write_text("\n".join(index_lines))
    (OUT / "FAILURES.md").write_text(
        f"# Render failures: {len(failures)}\n\n" + "\n\n".join(failures)
    )
    print(f"Rendered: {rendered}  |  Failures: {len(failures)}")
    print(f"Output: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
