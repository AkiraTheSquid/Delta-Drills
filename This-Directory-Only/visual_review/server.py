#!/usr/bin/env python3
"""Local visual audit server for Delta Drills image-output questions."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

try:
    import einops
except ImportError as exc:  # pragma: no cover - startup guard
    raise SystemExit("einops is required: python3 -m pip install einops") from exc


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
QUESTIONS_PATH = REPO / "This-Directory-Only" / "questions_full.json"
DELTA_NUMBERS_PATH = REPO / "Local_Deployed_Shared" / "delta_numbers.npy"
GENERATED_DIR = HERE / "generated"
MANIFEST_PATH = GENERATED_DIR / "manifest.json"
STATE_PATH = HERE / "review_state.json"
FLAGS_JSONL_PATH = HERE / "visual_malformed_flags.jsonl"
FLAGS_MD_PATH = HERE / "visual_malformed_flags.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(text: str, limit: int = 64) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"\s+", "_", text).strip("_")
    return (text or "untitled")[:limit].strip("_")


def localize_paths(code: str | None) -> str:
    return (code or "").replace("/delta_numbers.npy", str(DELTA_NUMBERS_PATH))


def starter_setup(code: str | None) -> str:
    lines: list[str] = []
    for line in (code or "").splitlines():
        if re.match(r"\s*def\s+solve\s*\(", line):
            break
        if re.match(r"\s*print\s*\(\s*solve\s*\(\s*\)\s*\)\s*$", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def shape_list(arr: np.ndarray) -> list[int]:
    return [int(x) for x in arr.shape]


def grid_dims_for_batch(n: int) -> tuple[int, int]:
    if n <= 1:
        return 1, 1
    for rows in range(int(np.sqrt(n)), 1, -1):
        if n % rows == 0:
            return rows, n // rows
    cols = int(np.ceil(np.sqrt(n)))
    return int(np.ceil(n / cols)), cols


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    return np.moveaxis(image, 0, -1)


def tile_batch_images(batch: np.ndarray) -> np.ndarray:
    batch = np.asarray(batch)
    batch_size = int(batch.shape[0])
    rows, cols = grid_dims_for_batch(batch_size)

    if batch.ndim == 3:
        height, width = int(batch.shape[1]), int(batch.shape[2])
        canvas = np.zeros((rows * height, cols * width), dtype=batch.dtype)
    elif batch.ndim == 4:
        height, width, channels = int(batch.shape[1]), int(batch.shape[2]), int(batch.shape[3])
        canvas = np.zeros((rows * height, cols * width, channels), dtype=batch.dtype)
    else:
        raise ValueError(f"cannot tile batch with shape {batch.shape}")

    for i in range(batch_size):
        row = i // cols
        col = i % cols
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width] = batch[i]
    return canvas


@dataclass
class DisplayArray:
    array: np.ndarray
    warnings: list[str]
    render_mode: str
    practice_compatible: bool


@dataclass
class EvaluationResult:
    array: np.ndarray
    warnings: list[str]


def normalize_for_display(value: Any) -> DisplayArray:
    original = np.asarray(value)
    arr = original
    warnings: list[str] = []
    practice_compatible = True
    mode = f"{arr.ndim}d"

    if arr.ndim == 0:
        warnings.append("scalar output expanded to 1x1")
        practice_compatible = False
        arr = arr.reshape(1, 1)
        mode = "scalar-to-2d"
    elif arr.ndim == 1:
        warnings.append("1D vector output rendered as a one-row strip")
        practice_compatible = False
        arr = arr.reshape(1, -1)
        mode = "1d-vector-strip"
    elif arr.ndim == 2:
        height, width = int(arr.shape[0]), int(arr.shape[1])
        aspect = max(height, width) / max(1, min(height, width))
        pixels = height * width
        if aspect > 32:
            warnings.append(f"extreme 2D aspect ratio {aspect:.1f}:1")
            practice_compatible = False
        if pixels > 4_000_000:
            warnings.append(f"large 2D image payload {pixels} pixels")
            practice_compatible = False
        mode = "2d"
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4):
            arr = chw_to_hwc(arr)
            mode = "chw-to-hwc"
        else:
            mode = "hwc-or-volume"
            if arr.shape[-1] not in (1, 3, 4):
                warnings.append(f"3D output has nonstandard trailing channel count {arr.shape[-1]}")
    elif arr.ndim == 4:
        if arr.shape[1] in (1, 3, 4):
            arr = np.moveaxis(arr, 1, -1)
            mode = "bchw-grid"
        else:
            mode = "bhwc-grid"
            if arr.shape[-1] not in (1, 3, 4):
                warnings.append(f"4D output has nonstandard trailing channel count {arr.shape[-1]}")
        arr = tile_batch_images(arr)
    else:
        warnings.append(f"{arr.ndim}D output flattened across leading axes for review")
        practice_compatible = False
        if arr.shape[-3] in (1, 3, 4):
            batch = arr.reshape((-1, *arr.shape[-3:]))
            arr = np.moveaxis(batch, 1, -1)
            mode = f"{original.ndim}d-leading-flatten-bchw-grid"
        else:
            batch = arr.reshape((-1, *arr.shape[-3:]))
            mode = f"{original.ndim}d-leading-flatten-grid"
        arr = batch
        arr = tile_batch_images(arr)

    return DisplayArray(array=np.asarray(arr), warnings=warnings, render_mode=mode, practice_compatible=practice_compatible)


def to_practice_png_array(display: np.ndarray) -> np.ndarray:
    arr = np.asarray(display)
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=255.0, neginf=0.0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"normalized display array must be 2D or 3D, got {arr.shape}")

    channels = int(arr.shape[-1])
    if channels == 1:
        return arr[..., 0]
    if channels == 2:
        out = np.zeros((*arr.shape[:2], 3), dtype=np.uint8)
        out[..., :2] = arr[..., :2]
        return out
    if channels == 3:
        return arr
    out = arr[..., :4]
    return out


def image_from_array(display: np.ndarray) -> Image.Image:
    png_array = to_practice_png_array(display)
    if png_array.ndim == 2:
        return Image.fromarray(png_array, mode="L")
    if png_array.shape[-1] == 4:
        return Image.fromarray(png_array, mode="RGBA")
    return Image.fromarray(png_array, mode="RGB")


def evaluate_question(question: dict[str, Any]) -> EvaluationResult:
    test_cases = question.get("test_cases") or []
    if not test_cases:
        raise ValueError("question has no test_cases")
    test_case = test_cases[0]
    expected_expr = localize_paths(test_case.get("expected_expr")).strip()
    if not expected_expr or expected_expr == "None":
        raise ValueError("question has no expected_expr")

    env: dict[str, Any] = {
        "np": np,
        "einops": einops,
        "rearrange": einops.rearrange,
        "reduce": einops.reduce,
        "repeat": einops.repeat,
        "einsum": einops.einsum,
        "display_array_as_img": lambda *args, **kwargs: None,
    }
    setup_chunks = [
        starter_setup(localize_paths(question.get("starter_code"))),
        localize_paths(test_case.get("setup_code")),
        localize_paths(test_case.get("expected_setup_code")),
    ]
    setup_code = "\n".join(chunk for chunk in setup_chunks if chunk.strip())
    if setup_code.strip():
        exec(setup_code, env)

    result = eval(expected_expr, env)
    warnings: list[str] = []
    if isinstance(result, (tuple, list)) and not hasattr(result, "tolist"):
        for part in result:
            if hasattr(part, "tolist") or isinstance(part, (np.ndarray, np.generic)):
                result = part
                warnings.append("expected_expr returned metadata tuple/list; rendered the ndarray component")
                break
    if not hasattr(result, "tolist") and not isinstance(result, (np.ndarray, np.generic)):
        raise TypeError(f"expected_expr returned {type(result).__name__}; target renderer expects ndarray-like output")
    return EvaluationResult(array=np.asarray(result), warnings=warnings)


def load_questions() -> list[dict[str, Any]]:
    questions = json.loads(QUESTIONS_PATH.read_text())
    return sorted(
        [
            q
            for q in questions
            if q.get("supports_visual_output") is True or q.get("expected_artifact_type") == "image"
        ],
        key=lambda q: int(q["id"]),
    )


def format_shape(shape: list[int] | None) -> str:
    return " x ".join(str(part) for part in (shape or [])) or "unknown"


def visual_description_for(question: dict[str, Any], item: dict[str, Any]) -> str:
    qtext = (question.get("question_text") or "").lower()
    shape = item.get("original_shape") or []
    warnings = item.get("warnings") or []

    if item.get("error"):
        return "No target image could be rendered from the canonical expression; this visual card needs review."

    shape_text = format_shape(shape)
    if len(shape) == 4 and shape[1] in (1, 3, 4):
        base = f"Should show a tiled batch of {shape[0]} channels-first digit images; each tile is {shape[2]} x {shape[3]} with {shape[1]} channel(s)."
    elif len(shape) == 4 and shape[-1] in (1, 3, 4):
        base = f"Should show a tiled batch of {shape[0]} channels-last digit images; each tile is {shape[1]} x {shape[2]} with {shape[3]} channel(s)."
    elif len(shape) == 3 and shape[0] in (1, 3, 4):
        base = f"Should show one channels-first digit or composite image, rendered from shape {shape_text}."
    elif len(shape) == 3 and shape[-1] in (1, 3, 4):
        base = f"Should show one channels-last digit or composite image, rendered from shape {shape_text}."
    elif len(shape) == 2:
        aspect = max(shape) / max(1, min(shape))
        if aspect > 20:
            base = f"Canonical output is a very thin 2D strip with shape {shape_text}; flag it if this diagnostic is supposed to show a recognizable digit."
        else:
            base = f"Should show a 2D grayscale-style array with shape {shape_text}."
    elif len(shape) == 1:
        base = f"Canonical output is a 1D feature vector of length {shape[0]}, so the rendered target will be a strip rather than a natural image."
    elif len(shape) > 4:
        base = f"Canonical output is {len(shape)}D with shape {shape_text}; the audit tool flattens leading axes into a review grid."
    else:
        base = f"Canonical output shape is {shape_text}; compare the rendered target against the prompt intent."

    if "flatten" in qtext or "feature" in qtext or "classifier" in qtext:
        intent = "Prompt intent: a flattened feature representation, not necessarily a readable digit."
    elif "grid" in qtext or "tile" in qtext or "composite" in qtext or "arranged" in qtext:
        intent = "Prompt intent: multiple digit images arranged into a grid or composite."
    elif "repeat" in qtext or "duplicate" in qtext or "increase" in qtext:
        intent = "Prompt intent: the same digit content repeated or stretched along one or more axes."
    elif "pool" in qtext or "average" in qtext or "reduce" in qtext:
        intent = "Prompt intent: a reduced or pooled version of the digit image."
    elif "swap" in qtext or "interchange" in qtext:
        intent = "Prompt intent: an axis swap; if the target becomes a thin strip, flag whether that is useful for this drill."
    elif "grayscale" in qtext or "maximum value" in qtext:
        intent = "Prompt intent: a grayscale-style projection of color channels."
    else:
        intent = "Prompt intent: preserve recognizable digit structure unless the shape transformation says otherwise."

    if warnings:
        return f"{base} {intent} Warning: {'; '.join(warnings)}."
    return f"{base} {intent}"


def render_manifest() -> dict[str, Any]:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    for png in GENERATED_DIR.glob("*.png"):
        png.unlink()

    items: list[dict[str, Any]] = []
    for question in load_questions():
        qid = int(question["id"])
        topic = question.get("topic") or ""
        subtopic = question.get("subtopic_key") or question.get("subtopic") or ""
        filename = f"q{qid:03d}_{slugify(subtopic, 28)}_{slugify(question.get('question_text', ''), 48)}.png"
        item: dict[str, Any] = {
            "id": qid,
            "topic": topic,
            "subtopic": subtopic,
            "difficulty": question.get("difficulty_label") or "",
            "question_text": question.get("question_text") or "",
            "answer_code": question.get("answer_code") or "",
            "source_path": question.get("source_path") or "",
            "expected_expr": (question.get("test_cases") or [{}])[0].get("expected_expr") or "",
            "image_file": filename,
            "image_url": f"/images/{filename}",
            "warnings": [],
            "error": None,
        }
        try:
            evaluation = evaluate_question(question)
            expected = evaluation.array
            display = normalize_for_display(expected)
            image = image_from_array(display.array)
            image.save(GENERATED_DIR / filename)
            item.update(
                {
                    "original_shape": shape_list(expected),
                    "dtype": str(expected.dtype),
                    "render_shape": shape_list(display.array),
                    "render_width": int(image.width),
                    "render_height": int(image.height),
                    "render_mode": display.render_mode,
                    "practice_compatible": display.practice_compatible,
                    "warnings": evaluation.warnings + display.warnings,
                }
            )
        except Exception as exc:  # keep the card visible
            item.update(
                {
                    "image_file": None,
                    "image_url": None,
                    "original_shape": [],
                    "dtype": "",
                    "render_shape": [],
                    "render_width": None,
                    "render_height": None,
                    "render_mode": "error",
                    "practice_compatible": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        item["visual_description"] = visual_description_for(question, item)
        items.append(item)

    manifest = {
        "generated_at": utc_now(),
        "question_source": str(QUESTIONS_PATH.relative_to(REPO)),
        "delta_numbers_source": str(DELTA_NUMBERS_PATH.relative_to(REPO)),
        "count": len(items),
        "items": items,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(regenerate: bool = False) -> dict[str, Any]:
    if regenerate or not MANIFEST_PATH.exists():
        return render_manifest()
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"version": 1, "updated_at": None, "reviews": {}}
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "updated_at": None, "reviews": {}}
    state.setdefault("version", 1)
    state.setdefault("reviews", {})
    return state


def write_flags_exports(state: dict[str, Any], manifest: dict[str, Any]) -> None:
    by_id = {str(item["id"]): item for item in manifest["items"]}
    flagged: list[dict[str, Any]] = []
    for qid, review in sorted(state.get("reviews", {}).items(), key=lambda pair: int(pair[0])):
        if review.get("status") != "needs_check":
            continue
        item = by_id.get(str(qid), {})
        flagged.append(
            {
                "id": int(qid),
                "status": "needs_check",
                "note": review.get("note", ""),
                "reviewed_at": review.get("updated_at"),
                "topic": item.get("topic"),
                "subtopic": item.get("subtopic"),
                "question_text": item.get("question_text"),
                "visual_description": item.get("visual_description"),
                "answer_code": item.get("answer_code"),
                "expected_expr": item.get("expected_expr"),
                "source_path": item.get("source_path"),
                "original_shape": item.get("original_shape"),
                "render_size": [item.get("render_width"), item.get("render_height")],
                "warnings": item.get("warnings") or [],
                "error": item.get("error"),
                "image_file": item.get("image_file"),
            }
        )

    FLAGS_JSONL_PATH.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in flagged),
        encoding="utf-8",
    )
    md_lines = [
        "# Delta Drills Visual Malformed Flags",
        "",
        f"Updated: {state.get('updated_at') or utc_now()}",
        f"Flagged: {len(flagged)}",
        "",
    ]
    for row in flagged:
        size = row["render_size"]
        md_lines.extend(
            [
                f"## Question {row['id']} — {row.get('subtopic') or row.get('topic') or 'Unknown'}",
                "",
                f"- Status: needs_check",
                f"- Render size: {size[0]} x {size[1]}",
                f"- Shape: {row.get('original_shape')}",
                f"- Warning: {', '.join(row.get('warnings') or []) or 'none'}",
                f"- Error: {row.get('error') or 'none'}",
                f"- Note: {row.get('note') or ''}",
                f"- Expected visual: {row.get('visual_description') or ''}",
                f"- Question: {row.get('question_text') or ''}",
                f"- Answer: `{row.get('answer_code') or ''}`",
                "",
            ]
        )
    FLAGS_MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")


def save_state(state: dict[str, Any], manifest: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    write_flags_exports(state, manifest)


def counts_for(manifest: dict[str, Any], state: dict[str, Any]) -> dict[str, int]:
    reviews = state.get("reviews", {})
    total = len(manifest["items"])
    needs_check = sum(1 for review in reviews.values() if review.get("status") == "needs_check")
    ok = sum(1 for review in reviews.values() if review.get("status") == "ok")
    errors = sum(1 for item in manifest["items"] if item.get("error"))
    return {
        "total": total,
        "ok": ok,
        "needs_check": needs_check,
        "unreviewed": total - ok - needs_check,
        "render_errors": errors,
    }


APP: dict[str, Any] = {"manifest": None, "state": None}


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Delta Drills Visual Review</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d7dce2;
      --text: #1e252d;
      --muted: #65717f;
      --ok: #1d7a46;
      --warn: #b54708;
      --bad: #b42318;
      --focus: #0969da;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      border-bottom: 1px solid var(--line);
      background: rgba(247, 248, 250, 0.96);
      backdrop-filter: blur(10px);
    }
    .bar {
      max-width: 1500px;
      margin: 0 auto;
      padding: 14px 18px;
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto;
      gap: 12px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      font-weight: 720;
    }
    .stats {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      color: var(--muted);
      font-size: 13px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      padding: 5px 9px;
      white-space: nowrap;
    }
    .filters {
      max-width: 1500px;
      margin: 0 auto;
      padding: 0 18px 14px;
      display: grid;
      grid-template-columns: minmax(180px, 2fr) minmax(140px, 1fr) minmax(140px, 1fr) minmax(120px, 0.7fr) auto auto;
      gap: 10px;
      align-items: center;
    }
    input, select, button, textarea {
      font: inherit;
      letter-spacing: 0;
    }
    input[type="search"], select {
      width: 100%;
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      padding: 0 10px;
    }
    button, .link-button {
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      padding: 0 11px;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      white-space: nowrap;
    }
    button:hover, .link-button:hover { border-color: #aab4c0; }
    button.active-ok {
      border-color: rgba(29, 122, 70, 0.45);
      background: #e9f7ef;
      color: var(--ok);
      font-weight: 650;
    }
    button.active-bad {
      border-color: rgba(180, 35, 24, 0.45);
      background: #fff0ee;
      color: var(--bad);
      font-weight: 650;
    }
    main {
      max-width: 1500px;
      margin: 0 auto;
      padding: 16px 18px 28px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 14px;
      align-items: start;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-width: 0;
    }
    .card.needs-check { border-color: rgba(180, 35, 24, 0.55); }
    .card.ok { border-color: rgba(29, 122, 70, 0.45); }
    .image-wrap {
      height: 230px;
      background:
        linear-gradient(45deg, #eef1f4 25%, transparent 25%),
        linear-gradient(-45deg, #eef1f4 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #eef1f4 75%),
        linear-gradient(-45deg, transparent 75%, #eef1f4 75%);
      background-color: #fff;
      background-position: 0 0, 0 10px, 10px -10px, -10px 0;
      background-size: 20px 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-bottom: 1px solid var(--line);
      padding: 10px;
    }
    .image-wrap img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      image-rendering: auto;
      border: 1px solid rgba(0, 0, 0, 0.08);
      background: #fff;
    }
    .error-box {
      width: 100%;
      min-height: 130px;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 12px;
      color: var(--bad);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .body { padding: 11px; }
    .meta {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 7px;
      background: #fafbfc;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .chip.bad { color: var(--bad); border-color: rgba(180, 35, 24, 0.35); background: #fff6f5; }
    .chip.warn { color: var(--warn); border-color: rgba(181, 71, 8, 0.35); background: #fff8ed; }
    .question {
      min-height: 58px;
      max-height: 78px;
      overflow: auto;
      margin: 0 0 10px;
      font-size: 13px;
      line-height: 1.35;
      color: #2c3540;
    }
    .visual-description {
      margin: 0 0 10px;
      padding: 8px;
      border: 1px solid #dfe4ea;
      border-radius: 6px;
      background: #fafbfc;
      color: #394553;
      font-size: 12px;
      line-height: 1.35;
    }
    .visual-description strong {
      font-weight: 700;
      color: #1e252d;
    }
    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 8px;
    }
    textarea {
      width: 100%;
      min-height: 38px;
      max-height: 110px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      color: var(--text);
      background: #fff;
      font-size: 13px;
    }
    .details {
      margin-top: 8px;
      border-top: 1px solid var(--line);
      padding-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    details summary {
      cursor: pointer;
      color: var(--muted);
    }
    pre {
      margin: 8px 0 0;
      padding: 8px;
      border-radius: 6px;
      background: #f4f6f8;
      overflow: auto;
      max-height: 120px;
      font-size: 12px;
    }
    .empty {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 20px;
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 860px) {
      .bar { grid-template-columns: 1fr; }
      .stats { justify-content: flex-start; }
      .filters { grid-template-columns: 1fr 1fr; }
      .filters .wide { grid-column: 1 / -1; }
    }
    @media (max-width: 560px) {
      .filters { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
      .image-wrap { height: 210px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <h1>Delta Drills Visual Review</h1>
      <div id="stats" class="stats"></div>
    </div>
    <div class="filters">
      <input id="search" class="wide" type="search" placeholder="Search id, text, shape, warning">
      <select id="topic"></select>
      <select id="subtopic"></select>
      <select id="status">
        <option value="all">All statuses</option>
        <option value="unreviewed">Unreviewed</option>
        <option value="needs_check">Needs check</option>
        <option value="ok">OK</option>
        <option value="render_error">Render error</option>
        <option value="warning">Warning</option>
      </select>
      <a class="link-button" href="/api/flags.md" target="_blank" rel="noreferrer">Flags MD</a>
      <a class="link-button" href="/api/flags.jsonl" target="_blank" rel="noreferrer">Flags JSONL</a>
    </div>
  </header>
  <main>
    <div id="grid" class="grid"></div>
  </main>
  <script>
    const state = { items: [], reviews: {}, counts: {} };
    const el = (id) => document.getElementById(id);

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function reviewFor(id) {
      return state.reviews[String(id)] || { status: "unreviewed", note: "" };
    }

    function statusFor(item) {
      const review = reviewFor(item.id);
      return review.status || "unreviewed";
    }

    function setOptions(select, values, label) {
      const current = select.value;
      select.innerHTML = `<option value="all">${label}</option>` +
        values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("");
      if ([...select.options].some((option) => option.value === current)) select.value = current;
    }

    function updateStats() {
      const counts = state.counts || {};
      el("stats").innerHTML = [
        `${counts.total || 0} total`,
        `${counts.unreviewed || 0} unreviewed`,
        `${counts.needs_check || 0} needs check`,
        `${counts.ok || 0} OK`,
        `${counts.render_errors || 0} render errors`,
      ].map((text) => `<span class="stat">${escapeHtml(text)}</span>`).join("");
    }

    function itemMatches(item) {
      const query = el("search").value.trim().toLowerCase();
      const topic = el("topic").value;
      const subtopic = el("subtopic").value;
      const status = el("status").value;
      const reviewStatus = statusFor(item);

      if (topic !== "all" && item.topic !== topic) return false;
      if (subtopic !== "all" && item.subtopic !== subtopic) return false;
      if (status === "render_error" && !item.error) return false;
      if (status === "warning" && !(item.warnings || []).length) return false;
      if (["unreviewed", "needs_check", "ok"].includes(status) && reviewStatus !== status) return false;
      if (!query) return true;

      const haystack = [
        item.id,
        item.topic,
        item.subtopic,
        item.visual_description,
        item.question_text,
        item.answer_code,
        item.expected_expr,
        (item.original_shape || []).join("x"),
        (item.render_shape || []).join("x"),
        ...(item.warnings || []),
        item.error || "",
      ].join(" ").toLowerCase();
      return haystack.includes(query);
    }

    function chip(text, cls = "") {
      if (!text) return "";
      return `<span class="chip ${cls}">${escapeHtml(text)}</span>`;
    }

    function cardHtml(item) {
      const review = reviewFor(item.id);
      const reviewedClass = review.status === "needs_check" ? "needs-check" : review.status === "ok" ? "ok" : "";
      const shape = item.original_shape?.length ? `shape ${item.original_shape.join(" x ")}` : "shape unavailable";
      const render = item.render_width ? `render ${item.render_width} x ${item.render_height}` : "not rendered";
      const warnings = (item.warnings || []).map((warning) => chip(warning, "warn")).join("");
      const image = item.image_url
        ? `<img src="${escapeHtml(item.image_url)}" alt="question ${item.id} rendered target">`
        : `<div class="error-box">${escapeHtml(item.error || "Render unavailable")}</div>`;
      return `
        <article class="card ${reviewedClass}" data-id="${item.id}">
          <div class="image-wrap">${image}</div>
          <div class="body">
            <div class="meta">
              ${chip("#" + item.id)}
              ${chip(item.subtopic || item.topic)}
              ${chip(shape)}
              ${chip(render)}
              ${item.error ? chip("render error", "bad") : ""}
              ${item.practice_compatible === false ? chip("practice warning", "warn") : ""}
              ${warnings}
            </div>
            <div class="visual-description"><strong>Expected visual:</strong> ${escapeHtml(item.visual_description || "No visual description available.")}</div>
            <p class="question">${escapeHtml(item.question_text)}</p>
            <div class="actions">
              <button type="button" class="${review.status === "ok" ? "active-ok" : ""}" data-action="status" data-status="ok">OK</button>
              <button type="button" class="${review.status === "needs_check" ? "active-bad" : ""}" data-action="status" data-status="needs_check">Needs check</button>
            </div>
            <textarea data-action="note" placeholder="Note">${escapeHtml(review.note || "")}</textarea>
            <div class="details">
              <details>
                <summary>Code</summary>
                <pre>${escapeHtml(item.answer_code || item.expected_expr || "")}</pre>
              </details>
            </div>
          </div>
        </article>
      `;
    }

    function renderGrid() {
      const filtered = state.items.filter(itemMatches);
      el("grid").innerHTML = filtered.length
        ? filtered.map(cardHtml).join("")
        : `<div class="empty">No matching visual questions.</div>`;
    }

    async function saveReview(id, patch, options = {}) {
      const rerender = options.rerender !== false;
      const existing = reviewFor(id);
      const next = { ...existing, ...patch };
      state.reviews[String(id)] = next;
      if (rerender) renderGrid();
      const response = await fetch("/api/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, ...next }),
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      state.reviews = payload.reviews;
      state.counts = payload.counts;
      updateStats();
      if (rerender) renderGrid();
    }

    let noteTimer = null;
    function queueNoteSave(textarea) {
      const card = textarea.closest(".card");
      const id = Number(card.dataset.id);
      clearTimeout(noteTimer);
      noteTimer = setTimeout(() => {
        saveReview(id, { note: textarea.value }, { rerender: false }).catch((err) => alert(err.message));
      }, 500);
    }

    function replaceTextareaSelection(textarea, nextValue, nextStart, nextEnd) {
      textarea.value = nextValue;
      textarea.selectionStart = nextStart;
      textarea.selectionEnd = nextEnd;
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    }

    function handleNoteTab(textarea, event) {
      const value = textarea.value;
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const lineStart = value.lastIndexOf("\n", start - 1) + 1;
      const selected = value.slice(start, end);

      event.preventDefault();
      if (event.shiftKey) {
        if (start !== end && selected.includes("\n")) {
          const blockEnd = end;
          const block = value.slice(lineStart, blockEnd);
          const lines = block.split("\n");
          let removedBeforeStart = 0;
          let removedTotal = 0;
          const outdented = lines.map((line, index) => {
            if (!line.startsWith("\t")) return line;
            removedTotal += 1;
            if (lineStart + lines.slice(0, index).join("\n").length + (index ? 1 : 0) < start) {
              removedBeforeStart += 1;
            }
            return line.slice(1);
          }).join("\n");
          replaceTextareaSelection(
            textarea,
            value.slice(0, lineStart) + outdented + value.slice(blockEnd),
            Math.max(lineStart, start - removedBeforeStart),
            Math.max(lineStart, end - removedTotal)
          );
          return;
        }
        if (value.slice(lineStart, lineStart + 1) === "\t") {
          replaceTextareaSelection(
            textarea,
            value.slice(0, lineStart) + value.slice(lineStart + 1),
            Math.max(lineStart, start - 1),
            Math.max(lineStart, end - 1)
          );
        }
        return;
      }

      if (start !== end && selected.includes("\n")) {
        const blockEnd = end;
        const block = value.slice(lineStart, blockEnd);
        const indented = block.split("\n").map((line) => "\t" + line).join("\n");
        const lineCount = block.split("\n").length;
        replaceTextareaSelection(
          textarea,
          value.slice(0, lineStart) + indented + value.slice(blockEnd),
          start + 1,
          end + lineCount
        );
        return;
      }

      replaceTextareaSelection(
        textarea,
        value.slice(0, start) + "\t" + value.slice(end),
        start + 1,
        start + 1
      );
    }

    document.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-action='status']");
      if (!button) return;
      const card = button.closest(".card");
      const id = Number(card.dataset.id);
      const status = button.dataset.status;
      saveReview(id, { status }).catch((err) => alert(err.message));
    });

    document.addEventListener("keydown", (event) => {
      if (!event.target.matches("textarea[data-action='note']")) return;
      if (event.key !== "Tab") return;
      handleNoteTab(event.target, event);
    });

    document.addEventListener("input", (event) => {
      if (event.target.matches("#search, #topic, #subtopic, #status")) {
        renderGrid();
        return;
      }
      if (!event.target.matches("textarea[data-action='note']")) return;
      queueNoteSave(event.target);
    });

    async function boot() {
      const response = await fetch("/api/items", { cache: "no-store" });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      state.items = payload.items;
      state.reviews = payload.reviews;
      state.counts = payload.counts;
      const topics = [...new Set(state.items.map((item) => item.topic).filter(Boolean))].sort();
      const subtopics = [...new Set(state.items.map((item) => item.subtopic).filter(Boolean))].sort();
      setOptions(el("topic"), topics, "All topics");
      setOptions(el("subtopic"), subtopics, "All subtopics");
      updateStats();
      renderGrid();
    }

    boot().catch((err) => {
      document.querySelector("main").innerHTML = `<div class="empty">${escapeHtml(err.message)}</div>`;
    });
  </script>
</body>
</html>
"""


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "DeltaDrillsVisualReview/1.0"

    def send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_text(self, data: str, content_type: str = "text/plain; charset=utf-8", status: int = 200) -> None:
        payload = data.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.send_text(INDEX_HTML, "text/html; charset=utf-8")
            return
        if path == "/api/items":
            manifest = APP["manifest"]
            state = APP["state"]
            self.send_json(
                {
                    **{k: manifest[k] for k in ("generated_at", "question_source", "delta_numbers_source", "count")},
                    "items": manifest["items"],
                    "reviews": state.get("reviews", {}),
                    "counts": counts_for(manifest, state),
                    "state_path": str(STATE_PATH),
                    "flags_jsonl_path": str(FLAGS_JSONL_PATH),
                    "flags_md_path": str(FLAGS_MD_PATH),
                }
            )
            return
        if path == "/api/flags.jsonl":
            self.send_text(FLAGS_JSONL_PATH.read_text(encoding="utf-8") if FLAGS_JSONL_PATH.exists() else "")
            return
        if path == "/api/flags.md":
            self.send_text(
                FLAGS_MD_PATH.read_text(encoding="utf-8") if FLAGS_MD_PATH.exists() else "",
                "text/markdown; charset=utf-8",
            )
            return
        if path.startswith("/images/"):
            name = Path(unquote(path.removeprefix("/images/"))).name
            image_path = GENERATED_DIR / name
            if not image_path.exists() or image_path.suffix.lower() != ".png":
                self.send_error(404)
                return
            payload = image_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(image_path))[0] or "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/review":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            qid = str(int(data["id"]))
            status = data.get("status") or "unreviewed"
            if status not in {"unreviewed", "ok", "needs_check"}:
                raise ValueError(f"invalid status {status!r}")
            note = str(data.get("note") or "")
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)
            return

        state = APP["state"]
        if status == "unreviewed" and not note.strip():
            state.setdefault("reviews", {}).pop(qid, None)
        else:
            state.setdefault("reviews", {})[qid] = {
                "status": status,
                "note": note,
                "updated_at": utc_now(),
            }
        save_state(state, APP["manifest"])
        self.send_json({"reviews": state.get("reviews", {}), "counts": counts_for(APP["manifest"], state)})

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), format % args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Delta Drills visual review app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--regen", action="store_true", help="Regenerate PNGs before serving.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    APP["manifest"] = load_manifest(regenerate=args.regen)
    APP["state"] = load_state()
    save_state(APP["state"], APP["manifest"])

    address = (args.host, args.port)
    with ThreadingHTTPServer(address, ReviewHandler) as httpd:
        print(f"Visual review: http://{args.host}:{args.port}")
        print(f"Questions: {APP['manifest']['count']}")
        print(f"Flags: {FLAGS_MD_PATH}")
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
