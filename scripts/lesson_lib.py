"""Shared parsing for the first-encounter lesson pipeline.

KP markdown format (see Local_Deployed_Shared/lessons/AUTHORING.md):
  - flat YAML-subset frontmatter between --- lines: scalars and inline [a, b] lists
  - segment sections: ## Concept, ## Watch out, ## Worked example,
    ## Faded practice
  - KP-level tail: ## Guided practice, ## Independent practice,
    ## Misconceptions (legacy single-segment fallback)
  - Faded/Guided practice contain ### q<id> subsections; faded ones carry
    ```python starter``` and ```python solution``` fences.
Used by scripts/compile_lessons.py and scripts/validate_lessons.py.
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LESSONS_DIR = REPO / "Local_Deployed_Shared" / "lessons"
REGISTRY_PATH = LESSONS_DIR / "kc_registry.json"
BANK_PATH = REPO / "Local_Deployed_Shared" / "questions_structured.json"

SECTIONS = [
    "Concept",
    "Watch out",
    "Worked example",
    "Faded practice",
    "Guided practice",
    "Independent practice",
    "Misconceptions",
]


def load_registry():
    return json.loads(REGISTRY_PATH.read_text())


def load_bank():
    bank = json.loads(BANK_PATH.read_text())
    return {q["id"]: q for q in bank}


def _parse_value(raw):
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(part) for part in inner.split(",")]
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    return raw.strip("'\"")


def parse_frontmatter(text, path):
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing frontmatter")
    end = text.index("\n---", 3)
    meta = {}
    for line in text[3:end].strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, raw = line.partition(":")
        meta[key.strip()] = _parse_value(raw)
    body = text[end + 4:]
    return meta, body


def split_sections(body, path):
    """Return ordered [(section_title, subtitle, content)] for ## headings.

    Concept / Watch out / Worked example / Faded practice may repeat — each
    `## Concept` starts a new single-concept segment (see AUTHORING.md). A
    section heading may carry a segment subtitle: `## Concept: np.diag`.
    """
    parts = re.split(r"^## (.+)$", body, flags=re.M)
    ordered = []
    for title, content in zip(parts[1::2], parts[2::2]):
        title = title.strip()
        subtitle = ""
        if ":" in title:
            title, _, subtitle = title.partition(":")
            title = title.strip()
            subtitle = subtitle.strip()
        if title not in SECTIONS:
            raise ValueError(f"{path}: unknown section '## {title}'")
        ordered.append((title, subtitle, content.strip()))
    return ordered


def build_segments(ordered, path):
    """Group ordered sections into single-concept segments plus KP-level tail.

    Each `## Concept` opens a segment; its `## Watch out`,
    `## Worked example`, and `## Faded practice` belong to it.
    Guided/Independent/Misconceptions are KP-level tail sections.
    """
    segments = []
    tail = {}
    current = None
    for title, subtitle, content in ordered:
        if title == "Concept":
            current = {
                "title": subtitle,
                "concept": content,
                "watch_out": "",
                "worked": "",
                "faded": "",
            }
            segments.append(current)
        elif title in ("Watch out", "Worked example", "Faded practice"):
            if current is None:
                raise ValueError(f"{path}: '## {title}' before any '## Concept'")
            key = {
                "Watch out": "watch_out",
                "Worked example": "worked",
                "Faded practice": "faded",
            }[title]
            if current[key]:
                raise ValueError(f"{path}: duplicate '## {title}' in one segment")
            current[key] = content
        else:
            if title in tail:
                raise ValueError(f"{path}: duplicate '## {title}'")
            tail[title] = content
    if not segments:
        raise ValueError(f"{path}: no '## Concept' section")
    return segments, tail


def split_items(section_text):
    """Return {qid: content} for ### q<id> subsections."""
    parts = re.split(r"^### q(\d+)\s*$", section_text, flags=re.M)
    return {int(qid): content.strip() for qid, content in zip(parts[1::2], parts[2::2])}


def code_fences(text, info=None):
    """Return code blocks; info filters the fence info string exactly."""
    blocks = re.findall(r"```([^\n]*)\n(.*?)```", text, flags=re.S)
    out = []
    for fence_info, code in blocks:
        fence_info = fence_info.strip()
        if info is None or fence_info == info:
            out.append(code)
    return out


def parse_kp(path):
    meta, body = parse_frontmatter(path.read_text(), path)
    ordered = split_sections(body, path)
    segments, tail = build_segments(ordered, path)
    # Aggregate view for consumers that don't care about segment boundaries
    # (qmatrix builder, coverage checks). Content order is preserved per kind.
    sections = dict(tail)
    sections["Concept"] = "\n\n".join(s["concept"] for s in segments if s["concept"])
    sections["Watch out"] = "\n\n".join(s["watch_out"] for s in segments if s["watch_out"])
    sections["Worked example"] = "\n\n".join(s["worked"] for s in segments if s["worked"])
    sections["Faded practice"] = "\n\n".join(s["faded"] for s in segments if s["faded"])
    kp = {
        "segments": segments,
        "file": str(path.relative_to(REPO)),
        "kc": meta.get("kc"),
        "title": meta.get("title"),
        "supporting": meta.get("supporting", []),
        "new_syntax": meta.get("new_syntax", []),
        "previews": meta.get("previews", []),
        "concepts": meta.get("concepts", []),
        "faded": meta.get("faded", []),
        "guided": meta.get("guided", []),
        "independent": meta.get("independent", []),
        "sections": sections,
    }
    return kp


def all_kp_paths():
    return sorted(LESSONS_DIR.glob("*/kp-*.md"))
