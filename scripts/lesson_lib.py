"""Shared parsing for the first-encounter lesson pipeline.

KP markdown format (see Local_Deployed_Shared/lessons/AUTHORING.md):
  - flat YAML-subset frontmatter between --- lines: scalars and inline [a, b] lists
  - sections: ## Concept, ## Worked example, ## Faded practice, ## Guided practice,
    ## Independent practice, ## Misconceptions
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
    """Return {section_title: content} for ## headings."""
    parts = re.split(r"^## (.+)$", body, flags=re.M)
    sections = {}
    for title, content in zip(parts[1::2], parts[2::2]):
        title = title.strip()
        if title not in SECTIONS:
            raise ValueError(f"{path}: unknown section '## {title}'")
        sections[title] = content.strip()
    return sections


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
    sections = split_sections(body, path)
    kp = {
        "file": str(path.relative_to(REPO)),
        "kc": meta.get("kc"),
        "title": meta.get("title"),
        "supporting": meta.get("supporting", []),
        "new_syntax": meta.get("new_syntax", []),
        "faded": meta.get("faded", []),
        "guided": meta.get("guided", []),
        "independent": meta.get("independent", []),
        "sections": sections,
    }
    return kp


def all_kp_paths():
    return sorted(LESSONS_DIR.glob("*/kp-*.md"))
