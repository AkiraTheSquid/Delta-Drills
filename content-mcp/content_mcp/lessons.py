"""Read and write the KP markdown pages.

A "KP" is one knowledge point: one markdown file under
`Local_Deployed_Shared/lessons/<course>/kp-<slug>.md`, addressed everywhere by
its KC id (`numpy.random-seeding`), never by path. Parsing goes through the
repo's own `scripts/lesson_lib.py` so this package and the validator can never
disagree about what a page says.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from . import paths

_LIB = None


def _lib():
    """Import the repo's canonical KP parser, lazily and once."""
    global _LIB
    if _LIB is None:
        if str(paths.SCRIPTS) not in sys.path:
            sys.path.insert(0, str(paths.SCRIPTS))
        import lesson_lib  # noqa: PLC0415 — deliberately deferred

        _LIB = lesson_lib
    return _LIB


def _kc_of(path: Path) -> str | None:
    """Cheap frontmatter read — used to build the kc -> path index."""
    try:
        head = path.read_text().split("---", 2)
    except OSError:
        return None
    if len(head) < 3:
        return None
    match = re.search(r"(?m)^kc:\s*(\S+)\s*$", head[1])
    return match.group(1) if match else None


def index() -> dict[str, Path]:
    """kc id -> markdown path, for every KP on disk."""
    found: dict[str, Path] = {}
    for path in paths.kp_files():
        kc = _kc_of(path)
        if kc:
            found[kc] = path
    return found


def resolve(kc: str) -> Path:
    mapping = index()
    if kc in mapping:
        return mapping[kc]
    near = [k for k in mapping if kc in k or k.endswith(kc)]
    hint = f" Did you mean: {', '.join(sorted(near)[:5])}?" if near else ""
    raise KeyError(f"No KP page for KC '{kc}'.{hint}")


def summarize(path: Path) -> dict:
    kp = _lib().parse_kp(path)
    return {
        "kc": kp["kc"],
        "title": kp["title"],
        "course": path.parent.name,
        "file": kp["file"],
        "segments": len(kp["segments"]),
        "new_syntax": kp["new_syntax"],
        "supporting": kp["supporting"],
        "rungs": {
            "faded": kp["faded"],
            "guided": kp["guided"],
            "solo": kp["independent"],
            "integrated": kp["integrated"],
        },
        "drill_count": len(
            set(kp["faded"]) | set(kp["guided"]) | set(kp["independent"]) | set(kp["integrated"])
        ),
    }


def list_kps(course: str | None = None) -> list[dict]:
    out = []
    for path in paths.kp_files():
        if course and path.parent.name != course:
            continue
        try:
            out.append(summarize(path))
        except Exception as err:  # a page mid-edit must not blind the listing
            out.append({"file": str(path.relative_to(paths.REPO)), "parse_error": str(err)})
    return out


def read_kp(kc: str, include_body: bool = True) -> dict:
    path = resolve(kc)
    record = summarize(path)
    if include_body:
        record["markdown"] = path.read_text()
    return record


def outline(kc: str) -> dict:
    path = resolve(kc)
    kp = _lib().parse_kp(path)
    return {
        "kc": kp["kc"],
        "title": kp["title"],
        "file": kp["file"],
        "headings": re.findall(r"(?m)^(#{2,3} .+)$", path.read_text()),
        "segments": [
            {
                "index": i,
                "has_concept": bool(seg.get("concept")),
                "has_worked": bool(seg.get("worked")),
                "faded_ids": seg.get("faded_ids", []),
            }
            for i, seg in enumerate(kp["segments"])
        ],
    }


def write_kp(kc: str, markdown: str) -> dict:
    """Replace a whole page. Frontmatter must still name the same KC."""
    path = resolve(kc)
    declared = re.search(r"(?m)^kc:\s*(\S+)\s*$", markdown.split("---", 2)[1] if markdown.count("---") >= 2 else "")
    if not declared:
        raise ValueError("New markdown has no `kc:` frontmatter key — refusing to write.")
    if declared.group(1) != kc:
        raise ValueError(
            f"Frontmatter says kc: {declared.group(1)} but you addressed {kc}. "
            "Rename through graph_update_kc, not by editing the page in place."
        )
    before = path.read_text()
    path.write_text(markdown)
    return {
        "kc": kc,
        "file": str(path.relative_to(paths.REPO)),
        "bytes_before": len(before),
        "bytes_after": len(markdown),
    }


def edit_kp(kc: str, old: str, new: str, expect_count: int = 1) -> dict:
    """Exact-string replacement, the Edit-tool contract: refuse on surprise."""
    path = resolve(kc)
    text = path.read_text()
    found = text.count(old)
    if found == 0:
        raise ValueError("`old` string not found in the page.")
    if expect_count and found != expect_count:
        raise ValueError(
            f"`old` appears {found} times, expected {expect_count}. "
            "Widen the string or pass expect_count."
        )
    path.write_text(text.replace(old, new))
    return {"kc": kc, "file": str(path.relative_to(paths.REPO)), "replacements": found}


def create_kp(course: str, kc: str, markdown: str) -> dict:
    """Write a brand-new page. The KC must already exist in the registry."""
    if course not in paths.courses():
        raise ValueError(f"Unknown course '{course}'. Known: {', '.join(paths.courses())}")
    if kc in index():
        raise ValueError(f"KP for {kc} already exists — use lesson_write.")

    # The registry is what makes a concept real: a page for an unregistered KC
    # is an orphan nothing serves, and a page whose frontmatter names a
    # different KC is indexed under that other name — both look like success.
    import json as _json

    registry = _json.loads(paths.KC_REGISTRY.read_text())
    if kc not in {node["id"] for node in registry["kcs"]}:
        raise ValueError(
            f"'{kc}' is not in kc_registry.json. Add the concept first with "
            "graph_add_kc, then write its page."
        )
    declared = re.search(
        r"(?m)^kc:\s*(\S+)\s*$",
        markdown.split("---", 2)[1] if markdown.count("---") >= 2 else "",
    )
    if not declared:
        raise ValueError("New markdown has no `kc:` frontmatter key — refusing to write.")
    if declared.group(1) != kc:
        raise ValueError(
            f"Frontmatter says kc: {declared.group(1)} but you asked to create {kc}."
        )
    slug = kc.split(".", 1)[-1]
    path = paths.LESSONS / course / f"kp-{slug}.md"
    if path.exists():
        raise ValueError(f"{path} already exists.")
    path.write_text(markdown)
    return {"kc": kc, "file": str(path.relative_to(paths.REPO)), "bytes": len(markdown)}


def authoring_guide() -> str:
    """The format contract every page must satisfy. Read it before authoring."""
    return paths.AUTHORING.read_text() if paths.AUTHORING.exists() else ""
