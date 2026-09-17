#!/usr/bin/env python3
"""Regenerate Local_Deployed_Shared/prereq_subtopics.json + prereq_subtopics.js
from canonical notebook metadata under arena-procedural-drills/prereqs_*.

Usage:
    python3 arena-procedural-drills/scripts/gen_prereq_subtopics.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRILL_ROOT = REPO / "arena-procedural-drills"
JSON_OUT = REPO / "Local_Deployed_Shared" / "prereq_subtopics.json"
JS_OUT = REPO / "Local_Deployed_Shared" / "concept-graph" / "prereq_subtopics.js"

BANK_VALID = {
    "CNN: Conv2d module mechanics", "CNN: Output shape & arithmetic",
    "CNN: Pooling, Flatten, BatchNorm", "CNN: nn.Module & training loop",
    "Einops: Deep Learning", "Einops: Rearrange", "Einops: Reduce", "Einops: Repeat",
    "Einsum: Applied patterns and advanced", "Einsum: Core array literacy",
    "Einsum: Indexing and selection", "Einsum: Vectorization and broadcasting",
}


def main() -> int:
    atom_to_sub: dict[str, str] = {}
    for nb_path in sorted(DRILL_ROOT.glob("prereqs_*/*/*.ipynb")):
        nb = json.loads(nb_path.read_text())
        dd = nb.get("metadata", {}).get("delta_drills", {})
        atom = dd.get("atom_id")
        sub = dd.get("subtopic")
        if not atom or not sub:
            continue
        if atom in atom_to_sub and atom_to_sub[atom] != sub:
            print(f"WARN: {atom} has inconsistent subtopic: "
                  f"{atom_to_sub[atom]!r} vs {sub!r}")
            continue
        atom_to_sub[atom] = sub

    sub_to_atoms: dict[str, list[str]] = {}
    for a, s in atom_to_sub.items():
        sub_to_atoms.setdefault(s, []).append(a)

    prereq_keys = sorted(k for k in sub_to_atoms if k not in BANK_VALID)

    registry = {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "source": "arena-procedural-drills/ notebook metadata (canonical walk)",
        "description": (
            "Prereq subtopic keys introduced by Colab procedural drills. "
            "These keys are NOT in the regular question bank (zero flashcards) "
            "— drills are the only practice surface. Backend questions.py + "
            "frontend atom_readiness.js both consume this file to know about them."
        ),
        "atom_to_subtopic": dict(sorted(atom_to_sub.items())),
        "subtopic_to_atoms": {k: sorted(v) for k, v in sorted(sub_to_atoms.items())},
        "prereq_subtopic_keys": prereq_keys,
    }

    JSON_OUT.write_text(json.dumps(registry, indent=2))
    print(f"wrote {JSON_OUT.relative_to(REPO)} "
          f"({len(atom_to_sub)} atoms, {len(sub_to_atoms)} subs, {len(prereq_keys)} prereq)")

    js = (
        "// Auto-generated bridge — DO NOT EDIT BY HAND.\n"
        "// Regenerate via: arena-procedural-drills/scripts/gen_prereq_subtopics.py\n"
        "// Source of truth: arena-procedural-drills/ notebook metadata (walked at build time).\n"
        "//\n"
        "// Lists the prereq subtopic keys introduced by Colab procedural drills.\n"
        "// These keys have NO flashcards in the regular bank — the drill beacon is\n"
        "// the only way to bump their EWMA. atom_readiness.js consumes this to map\n"
        "// drill-atom IDs to their direct subtopic for Case 0 resolution.\n"
        "\n"
        f"window.PREREQ_SUBTOPICS = {json.dumps(registry, indent=2)};\n"
    )
    JS_OUT.write_text(js)
    print(f"wrote {JS_OUT.relative_to(REPO)} ({len(js)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
