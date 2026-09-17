#!/usr/bin/env python3
"""Emit one trimmed authoring-input file per atom for the ERE author workflow.

Each opus author agent Reads its own ere/agent_input/<atomId>.json (atom def +
existing exercises as style/difficulty reference) and returns 3 worked + 3 faded
specs. Splitting per-atom keeps each agent's Read small.

Usage:
  python3 make_agent_input.py            # all atoms
  python3 make_agent_input.py --limit 3  # pilot: first 3 atoms
  python3 make_agent_input.py a1 a2      # specific atomIds
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BUNDLES = HERE / "ere_atom_bundles.json"
OUT_DIR = HERE / "agent_input"


def trim(bundle: dict) -> dict:
    exs = []
    for e in bundle["exercises"]:
        exs.append({
            "heading": e["heading"],
            "prompt_md": e["prompt_md"][:2500],
            "solution_code": e["solution_code"][:2500],
            # one representative refresher gives house teaching voice
        })
    return {
        "atomId": bundle["atomId"],
        "label": bundle["label"],
        "definition": bundle["definition"],
        "domain": bundle["domain"],
        "subtopic": bundle["subtopic"],
        "exampleRefresher": (bundle["exercises"][0]["refresher_md"][:1800]
                             if bundle["exercises"] else ""),
        "existingExercises": exs,
    }


def main() -> None:
    args = sys.argv[1:]
    limit = None
    ids = []
    i = 0
    while i < len(args):
        if args[i] == "--limit":
            limit = int(args[i + 1]); i += 2
        else:
            ids.append(args[i]); i += 1

    bundles = json.loads(BUNDLES.read_text())
    if ids:
        bundles = [b for b in bundles if b["atomId"] in ids]
    if limit:
        bundles = bundles[:limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for b in bundles:
        (OUT_DIR / f"{b['atomId']}.json").write_text(json.dumps(trim(b), indent=1))
    print(f"agent inputs: {len(bundles)} -> {OUT_DIR.relative_to(REPO)}/")
    print("atomIds:", json.dumps([b["atomId"] for b in bundles]))


if __name__ == "__main__":
    main()
