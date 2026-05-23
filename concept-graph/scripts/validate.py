"""Validate that every atom id referenced in exercises/ exists in vocab/atoms.json.

Run from repo root:
    python concept-graph/scripts/validate.py
"""

import json
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    vocab_path = repo_root / "vocab" / "atoms.json"
    exercises_dir = repo_root / "exercises"

    with vocab_path.open() as f:
        vocab = json.load(f)
    known_atom_ids = {a["id"] for a in vocab["atoms"]}

    exercise_files = sorted(exercises_dir.glob("*.json"))
    if not exercise_files:
        print(f"no exercise files in {exercises_dir}")
        return 0

    errors = 0
    proposed_atoms: dict[str, list[str]] = {}

    for ex_path in exercise_files:
        with ex_path.open() as f:
            ex = json.load(f)

        for atom_ref in ex.get("atoms", []):
            atom_id = atom_ref["id"]
            if atom_id not in known_atom_ids:
                proposed_atoms.setdefault(atom_id, []).append(ex_path.name)

        for atom_ref in ex.get("atoms_in_seed_but_not_actually_present", []):
            if atom_ref["id"] not in known_atom_ids:
                print(f"[ERROR] {ex_path.name}: removed atom '{atom_ref['id']}' not in vocab")
                errors += 1

    print(f"vocab: {len(known_atom_ids)} atoms")
    print(f"exercises: {len(exercise_files)} files")
    print()

    if proposed_atoms:
        print("PROPOSED NEW ATOMS (referenced in exercises, not yet in vocab):")
        for atom_id, sources in sorted(proposed_atoms.items()):
            print(f"  {atom_id}  <- {', '.join(sources)}")
        print()
        print("Either add these to vocab/atoms.json or rename them in the exercise file.")
        return 1

    if errors:
        return 1

    print("OK: every atom reference resolves to a known vocab entry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
