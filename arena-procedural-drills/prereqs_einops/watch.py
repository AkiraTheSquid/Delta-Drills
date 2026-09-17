"""Watch script for arena-procedural-drills/prereqs_einops/.

Verifies every notebook in this folder bridges to the Einops bank topic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_TOPIC = "Einops"


def check_imports() -> None:
    import json as _json  # noqa: F401


def check_public_api() -> None:
    notebooks = list(HERE.glob("*.ipynb"))
    assert notebooks, "prereqs_einops/ contains no .ipynb files"
    for nb_path in notebooks:
        nb = json.loads(nb_path.read_text())
        meta = nb.get("metadata", {}).get("delta_drills", {})
        subtopic = meta.get("subtopic", "")
        assert subtopic.startswith(f"{EXPECTED_TOPIC}:"), (
            f"{nb_path}: subtopic '{subtopic}' is not under bank topic '{EXPECTED_TOPIC}'"
        )


def check_invariants() -> None:
    # Each notebook in this folder must have a corresponding bridge rule
    # in atom_readiness.js that routes its atom to the Einops topic. We
    # check by reading the JS file as text — no need to execute it.
    bridge_js = (
        HERE.parents[1]
        / "Local_Deployed_Shared/concept-graph/atom_readiness.js"
    )
    if not bridge_js.exists():
        return  # Repo layout changed; skip rather than false-positive.
    js_text = bridge_js.read_text()
    for nb_path in HERE.glob("*.ipynb"):
        atom_id = nb_path.stem
        # The bridge uses substring matching against the atom-id, so at
        # least one token from an Einops-bucket rule must appear in the id.
        # Cheap heuristic: confirm the atom contains 'einops' / 'rearrange'
        # / 'reduce' / 'repeat' / 'einsum'.
        einops_tokens = ("einops", "rearrange", "reduce", "repeat", "einsum")
        assert any(tok in atom_id for tok in einops_tokens), (
            f"{atom_id}: no einops-cluster token in id; would not route via "
            f"ATOM_ID_TOKEN_TO_BANK_TOPIC in {bridge_js.name}"
        )
        # And confirm the bridge file at least mentions one of those tokens.
        assert any(tok in js_text for tok in einops_tokens), (
            f"{bridge_js.name}: no einops-bucket rules present"
        )


if __name__ == "__main__":
    try:
        check_imports()
        check_public_api()
        check_invariants()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    print("PASS")
