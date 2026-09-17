#!/usr/bin/env python3
"""Re-dialect the May-2026 solution authoring layer onto the torch bank.

WHY THIS EXISTS

`authored/*.jsonl` was written in May 2026 by 24 agents against a NumPy bank.
The July conversion rewrote every question into `import torch as t` but never
rebuilt the solution notebooks, so `Show Answer` on a torch question opened a
notebook that said `import numpy as np` — 397 of 455 fully, the other 58 in
their prose or their `%pip` line. This produces a torch authoring layer that
`build_solution_colabs.py` consumes in place of the stale one.

WHAT IS TRUSTED FROM WHERE

  solution_code  ALWAYS from the live bank's `answer_code`. The bank is the
                 graded contract; the May authoring is not. Never mechanically
                 translated — a translated answer that drifts from the bank is
                 a notebook that disagrees with the grader.
  question_text  from the live bank (already torch).
  explanation    the authored prose, mechanically re-dialected — but ONLY when
                 the bank's answer still matches the answer that prose was
                 written about. See DRIFT below.
  hint           the authored hint, mechanically re-dialected.

DRIFT IS THE WHOLE POINT

The July pass did not merely rename symbols. ~30 drills used a numpy function
torch cannot spell (`ogrid`, `nditer`, `apply_along_axis`, `argpartition`,
`intersect1d`, ufunc `out=`/`where=`) and were HAND-translated to a different
approach; a further handful were retired outright. For those, the authored
"Why this works" describes an algorithm the bank no longer poses. Renaming its
symbols yields fluent, confident, WRONG text — strictly worse than no text,
because a learner cannot tell it is wrong.

So every id is classified by comparing the authored solution against the live
`answer_code`, normalised for dialect. Below the similarity floor the
explanation is WITHHELD rather than guessed, and the id is reported for hand
authoring. Anything still naming a numpy symbol after conversion is withheld
too, whatever its similarity — that residue check is the backstop, since a
prose sentence can drift without the code drifting.

    .../python scripts/solution_build/retorch_authored.py
"""
from __future__ import annotations

import collections
import difflib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "This-Directory-Only" / "scripts"))

from torchify_np_prose import convert  # noqa: E402  (np.X -> t.X, same-name only)

QUESTIONS_JSON = HERE / "dd_questions.json"
AUTHORED_DIR = HERE / "authored"
OUT_LAYER = HERE / "authored_torch" / "layer.jsonl"
OUT_REPORT = HERE / "retorch_report.json"

# Below this normalised-similarity ratio the authored prose is describing a
# different algorithm. 0.90 chosen from the observed distribution: the corpus
# splits 274 exact / 121 in [0.90, 1.0) / 52 below, with a clear gap — the
# hand-translated drills land at 0.42-0.88.
DRIFT_FLOOR = 0.90

# Any of these surviving conversion means the sentence is still talking numpy.
RESIDUE = re.compile(r"\bnp\.|\bnumpy\b|\bndarray\b", re.IGNORECASE)

# Similarity is a TEXT measure, not a behaviour one. A one-character edit that
# scores ~0.99 can still invert what the answer does — `dim=0` to `dim=1`,
# `keepdim` flipping, a bound moving, `>` becoming `>=`, `descending` toggling.
# The prose explains exactly those choices, so where the two answers disagree
# on one of these tokens the explanation is withheld no matter how high the
# ratio. Numeric literals count: an axis or a shape is usually the whole point.
SEMANTIC_TOKEN = re.compile(
    r"\b(?:dim|axis|keepdim|correction|ddof|descending|largest|sorted|"
    r"unbiased|rounding_mode|start|end|step|dtype|device|p|eps)\s*=\s*[^,)]+"
    r"|[<>]=?|!=|==|\b\d+\b"
)


def semantic_tokens(code: str) -> collections.Counter:
    """Behaviour-bearing tokens, with the KNOWN-EQUIVALENT spellings folded.

    numpy's `axis=` is torch's `dim=` and numpy's `dtype=bool` is torch's
    `dtype=t.bool` — those are the rename the conversion performed, not a
    change of meaning, and counting them as drift would discard good prose.
    Numeric literals are deliberately NOT folded: a changed example value
    usually means the prompt's worked example changed, which the explanation
    quotes.
    """
    code = re.sub(r"\baxis\s*=", "dim=", code or "")
    code = re.sub(r"(dtype\s*=\s*)(?:t|torch|np|numpy)\.", r"\1", code)
    return collections.Counter(SEMANTIC_TOKEN.findall(code))

TORCH_IMPORT = re.compile(r"^\s*(?:import\s+torch\b|from\s+torch[\s.])", re.M)


def is_torch(*sources: str) -> bool:
    """Dialect DERIVED from the code, never read from `primary_library`.

    The backend infers `primary_library` before the torch-dialect overrides are
    layered on, so it still reports "numpy" for 437 of the 499 rows whose code
    is torch. `lessons.is_torch_dialect` derives for the same reason.
    """
    return bool(TORCH_IMPORT.search("\n".join(s or "" for s in sources)))


def redialect(text: str) -> str:
    """Rename numpy prose to torch prose. Conservative by construction.

    `convert()` only touches symbols torch spells identically. The word-level
    passes below run in a fixed order: the article fix precedes the bare-noun
    fix so "an array" becomes "a tensor" and not "an tensor", and the library
    rename runs LAST so "a numpy array" resolves to "a PyTorch tensor" rather
    than stalling at "a PyTorch array".
    """
    text = convert(text)
    text = re.sub(r"\ban (array|ndarray)\b", "a tensor", text)
    text = re.sub(r"\bAn (array|ndarray)\b", "A tensor", text)
    text = re.sub(r"\bndarrays\b", "tensors", text)
    text = re.sub(r"\bndarray\b", "tensor", text)
    text = re.sub(r"\barrays\b", "tensors", text)
    text = re.sub(r"\barray\b", "tensor", text)
    text = re.sub(r"\bArrays\b", "Tensors", text)
    text = re.sub(r"\bArray\b", "Tensor", text)
    text = re.sub(r"\bNumPy\b", "PyTorch", text)
    text = re.sub(r"\bnumpy\b", "PyTorch", text)
    text = re.sub(r"\bNumpy\b", "PyTorch", text)
    return text


def normalise_for_compare(code: str) -> str:
    """Collapse a solution to its shape, so dialect alone is not a difference."""
    code = re.sub(r"#.*", "", code or "")
    code = re.sub(r"\bimport\s+(?:numpy|torch)\s+as\s+\w+", "IMP", code)
    code = re.sub(r"\b(?:np|numpy)\.", "NS.", code)
    code = re.sub(r"\b(?:t|torch)\.", "NS.", code)
    code = re.sub(r"\b(?:array|tensor|ndarray)\b", "ARR", code)
    return re.sub(r"\s+", "", code)


def load_authored() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for f in sorted(AUTHORED_DIR.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                out[obj["id"]] = obj
    return out


def main() -> int:
    bank = {q["id"]: q for q in json.loads(QUESTIONS_JSON.read_text())}
    authored = load_authored()

    layer: list[dict] = []
    report: dict[str, list] = {
        "retired": [], "drifted": [], "residue": [],
        "clean": [], "no_authoring": [], "answer_not_torch": [],
    }

    for qid in sorted(bank):
        q = bank[qid]
        answer = q.get("answer_code") or ""

        if not is_torch(answer, q.get("starter_code") or ""):
            report["answer_not_torch"].append(qid)

        src = authored.get(qid)
        if not src:
            # Real case, not an error: the 75 curated additions (405-479) were
            # authored, but ids added since May have no prose at all.
            report["no_authoring"].append(qid)
            continue

        old_code = src.get("solution_code", "")
        ratio = difflib.SequenceMatcher(
            None,
            normalise_for_compare(old_code),
            normalise_for_compare(answer),
        ).ratio()
        # Two independent drift signals: the answer is broadly different, or it
        # is textually close but disagrees on a token that changes behaviour.
        semantic_drift = semantic_tokens(old_code) != semantic_tokens(answer)
        drifted = ratio < DRIFT_FLOOR or semantic_drift

        explanation = redialect(src.get("explanation") or "")
        hint = redialect(src.get("hint") or "")

        expl_residue = bool(RESIDUE.search(explanation))
        hint_residue = bool(RESIDUE.search(hint))

        # Withhold rather than guess, for BOTH fields. A drifted answer invalidates
        # the hint as surely as the explanation: a hand-translation that changed
        # the algorithm leaves a hint pointing at the approach the learner is no
        # longer meant to take, and a hint can do that without naming numpy at
        # all — so residue alone is not a sufficient test.
        keep_expl = bool(explanation) and not drifted and not expl_residue
        keep_hint = bool(hint) and not drifted and not hint_residue

        # Recorded independently: a row can be both drifted and residual, and an
        # elif here would hide one of them from the audit counts.
        if drifted:
            report["drifted"].append({
                "id": qid, "ratio": round(ratio, 3),
                "reason": "semantic-token" if ratio >= DRIFT_FLOOR else "shape",
                "subtopic": q.get("subtopic"),
            })
        if expl_residue or hint_residue:
            report["residue"].append({"id": qid, "explanation": expl_residue,
                                      "hint": hint_residue})
        if not drifted and not expl_residue and not hint_residue:
            report["clean"].append(qid)

        layer.append({
            "id": qid,
            "solution_code": answer,          # bank is authoritative, always
            "explanation": explanation if keep_expl else "",
            "hint": hint if keep_hint else "",
            "needs_authoring": not keep_expl,
            "drift_ratio": round(ratio, 3),
        })

    OUT_LAYER.parent.mkdir(parents=True, exist_ok=True)
    OUT_LAYER.write_text("\n".join(json.dumps(r) for r in layer) + "\n")

    # Retired = authored prose whose question no longer exists in the bank.
    report["retired"] = sorted(set(authored) - set(bank))
    OUT_REPORT.write_text(json.dumps(report, indent=2))

    print(f"layer rows          : {len(layer)}")
    print(f"  clean             : {len(report['clean'])}")
    print(f"  withheld (drift)  : {len(report['drifted'])}")
    print(f"  withheld (residue): {len(report['residue'])}")
    print(f"  no authoring      : {len(report['no_authoring'])}")
    print(f"retired (unmap)     : {len(report['retired'])} {report['retired']}")
    print(f"answer not torch    : {len(report['answer_not_torch'])}")
    print(f"\nlayer  -> {OUT_LAYER.relative_to(REPO)}")
    print(f"report -> {OUT_REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
