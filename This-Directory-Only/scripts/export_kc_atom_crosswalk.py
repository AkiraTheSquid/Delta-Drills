#!/usr/bin/env python3
"""Join the lesson graph's KC ids to the backend's concept-graph atom ids.

The lesson knowledge graph draws 64 KC ids (`lessons/kc_registry.json`, e.g.
`numpy.dtype-astype`). The practice backend tracks BKT belief per *atom*
(`arena_drillable_v1.json`, e.g. `argmax-prediction`) and ships it as
`atom_mastery`. The two id spaces are disjoint — measured overlap is zero — so
the graph currently cannot read the mastery it already has, and every node
falls back to an inferred estimate.

Questions are the bridge: both sides tag the same question bank.

    lessons/qmatrix_tags.json              question -> target_kcs
    backend/app/data/question_atom_tags.jsonl   question -> atoms (+confidence)

Only `target_kcs` count, matching export_kc_difficulty.py: `supporting_kcs`
are prerequisites the question leans on, not what it measures.

The raw join is noisy — a handful of atoms (broadcasting-rules, einops-einsum,
index-by-tensor) are tagged across a dozen KCs each, so their mastery says
almost nothing about any one KC. Each (kc, atom) pair therefore carries three
measured numbers:

    share  = P(atom | kc)  fraction of the KC's tagged evidence via this atom
    spec   = P(kc | atom)  how specific the atom is to this KC
    w      = share * spec, renormalized so a KC's weights sum to 1

`spec` is conditioned on the joined subset: its denominator sums support only
over questions that carry a registry KC, so it is P(kc | atom, question is
KC-tagged), not P(kc | atom) across the whole 455-question bank. That is the
quantity we want — it asks how an atom's *KC-bearing* evidence splits — but it
means `spec` will move if untagged questions later gain target_kcs.

and each KC carries

    reliability = sum_a share(k,a) * spec(k,a)   in (0, 1]
    shared_with = 1/reliability - 1

`reliability` is the evidence-weighted mean specificity of the atoms standing
in for a KC. Its reciprocal is a participation ratio — the same construction as
inverse-Simpson / effective sample size — so `shared_with` reads as "this KC's
measurement is *effectively* shared with N other KCs." The reading is exact
only in the uniform case (all of a KC's evidence via atoms shared equally by m
KCs gives reliability = 1/m and shared_with = m-1); with unequal spec it is an
effective count, always at or below the nominal number of sibling KCs.

That distinction decides the tier:

    measured    reliability >= 1/3, i.e. shared with at most ~2 other KCs.
                atom_mastery is a genuine per-node reading.
    topic-proxy below that. The atoms are coarser than the KC (twelve einsum
                KCs all resolve to `einops-einsum`), so reading atom_mastery
                per KC would hand several nodes one number and present a
                topic average as a per-node measurement. Those KCs stay in
                the inferred regime, where the existing subtopic fallback
                already gives the same information without the false
                precision. See docs/plan-graph-estimator-rev2.md (rev 3).

The 1/3 cut is a rounded interpretable boundary chosen by the author, not a
validated threshold; likewise `suggested_w_min`. Neither drops anything from
the file — both are emitted as advice, with the raw numbers alongside.

Output: Local_Deployed_Shared/concept-graph/kc_atom_crosswalk.json

Regenerate after editing the q-matrix or the atom tags:
    python3 This-Directory-Only/scripts/export_kc_atom_crosswalk.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "Local_Deployed_Shared"
BACKEND_DATA = ROOT / "This-Directory-Only" / "backend" / "app" / "data"

REGISTRY = SHARED / "lessons" / "kc_registry.json"
QMATRIX = SHARED / "lessons" / "qmatrix_tags.json"
ATOM_TAGS = BACKEND_DATA / "question_atom_tags.jsonl"
ATOM_GRAPH = BACKEND_DATA / "concept_graphs" / "arena_drillable_v1.json"
OUT = SHARED / "concept-graph" / "kc_atom_crosswalk.json"

# Author default, not sourced: a pair below this weight is more likely a
# co-tagging accident than a reading of the KC. Advice only — see module docs.
SUGGESTED_W_MIN = 0.15

# Rounded interpretable cut, not validated: reliability >= 1/3 means the KC's
# evidence is effectively shared with at most ~2 other KCs, so atom_mastery
# reads as a per-node measurement rather than a topic average.
MEASURED_RELIABILITY = 1.0 / 3.0


def _load_atom_tags() -> dict[str, list[dict]]:
    tags: dict[str, list[dict]] = {}
    for line in ATOM_TAGS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        qid = rec.get("question_id")
        if qid is None:
            continue
        tags[str(qid)] = rec.get("atoms") or []
    return tags


def main() -> int:
    registry = json.loads(REGISTRY.read_text())
    qmatrix = json.loads(QMATRIX.read_text())
    atom_tags = _load_atom_tags()
    atom_ids = {c["id"] for c in json.loads(ATOM_GRAPH.read_text()).get("concepts", [])}
    kc_ids = {k["id"] for k in registry.get("kcs", [])}

    support: dict[tuple[str, str], float] = defaultdict(float)
    pair_n: Counter[tuple[str, str]] = Counter()
    kc_questions: Counter[str] = Counter()
    joined = 0
    unknown_atoms: set[str] = set()

    for qid, entry in qmatrix.items():
        atoms = atom_tags.get(str(qid))
        if not atoms:
            continue
        kcs = [k for k in (entry.get("target_kcs") or []) if k in kc_ids]
        if not kcs:
            continue

        # Validate the atom side before counting anything. A question whose
        # atoms are all unknown ids or all zero/invalid confidence contributes
        # no evidence, so it must not inflate `questions_joined` or a KC's `n`
        # — those numbers are read as evidence volume. Confidence must be
        # finite and positive: a zero-confidence pair would enter `support` at
        # 0.0 and, if a KC had only such pairs, drive share = s/0.
        usable: list[tuple[str, float]] = []
        for atom in atoms:
            aid = atom.get("atom_id")
            if not aid:
                continue
            if aid not in atom_ids:
                unknown_atoms.add(aid)
                continue
            try:
                conf = float(atom.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if not (conf > 0.0) or conf != conf or conf == float("inf"):
                continue
            usable.append((aid, conf))
        if not usable:
            continue

        joined += 1
        for kc in kcs:
            kc_questions[kc] += 1
            for aid, conf in usable:
                # A question targeting several KCs splits its evidence between
                # them; the atom's own tag confidence scales the whole pair.
                support[(kc, aid)] += conf / len(kcs)
                pair_n[(kc, aid)] += 1

    if not support:
        raise SystemExit("no (kc, atom) pairs — check the q-matrix and atom tags")

    by_kc: dict[str, float] = defaultdict(float)
    by_atom: dict[str, float] = defaultdict(float)
    for (kc, aid), s in support.items():
        by_kc[kc] += s
        by_atom[aid] += s

    kcs_out: dict[str, dict] = {}
    for kc in sorted(by_kc):
        rows = []
        for (k, aid), s in support.items():
            if k != kc:
                continue
            share = s / by_kc[kc]
            spec = s / by_atom[aid]
            rows.append({"a": aid, "share": share, "spec": spec, "n": pair_n[(kc, aid)]})
        reliability = sum(r["share"] * r["spec"] for r in rows)
        total_w = reliability  # sum of share*spec before renormalization
        for r in rows:
            r["w"] = (r["share"] * r["spec"] / total_w) if total_w else 0.0
        rows.sort(key=lambda r: (-r["w"], r["a"]))
        # Tier off the *emitted* (rounded) reliability against the *emitted*
        # (rounded) threshold, so a consumer re-deriving the tier from this
        # file's own numbers always agrees with the `tier` field. Comparing
        # full precision against a rounded threshold lets 0.3332 export as
        # 0.333 >= 0.333 while being labelled topic-proxy.
        rel_out = round(reliability, 3)
        kcs_out[kc] = {
            "n": kc_questions[kc],
            "reliability": rel_out,
            "shared_with": round(1.0 / reliability - 1.0, 1) if reliability else None,
            "tier": "measured" if rel_out >= round(MEASURED_RELIABILITY, 3) else "topic-proxy",
            "atoms": [
                {
                    "a": r["a"],
                    "w": round(r["w"], 3),
                    "share": round(r["share"], 3),
                    "spec": round(r["spec"], 3),
                    "n": r["n"],
                }
                for r in rows
            ],
        }

    out = {
        "version": 1,
        "generated_from": [
            "lessons/qmatrix_tags.json target_kcs",
            "backend/app/data/question_atom_tags.jsonl",
            "backend/app/data/concept_graphs/arena_drillable_v1.json (atom id universe)",
        ],
        "method": (
            "support(kc,atom) = sum over shared questions of tag_confidence/|target_kcs|; "
            "share = P(atom|kc), spec = P(kc|atom), w = share*spec renormalized per kc; "
            "reliability = sum_a share*spec, the evidence-weighted mean specificity"
        ),
        "suggested_w_min": SUGGESTED_W_MIN,
        "measured_reliability_min": round(MEASURED_RELIABILITY, 3),
        "thresholds_note": "both cuts are author defaults, not sourced or validated",
        "coverage": {
            "questions_joined": joined,
            "kcs_covered": len(kcs_out),
            "kcs_total": len(kc_ids),
            "kcs_measured": sum(1 for v in kcs_out.values() if v["tier"] == "measured"),
            "atoms_used": len(by_atom),
            "pairs": len(support),
        },
        "kcs": kcs_out,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    cov = out["coverage"]
    print(
        f"[kc-atom-crosswalk] {cov['kcs_covered']}/{cov['kcs_total']} KCs joined from "
        f"{cov['questions_joined']} questions via {cov['atoms_used']} atoms ({cov['pairs']} pairs) "
        f"→ {OUT.relative_to(ROOT)}"
    )
    proxy = sorted(k for k, v in kcs_out.items() if v["tier"] == "topic-proxy")
    print(
        f"[kc-atom-crosswalk] {cov['kcs_measured']}/{cov['kcs_total']} KCs are per-node measurable; "
        f"{len(proxy)} are topic proxies and must stay inferred"
    )
    if proxy:
        print(f"[kc-atom-crosswalk] topic-proxy KCs: {', '.join(proxy)}")
    missing = sorted(kc_ids - set(kcs_out))
    if missing:
        print(f"[kc-atom-crosswalk] no joined questions for: {', '.join(missing)}")
    if unknown_atoms:
        print(f"[kc-atom-crosswalk] atom ids absent from the drillable graph: {', '.join(sorted(unknown_atoms))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
