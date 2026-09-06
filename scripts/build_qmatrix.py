#!/usr/bin/env python3
"""Build Local_Deployed_Shared/lessons/qmatrix_tags.json.

Tag sources, in priority order:
  1. KP frontmatter — a question referenced as faded/guided/independent/
     integrated by a KP gets that KP's KC as target, the KP's supporting list,
     and its new_syntax.
  2. LEFTOVER_TARGETS below — hand-assigned targets for easy-topic questions no
     KP references (reviewed against each question's text; see
     docs/spec-first-encounter-course-content.md).

Run after editing KP refs; validate with scripts/validate_lessons.py --coverage.
"""
import json
import sys
from lesson_lib import LESSONS_DIR, all_kp_paths, load_bank, load_registry, parse_kp

# "Python" joined these on 2026-08-28 with lesson py-0: the first-encounter
# course now starts BELOW numpy.ndarray-model, and its drills have to be
# tagged like any other course question or the crosswalk cannot measure the
# seven concepts under it.
EASY_TOPICS = ("Python", "Numpy", "Einsum", "Einops", "PyTorch")

# Hand-assigned target KCs for questions not referenced by any KP.
LEFTOVER_TARGETS = {
    # Trimmed 2026-08-30 with the cut back to the ARENA-necessary concepts:
    # 38 of these 77 entries named a concept the course no
    # longer teaches, or a drill retired with it. See
    # Local_Deployed_Shared/pipeline/retired_question_ids.json.
    # 103 and 198 kept back from the 2026-08-30 retirement because each was the
    # only trainer of a gating BKT atom; re-homed onto the surviving concept
    # that owns what they actually do. See pipeline/retired_question_ids.json.
    103: "numpy.axis-reductions", 198: "numpy.broadcasting-rules",
    1: "numpy.argmin-argmax", 48: "numpy.constructors",
    63: "numpy.elementwise-ufuncs", 74: "numpy.slicing-views",
    85: "numpy.boolean-masking", 87: "numpy.dtype-astype",
    95: "numpy.dot-matmul-patterns", 145: "numpy.boolean-masking",
    168: "numpy.argmin-argmax", 174: "numpy.axis-reductions",
    189: "numpy.slicing-views", 202: "numpy.boolean-masking",
    316: "einops.channel-groups-temporal", 319: "einops.pattern-language",
    321: "einops.patches-space-depth", 322: "einops.grids-montage",
    327: "einops.pattern-language", 331: "einops.split-axes",
    333: "einops.singleton-and-lists", 334: "einops.singleton-and-lists",
    336: "einops.pooling", 340: "einops.reduce-model",
    344: "einops.pattern-language", 346: "einops.merge-axes",
    352: "einops.repeat-model", 353: "einops.merge-axes",
    355: "einops.repeat-model", 359: "einops.channel-groups-temporal",
    365: "einops.singleton-and-lists", 370: "einops.reduce-model",
    371: "einops.grids-montage", 373: "einops.merge-axes",
    377: "einops.pooling", 380: "einops.merge-axes",
    392: "einops.merge-axes", 395: "einops.patches-space-depth",
    398: "einops.patches-space-depth", 400: "einops.merge-axes",
    403: "einops.patches-space-depth", 
}


def build():
    bank = load_bank()
    registry = load_registry()
    kc_meta = {kc["id"]: kc for kc in registry["kcs"]}
    kp_by_kc = {}
    for path in all_kp_paths():
        kp = parse_kp(path)
        kp_by_kc[kp["kc"]] = kp

    tags = {}

    def tag_for(kc_id, role):
        kp = kp_by_kc[kc_id]
        return {
            "target_kcs": [kc_id],
            "supporting_kcs": kp["supporting"],
            "new_syntax": kp["new_syntax"],
            "source": role,
        }

    for kc_id, kp in kp_by_kc.items():
        # `integrated` last: it is the top rung, and a question may sit on
        # exactly one rung (the duplicate check below enforces that).
        for role in ("faded", "guided", "independent", "integrated"):
            for qid in kp[role]:
                if qid in tags:
                    raise SystemExit(f"q{qid} referenced twice ({tags[qid]}, {kc_id})")
                tags[qid] = tag_for(kc_id, f"kp-{role}")

    for qid, kc_id in LEFTOVER_TARGETS.items():
        if qid in tags:
            # A KP has since claimed this question. That is the normal way a
            # leftover retires, and the KP reference wins — it carries the
            # role and the page's new_syntax, which the hand assignment
            # cannot. The entry stays in the table so the question keeps a
            # target if the KP ever drops it again. Only a DISAGREEMENT is an
            # error: two sources naming different KCs for one question means
            # one of them is wrong and the build must not pick silently.
            if tags[qid]["target_kcs"] != [kc_id]:
                raise SystemExit(
                    f"q{qid}: LEFTOVER_TARGETS says {kc_id}, but a KP claims it "
                    f"for {tags[qid]['target_kcs']} — fix one of them."
                )
            continue
        if kc_id not in kc_meta:
            raise SystemExit(f"q{qid}: unknown kc {kc_id}")
        kp = kp_by_kc.get(kc_id)
        tags[qid] = {
            "target_kcs": [kc_id],
            "supporting_kcs": kp["supporting"] if kp else [],
            "new_syntax": [],
            "source": "leftover-assignment",
        }

    easy = {qid for qid, q in bank.items() if q["curriculum"]["topic"] in EASY_TOPICS}
    missing = easy - set(tags)
    extra = set(tags) - easy
    if missing:
        raise SystemExit(f"{len(missing)} easy questions untagged: {sorted(missing)[:12]}")
    if extra:
        raise SystemExit(f"tags reference non-easy questions: {sorted(extra)}")

    out = LESSONS_DIR / "qmatrix_tags.json"
    out.write_text(json.dumps({str(k): tags[k] for k in sorted(tags)}, indent=1))
    print(f"wrote {len(tags)} question tags -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(build())
