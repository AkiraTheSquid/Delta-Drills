#!/usr/bin/env python3
"""Build Local_Deployed_Shared/lessons/qmatrix_tags.json.

Tag sources, in priority order:
  1. KP frontmatter — a question referenced as faded/guided/independent by a KP
     gets that KP's KC as target, the KP's supporting list, and its new_syntax.
  2. LEFTOVER_TARGETS below — hand-assigned targets for easy-topic questions no
     KP references (reviewed against each question's text; see
     docs/spec-first-encounter-course-content.md).

Run after editing KP refs; validate with scripts/validate_lessons.py --coverage.
"""
import json
import sys
from lesson_lib import LESSONS_DIR, all_kp_paths, load_bank, load_registry, parse_kp

EASY_TOPICS = ("Numpy", "Einsum", "Einops")

# Hand-assigned target KCs for questions not referenced by any KP.
LEFTOVER_TARGETS = {
    # Numpy: Vectorization and broadcasting
    1: "numpy.argmin-argmax", 48: "numpy.constructors",
    63: "numpy.elementwise-ufuncs", 68: "numpy.cumulative-diff",
    74: "numpy.slicing-views", 85: "numpy.boolean-masking",
    95: "numpy.dot-matmul-patterns", 120: "numpy.nan-handling",
    154: "numpy.centering", 155: "numpy.tile-repeat-meshgrid",
    160: "numpy.diag-triangles", 164: "numpy.index-grids",
    177: "numpy.fancy-indexing", 240: "numpy.where-select",
    # Numpy: Indexing and selection
    73: "numpy.ndarray-model", 78: "numpy.diag-triangles",
    87: "numpy.dtype-astype", 91: "numpy.random-sampling",
    103: "numpy.rescaling", 126: "numpy.window-stencil",
    129: "numpy.pairwise-metrics", 136: "numpy.fancy-indexing",
    145: "numpy.boolean-masking", 163: "numpy.cumulative-diff",
    168: "numpy.argmin-argmax", 174: "numpy.axis-reductions",
    202: "numpy.boolean-masking", 211: "numpy.set-combinatorics",
    # Numpy: Applied patterns and advanced
    133: "numpy.set-combinatorics", 147: "numpy.rescaling",
    158: "numpy.set-combinatorics", 170: "numpy.nan-handling",
    172: "numpy.onehot-bincount", 180: "numpy.diag-triangles",
    184: "numpy.fancy-indexing", 186: "numpy.pad-borders",
    189: "numpy.slicing-views", 198: "numpy.set-combinatorics",
    207: "numpy.set-combinatorics",
    # Einsum
    249: "einsum.reductions", 250: "einsum.batch-dims",
    253: "einsum.broadcast-scaling", 261: "einsum.reductions",
    279: "einsum.batch-dims", 286: "einsum.matvec-matmul",
    290: "einsum.reductions", 293: "einsum.diag-trace",
    294: "einsum.matvec-matmul", 300: "einsum.notation-model",
    302: "einsum.reductions", 303: "einsum.notation-model",
    # Einops: Rearrange
    319: "einops.pattern-language", 327: "einops.pattern-language",
    331: "einops.split-axes", 334: "einops.singleton-and-lists",
    344: "einops.pattern-language", 346: "einops.merge-axes",
    353: "einops.merge-axes", 373: "einops.merge-axes",
    392: "einops.merge-axes", 398: "einops.patches-space-depth",
    400: "einops.merge-axes", 403: "einops.patches-space-depth",
    # Einops: Reduce / Repeat
    336: "einops.pooling", 352: "einops.repeat-model",
    355: "einops.repeat-model",
    # Einops: Deep Learning
    316: "einops.channel-groups-temporal", 321: "einops.patches-space-depth",
    322: "einops.grids-montage", 333: "einops.singleton-and-lists",
    340: "einops.reduce-model", 359: "einops.channel-groups-temporal",
    365: "einops.singleton-and-lists", 370: "einops.reduce-model",
    371: "einops.grids-montage", 377: "einops.pooling",
    380: "einops.merge-axes", 387: "einops.grids-montage",
    395: "einops.patches-space-depth",
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
        for role in ("faded", "guided", "independent"):
            for qid in kp[role]:
                if qid in tags:
                    raise SystemExit(f"q{qid} referenced twice ({tags[qid]}, {kc_id})")
                tags[qid] = tag_for(kc_id, f"kp-{role}")

    for qid, kc_id in LEFTOVER_TARGETS.items():
        if qid in tags:
            raise SystemExit(f"q{qid} both referenced and in LEFTOVER_TARGETS")
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
