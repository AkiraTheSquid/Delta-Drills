"""kc_stats.py — the first READER of the attempt log: global per-KC aggregates.

`attempt_log.py` has recorded every prediction, feature vector and outcome since
2026-08-06, and until this module nothing ever read those files back. The two
numbers that motivated it (docs/spec-graph-metadata-audit-layer.md §5):

  * **Rung stall** — the trailing run of attempts one learner made on one
    concept at one rung. Seth's own log held a 63-attempt `faded` run at 16%
    accuracy on `numpy.ndarray-model`, dated the week BEFORE the `a.T`
    prerequisite bug was found by hand. This number, computed nightly, names a
    broken concept with a single learner's data.
  * **Served-while-predicting-failure** — a run of serves where the model's own
    `predicted_p` sat below a floor. The same log shows mean p of 0.21 on that
    concept: the model said "they will fail" ~60 times and the selector
    complied. Whichever of the two is wrong, the streak is the symptom.

Aggregation is read-only and cross-learner: it globs every `*.attempts.jsonl`
in the practice data dir. Counts, not identities — the per-learner extremes are
reported as lengths and rates with no user id attached, because the consumer is
a "which CONCEPT is broken" surface, not a learner dashboard.

Deliberately depends on `attempt_log` and `logistic_engine` only. `kc_graph`
would buy titles at the price of dragging the whole lattice (and its data
files) into what a local audit script imports; KC ids are already readable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from app import attempt_log
from app import logistic_engine as E

# A serve whose prediction sat below this is a "the model expected failure"
# serve. 0.35 rather than 0.5: near the boundary the selector choosing a
# stretch item is policy, not pathology; a third-chance is not.
LOW_P = 0.35

# Flag floors. Below these a streak is normal ladder traffic — a rung needs a
# few attempts before promotion by design, and one low-p serve is a stretch
# item. At/above them, a human should look. Both were chosen by eye against the
# 2026-09-01 logs (the a.T stall read 43 and 63) and should be re-derived when
# there is enough data to fit anything.
STALL_FLAG_MIN = 10
LOW_P_FLAG_MIN = 5


def _graded_rows(user_id: str, base_dir: Optional[Path]) -> List[attempt_log.AttemptRow]:
    """One learner's graded attempts, oldest first.

    Sorted by timestamp rather than trusted as-appended: `backfill_from_state`
    can write history behind live rows, and a stall is a claim about ORDER.
    """
    rows = [r for r in attempt_log.iter_rows(user_id, base_dir) if r.is_graded]
    rows.sort(key=lambda r: r.ts)
    return rows


def _trailing_same_stage(rows: List[attempt_log.AttemptRow]) -> dict:
    """The trailing run of one learner's attempts at one unchanged rung.

    Trailing, not maximal: the question this answers is "is this concept
    looping right now", and a run that a later promotion ended is a resolved
    story. A demotion starts a new trailing run, which is correct — being sent
    back down and held there is exactly the situation worth flagging.
    """
    n = 0
    correct = 0
    stage = None
    for row in reversed(rows):
        s = E.normalize_stage(row.stage)
        if stage is None:
            stage = s
        if s != stage:
            break
        n += 1
        correct += 1 if row.correct else 0
    return {"stage": stage, "length": n, "accuracy": (correct / n) if n else None}


def _max_low_p_run(rows: List[attempt_log.AttemptRow]) -> dict:
    """The longest run of consecutive serves with predicted_p < LOW_P.

    Maximal rather than trailing, unlike the stall: a stall that ended is
    resolved, but a selector that spent 20 serves on predicted failures has a
    policy bug whether or not it eventually moved on.
    """
    best_len = 0
    best_correct = 0
    best_p = 0.0
    run: List[attempt_log.AttemptRow] = []
    for row in rows + [None]:  # sentinel flushes the final run
        if row is not None and row.predicted_p is not None and row.predicted_p < LOW_P:
            run.append(row)
            continue
        if len(run) > best_len:
            best_len = len(run)
            best_correct = sum(1 for r in run if r.correct)
            best_p = sum(float(r.predicted_p) for r in run) / len(run)
        run = []
    return {
        "length": best_len,
        "accuracy": (best_correct / best_len) if best_len else None,
        "mean_predicted_p": round(best_p, 3) if best_len else None,
    }


def user_ids(base_dir: Optional[Path] = None) -> List[str]:
    """Every learner with an attempt log, from the filenames.

    The filename is the sanitised user id `log_path` wrote — good enough to
    read the file back, which is all this module does with it.
    """
    root = Path(base_dir) if base_dir is not None else attempt_log._data_dir()
    suffix = ".attempts.jsonl"
    return sorted(p.name[: -len(suffix)] for p in root.glob(f"*{suffix}"))


def kc_stats(base_dir: Optional[Path] = None) -> dict:
    """Global per-KC aggregates plus a flags list, over every learner's log.

    Shape: {"learners": N, "kcs": {kc: {...}}, "flags": [...]}. Per KC:
    attempt/learner counts, accuracy, mean predicted p, Brier, per-stage
    {n, accuracy}, and the worst per-learner `stall` / `low_p_run` (lengths and
    rates only — no user ids; see module docstring). A flag is emitted when a
    worst case clears STALL_FLAG_MIN / LOW_P_FLAG_MIN.
    """
    per_kc: Dict[str, dict] = {}
    learners = user_ids(base_dir)
    for uid in learners:
        by_kc: Dict[str, List[attempt_log.AttemptRow]] = {}
        for row in _graded_rows(uid, base_dir):
            if row.kc:
                by_kc.setdefault(row.kc, []).append(row)
        for kc, rows in by_kc.items():
            agg = per_kc.setdefault(
                kc,
                {
                    "learners": 0,
                    "attempts": 0,
                    "correct": 0,
                    "p_sum": 0.0,
                    "p_n": 0,
                    "brier_sum": 0.0,
                    "stages": {},
                    "worst_stall": {"stage": None, "length": 0, "accuracy": None},
                    "stalled_learners": 0,
                    "worst_low_p_run": {"length": 0, "accuracy": None, "mean_predicted_p": None},
                },
            )
            agg["learners"] += 1
            agg["attempts"] += len(rows)
            agg["correct"] += sum(1 for r in rows if r.correct)
            for r in rows:
                st = agg["stages"].setdefault(
                    E.normalize_stage(r.stage), {"n": 0, "correct": 0}
                )
                st["n"] += 1
                st["correct"] += 1 if r.correct else 0
                if r.predicted_p is not None:
                    agg["p_sum"] += float(r.predicted_p)
                    agg["p_n"] += 1
                    agg["brier_sum"] += (float(r.predicted_p) - (1.0 if r.correct else 0.0)) ** 2
            stall = _trailing_same_stage(rows)
            if stall["length"] >= STALL_FLAG_MIN:
                agg["stalled_learners"] += 1
            if stall["length"] > agg["worst_stall"]["length"]:
                agg["worst_stall"] = stall
            low = _max_low_p_run(rows)
            if low["length"] > agg["worst_low_p_run"]["length"]:
                agg["worst_low_p_run"] = low

    flags: List[dict] = []
    for kc, agg in sorted(per_kc.items()):
        n = agg["attempts"]
        agg["accuracy"] = round(agg["correct"] / n, 3) if n else None
        agg["mean_predicted_p"] = round(agg["p_sum"] / agg["p_n"], 3) if agg["p_n"] else None
        agg["brier"] = round(agg["brier_sum"] / agg["p_n"], 3) if agg["p_n"] else None
        for st in agg["stages"].values():
            st["accuracy"] = round(st["correct"] / st["n"], 3) if st["n"] else None
        for key in ("correct", "p_sum", "p_n", "brier_sum"):
            del agg[key]
        ws = agg["worst_stall"]
        if ws["length"] >= STALL_FLAG_MIN:
            flags.append(
                {
                    "kind": "rung_stall",
                    "kc": kc,
                    "stage": ws["stage"],
                    "length": ws["length"],
                    "accuracy": ws["accuracy"],
                    "learners_stalled": agg["stalled_learners"],
                }
            )
        wl = agg["worst_low_p_run"]
        if wl["length"] >= LOW_P_FLAG_MIN:
            flags.append(
                {
                    "kind": "served_while_predicting_failure",
                    "kc": kc,
                    "length": wl["length"],
                    "mean_predicted_p": wl["mean_predicted_p"],
                    "accuracy": wl["accuracy"],
                }
            )
    flags.sort(key=lambda f: -f["length"])
    return {"learners": len(learners), "kcs": per_kc, "flags": flags}
