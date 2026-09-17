#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime

P_ALPHA = 0.3
USER_DATA_DIR = Path(__file__).resolve().parents[1] / "user_data"

def parse_ts(ts: str) -> float:
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def recompute_p_for_state(state: dict) -> bool:
    changed = False
    subtopics = state.get("subtopic_states", {})
    for sub_name, sub_data in subtopics.items():
        history = sub_data.get("history", []) or []
        if not history:
            continue
        # Sort by timestamp for deterministic replay.
        history_sorted = sorted(history, key=lambda a: parse_ts(a.get("timestamp", "")))
        p = 0.5
        n = 0
        for attempt in history_sorted:
            n += 1
            indicator = 1.0 if attempt.get("grade", 0) > 85 else 0.0
            if n == 1:
                p = indicator
            else:
                p = P_ALPHA * indicator + (1 - P_ALPHA) * p
            attempt["p_after"] = p
        # Write back in original list order too.
        p_after_by_id = {(a.get("question_id"), a.get("timestamp")): a.get("p_after") for a in history_sorted}
        for attempt in history:
            key = (attempt.get("question_id"), attempt.get("timestamp"))
            if key in p_after_by_id:
                attempt["p_after"] = p_after_by_id[key]
        if abs(sub_data.get("p", 0.0) - p) > 1e-9:
            sub_data["p"] = p
            changed = True
    return changed


def main() -> None:
    if not USER_DATA_DIR.exists():
        raise SystemExit(f"User data directory not found: {USER_DATA_DIR}")
    files = sorted(USER_DATA_DIR.glob("*.json"))
    if not files:
        print("No user state files found.")
        return
    for path in files:
        state = json.loads(path.read_text(encoding="utf-8"))
        changed = recompute_p_for_state(state)
        if changed:
            path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            print(f"Updated: {path.name}")
        else:
            print(f"No change: {path.name}")

if __name__ == "__main__":
    main()
