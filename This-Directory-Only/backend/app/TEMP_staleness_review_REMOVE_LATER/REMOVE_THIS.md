# TEMP: Staleness Review Feature

This folder and its hooks are **temporary** and should be removed once FSRS
is integrated (phase 3). The feature is intentionally kept in one place so
it can be deleted cleanly.

## What this does

After a subtopic has not been studied for `STALENESS_THRESHOLD_DAYS` (default: 3),
the next session forces `STALENESS_REVIEW_COUNT` (default: 3) consecutive questions
from that subtopic before returning to normal priority-based selection.

This lets the EWMA learning rate update based on current performance, catching
cases where the user has forgotten material (score drops then rebounds = high
learning rate signal) or retained it (score holds = learning rate stays low).

Only subtopics past the cold-start window (≥ 3 questions answered) trigger
staleness reviews. Topics still in cold start are already high-priority via
the cold-start fix.

## How to remove

**Step 1:** Delete this entire folder:
```
backend/app/TEMP_staleness_review_REMOVE_LATER/
```

**Step 2:** In `backend/app/prioritization.py`, remove these 3 things:

1. The feature flag near the top of the file:
```python
# TEMP: staleness review — remove with TEMP_staleness_review_REMOVE_LATER/
ENABLE_STALENESS_REVIEW: bool = True
```

2. The conditional import block below it:
```python
if ENABLE_STALENESS_REVIEW:
    from app.TEMP_staleness_review_REMOVE_LATER.staleness import get_staleness_override
```

3. The override block at the top of `select_next_subtopic`:
```python
    # TEMP: staleness review override
    if ENABLE_STALENESS_REVIEW:
        available = [st for st in subtopics if get_questions_by_subtopic(st)]
        override = get_staleness_override(user_state, available)
        if override is not None:
            return override
```

**Step 3:** Nothing else needs to change. `adaptive.py`, `practice_router.py`,
and all other files are unmodified by this feature.
