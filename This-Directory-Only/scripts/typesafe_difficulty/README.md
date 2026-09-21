# typesafe_difficulty

## Purpose
- Re-rates the difficulty of every drill in the bank with TypeSafe's Jev model, ONE
  CONCEPT AT A TIME: within each concept the hardest drill is found (a champion) and
  every other drill is placed by how likely Jev judges it harder than that champion.
- The domain concern is the difficulty LADDER the picker walks inside a concept
  (`solo_progress.next_band`, `prioritization.target_difficulty`): bands only mean
  something if the scores inside a concept order its drills sensibly and the top of
  the scale is that concept's own ceiling.

## Owns
- Grouping the bank into concepts (`bank.py`): q-matrix `target_kcs` first, fallback
  `subtopic_key` for the 65 untagged questions.
- The pairwise judgment + cache (`judge.py`): one pair per request, both framings,
  logit-averaged; requests fan out on a thread pool (`TYPESAFE_WORKERS`, default 12).
- The champion search and final pass (`rate.py`), the per-concept rescale into
  `difficulty_score` bands (`scale.py`), and the pre-apply sanity report
  (`validate.py`).
- The override layer `This-Directory-Only/chatgpt/typesafe_difficulty_overrides.jsonl`
  (`difficulty_score` only).

## Does NOT own
- The picker's use of `difficulty_score` — `backend/app/prioritization.py`,
  `solo_progress.py`, `practice/grading.py`.
- The override merge — `Local_Deployed_Shared/pipeline/export_questions_json.py` and
  `backend/app/questions.py` (the layer list is duplicated; register in BOTH).
- Per-KC difficulty means for the graph — `../export_kc_difficulty.py`.
- The legacy champion rater this replaces:
  `~/Applications/pdf_2_problem/Problem Generator/19. find_champion` + `20. rate_problem_difficulty`
  (GPT logprobs, two calls per ordering, one pair per call).

## Key Files
- `bank.py`: loads `questions_full.json` + `qmatrix_tags.json`, builds `Problem`s and concept groups.
- `judge.py`: `Judge.p_harder(champion, challengers)`; one challenger per request
  (`TYPESAFE_BATCH` must stay 1 — see Known Issues), appends every raw Noul to `cache.jsonl`.
- `rate.py`: per concept, king-of-the-hill until the champion is unbeaten; on a
  cycle (judgments are not transitive) the title goes to the candidate the fewest
  drills beat, and that pass is the rating. Writes `ratings.csv`.
- `scale.py`: logit(p) → 15..100 per concept, champion = 100, rounded to bands of 5;
  writes `difficulty_scores.csv` and the override JSONL.
- `validate.py`: Spearman vs the current `difficulty_score`, band-size report per
  concept, biggest movers. Run before registering the layer.

## Data & External Dependencies
- Reads `This-Directory-Only/questions_full.json` (the exported bank, so run the
  export first if the bank changed) and `Local_Deployed_Shared/lessons/qmatrix_tags.json`.
- Writes under `This-Directory-Only/typesafe_difficulty/` (cache + CSVs).
- `typesafe-sdk` (installed in `backend/.venv`); `TYPESAFE_API_KEY` in the env;
  model pinned to `jev-1.13.0` (`TYPESAFE_MODEL`) so a rescale is reproducible.
- Cost: input tokens only, $0.042/Mtok — a whole-bank run is ~2600 requests, ~$0.10, ~50 s.

## How It Works (Flow)
1. `rate.py`: for each concept, seed the champion with the highest current score.
   Ask Jev, for every other drill in the concept, "is X harder than champion?" and
   "is champion harder than X?" (one request per pair, 2 Nouls, 12 in flight). If any
   drill beats the champion (combined p > 0.5), the strongest beater becomes
   champion and the round repeats; capped at 6 rounds, and a title returning to
   a drill that already held it is a cycle — then the champion is whichever
   holder the fewest drills beat. That holder's pass IS the final rating.
2. `scale.py`: `score = 15 + 85 * (logit(p) - min_logit) / (0 - min_logit)` within
   the concept, so the champion sits at 100 (logit 0 = a coin flip against itself)
   and the concept's easiest drill at 15; then round to the nearest 5.
3. `validate.py` reports; if sane, register the JSONL LAST in both layer lists,
   re-export, run `export_kc_difficulty.py`, deploy.

## Invariants & Constraints
- Scores are quantised to bands of 5. `next_band` gates a band on TWO successes
  in it; a band of one drill can never clear, so a unique score per drill turns
  the rung into a repeat loop.
- Champion = 100 and the floor = 15 within EVERY concept. Cross-concept
  comparability is deliberately not carried by `difficulty_score` (Seth's call,
  2026-09-21: scale to the hardest problem within each concept).
- Never a Noul at exactly 0.5 for the champion vs itself: `judge.py` returns 0.5
  without a call, and `scale.py` maps it to 100.
- The cache is append-only and keyed by model id + a fingerprint of everything
  the model saw (batch size, both drills' text, `CONTEXT`, `CRITERIA`, `FRAMINGS`); edit any
  of those, or a drill, and only the affected pairs are re-asked. Do not
  hand-edit `cache.jsonl`.
- `scale.py` refuses a partial `ratings.csv` (a rateable concept missing, or a
  concept whose drill set no longer matches the bank) unless `--allow-partial`;
  one-drill concepts and concepts with nothing below their champion keep their
  current scores and are listed on stderr.
- The override layer carries ONLY `difficulty_score`, so layering it after
  `curated_overrides.jsonl` cannot clobber a hand-authored prompt or case.

## Extension Points
- Different notion of "harder": edit `CONTEXT` / `CRITERIA` / `FRAMINGS` in
  `judge.py`; the fingerprint changes and every pair is re-asked on the next run.
- Cross-concept ceilings: rate the 72 champions against each other with the same
  `Judge` and feed `export_kc_difficulty.py` — not built.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Batching challengers corrupts the judgment** — `FIXED 2026-09-21`
  - When it happens: several challengers in one state, addressed as `challengers[i]`.
  - Symptom: the first full run made `return sum(numbers)` (q780) the hardest
    calling-functions drill (old 10 → 100); 27 of 92 concepts cycled.
  - Root cause: Jev's positional reference degrades with position — q780 vs q797 alone
    0.03/0.91, at index 3 of 8 0.14/0.64, at index 6 0.68/0.17. Both framings
    agreed, so the reverse framing does not catch it.
  - Prevention/fix: `BATCH=1`, batch size in the cache fingerprint (rows judged under
    a wider batch are never reused), thread pool for throughput. Never raise `BATCH`
    without re-running the position probe.
  - Status: `FIXED`.

- **Calibrated p compresses the scale** — `ACTIVE`
  - When it happens: Jev is calibrated, so "easy drill harder than the hardest?" comes back near 0.02–0.1 for most drills, unlike GPT logprobs (1e-7).
  - Symptom: raw odds^0.8 (the legacy formula) piles everything at the floor.
  - Root cause: the legacy formula assumed the wide logprob range.
  - Prevention/fix: `scale.py` rescales logit affinely per concept (floor→15, champion→100) instead of trusting raw odds. Check `validate.py`'s band counts before applying.
  - Status: `ACTIVE` by design; watch for concepts where every band has one drill.

## Recent Changes
- 2026-09-21 (later): first real run. Batching found unsafe (see Known Issues) →
  `BATCH=1` + thread pool + batch size in the fingerprint; `scale.py` takes the
  champion from `champion_id` instead of asserting a single p=0.5 (a challenger can
  tie at 0.5). Full bank: 2598 requests, $0.10, 51 s, 13 cyclic concepts.
- 2026-09-21: Folder created: bank grouping, cached pairwise judge, rate / scale /
  validate. Codex critic pass: champion search no longer filters "beaten" drills
  (it could return a drill the pass itself showed beaten), cache keys carry a
  content fingerprint, scale checks coverage + one champion per concept and
  writes atomically, flat concepts keep their scores.
