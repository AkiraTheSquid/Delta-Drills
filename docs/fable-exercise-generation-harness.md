# Fable 5 Exercise-Generation Harness for Delta Drills

> Status: DESIGN. Written 2026-07-04, after the bank-hardening pass (157
> gameable gradings fixed, `audit_question_bank.py --gate` wired into deploy).
> Generator model: **Claude Fable 5 driven through the Claude Code harness**
> (inline agents / workflows, full repo access). This doc adapts the two
> research reports to that setup:
>   - `docs/exercise-generation-guidelines.md` (2026-07 survey; rubrics,
>     failure modes, test hardening)
>   - `what-is-the-best-way-to-build-exercises-for-a-TTS/Standalone_Deep_
>     Research_Report...md` (2026-05; pipeline spec, KC decomposition,
>     CBIT generators) — its claim audit lives beside it.

## Non-negotiables (ordered by evidence strength)

1. **Deterministic execution gate innermost** (Sarsa 2022: only ~31% of
   unvalidated generated exercises passed their own tests). Nothing ships
   unless `audit_question_bank.py --gate` passes on the merged bank:
   reference solution passes the real `run_function_tests` harness, every
   bare-fixture cheat fails, starter parses, expected value is non-degenerate.
   The gate already exists and runs in `deploy_delta_drills` Step 2b — the
   generator only needs to feed it.
2. **Blind spec-first generation** (anchoring: Lou & Sun 2024 — stronger
   models anchor MORE consistently, and prompt-level "ignore the anchor"
   instructions do not work; only structural removal of the anchor does).
   When regenerating or improving an existing question, the generating agent
   receives the SPEC ONLY — never the old question text or starter. Judging
   old-vs-new happens in a separate agent that sees both blind (no "which is
   the incumbent" label).
3. **Compact prompts** (Scaria 2024, audit-corrected finding: PS4 — CoT +
   one-sentence skill description + ONE example — beat PS5, which added
   explanations and *reduced* quality for 3 of 5 models). Do not stuff
   pedagogy essays into the generation prompt. One exemplar of house style,
   one KC sentence, the format contract, the constraint list. Done.
4. **Single-KC per drill, integration explicit and late** (Tutor Kai: concept
   coverage collapses ~94%→40% from 1→3 concepts). Default every generated
   drill to ONE knowledge component; only ladder-final items may compose 2–3,
   and those get a fallback narrower variant.
5. **Multi-agent validation, not self-critique alone** (PyTaskSyn: single
   agents self-correct poorly; separate expert/student roles beat one judge).
   In Claude Code terms: generation agent ≠ validation agent, always separate
   contexts, validator returns structured JSON findings, generator retries on
   reject, pipeline **abstains** (emits nothing) after N=3 rejects.

## The exercise spec (generation input)

```json
{
  "atom": "broadcasting-rules",
  "kc": "singleton-axis-expansion",
  "kc_description": "Expand a size-1 axis against a larger shape without copying.",
  "bloom": "Apply",
  "difficulty_rung": 3,
  "surface_format": "function-mode | stdout-mode | colab-torch",
  "fixture_style": "small deterministic numpy arrays, seeded randomness only",
  "print_contract": "print(solve())  — list of floats rounded to 4dp",
  "banned": ["answer expression in text/comments", "derivation in setup",
             "identity/no-op transforms", "unseeded randomness"],
  "exemplar_id": 74
}
```

The spec is extracted from the existing bank (for regeneration) or authored
per atom (for new drills). `exemplar_id` points at ONE house-style question
the prompt may show — never the question being replaced.

## Generation prompt skeleton (PS4-shaped)

```
You are writing one Delta Drills exercise. Think through the design first,
then emit strict JSON.

Skill (one sentence): {kc_description}
Bloom target: {bloom}. Difficulty rung: {difficulty_rung}/5.
House-style example (format only — do not copy its content): {exemplar}

Emit JSON: {question_text, starter_code, test_cases:[{setup_code,
expected_setup_code, call, expected_expr}], solution_notes,
expected_failure_modes}

Hard constraints:
- setup_code = input fixtures ONLY. Every derivation line lives in
  expected_setup_code/expected_expr (the graded side). A learner submitting
  `return <any setup var>` must fail.
- question_text states the task; it never contains the answer expression or
  a step-by-step recipe of the exact calls to make.
- starter_code parses, defines solve() returning None, and re-states fixtures
  exactly as setup_code does.
- Tests must reject the do-nothing answer AND one plausible wrong approach
  you name in expected_failure_modes.
- 3–7 minutes of learner work. One KC only: {kc}.
```

## Pipeline (Claude Code realization)

```
spec ──► GENERATE (Fable agent, blind)          ──► candidate JSON
      ─► MECHANICAL GATE (no LLM):
           ast.parse starter; exec fixtures; honest-solve passes
           run_function_tests; every bare-fixture cheat fails;
           dtype/type-blind equality trap checked (wrap `call` when needed —
           see ids 230/361 for the (dtype.name, value) / (type, value) idiom)
      ─► EXPERT VALIDATOR (separate Fable agent, structured verdict):
           {accept, failure_type, notes, suggested_fix}
           checks: KC match, statement-test agreement, multiple-valid-solution
           traps, Bloom plausibility, leak by paraphrase
      ─► reject → regenerate with suggested_fix appended (max 3), else ABSTAIN
      ─► append to override layer (chatgpt/*.jsonl) → regen chain → diff guard
           (only intended fields change) → audit --gate on merged bank
```

- Orchestrate wide runs with the Workflow tool (one agent per spec,
  validator fan-out per candidate); single questions inline.
- The regen chain + diff-guard + gate steps are the SAME ones used in the
  2026-07-04 hardening pass; they are already scripted and proven.

## Quality rubric for the validator (adapted from Doughty's 6 + failure modes)

1. Statement gives sufficient information, in prose, without the answer
   expression (leakage ban).
2. Exactly one behavior satisfies the tests; alternative correct
   implementations of THAT behavior also pass (no over-constraint).
3. Starter runs, is fixture-only, and compiles.
4. Tests discriminate: do-nothing fails, named-plausible-wrong fails.
5. KC alignment: the shortest passing solution exercises the named KC and
   not a neighbor.
6. Cognitive demand ≥ Apply unless difficulty_rung 1 (ICAP-Constructive:
   the learner produces the transformation, never transcribes it).

## Where to point it first

- The 6 re-authored questions (61, 206, 230, 322, 358, 361) are repaired but
  minimal — good first regeneration targets to compare harness output against
  hand fixes.
- New ERE faded variants and five-rung ladders for atoms with thin coverage.
- CBIT-style parameterized generators (old report Phase 4) for
  broadcasting / rearrange / indexing families: have Fable write the
  *generator program*, gate every emitted variant.

## Measurement before scale-up

Stage the rollout (both reports agree): (i) offline generation + gate;
(ii) blind A/B against existing items (Denny 2023 protocol — a judge panel
rates correctness+helpfulness without knowing which is incumbent);
(iii) small learner pilot watching the existing difficulty-feedback buttons;
(iv) only then IRT-style calibration (Isley 2025) once real response data
accumulates. LM judge panels screen difficulty; they do not replace learner
data (audit-corrected Liu 2026: max ρ=0.537, not the survey's stronger read).
