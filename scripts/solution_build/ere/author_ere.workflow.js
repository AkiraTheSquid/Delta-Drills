export const meta = {
  name: 'author-ere-tiers',
  description: 'Batch-author + self-validate 3 worked + 3 faded ERE problems per atom (sonnet, checkpointed)',
  phases: [
    { title: 'Author', detail: 'one sonnet agent per ~12-atom batch; writes authored/<atom>.json per atom as it goes' },
  ],
}

const REPO = '/home/stellar-thread/Applications/Delta-Drills-Local'
const ERE = `${REPO}/scripts/solution_build/ere`
const BATCH = 12

let atomIds = args
if (typeof atomIds === 'string') {
  try { atomIds = JSON.parse(atomIds) } catch (e) { atomIds = atomIds.split(/[,\s]+/).filter(Boolean) }
}
if (!Array.isArray(atomIds)) atomIds = (atomIds && atomIds.atomIds) ? atomIds.atomIds : []
if (!atomIds.length) throw new Error('pass atomIds as args (array or comma string)')

const batches = []
for (let i = 0; i < atomIds.length; i += BATCH) batches.push(atomIds.slice(i, i + BATCH))

const SUMMARY = {
  type: 'object',
  required: ['done'],
  additionalProperties: true,
  properties: {
    done: { type: 'array', items: { type: 'string' } },
    failed: { type: 'array', items: { type: 'string' } },
    note: { type: 'string' },
  },
}

const prompt = (atoms) => `Author Expertise-Reversal-Effect tier content for ${atoms.length} Delta Drills atoms.

STYLE: caveman-ultra for ALL your OWN output — reasoning, notes, tool-call commentary, final summary. Drop articles/filler/hedging, fragments OK, abbreviations (fn/impl/req/res), arrows (X → Y), one word where one word works. This keeps your non-deliverable tokens cheap.
EXCEPTION — the learner-facing notebook content you author (concept_md, walkthrough_md, prompt_md) must be NORMAL, clear, well-written teaching English, NOT caveman. Caveman applies only to YOUR thinking + status, never to the content fields.

ATOMS (process each, independently): ${atoms.join(', ')}

For EACH atom, in order:
1. Read ${ERE}/agent_input/<atomId>.json — gives label/definition/domain/subtopic, an example refresher (house voice), and the existing full-difficulty exercises (prompt + canonical solution) defining house style, difficulty band, library (numpy / einops / torch).
2. Author 3 WORKED + 3 FADED examples — NEW problems, similar-but-different to the existing ones (same atom/concept, same difficulty band, same libraries). Not copies, not same numbers. The 3 within each tier distinct from each other.

   WORKED (study-only, fully solved):
   - slug (kebab, ≤6 words), title
   - concept_md: 2-4 sentence refresher (clear teaching English)
   - walkthrough_md: step-by-step WHY-each-step explanation (clear teaching English — the novice reads this instead of solving)
   - solution_code: COMPLETE runnable python. Define the computation, then exercise it + print() a small result. NO NotImplementedError/blanks. Runs top-to-bottom with standard imports present (numpy as np, torch as t, einops + rearrange/reduce/repeat, seeds set). Reseed inside before any draw.

   FADED (completion problem — most given, ONE step blanked):
   - slug, title
   - concept_md: 2-4 sentence refresher
   - prompt_md: task + what to complete
   - blank_description: one sentence naming the blanked step
   - scaffold_code: full solution with ONE key step replaced by a SYNTACTICALLY VALID stub. Use \`raise NotImplementedError()  # TODO: <blank_description>\` as a standalone statement, OR \`var = None  # TODO: <blank_description>\` (standalone). NEVER \`x = raise ...\` (syntax error). Everything else stays filled.
   - reference_fill: COMPLETE correct code (scaffold with blank filled) — runs top-to-bottom, passes the test.
   - test_code: defines EXACTLY \`def _test():\` with asserts checking the filled solution vs an independent ground truth. Do NOT call _test(). Asserts pass when reference_fill in place.

   Correctness bar: every solution_code/reference_fill/test genuinely correct torch/numpy/einops. Verify shapes/dtypes/broadcasting/seeding. Grader seeds then runs setup WITHOUT reseeding → reseed inside before draws. Self-check each before writing.

3. CHECKPOINT: immediately Write ${ERE}/authored/<atomId>.json = {"atomId":"<id>","worked":[3],"faded":[3]} (valid JSON, escape newlines). Write it RIGHT AFTER finishing that atom, BEFORE starting the next — so progress survives interruption. Then move to next atom.

After all atoms: return {done:[atomIds written], failed:[atomIds you could not complete], note:"<terse>"}.`

phase('Author')
const results = await parallel(batches.map((b) => () =>
  agent(prompt(b), { label: `author:${b[0]}+${b.length - 1}`, phase: 'Author', model: 'sonnet', schema: SUMMARY })
))

const done = results.filter(Boolean).flatMap((r) => r.done || [])
const failed = results.filter(Boolean).flatMap((r) => r.failed || [])
log(`batches ${batches.length} · atoms authored ${done.length} · failed ${failed.length}`)
return { batches: batches.length, authored: done.length, failed }
