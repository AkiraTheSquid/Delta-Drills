# Survey: Primary Sources for LLM-Generated High-Quality CS/Coding Exercises with Pedagogical Rigor

## TL;DR
- The strongest empirical evidence that AI can approach human-authored CS assessment quality comes from a small cluster of computing-education papers (Doughty et al. 2024; Sarsa et al. 2022; Del Carpio Gutierrez et al. 2024) and one large IRT-based field study (Isley et al. 2025); true blinded "expert-parity" in CS coding tasks is real but narrow, and every study still reports concrete failure modes (multiple correct answers, weak distractors, broken tests) that require validation.
- The most valuable extractables are: (a) generate-then-validate pipelines with execution and multi-agent checks (PyTaskSyn; AlphaCode/CodeContests), (b) item-writing rubrics (Haladyna–Downing's 31 rules; Scaria's 9-item rubric; Doughty's 6-criterion rubric), and (c) pedagogical frameworks to raise cognitive demand (revised Bloom's taxonomy, ICAP, PRIMM, Parsons/faded scaffolding).
- For the ARENA/mech-interp use case, combine ARENA's own exercise-design conventions (exercise/solution pairs, test functions, difficulty + importance ratings, time annotations) with an execution-validated generation loop and a rubric that explicitly bans the four failure modes (answer leakage, mixed exposition/task, non-runnable scaffolds, low cognitive demand).

## Key Findings
- **Expert-parity in CS is demonstrated but qualified.** Doughty et al. (2024) found GPT-4 MCQs "comparable" overall to instructor MCQs and *significantly better* aligned to learning objectives, but statistically worse on single-correct-answer and distractor quality. This is the closest thing to a CS-specific parity result with real numbers, and it is honest about defects. Note the raters were not explicitly blinded to item source.
- **Execution-based validation is the single highest-leverage technique.** AlphaCode's CodeContests cut its own false-positive rate (incorrect solutions passing tests) from 62% to 4% by generating extra tests and filtering; PyTaskSyn uses expert/tutor/student agents to check that a reference solution passes tests before showing a task. Both directly address the "reference solution doesn't run / tests don't discriminate" failure mode.
- **Rubrics and taxonomies are the raw material for a system prompt.** Haladyna–Downing's 31-rule item-writing taxonomy, Scaria et al.'s validated 9-item question-quality rubric, and Doughty's 6-criterion CS rubric can be transcribed nearly verbatim into a generation-and-validation checklist.
- **The field warns against overselling.** Multiple medical/health-education studies found AI MCQs worse than experts; Sarsa et al. found only ~31% of generated exercises had a sample solution that passed the generated tests. The literature supports "AI-assisted, human/execution-validated," not "AI-autonomous."

## Details

### Theme 1 — Empirical studies of LLM exercise/question generation with outcome data

**1. Sarsa, Denny, Hellas & Leinonen (2022). "Automatic Generation of Programming Exercises and Code Explanations Using Large Language Models." ICER '22, pp. 27–43.**
- Link: https://arxiv.org/abs/2206.11861 · DOI: 10.1145/3501385.3543957 · Open PDF: https://arxiv.org/pdf/2206.11861
- Evidence (verified from full text): The foundational paper. Using OpenAI Codex (code-davinci-001, GPT-3-based), generated 240 programming exercises. Results: 84.6% (203/240) had a sample solution; of those, 89.7% (182/203) executed without errors; 70.8% (170/240) included automated tests; but only **30.9% (51/165 with both a solution and tests) had a sample solution that passed its own generated tests**. Of the 51 that passed, average statement coverage was 98.0% (48 at 100%). A qualitative sample of 120 exercises: 75.0% sensible, 81.8% novel, 76.7% had a matching sample solution. Code explanations: 90% explained all parts; 117/174 (67.2%) line-by-line explanations correct. The authors state the exercises were largely "novel and sensible" but "Much less impressive was the quality of the test suites and the performance of the code against those tests."
- What to extract: (a) The failure statistic — roughly 69% of generated exercises had a solution that did NOT pass its own tests — as the empirical justification for a mandatory execution-validation gate. (b) Their finding that supplying keywords (concept + contextual theme) reliably steers content: adopt a structured input schema {concept, theme, difficulty}. (c) The canonical generated artifact bundle: exercise + sample solution + test cases + explanation. Caveat: uses Codex (2022-era), so treat the raw pass rates as a floor, not current capability.

**2. Doughty, Wan, et al. (2024). "A Comparative Study of AI-Generated (GPT-4) and Human-crafted MCQs in Programming Education." ACE '24, pp. 114–123.**
- Link: DOI 10.1145/3636243.3636256 · arXiv: https://arxiv.org/abs/2312.03173 · ResearchGate: https://www.researchgate.net/publication/377774502
- Evidence (verified from full text): Evaluated **651 GPT-4-generated and 449 human-crafted MCQs aligned to 246 learning objectives from 6 Python courses**. Model: gpt-4-0613, temperature 1.0, max 2,000 tokens. 13 raters (7 students, 6 CS instructors, all authors) produced 3,076 annotations against a 6-criterion rubric. Statistics via Fisher's exact test; inter-rater Fleiss κ=0.22 (fair), Gwet's AC1 0.62–0.96 per item. Findings: generated MCQs "comparable" quality overall; **81.7% passed all evaluation criteria** (so fewer than 1 in 5 need instructor edits); GPT MCQs significantly WORSE on single-correct-answer (**4.9% had multiple correct answers vs 1.1% human, p=0.002**) and distractor quality (**4.0% had distractors that gave away the answer vs 0.9% human, p=0.002**); but GPT MCQs significantly BETTER aligned to LOs (**p<10⁻⁹**). Student raters answered 71.5% of generated MCQs correctly (62.6% human); instructors 80.1% (76.3% human). Raters were **not explicitly stated to be blinded** to item source; not deployed with real students (pedagogical impact untested), and difficulty was not compared.
- What to extract: (a) The full 6-criterion rubric — sufficient information in clear language; single correct answer; unique choices; no obviously wrong choice (distractor plausibility); syntactically/logically correct code; alignment with the learning objective — transcribe into the validation checklist. (b) The generation pipeline: predict the Bloom's-taxonomy level of the LO (they fine-tuned per-level BERT classifiers on 21,380 LOs from 5,558 courses), map level → question type (recall, fill-in-the-blank, identify-correct-output, trace/analyze code, scenario), inject "MCQ principles" (distractors should be plausible and limited, often just two), and output JSON {stem, key, distractors} with a code_in_stem boolean, exactly 2 distractors + 1 key. (c) The specific defect rates (4.9% multiple-correct; 4.0% answer-giveaway) as the automated checks your validator must run — these map directly to the "answer leakage/triviality" failure mode.

**3. Del Carpio Gutierrez, Denny & Luxton-Reilly (2024). "Evaluating Automatically Generated Contextualised Programming Exercises." SIGCSE '24, pp. 289–295.**
- Link: DOI 10.1145/3626252.3630863
- Evidence: Used GPT-4 to generate 500 contextualised programming exercises; reported exercise quality was high and that personalising the contextual framing to student interests is feasible without degrading quality. A follow-up (PuzzleMakerPy / "Automating Personalized Parsons Problems," ITiCSE '24, DOI 10.1145/3649217.3653568) deployed the tool in a large intro course; students found the ability to personalise context and topic engaging and useful.
- What to extract: the contextualization method (theme + concept parameters) and evidence that theming preserves quality — useful for varying mech-interp exercises without losing rigor.

**4. Denny, Khosravi, Hellas, Leinonen & Sarsa (2023). "Can We Trust AI-Generated Educational Content? Comparative Analysis of Human and AI-Generated Learning Resources." arXiv:2306.10509.**
- Link: https://arxiv.org/abs/2306.10509
- Evidence: Blind student evaluation of correctness and helpfulness in an intro programming context; AI-generated resources rated equivalent in quality to peer (learnersourced) resources after both were seeded with identical exemplars.
- What to extract: the blind A/B evaluation protocol (students rate correctness + helpfulness) as a template for validating your generated exercises against a human baseline.

**5. Isley, Gilbert, et al. (2025). "Assessing the Quality of AI-Generated Exams: A Large-Scale Field Study." arXiv:2508.08314.**
- Link: https://arxiv.org/html/2508.08314v1 · Code/data: https://github.com/calisley/ai_exams
- Evidence: Largest field study to date — 91 classes (including CS, mathematics, chemistry), ~1,700 students. Introduced an iterative refinement strategy (Self-Refine-style generate→critique→revise). IRT analysis found AI-generated questions performed on par with expert-created standardized (AP) exam items on difficulty and discrimination; AI items were on average somewhat easier but more discriminating. 71 courses (~1,200 students) tested the generated items; 20 courses (~500 students) benchmarked against AP items.
- What to extract: (a) the iterative-refinement (self-critique/revision) loop as the core generation architecture. (b) The IRT evaluation methodology (difficulty + discrimination) as the gold-standard way to prove item quality with real student data. (c) The prompt templates and test-generation pseudocode in their appendix. This is the best evidence that self-critique loops yield psychometrically sound items — though it spans mixed STEM, not CS coding tasks specifically.

**6. Scaria, Chenna & Subramani (2024). "Automated Educational Question Generation at Different Bloom's Skill Levels Using Large Language Models: Strategies and Evaluation." AIED 2024, LNCS 14830, pp. 165–179. arXiv:2408.04394.**
- Link: https://arxiv.org/abs/2408.04394 · Code: https://github.com/nicyscaria/AEQG_Blooms_Evaluation_LLMs
- Evidence: Compared 5 LLMs (Mistral 7B, Llama2 70B, PaLM 2, GPT-3.5, GPT-4), evaluated with a validated hierarchical 9-item rubric (e.g., Understandable; topic relevance; grammatical correctness; clarity; answerability; appropriate difficulty; Bloom-level alignment). Found that reliably hitting higher Bloom levels requires multi-stage/complex prompting; quality varies substantially by model and cognitive level.
- What to extract: the 9-item rubric (transcribe verbatim) and the finding that higher cognitive levels require explicit multi-stage prompting — directly addresses the "low cognitive demand" failure mode.

### Theme 2 — Pedagogical frameworks and item-writing rubrics worth encoding into prompts

**7. Haladyna, Downing & Rodriguez (2002). "A Review of Multiple-Choice Item-Writing Guidelines for Classroom Assessment." Applied Measurement in Education 15(3), 309–334** (and the original Haladyna & Downing 1989 taxonomy of 43 rules, Applied Measurement in Education 2(1), 37–50).
- Links: https://www.tandfonline.com/doi/abs/10.1207/S15324818AME1503_5 · ERIC: https://eric.ed.gov/?id=EJ660246 · Open one-page summary: https://www.schreyerinstitute.psu.edu/pdf/Multiple_Choice_Item_Writing_Rules.pdf
- Evidence: The authoritative, consensus-validated taxonomy of 31 item-writing guidelines (derived from 27 textbooks + 27 empirical studies). Rules include: base each item on one instructional objective; avoid trick items; keep options homogeneous in content, length, and complexity; avoid "all/none of the above"; make all distractors plausible; minimize examinee reading time; format vertically.
- What to extract: the full 31-rule list as hard constraints in the system prompt. The rules on option length/complexity homogeneity directly prevent answer-leakage (the "longest/most-detailed option is the key" tell).

**8. Anderson & Krathwohl (2001), revised Bloom's taxonomy.**
- Six cognitive levels: remember, understand, apply, analyze, evaluate, create.
- What to extract: use as the "target cognitive level" parameter. To fight low cognitive demand, require exercises at apply/analyze or higher (not remember/transcribe). Both Doughty and Scaria operationalize this with level→question-type mappings you can copy.

**9. Chi & Wylie (2014), the ICAP framework (Educational Psychologist 49(4)).** Passive < Active < Constructive < Interactive engagement.
- Open explainer PDF: https://www.unh.edu/teaching-learning-resource-hub/sites/default/files/media/2023-05/itow-applying-the-icap-framework-to-improve-classroom-learning-chi-boucher.pdf
- Evidence: In their materials-science study, Chi & Wylie report improved learning of roughly 8–10% with each progressive step up the engagement hierarchy (ordered I>C>A>P).
- What to extract: define a "good exercise" as one requiring Constructive engagement (the student generates something beyond what is given) rather than merely Active (copying/transcribing). This is the theoretical grounding for banning "just tell the student exactly what to type."

**10. Sentance & Waite (2017), PRIMM (Predict–Run–Investigate–Modify–Make). WiPSCE '17.**
- Open PDF: https://kclpure.kcl.ac.uk/portal/files/79583213/version_pure_primm_1.pdf · Project page: https://computingeducationresearch.org/projects/primm/
- What to extract: PRIMM's staged progression as a template for scaffolded exercise sequences — start with Predict/Investigate on working code, then Modify, then Make. Maps naturally onto faded scaffolding.

**11. Ericson, Denny, Prather, et al. (2022). "Parsons Problems and Beyond: Systematic Literature Review and Empirical Study Designs." ITiCSE-WGR '22, pp. 191–234.**
- Link: DOI 10.1145/3571785.3574127 · Open PDF: https://juholeinonen.com/assets/pdf/ericson2022parsons.pdf
- Evidence: Comprehensive review of Parsons problems (drag-and-drop code ordering) as low-cognitive-load scaffolding, documenting learning benefits and efficiency gains versus writing code from scratch.
- What to extract: Parsons and faded-Parsons problems as a scaffolding format between worked examples and free code-writing. See also Weinman et al. (Faded Parsons Problems, CHI 2021) and PrairieLearn's auto-gradable faded-Parsons generation (SIGCSE 2024, DOI 10.1145/3626252.3630786) for how to auto-grade them.

### Theme 3 — Generation+validation pipelines and execution-based quality control

**12. Nguyen, Pădurean, Gotovos, Tschiatschek & Singla (2025). "Synthesizing High-Quality Programming Tasks with LLM-based Expert and Student Agents" (PyTaskSyn). arXiv:2504.07655.**
- Link: https://arxiv.org/pdf/2504.07655
- Evidence (verified from full text): Two-stage pipeline — generate task (description + test suite), then validate via simulated expert, tutor, and student agents. Explicitly motivated by the finding that single agents self-correct poorly: "research has shown that they struggle with self-correction. These limitations in single-agent validation motivated our multi-agent based approach." Validation uses SIMEXPERT, SIMTUTOR, and SIMSTUDENT agents built on GPT-4o (strong) and GPT-4o-mini (weak); the pipeline retries up to N times and abstains if validation fails. Reports significant precision improvement over single-agent baselines while maintaining coverage; user studies show synthesized tasks match expert-created quality.
- What to extract: (a) the two-stage architecture and the metric definitions — Coverage (% of times a task is delivered) and Precision (% of delivered tasks that are high-quality). (b) The expert quality-assessment procedure: expert formulates a solution, verifies the tests validate it including base and corner cases, verifies concept coverage, and checks description comprehensibility — a ready-made validation checklist. (c) The "abstain and retry up to N times" design for when validation fails.

**13. Li et al. (2022). "Competition-Level Code Generation with AlphaCode" (introduces CodeContests). arXiv:2203.07814.**
- Link: https://arxiv.org/pdf/2203.07814
- Evidence (verified verbatim, §3.2.1): "generated tests and filtering reduced our false positive rates from 62% to 4%" (the widely cited ~30–60% range refers to prior datasets APPS/HumanEval, not CodeContests' own starting point). Method: generate additional test cases by mutating existing inputs (bit flips on binary inputs, randomly incrementing/decrementing integers, swapping/changing string characters); mutated inputs are "verified by running 30 correct solutions on them, and checking that all solutions produce the same output"; then keep "only problems with at least 5 hidden or generated test cases that result in at least 2 different outputs. This ensures a model cannot trivially solve problems by always outputting a constant."
- What to extract: the concrete test-hardening recipe — mutation-based test generation, cross-checking mutated outputs against multiple reference solutions, and the "≥5 tests, ≥2 distinct outputs" coverage filter. Directly addresses the "tests don't discriminate / answer is trivial" failure modes.

**14. Wei, Wang, Liu, Ding & Zhang (2024). "Magicoder: Empowering Code Generation with OSS-Instruct." ICML 2024. arXiv:2312.02120.** and **Luo et al. (2023), "WizardCoder" (Code Evol-Instruct). ICLR 2024. arXiv:2306.08568.**
- Links: https://arxiv.org/abs/2312.02120 · https://arxiv.org/abs/2306.08568
- Evidence: OSS-Instruct seeds problem generation with real open-source code snippets to reduce LLM bias and increase realism/diversity; Code Evol-Instruct systematically increases problem complexity via explicit operators (add constraints, replace broad requirements with detailed ones, extend reasoning steps, add deceptive/debugging elements, increase time/space complexity constraints). These are training-data-synthesis papers, not education studies — their "quality" is downstream benchmark pass@1 (e.g., MagicoderS-CL-7B reached 66.5 vs ChatGPT's 65.9 on HumanEval+), not pedagogical value.
- What to extract: (a) OSS-Instruct's "seed with real code" idea — for ARENA, seed generation with real snippets from TransformerLens/PyTorch. (b) Evol-Instruct's explicit complexity-increasing operators as a menu for raising exercise difficulty and cognitive demand. Use the heuristics, not the quality claims.

**15. Moore, Nguyen, Chen & Stamper (2023). "Assessing the Quality of Multiple-Choice Questions Using GPT-4 and Rule-Based Methods." EC-TEL 2023, LNCS 14200, pp. 229–245.**
- Evidence: Combines rule-based checks (item-writing flaw detection) with GPT-4 evaluation of MCQ quality.
- What to extract: the automatic item-flaw detection rules (a computable subset of Haladyna–Downing) for a fast pre-filter before human or LLM review.

### Theme 4 — Practitioner curricula and exercise design patterns (ARENA / mech-interp)

**16. ARENA 3.0 curriculum (Callum McDougall).**
- Repo: https://github.com/callummcdougall/ARENA_3.0 · Site: https://www.arena.education/curriculum · Chapter 1 (transformers/interp) home file: https://github.com/callummcdougall/arena_3.0/blob/main/chapter1_transformer_interp/instructions/Home.py
- Evidence/conventions (from repo and community docs): Exercises isolate a small concept; every exercise ships with a provided solution; exercises carry difficulty ratings and importance ratings (a widely shared community guide notes importance ratings should drive skip decisions while difficulty ratings should not); content is annotated with estimated time; everything is built around PyTorch and TransformerLens; sections are tagged compulsory vs optional/bonus (e.g., §1.1–1.2 compulsory). Designed for driver/navigator pair-programming. This is practitioner documentation, not a peer-reviewed learning-outcomes study.
- What to extract: replicate ARENA's exercise artifact schema — {concept, difficulty rating, importance rating, estimated time, exercise text, starter/scaffold code, solution, test function, hints}. Match ARENA's house style so generated exercises drop into the curriculum. Use TransformerLens/PyTorch as the execution environment for validation.

**17. Neel Nanda, "200 Concrete Open Problems in Mechanistic Interpretability" and the "Getting Started"/Quickstart guides.**
- Links: https://www.neelnanda.io/concrete-open-problems · https://www.lesswrong.com/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability · Quickstart: https://www.neelnanda.io/mechanistic-interpretability/quickstart-old
- Evidence: A curated, difficulty-rated (A/B/…) list of concrete mech-interp research problems, organized by category (e.g., interpreting algorithmic problems, toy language models).
- What to extract: use as a domain-specific source of authentic problem seeds and difficulty calibration for mech-interp exercises; the A/B difficulty-grading convention is a model for rating generated exercises.

**18. PrairieLearn (West, Herman & Zilles, 2015; Python autograder docs).**
- Docs: https://prairielearn.readthedocs.io/en/latest/python-grader/ · Research case studies: https://www.prairielearn.com/research
- Evidence: Production autograding platform; the Python external grader runs student code in a sandbox against instructor test functions, supports disallowing specific library functions (Feedback.not_allowed), and supports algorithmically generated question variants.
- What to extract: the autograder architecture (external grading image, names_from_user contract, per-question test files) as a concrete target format for generated test functions, and the sandbox execution model as the validation harness for runnability.

### Theme 5 — Prioritized extraction guide

**Reading order for the downstream AI:**
1. Sarsa et al. 2022 (arXiv:2206.11861) — the canonical artifact bundle and the empirical case that validation is mandatory (only ~31% passed their own tests).
2. Doughty et al. 2024 (arXiv:2312.03173) — the CS-specific 6-criterion rubric, the Bloom→question-type pipeline, and the exact defect rates to check for.
3. PyTaskSyn (arXiv:2504.07655) — the generate-then-validate multi-agent architecture and expert quality-assessment procedure.
4. Isley et al. 2025 (arXiv:2508.08314) — the self-critique iterative-refinement loop and IRT evaluation.
5. AlphaCode/CodeContests (arXiv:2203.07814) — the test-hardening recipe (62%→4% false positives).
6. Haladyna–Downing (2002) + Scaria et al. 9-item rubric (arXiv:2408.04394) — the rubric text to transcribe.
7. Bloom / ICAP / PRIMM / Parsons — cognitive-demand and scaffolding frameworks.
8. ARENA 3.0 repo + Nanda's open problems — house style, artifact schema, and domain-specific seeds.

**How to turn these into a generation system prompt + quality rubric + validation checklist (mapping to the four failure modes):**

- **(a) Answer leakage / triviality** → Encode Haladyna–Downing's rules on option homogeneity plus Doughty's defect checks (no distractor that is actually correct — GPT-4 baseline 4.9%; no distractor that gives away the key — baseline 4.0%). For code exercises, forbid including the reference solution or the exact API-call sequence in the task statement. Require ICAP-Constructive engagement. Automated check: does the prompt text contain solution tokens? Is the answer inferable without reasoning?
- **(b) Mixed exposition/code/task in one block** → Enforce ARENA's structured artifact schema with separated, labeled fields (background/exposition, task statement, starter code, hints, solution, tests). Automated check: are exposition, task, and code in distinct labeled sections?
- **(c) Non-runnable scaffold / reference solution** → Adopt the PyTaskSyn + AlphaCode execution-validation approach: the reference solution MUST run in the target environment (a PyTorch/TransformerLens sandbox) and pass all tests; generate ≥5 discriminating tests with ≥2 distinct outputs; cross-check outputs against multiple reference solutions; abstain and retry up to N times on failure. This directly fixes "import torch times out in the sandbox" by executing in the real environment before delivery.
- **(d) Low cognitive demand** → Set the target Bloom level to apply/analyze or higher; use Evol-Instruct complexity operators; require the exercise to demand reasoning/implementation rather than transcription; use PRIMM and faded-Parsons scaffolding so difficulty fades appropriately. Automated check: classify the generated exercise's Bloom level and reject if it is mere recall/transcription (Scaria shows this needs explicit multi-stage prompting).

## Recommendations
1. **Hard-code two rubrics into the generation prompt first:** Haladyna–Downing's 31 rules (for any MCQ) and Doughty's 6 criteria (for CS specifically). Threshold to change: if generated items still fail the single-correct-answer check more than ~5% of the time (Doughty's GPT-4 baseline was 4.9%), add a dedicated distractor-validation agent.
2. **Make execution validation non-negotiable.** Implement the PyTaskSyn generate→validate loop with a real PyTorch/TransformerLens sandbox and AlphaCode-style test hardening. Benchmark: the reference solution must pass 100% of ≥5 discriminating tests (≥2 distinct outputs); abstain if not. This is the concrete fix for the sandbox/verifiability failure mode.
3. **Adopt a self-critique refinement loop** (Isley et al.): generate → LLM critiques against the rubric → revise, for at least one cycle. Evaluate a sample with IRT (difficulty + discrimination) once you have student response data.
4. **Enforce ARENA's artifact schema** so exercises are drop-in and never mix exposition/task/code; rate every exercise for difficulty + importance + estimated time.
5. **Set cognitive-demand floors** using Bloom/ICAP and reject transcription-level tasks; use PRIMM/Parsons for scaffolded sequences.
6. **Stage the rollout:** (i) offline generation with execution validation + automated rubric checks; (ii) instructor spot-review of a sample (Doughty found fewer than 1 in 5 items need edits); (iii) small student pilot with blind A/B rating (Denny 2023 protocol); (iv) IRT analysis at scale (Isley). Advance a stage only when the prior stage's defect rate is acceptable.

## Caveats
- **True blinded expert-parity in CS is rare and qualified.** Doughty et al. did NOT explicitly blind raters to item source, and found GPT-4 items significantly worse on two of five quality criteria even while "comparable" overall. The strongest parity claim (Isley et al., IRT on-par with AP items) spans mixed STEM, not CS coding tasks specifically.
- **Contrary evidence exists and is substantial.** Remick et al. (2024), "ChatGPT 3.5 fails to write appropriate multiple choice practice exam questions," *Academic Pathology* 11(1), found ChatGPT-3.5 produced fully correct MCQs with explanations in only 32% of cases (19/60) — "a grade of 32% would be considered failing." A dental-education benchmarking study in *Scientific Reports* (2025) found instructor MCQs rated significantly higher than ChatGPT-3.5 MCQs (t(545)=19.22, p<0.001; instructor person-reliability 0.845 vs 0.778). Model generation matters enormously — Codex/GPT-3.5-era results badly understate GPT-4/o1-era capability, so weight recent, higher-model studies more heavily.
- **The training-data synthesis papers (Magicoder, WizardCoder) are not education studies.** Their quality is measured by downstream benchmark pass@1, not pedagogical value; extract their generation heuristics, not their quality claims.
- **ARENA-specific evidence is practitioner documentation, not peer-reviewed evaluation.** ARENA's design conventions are well-regarded community practice with no published learning-outcomes study; treat them as a style/format standard, not validated pedagogy.
- **Verification note:** All headline statistics above were checked against the primary-source full texts (Sarsa 2022; Doughty 2024; PyTaskSyn 2025; AlphaCode 2022; Isley 2025; Scaria 2024). A few very recent (2025–2026) papers surfaced only in reference lists and were not independently opened; they are not relied upon for any numeric claim here.