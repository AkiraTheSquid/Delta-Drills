# Question Pipeline And Pedagogy Architecture

## Purpose

This document explains the full question architecture across both systems:

1. the older upstream content-generation pipeline at:
   - `/home/stellar-thread/Applications/pdf_2_problem/Problem Generator`
2. the current Delta Drills app in this repo

The first system explains where much of the original question inventory came from.
The second system explains how that content was converted into runnable coding drills and adaptively served to learners.

This is the right document to read if the question is:

- where did the questions come from?
- how were they pedagogically shaped?
- how do those earlier outputs relate to the current app?
- what columns and schemas matter at each stage?

## The Two Systems

There are two distinct but connected layers.

### 1. Legacy Upstream Generator

The legacy generator is a multi-stage authoring and synthesis pipeline.

Its job was to:

- ingest PDFs and source materials
- split and regroup source content into coherent sections
- generate many candidate questions from those sections
- answer those questions
- filter duplicates
- add rough relative difficulty
- emit a final large question/answer bank

It is not the runtime learner-facing drill engine.

### 2. Current Delta Drills App

The current Delta Drills app is the learner-facing adaptive system.

Its job is to:

- load curated CSV banks
- convert them into function-style drill items
- build starter code and tests
- validate and repair question scaffolds
- serve questions by subtopic
- adapt difficulty from performance and user feedback

It is not the original source-content generator.

## One-Sentence Relationship

The old generator created and refined broad candidate question banks from long-form educational content; the current Delta Drills app takes curated descendants of that content and turns them into executable adaptive coding drills.

## End-To-End Architecture

At the highest level, the full architecture looks like this:

1. source PDFs and reference materials
2. legacy content-distillation pipeline
3. large cleaned Q/A bank with difficulty scores
4. later manual curation into topic-specific Delta Drills CSVs
5. current export / scaffold / validation / repair pipeline
6. current runtime adaptive practice engine

So the true "question pipeline" spans both repos and both eras.

## Part I: Legacy Generator

## High-Level Flow

The legacy generator is a sequential refinement pipeline. Its rough flow is:

1. convert PDFs to markdown
2. clean markdown
3. split markdown into atomic sections
4. merge small sections into larger context windows
5. group related sections into conceptual units
6. resolve grouping conflicts
7. batch grouped sections for model processing
8. merge undersized groups
9. classify sections by procedural vs conceptual balance
10. generate questions from sections
11. convert outputs into structured CSV rows
12. clean malformed question text
13. answer generated questions
14. filter duplicates
15. merge batches
16. filter duplicates again
17. merge again
18. clean numbering and attach provenance
19. choose a champion problem
20. rate all problems relative to the champion
21. assemble final columns

This is a progressive-refinement curriculum pipeline, not a one-shot generator.

## Pedagogical Intent Of The Legacy Generator

The main pedagogical goals were:

### Preserve conceptual context

Questions were generated from real sections, not just keywords. That means the pipeline tried to preserve:

- local conceptual framing
- examples
- definitions
- procedures
- relationships between ideas

### Generate multiple forms of practice

The generator did not treat one section as one question. It tried to produce a local question set with:

- `Basic`
- `Intermediate`
- `Advanced`
- multi-part questions
- far-transfer questions

That means it was trying to create a small curriculum around each source section, not just isolated prompts.

### Balance procedural and conceptual work

Earlier stages classify sections partly in terms of procedural vs conceptual emphasis. That classification then informs how sections should be represented in the generated bank.

Pedagogically, this matters because a section that mostly teaches procedure should not generate exactly the same kind of assessment as a section that mostly teaches conceptual interpretation.

### Encourage transfer, not only paraphrase

The far-transfer formats are important. They show the generator was not just trying to restate the text; it was also trying to force the learner to map the same underlying idea into a slightly different context.

### Separate question generation from answer generation

Questions are generated first, then answered later in a different stage. This improves modularity and allows later filtering based on the quality of the resulting Q/A pair, rather than trusting a single giant prompt.

### Remove redundancy

Duplicate filtering is a pedagogical step, not just data hygiene. A bank with many near-clones gives fake breadth and trains recall of surface form instead of transfer.

### Estimate relative challenge

The later champion-comparison stages give a rough scalar difficulty signal. It is not a rigorous psychometric measure, but it gives enough ordering signal to separate easier and harder items.

## The Key Generation Stage

The most important middle stage is:

- `10. problem_generator`

That stage decides how many questions a section gets, chooses prompt families, and asks the model to emit structured question headers such as:

- `# N (Basic)`
- `# N (Intermediate)`
- `# N (Advanced)`
- `# N (Basic-Multi-Part)`
- `# N (Intermediate-Far-Transfer)`

This is where a lot of the pedagogy is concretely encoded:

- how much coverage a section deserves
- how much of that coverage should be basic vs advanced
- when to scaffold with multiple parts
- when to ask transfer questions

## Legacy Schema Evolution

Following the schemas is the easiest way to understand the pipeline.

### Temporary generated questions

Files like:

- `12. clean_csv_problems/1_temp_question_storage.csv`

use:

- `section`
- `base_question_id`
- `subpart_id`
- `question`

Meaning:

- `section`: source conceptual unit
- `base_question_id`: shared ID for a question family
- `subpart_id`: part number if the question is multi-part, else `NA`
- `question`: student-facing prompt text

At this stage, the system has prompt structure but not answers.

### Question-answer pairs

Files like:

- `14. Filter Questions/1_question_answer_pairs.csv`

use:

- `section`
- `base_question_id`
- `subpart_id`
- `question`
- `answer`

Now the bank is a real Q/A bank.

### Cleaned numbered bank with provenance

File:

- `19. find_champion/clean_1_2_3_question_answer_pairs.csv`

uses:

- `section`
- `section_path`
- `base_question_id`
- `subpart_id`
- `number`
- `question`
- `answer`

Added meaning:

- `section_path`: provenance back to the source section file
- `number`: stable presentation order

### Difficulty-comparison layer

File:

- `20. rate_problem_difficulty/champion_difficulty_ratings.csv`

uses:

- `champion_id`
- `challenger_id`
- `winner`
- `prob_challenger_wins`
- `score`

This is not the bank itself. It is the relative-difficulty layer.

### Final legacy export

File:

- `22. final/gs_q_and_a.csv`

uses:

- `section`
- `base_question_id`
- `subpart_id`
- `number`
- `question`
- `answer`
- `score`

This is the final assembled legacy export:

- cleaned Q/A rows
- stable numbering
- rough difficulty score

## What The Legacy Generator Actually Produced

Pedagogically, the output of the legacy generator is best thought of as:

- a large candidate bank
- organized by source section rather than current app topic/subtopic
- containing single-part and multi-part prompts
- already partly filtered for duplication
- already roughly ordered by challenge
- still not yet in the exact shape needed for executable coding drills

That last point matters. The legacy generator made educational content, but not yet the full Delta Drills runtime object model.

## Part II: Current Delta Drills Pipeline

## Current Source Of Truth

In the current app, the source-of-truth content now lives in local CSVs under:

- `This-Directory-Only/csv files of problems/`

The active bank is narrower and more specialized than the legacy generator output. The current bank is focused on concrete coding subskills such as:

- NumPy
- einsum
- einops

These are organized into explicit topics and subtopics rather than raw source sections.

## Current Content Pipeline

The current committed pipeline code lives under:

- `Local_Deployed_Shared/pipeline/`

Its role is different from the legacy generator.

It does not primarily invent the content from scratch. Instead, it:

- exports current CSV banks into richer JSON formats
- infers task types
- derives starter code
- derives test cases
- validates function-mode exercises
- uses LLM repair passes when scaffolds or tests are broken
- rewrites prompts that leak the answer too directly

This is more like drill normalization and runtime preparation than question invention.

## Current Core Columns

The current CSV banks use columns like:

- `Topic`
- `Subtopic`
- `Question`
- `Answer`
- `Problem difficulty`
- `Output`

These mean:

- `Topic`: broad family such as `Numpy`, `Einsum`, or `Einops`
- `Subtopic`: narrower skill bucket within the topic
- `Question`: learner-facing prompt
- `Answer`: canonical solution or canonical expression
- `Problem difficulty`: scalar numeric difficulty used by the current app
- `Output`: expected runtime output where applicable

These are much closer to the app's runtime needs than the legacy `section / question / answer / score` format.

## Current Derived Runtime Fields

The current app derives additional fields from those CSV columns, including:

- `difficulty_label`
- `primary_library`
- `task_type`
- `function_name`
- `starter_code`
- `test_cases`
- `submission_mode`
- `expected_artifact_type`
- `supports_visual_output`

Those derived fields are what let the app act like a coding drill platform rather than a static Q/A sheet.

## Current Repair And Pedagogy Cleanup

The current app also has a second layer of pedagogy cleanup that the legacy system did not fully solve in runtime form.

Examples:

- function-mode scaffold repair
- broken test-case repair
- prompt rewrites for einops arrow leakage
- prompt rewrites for numpy/einsum prompts that reveal the answer function

So the current pipeline is doing a different kind of pedagogical work:

- less "generate a lot of questions from content"
- more "make the selected questions teachable, runnable, and non-leaky inside the app"

## Part III: Current Delta Drills Runtime Pedagogy

## What The Live App Actually Does

Once the bank is loaded, the runtime app does three pedagogical things:

1. choose the next subtopic
2. choose the next difficulty inside that subtopic
3. choose a question near that target difficulty

It then grades the submission and updates the learner model.

## Runtime Unit Of Adaptation

The current learner model adapts per subtopic, not just globally.

For each user and subtopic, it tracks:

- how many questions have been answered there
- a running performance baseline
- a running correctness rate `p`
- the target difficulty for the next question
- a history of attempts
- which question IDs have already been served

Pedagogically, this means the app is trying to model skill growth locally rather than treating "NumPy" or "Python" as one undifferentiated competence.

## Subtopic Selection Logic

The next subtopic is chosen by a weighted learning-gradient idea.

Each subtopic gets:

- a weight
- a learning-rate estimate
- a gradient equal to `weight * learning_rate`

The app selects the subtopic with the highest gradient.

Interpretation:

- user weights say what matters
- learning rate says where growth is happening
- the app chooses where practice is currently most worthwhile

That is a very different pedagogical function from the legacy generator.

The legacy generator created candidate material.
The runtime app decides what to serve next.

## Difficulty Adaptation Logic

Within each subtopic, the runtime system does a cold start first.

The first three questions are fixed target difficulties:

- `25`
- `50`
- `75`

After that, the app updates:

- `baseline`: a performance-weighted estimate
- `p`: a smoothed correctness rate

Then it computes:

- `target_difficulty = baseline * difficulty_multiplier(p)`

The learner also gives feedback about how much the last result should change the next question:

- `not_much`
- `somewhat`
- `a_lot`

That feedback changes how aggressively the system updates its performance baseline.

Pedagogically, that means the runtime engine is not just reading correctness. It also asks how informative the learner thinks the last result was.

## Question Selection Inside A Subtopic

Once the system knows:

- which subtopic to train
- what target difficulty it wants

it picks a question from that subtopic whose stored difficulty is close to the target.

So the final learner-facing logic is:

1. pick the most worthwhile subtopic
2. estimate the right challenge level there
3. pick a nearby question
4. grade it
5. update the learner state

## Part IV: How The Systems Connect

## What Carried Over

The current app does not literally run the old generator, but several ideas carried over:

- use real educational source material
- value multiple difficulty bands
- care about far transfer and not just recall
- care about removing duplicates and answer leakage
- use scalar difficulty as an organizing signal

## What Changed

The current app changed the representation and the delivery model.

The legacy system was:

- source-section-centered
- long sequential offline batch processing
- broad candidate generation
- final output mostly as Q/A rows plus score

The current app is:

- topic/subtopic-centered
- narrower curated bank
- coding-drill-oriented
- runtime-executable
- adaptively sequenced per learner

## Clean Mental Model

The best way to understand the full architecture is:

- legacy generator = upstream curriculum distillation and bulk question synthesis
- current Delta Drills pipeline = drill normalization, scaffold construction, validation, and repair
- current Delta Drills runtime = adaptive sequencing and learner modeling

All three are part of the full system history, but they solve different problems.

## Final Summary

The full question architecture is not one pipeline but two connected layers.

The older `pdf_2_problem/Problem Generator` system transformed PDFs and reference materials into a large, pedagogy-shaped, deduplicated, roughly difficulty-ranked question/answer bank. It was responsible for breadth, initial pedagogical shaping, and source-driven question creation.

The current Delta Drills app then took curated descendants of that content and restructured them into explicit topic/subtopic coding drills with starter code, tests, canonical solutions, prompt rewrites, validation, and adaptive delivery. It is responsible for runtime teachability, execution, grading, and personalized sequencing.

So if someone asks "what is the architecture of the question system?", the correct answer is:

- an upstream offline generator produced the raw pedagogically-shaped candidate bank
- a downstream adaptive drill system converted selected content into executable practice and decided what to serve each learner next
