# Extraction Pipeline

## Current intent

Delta Drills should preserve the exercise contract from source curricula rather than flattening everything into short prompt/output pairs.

This repo now has two extraction layers:

1. `scripts/export_questions_json.py`
   - Exports the existing CSV bank into:
     - `questions.json` for the frontend
     - `questions_structured.json` for curation and future generation
   - Includes NumPy, Einsum, and Einops instead of exporting only NumPy.

2. `scripts/extract_arena_prereqs.py`
   - Extracts structured records from the ARENA prerequisites notebook.
   - Preserves source cell references, starter code, canonical solutions, and image-task metadata.

## Why this matters

The old pipeline lost several pieces of information that matter for coding quality:

- exact function signatures
- starter code
- whether the task is an image transformation
- whether the output is stdout vs a rendered artifact
- source provenance back to the notebook

Without these, Delta Drills cannot reliably preserve ARENA-style exercises.

## Generated artifacts

- `questions.json`
- `questions_structured.json`
- `arena_prereqs_structured.json`

## Recommended next step for ChatGPT batch extraction

Run LLM extraction over *structured source records*, not raw notebooks. The LLM should receive one record at a time and emit:

- normalized prompt text
- function signature
- starter code
- evaluation mode (`stdout`, `function`, `image`)
- canonical solution
- expected output or visual target description
- topic / subtopic labels
- difficulty estimate

Then validate each generated item automatically:

- function exercises: run tests or compare against canonical implementation
- stdout exercises: execute canonical solution and capture output
- image exercises: execute canonical solution and compare produced arrays or saved renders

## Basic usage

```bash
python3 scripts/export_questions_json.py
python3 scripts/extract_arena_prereqs.py
```
