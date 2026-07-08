# Visual Review

Local-only review app for Delta Drills image-output questions.

Run from the repo root:

```bash
python3 This-Directory-Only/visual_review/server.py --regen --port 8765
```

Open `http://127.0.0.1:8765`.

The server renders every `supports_visual_output` / `expected_artifact_type=image`
question from `This-Directory-Only/questions_full.json` into
`visual_review/generated/`. Review decisions are saved locally:

- `review_state.json`: all current statuses and notes.
- `visual_malformed_flags.jsonl`: machine-readable rows for questions marked
  `Needs check`.
- `visual_malformed_flags.md`: human-readable flag log.

Generated files and review state are intentionally local audit artifacts.
