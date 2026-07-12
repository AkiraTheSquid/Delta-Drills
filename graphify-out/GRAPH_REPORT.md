# Graph Report - Delta-Drills-Deployed  (2026-07-11)

## Corpus Check
- 886 files · ~9,700,471 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 122 nodes · 149 edges · 5 communities
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d679ee1a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- dom.js
- audit_question_bank.py
- PracticeAPI
- timer.js

## God Nodes (most connected - your core abstractions)
1. `audit()` - 9 edges
2. `scan_function_question()` - 7 edges
3. `PracticeAPI` - 7 edges
4. `check_atom_graph()` - 5 edges
5. `startTimer()` - 5 edges
6. `base_namespace()` - 4 edges
7. `selftest_atom_graph()` - 4 edges
8. `_loadNextPracticeQuestion()` - 4 edges
9. `initPractice()` - 4 edges
10. `updateTimerDisplay()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `refreshPracticeQuestionForPreferences()` --references--> `PracticeAPI`  [EXTRACTED]
  Local_Deployed_Shared/practice/init.js → Local_Deployed_Shared/practice/api.js
- `_loadNextPracticeQuestion()` --references--> `PracticeAPI`  [EXTRACTED]
  Local_Deployed_Shared/practice/events.js → Local_Deployed_Shared/practice/api.js
- `_notifyIfPlacementDone()` --references--> `PracticeAPI`  [EXTRACTED]
  Local_Deployed_Shared/practice/events.js → Local_Deployed_Shared/practice/api.js
- `_rateTorchAndAdvance()` --references--> `PracticeAPI`  [EXTRACTED]
  Local_Deployed_Shared/practice/events.js → Local_Deployed_Shared/practice/api.js
- `refreshPlacementStartBtn()` --references--> `PracticeAPI`  [EXTRACTED]
  Local_Deployed_Shared/practice/events.js → Local_Deployed_Shared/practice/api.js

## Import Cycles
- None detected.

## Communities (5 total, 0 thin omitted)

### Community 0 - "dom.js"
Cohesion: 0.03
Nodes (72): aiExplanationSection, aiExplanationText, answerAids, codeEditor, colabSolutionLink, coldStartBadge, coldStartLabel, coldStartNote (+64 more)

### Community 1 - "audit_question_bank.py"
Cohesion: 0.13
Nodes (24): audit(), base_namespace(), check_atom_graph(), check_starter_syntax(), check_stdout_expected(), check_todo_answer_leak(), confirm_gameable(), exec_seeded() (+16 more)

### Community 2 - "PracticeAPI"
Cohesion: 0.22
Nodes (11): PracticeAPI, _loadNextPracticeQuestion(), _notifyIfPlacementDone(), _rateTorchAndAdvance(), NOTE: ARENA unlock interstitial does NOT fire on Submit — student, refreshPlacementStartBtn(), _resetProblemFeedbackRow(), initPractice() (+3 more)

### Community 3 - "timer.js"
Cohesion: 0.62
Nodes (6): formatTimer(), parseTimerInput(), resetTimerToInput(), startTimer(), stopTimer(), updateTimerDisplay()

## Knowledge Gaps
- **72 isolated node(s):** `timedModeToggle`, `timerControls`, `timerInput`, `questionMetaTop`, `questionNumber` (+67 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `Mirror of the grading harness's _delta_equal (+ torch, which the     harness nev`, `Namespace seeded with the grading harness's own CODE_PREAMBLE     (numpy, einops`, `Exec code the way the grading harness does: seeded, stdout swallowed.` to the rest of the system?**
  _82 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `dom.js` be split into smaller, more focused modules?**
  _Cohesion score 0.0273972602739726 - nodes in this community are weakly interconnected._
- **Should `audit_question_bank.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._