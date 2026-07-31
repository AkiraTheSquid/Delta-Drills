/* ================================================================
   PRACTICE DOM — element references
   ================================================================ */

// Resumable-session controls (setup/resume panel + in-session status; timer.js)
const sessionSetupPanel = document.getElementById("practice-session-setup");
const sessionQuestionCountInput = document.getElementById("session-question-count");
const sessionAnswerTimeInput = document.getElementById("session-answer-time");
const sessionReviewTimeInput = document.getElementById("session-review-time");
const sessionTimeEstimate = document.getElementById("session-time-estimate");
const sessionStartBtn = document.getElementById("session-start-btn");
const sessionSummary = document.getElementById("session-summary");
const sessionResumePanel = document.getElementById("session-resume-panel");
const sessionResumeSummary = document.getElementById("session-resume-summary");
const sessionResumeBtn = document.getElementById("session-resume-btn");
const sessionDiscardBtn = document.getElementById("session-discard-btn");
const sessionStatusRow = document.getElementById("session-status-row");
const sessionProgressLabel = document.getElementById("session-progress");
const sessionPhaseLabel = document.getElementById("session-phase");
const sessionCountdown = document.getElementById("session-countdown");
const sessionPauseBtn = document.getElementById("session-pause-btn");
const sessionEndBtn = document.getElementById("session-end-btn");
const questionMetaTop = document.getElementById("question-meta-top");
const questionNumber = document.getElementById("question-number");
const questionText = document.getElementById("question-text");
const questionImports = document.getElementById("question-imports");
const questionImportsList = document.getElementById("question-imports-list");
const questionVisual = document.getElementById("question-visual");
const questionVisualNote = document.getElementById("question-visual-note");
const questionVisualCanvas = document.getElementById("question-visual-canvas");
const subtopicLabel = document.getElementById("subtopic-label");
const difficultyLabel = document.getElementById("difficulty-label");
const questionIdChip = document.getElementById("question-id-chip");
const targetDifficultyTitle = document.getElementById("target-difficulty-title");
const targetDifficultyFill = document.getElementById("target-difficulty-fill");
const targetDifficultyDelta = document.getElementById("target-difficulty-delta");
const targetDifficultyMarkerOld = document.getElementById("target-difficulty-marker-old");
const targetDifficultyNumberOld = document.getElementById("target-difficulty-number-old");
const targetDifficultyMarkerNew = document.getElementById("target-difficulty-marker-new");
const targetDifficultyNumberNew = document.getElementById("target-difficulty-number-new");
const targetDifficultyValue = document.getElementById("target-difficulty-value");
const practiceSubmitArea = document.getElementById("practice-submit-area");
// The two self-report buttons. NO is first because it is also the timeout
// default — practice/timer.js clicks it when the answer countdown expires.
const selfReportNoBtn = document.getElementById("self-report-no");
const selfReportYesBtn = document.getElementById("self-report-yes");
const practiceSkipBtn = document.getElementById("practice-skip-btn");
const practiceDontKnowBtn = document.getElementById("practice-dontknow-btn");
const placementStartBtn = document.getElementById("placement-start-btn");
// Where this problem is worked. Every question routes to Colab now, so this
// card is part of the normal flow rather than a per-library special case.
const colabCard = document.getElementById("colab-card");
const colabCardLabel = document.getElementById("colab-card-label");
const colabOpenLink = document.getElementById("colab-open-link");
const colabCardNote = document.getElementById("colab-card-note");
// Difficulty-rating helpers (clear default + "missed one concrete thing").
const feedbackHelp = document.getElementById("feedback-help");
const missedFactRow = document.getElementById("missed-fact-row");
const missedFactBtn = document.getElementById("missed-fact-btn");
const missedFactStatus = document.getElementById("missed-fact-status");
const ewmaAccuracy = document.getElementById("ewma-accuracy");
const ewmaAccuracyLabel = document.getElementById("ewma-accuracy-label");
const ewmaAccuracyFill = document.getElementById("ewma-accuracy-fill");
const ewmaAccuracyValue = document.getElementById("ewma-accuracy-value");
const ewmaAccuracyDelta = document.getElementById("ewma-accuracy-delta");
const ewmaAccuracyMarkerOld = document.getElementById("ewma-accuracy-marker-old");
const ewmaAccuracyMarkerNew = document.getElementById("ewma-accuracy-marker-new");
const practiceFeedbackArea = document.getElementById("practice-feedback-area");
const resultBadge = document.getElementById("result-badge");
const overrideRow = document.getElementById("override-row");
const overrideCorrectBtn = document.getElementById("override-correct-btn");
const nextProblemBtn = document.getElementById("next-problem-btn");
const solutionCode = document.getElementById("solution-code");
const aiExplanationSection = document.getElementById("ai-explanation-section");
const aiExplanationText = document.getElementById("ai-explanation-text");
const coldStartBadge = document.getElementById("cold-start-badge");
const coldStartLabel = document.getElementById("cold-start-label");
const coldStartNote = document.getElementById("cold-start-note");
// No codeEditor / runBtn / outputArea / outputVisual* here any more — the
// editor panel was removed on 2026-07-31. Code is written and run in Colab.
// If you are adding one back, you are rebuilding the runner; read
// practice/README.md first.
const feedbackPrompt = document.getElementById("feedback-prompt");
const feedbackButtons = document.querySelectorAll(".feedback-btn");
const problemFeedbackRow = document.getElementById("problem-feedback-row");
const problemFlagButtons = document.querySelectorAll(".problem-flag-btn");
const problemFeedbackNote = document.getElementById("problem-feedback-note");
const problemFeedbackStatus = document.getElementById("problem-feedback-status");
const showHintBtn = document.getElementById("show-hint-btn");
const showAnswerBtn = document.getElementById("show-answer-btn");
const hintSection = document.getElementById("hint-section");
const hintText = document.getElementById("hint-text");
const answerAids = document.getElementById("answer-aids");
const colabSolutionLink = document.getElementById("colab-solution-link");
