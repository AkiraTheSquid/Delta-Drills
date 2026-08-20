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
const practiceSubmitArea = document.getElementById("practice-submit-area");
const practiceSubmitBtn = document.getElementById("practice-submit-btn");
const practiceSkipBtn = document.getElementById("practice-skip-btn");
const practiceDontKnowBtn = document.getElementById("practice-dontknow-btn");
const placementStartBtn = document.getElementById("placement-start-btn");
// Torch-drill Colab routing (torch can't run in the in-app sandbox).
const torchColabNotice = document.getElementById("torch-colab-notice");
const torchColabLink = document.getElementById("torch-colab-link");
const torchSolutionLink = document.getElementById("torch-solution-link");
const torchRateSolved = document.getElementById("torch-rate-solved");
const torchRateLookedUp = document.getElementById("torch-rate-lookedup");
// Difficulty-rating helpers (clear default + "missed one concrete thing").
const feedbackHelp = document.getElementById("feedback-help");
const missedFactRow = document.getElementById("missed-fact-row");
const missedFactBtn = document.getElementById("missed-fact-btn");
const missedFactStatus = document.getElementById("missed-fact-status");
const practiceFeedbackArea = document.getElementById("practice-feedback-area");
const resultBadge = document.getElementById("result-badge");
const overrideRow = document.getElementById("override-row");
const overrideCorrectBtn = document.getElementById("override-correct-btn");
const nextProblemBtn = document.getElementById("next-problem-btn");
// Top progress lives in StageLadder. Legacy ewma-accuracy DOM below review now
// carries scoped KC understanding with tier + coverage, never broad EWMA.
const solutionCode = document.getElementById("solution-code");
const aiExplanationSection = document.getElementById("ai-explanation-section");
const aiExplanationText = document.getElementById("ai-explanation-text");
const tutorSection = document.getElementById("tutor-section");
const tutorThread = document.getElementById("tutor-thread");
const tutorEmpty = document.getElementById("tutor-empty");
const tutorSuggestions = document.getElementById("tutor-suggestions");
const tutorInput = document.getElementById("tutor-input");
const tutorSendBtn = document.getElementById("tutor-send-btn");
const coldStartBadge = document.getElementById("cold-start-badge");
const coldStartLabel = document.getElementById("cold-start-label");
const coldStartNote = document.getElementById("cold-start-note");
const codeEditor = document.getElementById("code-editor");
const runBtn = document.getElementById("run-btn");
const runtimeStatus = document.getElementById("runtime-status");
const runtimeResetBtn = document.getElementById("runtime-reset-btn");
const outputArea = document.getElementById("output-area");
const outputVisual = document.getElementById("output-visual");
const outputVisualNote = document.getElementById("output-visual-note");
const outputVisualCanvas = document.getElementById("output-visual-canvas");
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
