/* ================================================================
   PRACTICE TIMER — timed mode controls
   ================================================================ */

let timerInterval = null;
let timerSeconds = 10; // 10 seconds default
let timerTargetSeconds = 10; // the user-set value to reset to

const parseTimerInput = (value) => {
  const raw = String(value || "").trim();
  if (!raw) return 10;
  if (raw.includes(":")) {
    const [mStr, sStr] = raw.split(":");
    const m = Number(mStr);
    const s = Number(sStr);
    if (!Number.isFinite(m) || !Number.isFinite(s)) return 10;
    return Math.max(1, Math.min(3600, m * 60 + s));
  }
  const asNumber = Number(raw);
  if (!Number.isFinite(asNumber)) return 10;
  return Math.max(1, Math.min(3600, Math.round(asNumber)));
};

const formatTimer = (value) => {
  const clamped = Math.max(0, Math.min(3600, Math.round(value)));
  const m = Math.floor(clamped / 60);
  const s = clamped % 60;
  return String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
};

function updateTimerDisplay() {
  timerInput.value = formatTimer(timerSeconds);
}

function stopTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
}

function resetTimerToInput() {
  stopTimer();
  timerSeconds = timerTargetSeconds;
  updateTimerDisplay();
}

function startTimer() {
  stopTimer();
  timerTargetSeconds = parseTimerInput(timerInput.value);
  timerSeconds = timerTargetSeconds;
  updateTimerDisplay();
  if (timerSeconds <= 0) return;
  timerInterval = setInterval(() => {
    timerSeconds--;
    updateTimerDisplay();
    if (timerSeconds <= 0) {
      stopTimer();
      resetTimerToInput();
      // Auto-submit
      if (!practiceSubmitArea.classList.contains("hidden") && !practiceSubmitBtn.disabled) {
        practiceSubmitBtn.click();
      }
    }
  }, 1000);
}

timedModeToggle.addEventListener("change", () => {
  timerControls.classList.toggle("hidden", !timedModeToggle.checked);
  if (timedModeToggle.checked) {
    startTimer();
  } else {
    stopTimer();
    timerSeconds = parseTimerInput(timerInput.value);
    updateTimerDisplay();
  }
});

timerInput.addEventListener("change", () => {
  timerTargetSeconds = parseTimerInput(timerInput.value);
  timerSeconds = timerTargetSeconds;
  updateTimerDisplay();
  if (timedModeToggle.checked) {
    startTimer();
  }
});
