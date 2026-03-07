/* ================================================================
   PRACTICE BARS — target difficulty + accuracy bar rendering/animation
   ================================================================ */

/* ── Shared helpers ──────────────────────────────────────────── */

function clampDifficulty(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

function formatDifficulty(value) {
  if (!Number.isFinite(value)) return "--";
  return value.toFixed(1);
}

/* ── Target difficulty bar ───────────────────────────────────── */

let targetRafId = null;

function cancelTargetAnimation() {
  if (targetRafId !== null) {
    cancelAnimationFrame(targetRafId);
    targetRafId = null;
  }
}

function setTargetDifficultyInitial(targetDifficulty) {
  cancelTargetAnimation();
  const clamped = clampDifficulty(targetDifficulty);
  targetDifficultyTitle.textContent = "Target difficulty of " + (PracticeAPI.currentQuestion?.subtopic || "");
  targetDifficultyValue.textContent = `Old ${formatDifficulty(clamped)}`;
  targetDifficultyFill.style.width = `${clamped}%`;
  targetDifficultyDelta.classList.add("hidden");
  targetDifficultyDelta.style.width = "0%";
  targetDifficultyMarkerOld.style.left = `${clamped}%`;
  targetDifficultyNumberOld.textContent = formatDifficulty(clamped);
  targetDifficultyMarkerNew.classList.add("hidden");
}

function setTargetDifficultyFinal(oldTarget, newTarget) {
  const oldClamped = clampDifficulty(oldTarget);
  const newClamped = clampDifficulty(newTarget);
  const diff = Math.abs(newClamped - oldClamped);
  targetDifficultyTitle.textContent = "Target difficulty of " + (PracticeAPI.currentQuestion?.subtopic || "");
  targetDifficultyValue.textContent = `Old ${formatDifficulty(oldClamped)} -> New ${formatDifficulty(newClamped)}`;
  targetDifficultyFill.style.width = `${newClamped}%`;
  targetDifficultyMarkerOld.style.left = `${oldClamped}%`;
  targetDifficultyNumberOld.textContent = formatDifficulty(oldClamped);
  targetDifficultyMarkerNew.classList.remove("hidden");
  targetDifficultyMarkerNew.style.left = `${newClamped}%`;
  targetDifficultyNumberNew.textContent = formatDifficulty(newClamped);

  if (diff < 0.01) {
    targetDifficultyDelta.classList.add("hidden");
    targetDifficultyDelta.style.width = "0%";
    return;
  }
  const left = Math.min(oldClamped, newClamped);
  targetDifficultyDelta.style.left = `${left}%`;
  targetDifficultyDelta.style.width = `${diff}%`;
  targetDifficultyDelta.classList.remove("hidden");
  targetDifficultyDelta.classList.toggle("up", newClamped > oldClamped);
  targetDifficultyDelta.classList.toggle("down", newClamped < oldClamped);
}

function animateTargetDifficulty(oldTarget, newTarget, onComplete) {
  cancelTargetAnimation();
  const oldClamped = clampDifficulty(oldTarget);
  const newClamped = clampDifficulty(newTarget);
  const isUp = newClamped > oldClamped;
  const start = performance.now();
  const duration = 900;

  targetDifficultyMarkerNew.classList.remove("hidden");
  targetDifficultyMarkerNew.style.left = `${oldClamped}%`;
  targetDifficultyNumberNew.textContent = formatDifficulty(oldClamped);
  targetDifficultyTitle.textContent = "Target difficulty of " + (PracticeAPI.currentQuestion?.subtopic || "");
  targetDifficultyValue.textContent = `Old ${formatDifficulty(oldClamped)} -> New ${formatDifficulty(oldClamped)}`;
  targetDifficultyDelta.classList.toggle("up", isUp);
  targetDifficultyDelta.classList.toggle("down", !isUp && newClamped !== oldClamped);
  targetDifficultyDelta.classList.remove("hidden");

  const tick = (now) => {
    const progress = Math.min((now - start) / duration, 1);
    const value = oldClamped + (newClamped - oldClamped) * progress;
    targetDifficultyFill.style.width = `${value}%`;
    targetDifficultyMarkerNew.style.left = `${value}%`;
    targetDifficultyNumberNew.textContent = formatDifficulty(value);
    const left = Math.min(oldClamped, value);
    const width = Math.abs(value - oldClamped);
    targetDifficultyDelta.style.left = `${left}%`;
    targetDifficultyDelta.style.width = `${width}%`;

    if (progress < 1) {
      targetRafId = requestAnimationFrame(tick);
      return;
    }
    targetRafId = null;
    targetDifficultyTitle.textContent = "Target difficulty of " + (PracticeAPI.currentQuestion?.subtopic || "");
    targetDifficultyValue.textContent = `Old ${formatDifficulty(oldClamped)} -> New ${formatDifficulty(newClamped)}`;
    targetDifficultyFill.style.width = `${newClamped}%`;
    targetDifficultyMarkerNew.style.left = `${newClamped}%`;
    targetDifficultyNumberNew.textContent = formatDifficulty(newClamped);
    if (Math.abs(newClamped - oldClamped) < 0.01) {
      targetDifficultyDelta.classList.add("hidden");
      targetDifficultyDelta.style.width = "0%";
    }
    if (typeof onComplete === "function") onComplete();
  };

  targetRafId = requestAnimationFrame(tick);
}

/* ── Accuracy bar ────────────────────────────────────────────── */

// Captured in renderQuestion so the feedback handler always has the correct
// pre-question EWMA, regardless of any state changes that happen after submit.
let ewmaAccuracyPBefore = null;

// Track the rAF ID so stale animations can be cancelled.
let ewmaRafId = null;

function cancelEwmaAnimation() {
  if (ewmaRafId !== null) {
    cancelAnimationFrame(ewmaRafId);
    ewmaRafId = null;
  }
}

// Initial state shown after submit (mirrors setTargetDifficultyInitial).
function showEwmaAccuracyInitial(p, subtopic) {
  cancelEwmaAnimation();
  // Always write the current subtopic so stale text from a previous question
  // never leaks through, regardless of whether we show the bar.
  ewmaAccuracyLabel.textContent = "Accuracy of " + (subtopic || "");
  ewmaAccuracyDelta.classList.add("hidden");
  ewmaAccuracyDelta.style.width = "0%";
  ewmaAccuracyMarkerNew.classList.add("hidden");
  if (!Number.isFinite(p)) {
    // No history yet — keep the bar hidden and clear any stale values.
    ewmaAccuracyValue.textContent = "";
    ewmaAccuracyFill.style.width = "0%";
    ewmaAccuracy.classList.add("hidden");
    return;
  }
  const pct = Math.round(p * 1000) / 10;
  ewmaAccuracyValue.textContent = "Old " + pct.toFixed(1) + "%";
  ewmaAccuracyFill.style.width = pct + "%";
  ewmaAccuracyMarkerOld.style.left = pct + "%";
  ewmaAccuracy.classList.remove("hidden");
}

// Animated old → new after feedback button (mirrors animateTargetDifficulty).
function showEwmaAccuracy(pBefore, pAfter, subtopic) {
  if (!Number.isFinite(pAfter)) return;
  cancelEwmaAnimation();
  const newPct = Math.round(pAfter * 1000) / 10;

  ewmaAccuracyLabel.textContent = "Accuracy of " + (subtopic || "");
  ewmaAccuracy.classList.remove("hidden");

  // No prior history: grow fill bar in blue, no delta or markers.
  if (!Number.isFinite(pBefore)) {
    ewmaAccuracyMarkerOld.style.left = "0%";
    ewmaAccuracyMarkerNew.classList.add("hidden");
    ewmaAccuracyDelta.classList.add("hidden");
    ewmaAccuracyDelta.style.width = "0%";
    ewmaAccuracyFill.style.width = "0%";
    ewmaAccuracyValue.textContent = "0.0%";
    const start = performance.now();
    const duration = 900;
    const tick = (now) => {
      const progress = Math.min((now - start) / duration, 1);
      const value = newPct * progress;
      ewmaAccuracyFill.style.width = value + "%";
      ewmaAccuracyValue.textContent = value.toFixed(1) + "%";
      if (progress < 1) { ewmaRafId = requestAnimationFrame(tick); return; }
      ewmaRafId = null;
      ewmaAccuracyFill.style.width = newPct + "%";
      ewmaAccuracyMarkerOld.style.left = newPct + "%";
      ewmaAccuracyValue.textContent = newPct.toFixed(1) + "%";
    };
    ewmaRafId = requestAnimationFrame(tick);
    return;
  }

  // Has prior history: full old → new animation with delta.
  const oldPct = Math.round(pBefore * 1000) / 10;
  ewmaAccuracyFill.style.width = oldPct + "%";
  ewmaAccuracyMarkerOld.style.left = oldPct + "%";
  ewmaAccuracyMarkerNew.classList.remove("hidden");
  ewmaAccuracyMarkerNew.style.left = oldPct + "%";

  const isUp = newPct > oldPct;
  ewmaAccuracyDelta.classList.toggle("up", isUp);
  ewmaAccuracyDelta.classList.toggle("down", !isUp && newPct !== oldPct);
  ewmaAccuracyDelta.classList.remove("hidden");
  ewmaAccuracyValue.textContent = `Old ${oldPct.toFixed(1)}% → New ${oldPct.toFixed(1)}%`;

  const start = performance.now();
  const duration = 900;

  const tick = (now) => {
    const progress = Math.min((now - start) / duration, 1);
    const value = oldPct + (newPct - oldPct) * progress;
    ewmaAccuracyFill.style.width = value + "%";
    ewmaAccuracyMarkerNew.style.left = value + "%";
    const left = Math.min(oldPct, value);
    const width = Math.abs(value - oldPct);
    ewmaAccuracyDelta.style.left = left + "%";
    ewmaAccuracyDelta.style.width = width + "%";
    ewmaAccuracyValue.textContent = `Old ${oldPct.toFixed(1)}% → New ${value.toFixed(1)}%`;

    if (progress < 1) {
      ewmaRafId = requestAnimationFrame(tick);
      return;
    }
    ewmaRafId = null;
    ewmaAccuracyFill.style.width = newPct + "%";
    ewmaAccuracyMarkerNew.style.left = newPct + "%";
    ewmaAccuracyValue.textContent = `Old ${oldPct.toFixed(1)}% → New ${newPct.toFixed(1)}%`;
    if (Math.abs(newPct - oldPct) < 0.01) {
      ewmaAccuracyDelta.classList.add("hidden");
      ewmaAccuracyDelta.style.width = "0%";
    }
  };

  ewmaRafId = requestAnimationFrame(tick);
}

// Instant final state, no animation (used when restoring pending feedback on reload).
function setEwmaAccuracyFinal(pBefore, pAfter, subtopic) {
  if (!Number.isFinite(pAfter)) return;
  const newPct = Math.round(pAfter * 1000) / 10;
  const oldPct = Number.isFinite(pBefore) ? Math.round(pBefore * 1000) / 10 : newPct;
  const diff = Math.abs(newPct - oldPct);
  ewmaAccuracyLabel.textContent = "Accuracy of " + (subtopic || "");
  ewmaAccuracyValue.textContent = `Old ${oldPct.toFixed(1)}% → New ${newPct.toFixed(1)}%`;
  ewmaAccuracyFill.style.width = newPct + "%";
  ewmaAccuracyMarkerOld.style.left = oldPct + "%";
  ewmaAccuracyMarkerNew.classList.remove("hidden");
  ewmaAccuracyMarkerNew.style.left = newPct + "%";
  if (diff < 0.01) {
    ewmaAccuracyDelta.classList.add("hidden");
    ewmaAccuracyDelta.style.width = "0%";
  } else {
    const left = Math.min(oldPct, newPct);
    ewmaAccuracyDelta.style.left = left + "%";
    ewmaAccuracyDelta.style.width = diff + "%";
    ewmaAccuracyDelta.classList.toggle("up", newPct > oldPct);
    ewmaAccuracyDelta.classList.toggle("down", newPct < oldPct);
    ewmaAccuracyDelta.classList.remove("hidden");
  }
  ewmaAccuracy.classList.remove("hidden");
}
