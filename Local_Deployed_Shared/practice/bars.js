function clampDifficulty(value) {
  return Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
}

function setTargetDifficultyUnavailable(note, currentValue) {
  const known = Number.isFinite(currentValue);
  window.StageLadder?.setDifficulty(undefined, known ? clampDifficulty(currentValue) : null);
}

function setTargetDifficultyInitial(targetDifficulty) {
  window.StageLadder?.setDifficulty(undefined, clampDifficulty(targetDifficulty));
}

function setTargetDifficultyFinal(oldTarget, newTarget) {
  window.StageLadder?.setDifficulty(undefined, clampDifficulty(newTarget));
}

function animateTargetDifficulty(oldTarget, newTarget, onComplete) {
  if (typeof onComplete === "function") onComplete();
}

function setConceptUnderstanding({ mastery, coverage, tier, title } = {}) {
  const host = document.getElementById("ewma-accuracy");
  if (!host) return;
  if (!Number.isFinite(mastery) || !Number.isFinite(coverage) || !tier) {
    host.classList.add("hidden");
    return;
  }
  const pct = Math.max(0, Math.min(100, mastery * 100));
  const covered = Math.max(0, Math.min(100, coverage * 100));
  const tierLabel = tier === "measured" ? "Measured concept" : tier === "topic-proxy" ? "Topic proxy" : "Unmapped";
  const label = document.getElementById("ewma-accuracy-label");
  const meta = document.getElementById("ewma-accuracy-meta");
  const value = document.getElementById("ewma-accuracy-value");
  const fill = document.getElementById("ewma-accuracy-fill");
  const oldMarker = document.getElementById("ewma-accuracy-marker-old");
  const newMarker = document.getElementById("ewma-accuracy-marker-new");
  if (label) label.textContent = `Understanding of ${title || "this concept"}`;
  if (meta) meta.textContent = `${tierLabel} · ${covered.toFixed(0)}% evidence coverage`;
  if (value) value.textContent = `${pct.toFixed(1)}%`;
  if (fill) fill.style.width = `${pct}%`;
  if (oldMarker) oldMarker.style.left = `${pct}%`;
  newMarker?.classList.add("hidden");
  host.classList.remove("hidden");
}
