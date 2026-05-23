/* ================================================================
   STATS PREDICTED — data shaping helpers

   Reusable section-sort, key, and aggregation helpers that take
   an ARENA problem (or a list) and return primitives the renderer
   in predicted.js can use without juggling labels or skill arrays.

   Loaded BEFORE predicted.js in index.html.
   ================================================================ */

// ── per-problem score lookup ─────────────────────────────────
const computeProblemScore = (problem) => {
  const fallback = Number(problem?.readinessScore) || 0;
  if (typeof window.computeArenaReadiness !== "function") return fallback;
  const s = window.computeArenaReadiness(problem?.skillWeights, fallback);
  return Number.isFinite(s) ? s : fallback;
};

const exercisesForProblem = (problem) => {
  const reg = (window.ARENA_EXERCISES_BY_NOTEBOOK && typeof window.ARENA_EXERCISES_BY_NOTEBOOK === "object")
    ? window.ARENA_EXERCISES_BY_NOTEBOOK
    : {};
  return Array.isArray(reg[problem?.notebookPath]) ? reg[problem.notebookPath] : [];
};

// ── section label parsing + sort ─────────────────────────────
const sectionNumberFromLabel = (label) => {
  const raw = String(label || "").trim();
  return raw.split(/\s+/)[0] || "";
};

const parseSectionPath = (label) => {
  const parts = sectionNumberFromLabel(label)
    .split(".")
    .map((part) => Number(part))
    .filter((part) => Number.isFinite(part));
  return parts;
};

const compareSectionLabels = (left, right) => {
  const a = parseSectionPath(left);
  const b = parseSectionPath(right);
  const length = Math.max(a.length, b.length);
  for (let i = 0; i < length; i += 1) {
    const av = a[i] ?? -1;
    const bv = b[i] ?? -1;
    if (av !== bv) return av - bv;
  }
  return sectionNumberFromLabel(left).localeCompare(sectionNumberFromLabel(right));
};

const subsectionKeyForProblem = (problem) => {
  const parts = parseSectionPath(problem.sectionLabel);
  if (parts.length >= 3) return parts.slice(0, 2).join(".");
  return sectionNumberFromLabel(problem.sectionLabel) || problem.id;
};

const sectionLabelForProblem = (problem) =>
  sectionNumberFromLabel(problem.sectionLabel) || problem.id;

// ── skill aggregation ────────────────────────────────────────
const aggregateTopSkill = (problems) => {
  const totals = new Map();
  let problemCount = 0;
  problems.forEach((problem) => {
    problemCount += 1;
    (Array.isArray(problem.skillWeights) ? problem.skillWeights : []).forEach((entry) => {
      if (!entry?.skill) return;
      totals.set(entry.skill, (totals.get(entry.skill) || 0) + (Number(entry.weight) || 0));
    });
  });
  if (!totals.size) return "—";
  let bestSkill = null;
  let bestWeight = -Infinity;
  totals.forEach((weight, skill) => {
    if (weight > bestWeight) {
      bestSkill = skill;
      bestWeight = weight;
    }
  });
  const normalized = problemCount > 0 ? bestWeight / problemCount : bestWeight;
  return `${bestSkill} (${Math.round(normalized * 100)}%)`;
};

const topSkillLabel = (problem) => {
  const skills = Array.isArray(problem.skillWeights) ? problem.skillWeights : [];
  if (!skills.length) return "—";
  const top = skills.reduce((a, b) => (a.weight >= b.weight ? a : b));
  return `${top.skill} (${Math.round(top.weight * 100)}%)`;
};
