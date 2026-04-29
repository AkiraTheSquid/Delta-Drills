/* ================================================================
   PRACTICE VISUALS — target image rendering for visual exercises
   ================================================================ */

const ARENA_NUMBERS_PATH = "/delta_numbers.npy";
const ARENA_NUMBERS_PNG_PATH = "/numbers_stacked.png";
const DELTA_VISUAL_DEBUG = true;
let deltaVisualDebugReportTimer = null;
let deltaVisualDebugLastSignature = "";

function getArenaNumbersPathCandidates() {
  const currentDir = window.location.pathname.replace(/[^/]*$/, "");
  return Array.from(new Set([
    ARENA_NUMBERS_PATH,
    `${currentDir}delta_numbers.npy`,
    "/Local_Deployed_Shared/delta_numbers.npy",
    "delta_numbers.npy",
  ]));
}

async function fetchArenaNumbersAsset() {
  const candidates = getArenaNumbersPathCandidates();
  let lastError = null;

  for (const path of candidates) {
    try {
      const response = await fetch(path, { cache: "no-store" });
      if (!response.ok) {
        lastError = new Error(`Failed to fetch ${path} (${response.status})`);
        continue;
      }
      setVisualDebug({ arenaNumbersPath: path });
      return response;
    } catch (err) {
      lastError = err;
    }
  }

  throw lastError || new Error(`Failed to fetch ${ARENA_NUMBERS_PATH}`);
}

function getArenaNumbersPngCandidates(question) {
  const currentDir = window.location.pathname.replace(/[^/]*$/, "");
  const explicit = question?.fallback_image_url;
  return Array.from(new Set([
    explicit,
    ARENA_NUMBERS_PNG_PATH,
    `${currentDir}numbers_stacked.png`,
    "/Local_Deployed_Shared/numbers_stacked.png",
    "numbers_stacked.png",
  ].filter(Boolean)));
}

function drawImageToCanvas(canvasEl, img) {
  const ctx = canvasEl.getContext("2d");
  canvasEl.width = img.naturalWidth || img.width;
  canvasEl.height = img.naturalHeight || img.height;
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  ctx.drawImage(img, 0, 0);
  canvasEl.classList.remove("hidden");
}

function loadImageFromCandidates(candidates) {
  return new Promise((resolve, reject) => {
    const remaining = [...candidates];
    function tryNext() {
      if (!remaining.length) {
        reject(new Error("No fallback image candidate succeeded."));
        return;
      }
      const path = remaining.shift();
      const img = new Image();
      img.onload = () => resolve({ img, path });
      img.onerror = () => tryNext();
      img.src = path;
    }
    tryNext();
  });
}

async function renderFallbackImage(question) {
  const candidates = getArenaNumbersPngCandidates(question);
  const { img, path } = await loadImageFromCandidates(candidates);
  drawImageToCanvas(questionVisualCanvas, img);
  setVisualDebug({ fallbackImagePath: path, fallbackImageRendered: true });
  return path;
}

function scheduleVisualDebugReport() {
  if (practiceMode !== "backend") return;
  const payload = window.__deltaLastVisualDebug || {};
  if (!payload) return;
  const terminal = payload.rendered || payload.error || payload.reason;
  if (!terminal) return;
  const signature = JSON.stringify(payload);
  if (signature === deltaVisualDebugLastSignature) return;
  deltaVisualDebugLastSignature = signature;
  if (deltaVisualDebugReportTimer) clearTimeout(deltaVisualDebugReportTimer);
  deltaVisualDebugReportTimer = setTimeout(async () => {
    try {
      await apiFetch("/api/practice/visual-debug", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ payload }),
      });
    } catch (err) {
      if (DELTA_VISUAL_DEBUG) {
        console.debug("[delta visual] failed to report", err);
      }
    }
  }, 150);
}

function setVisualDebug(info) {
  window.__deltaLastVisualDebug = {
    ...(window.__deltaLastVisualDebug || {}),
    ...info,
    timestamp: new Date().toISOString(),
  };
  if (DELTA_VISUAL_DEBUG) {
    console.debug("[delta visual]", window.__deltaLastVisualDebug);
  }
  scheduleVisualDebugReport();
}

function questionNeedsEinops(question = PracticeAPI?.currentQuestion) {
  return (
    question?.primary_library === "einops" ||
    question?.primary_library === "einops.einsum" ||
    question?.topic === "Einops" ||
    question?.topic === "Einsum" ||
    question?.supports_visual_output
  );
}

function questionNeedsArenaArray(question = PracticeAPI?.currentQuestion) {
  if (!question) return false;
  if (question.supports_visual_output) return true;

  const haystacks = [
    question.question_text,
    question.solution_code,
    question.starter_code,
    ...(Array.isArray(question.test_cases)
      ? question.test_cases.flatMap((test) => [
          test?.setup_code,
          test?.expected_setup_code,
          test?.expected_expr,
          test?.call,
        ])
      : []),
  ]
    .filter(Boolean)
    .join("\n");

  return haystacks.includes("/delta_numbers.npy");
}

function hideQuestionVisual() {
  setVisualDebug({ hidden: true, reason: "question_not_visual" });
  questionVisual.classList.add("hidden");
  questionVisualCanvas.classList.add("hidden");
  questionVisualNote.textContent = "";
  const ctx = questionVisualCanvas.getContext("2d");
  ctx.clearRect(0, 0, questionVisualCanvas.width || 1, questionVisualCanvas.height || 1);
}

function extractVisualVariableName(code) {
  if (!code) return null;
  const displayMatch = code.match(/display_array_as_img\(([^)]+)\)/);
  if (displayMatch) return displayMatch[1].trim();
  const assignMatch = code.match(/\b(arr\d+)\s*=/);
  return assignMatch ? assignMatch[1] : null;
}

function stripDisplayCalls(code) {
  return (code || "")
    .split("\n")
    .filter((line) => !line.includes("display_array_as_img("))
    .join("\n");
}

function extractStarterSetupCode(code) {
  if (!code) return "";
  const lines = [];
  for (const line of code.split("\n")) {
    if (/^\s*def\s+solve\s*\(/.test(line)) break;
    if (/^\s*print\s*\(\s*solve\s*\(\s*\)\s*\)\s*$/.test(line)) continue;
    lines.push(line);
  }
  return lines.join("\n").trim();
}

function augmentVisualSetupCode(setupCode, resultExpr) {
  const setupLines = (setupCode || "").trim();
  const needsHwcs = /\bhwcs\b/.test(resultExpr || "");
  const definesHwcs = /\bhwcs\s*=/.test(setupLines);
  const definesArr = /\barr\s*=/.test(setupLines);
  const definesB = /\bb\s*=/.test(setupLines);

  if (needsHwcs && !definesHwcs && definesArr) {
    const hwcsLine = definesB ? "hwcs = arr[:b]" : "hwcs = arr";
    return [setupLines, hwcsLine].filter(Boolean).join("\n");
  }

  return setupLines;
}

function getVisualExecutionPlan(question) {
  const testCase = Array.isArray(question?.test_cases) ? question.test_cases[0] : null;
  const starterSetupCode = extractStarterSetupCode(question?.starter_code || "");
  if (testCase?.expected_expr) {
    const setupCode = [
      starterSetupCode,
      testCase.setup_code || "",
      testCase.expected_setup_code || "",
    ]
      .filter(Boolean)
      .join("\n");
    return {
      setupCode: augmentVisualSetupCode(setupCode, testCase.expected_expr),
      resultExpr: testCase.expected_expr,
    };
  }

  const solutionCode = question?.solution_code || "";
  const variableName = extractVisualVariableName(solutionCode);
  if (!variableName) return null;
  return {
    setupCode: augmentVisualSetupCode(
      [starterSetupCode, stripDisplayCalls(solutionCode)].filter(Boolean).join("\n"),
      variableName
    ),
    resultExpr: variableName,
  };
}

function renderImageArrayToCanvas(arrayData) {
  return renderArrayToCanvas(questionVisualCanvas, arrayData);
}

function renderArrayToCanvas(canvasEl, arrayData) {
  function inferShape(value) {
    const shape = [];
    let cur = value;
    while (Array.isArray(cur)) {
      shape.push(cur.length);
      cur = cur[0];
    }
    return shape;
  }

  function chwToHwc(image) {
    const channels = image.length;
    const height = image[0].length;
    const width = image[0][0].length;
    const out = Array.from({ length: height }, () =>
      Array.from({ length: width }, () => Array(channels).fill(0))
    );
    for (let c = 0; c < channels; c++) {
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          out[y][x][c] = image[c][y][x];
        }
      }
    }
    return out;
  }

  function tileBatchImages(batch) {
    const batchSize = batch.length;
    const cols = Math.ceil(Math.sqrt(batchSize));
    const rows = Math.ceil(batchSize / cols);
    const height = batch[0].length;
    const width = batch[0][0].length;
    const channels = Array.isArray(batch[0][0][0]) ? batch[0][0][0].length : 1;
    const blankPixel = channels === 1 ? 0 : Array(channels).fill(0);
    const canvas = Array.from({ length: rows * height }, () =>
      Array.from({ length: cols * width }, () =>
        channels === 1 ? 0 : [...blankPixel]
      )
    );

    for (let i = 0; i < batchSize; i++) {
      const row = Math.floor(i / cols);
      const col = i % cols;
      const image = batch[i];
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          canvas[row * height + y][col * width + x] = image[y][x];
        }
      }
    }
    return canvas;
  }

  function normalizeForDisplay(value) {
    const shape = inferShape(value);
    setVisualDebug({ rawShape: shape });
    if (shape.length === 2) return value;
    if (shape.length === 3) {
      if (shape[0] === 1 || shape[0] === 3 || shape[0] === 4) return chwToHwc(value);
      return value;
    }
    if (shape.length === 4) {
      let batch = value;
      if (shape[1] === 1 || shape[1] === 3 || shape[1] === 4) {
        batch = value.map((image) => chwToHwc(image));
      }
      setVisualDebug({ tiledBatch: true, batchSize: shape[0] });
      return tileBatchImages(batch);
    }
    throw new Error("Unsupported image payload.");
  }

  const ctx = canvasEl.getContext("2d");
  let height;
  let width;
  let imageData;
  const normalized = normalizeForDisplay(arrayData);

  if (!Array.isArray(normalized) || !Array.isArray(normalized[0])) {
    throw new Error("Unsupported image payload.");
  }

  if (Array.isArray(normalized[0][0])) {
    height = normalized.length;
    width = normalized[0].length;
    imageData = ctx.createImageData(width, height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixel = normalized[y][x];
        const idx = (y * width + x) * 4;
        imageData.data[idx] = Math.max(0, Math.min(255, pixel[0] ?? 0));
        imageData.data[idx + 1] = Math.max(0, Math.min(255, pixel[1] ?? 0));
        imageData.data[idx + 2] = Math.max(0, Math.min(255, pixel[2] ?? 0));
        imageData.data[idx + 3] = pixel.length > 3 ? Math.max(0, Math.min(255, pixel[3])) : 255;
      }
    }
  } else {
    height = normalized.length;
    width = normalized[0].length;
    imageData = ctx.createImageData(width, height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const value = Math.max(0, Math.min(255, normalized[y][x] ?? 0));
        const idx = (y * width + x) * 4;
        imageData.data[idx] = value;
        imageData.data[idx + 1] = value;
        imageData.data[idx + 2] = value;
        imageData.data[idx + 3] = 255;
      }
    }
  }

  canvasEl.width = width;
  canvasEl.height = height;
  ctx.putImageData(imageData, 0, 0);
  canvasEl.classList.remove("hidden");
  setVisualDebug({ renderedWidth: width, renderedHeight: height, rendered: true });
}

async function renderQuestionVisual(question) {
  setVisualDebug({
    questionId: question?.question_id || question?.id || null,
    questionText: question?.question_text || "",
    supportsVisualOutput: !!question?.supports_visual_output,
    taskType: question?.task_type || null,
    expectedArtifactType: question?.expected_artifact_type || null,
  });
  if (!question?.supports_visual_output) {
    hideQuestionVisual();
    return;
  }

  questionVisual.classList.remove("hidden");
  questionVisualCanvas.classList.add("hidden");
  questionVisualNote.textContent = "Loading image preview...";

  const plan = getVisualExecutionPlan(question);
  setVisualDebug({
    setupCode: plan?.setupCode || "",
    resultExpr: plan?.resultExpr || "",
  });
  if (!plan?.setupCode || !plan?.resultExpr) {
    setVisualDebug({ rendered: false, reason: "missing_execution_plan" });
    questionVisualNote.textContent = "Image preview unavailable for this question.";
    return;
  }

  try {
    const pyodide = await initPyodide();
    if (!pyodide) throw new Error("Python runtime unavailable.");

    await ensurePyodidePracticePackages({ needsEinops: questionNeedsEinops(question) });
    if (questionNeedsArenaArray(question)) {
      await ensureArenaNumbersInPyodide();
    }

    const payload = await pyodide.runPythonAsync(`
import json
import numpy as np
${questionNeedsEinops(question) ? "import einops" : ""}
${questionNeedsArenaArray(question) ? "arr = np.load('/delta_numbers.npy')" : ""}
def display_array_as_img(*args, **kwargs):
    return None
${plan.setupCode}
_delta_result = ${plan.resultExpr}
if getattr(_delta_result, "ndim", None) == 3 and _delta_result.shape[0] in (1, 3, 4):
    _delta_result = np.moveaxis(_delta_result, 0, -1)
json.dumps(_delta_result.tolist())
`);

    const parsedPayload = JSON.parse(payload);
    setVisualDebug({ payloadPreviewType: Array.isArray(parsedPayload) ? "array" : typeof parsedPayload });
    renderImageArrayToCanvas(parsedPayload);
    questionVisualNote.textContent = "Reference image generated from the canonical solution.";
  } catch (err) {
    setVisualDebug({ rendered: false, error: err.message || String(err) });
    try {
      await renderFallbackImage(question);
      questionVisualNote.textContent =
        "Live preview unavailable; showing static reference image of the source data.";
    } catch (fallbackErr) {
      setVisualDebug({ fallbackImageError: fallbackErr.message || String(fallbackErr) });
      questionVisualNote.textContent = "Unable to render image preview: " + err.message;
    }
  }
}

window.renderDeltaArrayToCanvas = renderArrayToCanvas;
