/* ================================================================
   REAL IMAGES FOR THE IMAGE DRILLS
   ================================================================

   🔴 AN IMAGE DRILL HAS TO SHOW AN IMAGE. Seth, 2026-09-06, on one of the
   ARENA 0.0 einops exercises: "whenever I solved it, it returned the matrix
   rather than the image itself."

   He was right, and it was not one bug but the shape of the whole variant
   set. Every 0.0 einops variant (q889-q920) is graded as a FUNCTION over
   nested lists, because the grader compares values and a 150x150 picture is
   not a value anybody wants to diff. So the canonical test case hands
   `solve` a 2x2 toy — `[[[[0, 1], [2, 3]]], [[[4, 5], [6, 7]]]]` — and
   practice/runner.js rendered exactly that to the output canvas: four
   near-black pixels. A matrix, not an image.

   The fix is not to change what is GRADED. It is to run the learner's own
   `solve` a second time, for the preview only, on the real ARENA digits —
   `arr`, shape (6, 3, 150, 150), already loaded into Pyodide for these
   questions. The learner's function is written against axis NAMES, so it
   does not care that h and w suddenly became 120 instead of 2, and the
   picture that comes back is the one the exercise is actually about:
   six digits laid side by side, stacked, tiled, transposed, pooled.

   The other arguments are passed through untouched. That matters: `rows`,
   `n`, `k`, `cols` are all tied to the BATCH size or to a pooling factor,
   so the substitute keeps every non-spatial dimension exactly as the toy
   had it and only grows h and w. `solve(imgs, rows=2)` on a 4-image toy
   stays a 4-image call.

   120, not 150: the pooling variants divide h and w by 2, 3 or 4, and 150
   is not divisible by 4. A centred 120x120 crop keeps the digit whole and
   divides by 2, 3, 4, 5, 6 and 8. A factor it does not divide raises inside
   einops, the caller catches it, and the pane simply stays hidden — a
   missing preview, never a wrong one. */

(function () {
  // Axis tuples as the questions declare them, e.g. "a batch of images
  // `(b, c, h, w)`" or "a channels-LAST image `(h, w, c)`". The prompt is the
  // right place to read this from: it is what the learner was told, so a
  // preview built from it can never contradict the question.
  const DECLARED_SHAPE = /`\(\s*([bchw](?:\s*,\s*[bchw])+)\s*\)`/;

  // The one drill in the set that says "same grid" instead of repeating the
  // tuple (q910) falls through to this, and so would any future variant that
  // leaves the shape implicit. Matches the Python fallback below.
  const BY_RANK = { 2: ["h", "w"], 3: ["c", "h", "w"], 4: ["b", "c", "h", "w"] };

  function layoutFor(question) {
    const match = DECLARED_SHAPE.exec(question?.question_text || "");
    if (!match) return null;
    return match[1].split(",").map((axis) => axis.trim());
  }

  function isImageDrill(question) {
    return (
      !!question &&
      question.expected_artifact_type === "image" &&
      question.submission_mode === "function"
    );
  }

  /* Defines `_delta_real_image_arg(toy, layout)`: same rank and same
     non-spatial sizes as the toy argument, real pixels in h and w. Each
     (batch, channel) slot gets its own plane of `arr`, wrapping round when the
     drill asks for more than the six images or three channels there are. */
  function pythonHelper() {
    return `
import itertools as _delta_itertools


def _delta_to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_delta_to_jsonable(v) for v in value]
    return value


def _delta_real_image_arg(toy, layout=None):
    a = np.asarray(toy)
    if a.ndim < 2 or a.ndim > 4:
        return toy
    if not layout or len(layout) != a.ndim:
        layout = {2: ("h", "w"), 3: ("c", "h", "w"), 4: ("b", "c", "h", "w")}[a.ndim]
    layout = list(layout)
    if "h" not in layout or "w" not in layout:
        return toy
    side = min(120, int(arr.shape[-2]), int(arr.shape[-1]))
    top = (arr.shape[-2] - side) // 2
    left = (arr.shape[-1] - side) // 2
    src = arr[:, :, top:top + side, left:left + side]
    target = [side if ax in ("h", "w") else int(n) for n, ax in zip(a.shape, layout)]
    out = np.zeros(target, dtype=np.float64 if a.dtype.kind == "f" else np.int64)
    hi, wi = layout.index("h"), layout.index("w")
    bi = layout.index("b") if "b" in layout else None
    ci = layout.index("c") if "c" in layout else None
    lead = [i for i in range(len(layout)) if i not in (hi, wi)]
    for combo in _delta_itertools.product(*[range(target[i]) for i in lead]):
        at = dict(zip(lead, combo))
        plane = src[at.get(bi, 0) % src.shape[0], at.get(ci, 0) % src.shape[1]]
        index = [slice(None)] * len(layout)
        for axis, value in at.items():
            index[axis] = value
        out[tuple(index)] = plane
    return out.tolist()


def _delta_capture(*args, **kwargs):
    # KEYWORDS SURVIVE. No drill in the 0.0 set passes one today, but the
    # substitution above is documented to handle solve(imgs, rows=2) and a
    # *args-only capture would have turned that into a TypeError and a silently
    # hidden pane the day somebody authored it. (codex, 2026-09-07.)
    return [list(args), kwargs]
`;
  }

  /* The canonical call with its FIRST argument swapped for real pixels.
     `_delta_capture` reuses the test case's own argument list verbatim rather
     than re-parsing a nested literal out of it — the fixture variables the
     setup defined (`img`, `imgs`, `hwcs`) resolve exactly as they do when the
     grader runs. */
  function realImageCall(question) {
    if (!isImageDrill(question)) return null;
    const call = question?.test_cases?.[0]?.call || "";
    if (!/^\s*solve\s*\(/.test(call)) return null;
    const layout = layoutFor(question);
    const layoutLiteral = layout
      ? "[" + layout.map((axis) => JSON.stringify(axis)).join(", ") + "]"
      : "None";
    return `
_delta_args, _delta_kwargs = ${call.replace(/^\s*solve\s*\(/, "_delta_capture(")}
if _delta_args:
    _delta_args[0] = _delta_real_image_arg(_delta_args[0], ${layoutLiteral})
_delta_output_value = solve(*_delta_args, **_delta_kwargs)
`;
  }

  /* THE PREVIEW HAS TO RUN WHERE THE CODE RAN. Every drill in the modern bank
     is torch, torch is not in Pyodide, and so `runSnippet` sends these to the
     backend and hands back `pyodide: null` — at which point the output canvas
     was skipped entirely and the learner's only feedback was the printed list.
     That is the other half of "it returned the matrix rather than the image".

     So when the graded run went to the backend, the preview goes there too:
     one more `/api/practice/run-code` with the learner's own code, the fixture,
     and a single sentinel line carrying the result as JSON. The backend
     preamble already defines `arr` (backend/app/code_runner.py), which is why
     no fixture data has to travel with the request. */
  const SENTINEL = "__DELTA_VISUAL__";

  function backendProgram(question, learnerCode) {
    const body = realImageCall(question);
    if (!body) return null;
    return `${learnerCode}

import json as _delta_json
import numpy as np
${pythonHelper()}
${body}
print(${JSON.stringify(SENTINEL)} + _delta_json.dumps(_delta_to_jsonable(_delta_output_value)))
`;
  }

  // The learner's own prints come back in the same stdout, so read the LAST
  // sentinel line and ignore everything around it.
  function parseSentinel(stdout) {
    const lines = String(stdout || "").split("\n");
    for (let i = lines.length - 1; i >= 0; i -= 1) {
      const line = lines[i].trim();
      if (line.startsWith(SENTINEL)) {
        try {
          return JSON.parse(line.slice(SENTINEL.length));
        } catch (err) {
          return null;
        }
      }
    }
    return null;
  }

  window.DeltaVisualFixture = {
    isImageDrill,
    layoutFor,
    pythonHelper,
    realImageCall,
    backendProgram,
    parseSentinel,
    BY_RANK,
  };
})();
