/* ARENA section compute: which machine a section's cells run on, and the
   banner that says so. Spec: This-Directory-Only/SPEC_MODAL_COMPUTE_BUDGET.md,
   pass 2.

   The server's `GET /api/practice/kernel/compute` is the one list of what each
   section needs (backend/app/modal_gpu.py `SECTIONS`):

     cpu      nothing to say; the default sandbox runs it.
     api      calls OpenAI / Anthropic with the learner's own key.
     gpu      opens on the CPU kernel; from its `gpu_from` cell on, Python moves
              to a GPU sandbox BY ITSELF. The context gains `#gpu`, which the
              server treats like a notebook switch: the CPU sandbox closes, a
              GPU one starts, setup is restored on it. Sticky for this open —
              hopping back per cell would lose the model in memory.
     gpu-big  needs more GPU than notebooks here have; a banner says where to
              run it. 4.4 still has a `gpu_from`: its Qwen2.5-7B route fits.

   arena-notebook.js keeps the wiring small: `attach` per render, `context()`
   for every request, `enter(node)` before a run, `leave()` when the server
   refused the move (GPUs busy, monthly hours used) so earlier cells keep
   running on the CPU kernel the learner still has. */
window.ArenaCompute = (() => {
  const PATH = "/api/practice/kernel/compute";
  const GPU_SUFFIX = "#gpu";

  /* Read once per notebook open, never cached across opens: the GPU hours
     left move every time the learner uses the GPU. */
  const load = async () => {
    if (typeof apiFetch !== "function" || !window.DeltaKernel?.available()) return null;
    try {
      const res = await apiFetch(PATH);
      return res.ok ? await res.json() : null;
    } catch (_err) {
      return null;
    }
  };

  const hours = (n) => `${Math.round(n * 10) / 10}`;

  const bannerText = (section, view) => {
    const kind = section && section.class;
    if (kind === "api") {
      return "This section calls the OpenAI and Anthropic APIs with your own key, which " +
        "ARENA's setup reads from a .env file. Without a key the cells that call a model " +
        "fail; everything else runs.";
    }
    if (kind === "gpu-big") {
      const route = section.gpu_from && view.gpu_enabled
        ? " The smaller-model route (Qwen2.5-7B) runs here: from the marked cell on, Python " +
          `moves to an ${view.gpu_type} GPU by itself.`
        : "";
      return "This section's main models need a far bigger GPU (30 GB and up) than notebooks " +
        "here have. Read along here, and run those parts on Colab Pro or RunPod with an " +
        `A100.${route}`;
    }
    if (kind !== "gpu") return "";
    if (!view.gpu_enabled) {
      return "From the marked cell on, this section loads a model that needs a GPU, and this " +
        "server has none. The cells before it run here.";
    }
    const left = view.gpu_hours_left > 0
      ? `${hours(view.gpu_hours_left)} of your ${hours(view.gpu_hours_cap)} GPU hours are left this month.`
      : `Your ${hours(view.gpu_hours_cap)} GPU hours for this month are used; the GPU part comes back on the 1st.`;
    return `From the marked cell on, Python moves to an ${view.gpu_type} GPU by itself ` +
      "(about a minute to start; setup is restored on it). It closes after " +
      `${Math.round(view.gpu_idle_seconds / 60)} minutes without a run. ${left}`;
  };

  const mark = (node, view) => {
    if (!node || node.previousElementSibling?.classList.contains("arena-nb-gpu-mark")) return;
    const flag = document.createElement("div");
    flag.className = "arena-nb-gpu-mark";
    flag.textContent = view.gpu_enabled
      ? `GPU from here — running this cell or any below it moves Python to an ${view.gpu_type} GPU.`
      : "GPU from here — the cells below need a GPU this server does not have.";
    node.parentNode.insertBefore(flag, node);
  };

  /* `at or below` in the notebook as it is NOW, so a cell the learner added
     after the marked one counts as after it. */
  const atOrAfter = (node, anchor) =>
    node === anchor ||
    !!(anchor.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING);

  const attach = ({ slug, host, body, cellIdOf, setupIds, baseContext, stillOpen }) => {
    const ctl = {
      gpu: false,
      moved: false,
      section: null,
      context: () => baseContext + (ctl.gpu ? GPU_SUFFIX : ""),
    };
    const anchor = () =>
      ctl.section && ctl.section.gpu_from
        ? Array.from(body.children).find((n) => cellIdOf(n) === ctl.section.gpu_from)
        : null;

    ctl.ready = (async () => {
      const view = await load();
      const section = view && view.sections ? view.sections[slug] : null;
      if (!section || !stillOpen()) return;
      ctl.section = section;
      ctl.view = view;
      const text = bannerText(section, view);
      const slot = host.querySelector(".arena-nb-compute");
      if (slot && text) {
        slot.textContent = text;
        slot.hidden = false;
      }
      const from = anchor();
      if (from) mark(from, view);
      if (!section.gpu_from || !view.gpu_enabled) return;
      // Setup that already crosses the line (1.3.1) starts on the GPU, and a
      // GPU kernel still alive from before a reload is kept rather than closed
      // by the first early cell the learner re-runs.
      const setupCrosses = from && setupIds.some((id) => {
        const n = Array.from(body.children).find((c) => cellIdOf(c) === id);
        return n && atOrAfter(n, from);
      });
      if (setupCrosses || (await window.DeltaKernel?.contextAlive?.(baseContext + GPU_SUFFIX))) {
        ctl.gpu = true;
      }
    })();

    /* Before a run. True when THIS run is the move to the GPU. */
    ctl.enter = (node) => {
      if (ctl.gpu || !ctl.view?.gpu_enabled) return false;
      const from = anchor();
      if (!from || !atOrAfter(node, from)) return false;
      ctl.gpu = true;
      ctl.moved = true;
      return true;
    };
    // The server refused the move; the CPU kernel is untouched.
    ctl.leave = () => {
      ctl.gpu = false;
      ctl.moved = false;
    };
    // What the fresh-kernel banner says when the fresh kernel IS the move.
    ctl.freshText = () => {
      if (!ctl.moved) return "";
      ctl.moved = false;
      return `Starting an ${ctl.view.gpu_type} GPU for this part — about a minute. Setup is ` +
        "restored on it; earlier answers stay above, re-run any whose names you need.";
    };
    return ctl;
  };

  return { attach, load, bannerText };
})();
