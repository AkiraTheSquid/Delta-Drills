/* ================================================================
   ARENA NOTEBOOK OUTPUTS — what the cells you ran already answered
   ================================================================

   Seth, 2026-09-06: "it also has like the saving of which code cells you ran
   rather than having to run it every single time ... it would have already had
   the code cells that you ran and all that so that you don't have to run them
   again."

   `arena-notebook.js` already persists the cell SOURCE (`dd_arena_cells:*`).
   What it threw away on every reload was the other half of a notebook: the
   `[n]` counters, the printed output, the figures. Coming back to a section
   you had worked through showed a page of untouched cells, which reads as "you
   did none of this" — and made re-running from the top the only way to get the
   page back.

   This file stores one record per code cell, keyed by notebook slug, and paints
   them back when the notebook renders.

   🔴 A RESTORED OUTPUT IS A PHOTOGRAPH, NOT A LIVE SESSION. The Python names
   those cells bound live in a Modal sandbox that idles out after 30 minutes and
   is capped at four hours. Painting the outputs back does not bring `t`,
   `model` or `rays` back with them, and a page that looked fully run while the
   kernel was gone would be the worst of the three states. So the view asks
   `DeltaKernel.status()` whether the learner's live session is still the one
   these outputs came from, and marks every restored cell `is-stale` when it is
   not — the same class the Restart-session path uses, which `syncCompletion`
   already refuses to count. The outputs stay readable either way; what changes
   is whether the notebook claims to be current.

   🔴 THIS IS A CACHE IN A 5 MB BOX. An ARENA notebook is up to 656 cells and a
   single matplotlib figure is a base64 PNG; plotly figures are worse. Storing
   them naively fills the origin's localStorage and then EVERY other feature's
   `setItem` starts throwing — the cell edits above, the scroll position, the
   practice queue. Hence three defences, in this order: trim each record as it
   is written (head+tail of the text, a size ceiling per mimebundle), shrink the
   whole store to a byte budget by dropping figures from the oldest cells first,
   and on an actual quota error evict OTHER notebooks' stores one at a time
   before giving up. Losing a remembered output costs one re-run. Losing the
   learner's edits costs their work.
   ================================================================ */

const ArenaNotebookOutputs = (() => {
  const PREFIX = "dd_arena_out:";
  const KEY = (slug) => `${PREFIX}${slug}`;
  const VERSION = 1;

  /* Jupyter truncates the MIDDLE of a long output, and so does this: the first
     lines say what the cell started doing and the last say how it ended, which
     is where a traceback lives. Keeping only the head loses every error. */
  const TEXT_HEAD = 1500;
  const TEXT_TAIL = 2500;
  const ELISION = "\n\n… output trimmed to fit this browser's storage — re-run the cell to see all of it …\n\n";

  const MAX_BUNDLE_CHARS = 48000; // one figure
  const MAX_STORE_CHARS = 400000; // one notebook
  const MAX_NOTEBOOKS = 4; // how many sections keep their outputs at once

  /* One notebook is on screen at a time, so one store is cached at a time.
     Everything else re-reads. */
  let slugLoaded = null;
  let store = null;
  let flushTimer = 0;
  let refused = false; // this browser told us no, and the view has been told once
  /* Cells this tab deleted. localStorage is shared by every tab on the origin
     and the flush below merges what it finds there back in, so without this a
     deleted cell's record is resurrected by the next write. */
  let dropped = new Set();

  const _blank = () => ({ version: VERSION, at: Date.now(), cells: {} });

  /* A cheap 32-bit hash of the source the record came from.

     🔴 A CELL ID IS NOT A PROMISE ABOUT THE CODE UNDER IT. These notebooks are
     compiled from an upstream checkout (`scripts/compile_arena_notebooks.py`),
     and a recompile can hand the SAME id a different body — upstream edited the
     exercise, we bumped `ARENA_SHA`. Restoring by id alone then paints last
     week's output under this week's code and calls it current, with nothing
     anywhere throwing. Storing what the record was produced FROM makes that
     detectable, and a mismatch is exactly the state `is-stale` names. Codex
     raised it, 2026-09-06. */
  const _fingerprint = (source) => {
    const text = String(source == null ? "" : source);
    let hash = 5381;
    for (let i = 0; i < text.length; i += 1) hash = ((hash * 33) ^ text.charCodeAt(i)) >>> 0;
    return `${text.length}:${hash.toString(36)}`;
  };

  const _readRaw = (slug) => {
    try {
      const parsed = JSON.parse(localStorage.getItem(KEY(slug)) || "null");
      if (!parsed || parsed.version !== VERSION || typeof parsed.cells !== "object") return null;
      return parsed;
    } catch (_) {
      return null;
    }
  };

  /* Every stored notebook and when it was last written, oldest first. Used to
     decide what to evict — both routinely and under a quota error. */
  const _others = (slug) => {
    const rows = [];
    let index = 0;
    try {
      for (; index < localStorage.length; index += 1) {
        const key = localStorage.key(index);
        if (!key || !key.startsWith(PREFIX) || key === KEY(slug)) continue;
        let at = 0;
        try {
          at = JSON.parse(localStorage.getItem(key) || "{}").at || 0;
        } catch (_) {
          at = 0; // unparseable is the first thing worth dropping
        }
        rows.push({ key, at });
      }
    } catch (_) {
      return [];
    }
    return rows.sort((a, b) => a.at - b.at);
  };

  const _drop = (key) => {
    try {
      localStorage.removeItem(key);
    } catch (_) {}
  };

  const _trimText = (text) => {
    const value = String(text == null ? "" : text);
    if (value.length <= TEXT_HEAD + TEXT_TAIL) return value;
    return value.slice(0, TEXT_HEAD) + ELISION + value.slice(-TEXT_TAIL);
  };

  /* A mimebundle too big to keep becomes a note that says so, rather than
     disappearing: "this cell drew something" is information, and a cell whose
     figure silently vanished looks like a cell that printed nothing. */
  const _trimBundles = (outputs) => {
    if (!Array.isArray(outputs) || !outputs.length) return [];
    const kept = [];
    for (const bundle of outputs) {
      if (!bundle || typeof bundle !== "object") continue;
      let size = 0;
      try {
        size = JSON.stringify(bundle).length;
      } catch (_) {
        continue; // not serialisable, so it was never storable
      }
      kept.push(
        size <= MAX_BUNDLE_CHARS
          ? bundle
          : { "text/plain": "(figure too large to remember — re-run this cell to draw it)" },
      );
    }
    return kept;
  };

  /* Bring the whole store under its byte budget. Figures go before text, and
     the oldest runs go before the newest, because the cells a learner is
     working on are the ones they just ran. */
  const _shrink = (value) => {
    const size = () => JSON.stringify(value).length;
    if (size() <= MAX_STORE_CHARS) return value;
    const order = Object.keys(value.cells).sort(
      (a, b) => (value.cells[a].seq || 0) - (value.cells[b].seq || 0),
    );
    for (const id of order) {
      if (size() <= MAX_STORE_CHARS) return value;
      if (value.cells[id].outputs?.length) value.cells[id].outputs = [];
    }
    for (const id of order) {
      if (size() <= MAX_STORE_CHARS) return value;
      delete value.cells[id];
    }
    return value;
  };

  /* Write, and treat a quota error as something to make room for rather than
     something to report. Returns false only once every room-making move has
     been tried — that is the one case the view tells the learner about. */
  const _write = (slug, value) => {
    value.at = Date.now();
    _shrink(value);
    const attempt = () => {
      try {
        localStorage.setItem(KEY(slug), JSON.stringify(value));
        return true;
      } catch (_) {
        return false;
      }
    };
    if (attempt()) return true;
    for (const row of _others(slug)) {
      _drop(row.key);
      if (attempt()) return true;
    }
    // Last resort: this notebook's own figures. Its text is the cheap half and
    // the half that says whether a cell passed.
    for (const id of Object.keys(value.cells)) value.cells[id].outputs = [];
    if (attempt()) return true;
    _drop(KEY(slug));
    return false;
  };

  /* Routine housekeeping, not an emergency: keep the newest MAX_NOTEBOOKS
     stores and let the rest go. Runs after a successful write so the notebook
     just written is never the one dropped. */
  const _prune = (slug) => {
    const rows = _others(slug);
    const excess = rows.length - (MAX_NOTEBOOKS - 1);
    for (let i = 0; i < excess; i += 1) _drop(rows[i].key);
  };

  /* 🔴 TWO TABS ON THE SAME NOTEBOOK MUST NOT ERASE EACH OTHER. This module
     caches the whole store for as long as the notebook is open and then writes
     the whole thing back, so a tab that loaded an hour ago and flushes now
     would delete every record the other tab wrote in between — the same
     whole-document-overwrite shape that has bitten this app before. The write
     is therefore a MERGE: whatever is on disk and newer than our copy of the
     same cell wins, cell by cell, and only cells this tab deliberately dropped
     stay dropped. Every mutation stamps `at`, which is what "newer" means.
     Found by codex, 2026-09-06. */
  const _merged = (slug, mine) => {
    const disk = _readRaw(slug);
    if (!disk) return mine;
    for (const id of Object.keys(disk.cells)) {
      if (dropped.has(id)) continue;
      const ours = mine.cells[id];
      if (!ours || (disk.cells[id].at || 0) > (ours.at || 0)) mine.cells[id] = disk.cells[id];
    }
    return mine;
  };

  const _flush = (slug) => {
    clearTimeout(flushTimer);
    flushTimer = 0;
    if (slug !== slugLoaded || !store) return;
    _merged(slug, store);
    if (_write(slug, store)) {
      refused = false;
      _prune(slug);
    } else {
      refused = true;
    }
  };

  const _queue = (slug) => {
    clearTimeout(flushTimer);
    flushTimer = setTimeout(() => _flush(slug), 300);
  };

  /* ---------- the surface --------------------------------------------- */

  /* The records for one notebook. Cached, because `_render` asks once per cell
     and a 656-cell notebook must not parse the same JSON 656 times. */
  const load = (slug) => {
    if (!slug) return null;
    if (slugLoaded !== slug) {
      _flush(slugLoaded); // the notebook being left keeps what it had pending
      slugLoaded = slug;
      store = _readRaw(slug) || _blank();
      dropped = new Set();
    }
    return store;
  };

  const get = (slug, cellId) => {
    const value = load(slug);
    return (value && cellId && value.cells[cellId]) || null;
  };

  /* One cell finished. `text` is what the view PAINTED, captured before the
     rich outputs were appended — reading it back off the node afterwards would
     fold every figure's alt text into the stored text. */
  const record = (slug, cellId, { text = "", failed = false, seq = 0, outputs = [], source = "" } = {}) => {
    if (!slug || !cellId) return;
    const value = load(slug);
    if (!value) return;
    value.cells[cellId] = {
      text: _trimText(text),
      failed: !!failed,
      seq: Number(seq) || 0,
      stale: false,
      src: _fingerprint(source),
      at: Date.now(),
      outputs: _trimBundles(outputs),
    };
    dropped.delete(cellId);
    _queue(slug);
  };

  /* "This output no longer describes the code above it." Typing into a cell
     that has run, and the kernel restarting under the whole notebook, are the
     two ways to get here — the class is applied by the view either way, and
     this is how the fact survives a reload. */
  const markCellStale = (slug, cellId) => {
    const rec = get(slug, cellId);
    if (!rec || rec.stale) return;
    rec.stale = true;
    rec.at = Date.now();
    _queue(slug);
  };

  const markStale = (slug) => {
    const value = load(slug);
    if (!value) return;
    let changed = false;
    for (const id of Object.keys(value.cells)) {
      if (!value.cells[id].stale) {
        value.cells[id].stale = true;
        value.cells[id].at = Date.now();
        changed = true;
      }
    }
    if (changed) _queue(slug);
  };

  const forget = (slug, cellId) => {
    const value = load(slug);
    if (!value || !value.cells[cellId]) return;
    delete value.cells[cellId];
    dropped.add(cellId);
    _queue(slug);
  };

  const clear = (slug) => {
    clearTimeout(flushTimer);
    flushTimer = 0;
    if (slugLoaded === slug) {
      store = _blank();
      dropped = new Set();
    }
    _drop(KEY(slug));
  };

  /* Put one record back on screen. The classes matter as much as the text:
     `has-run` is what the contents rail counts, and `is-stale` is what stops
     it counting a cell whose session is gone. */
  const paint = (node, rec, stale = false) => {
    if (!node || !rec) return false;
    const out = node.querySelector(".nbv-out");
    if (!out) return false;
    out.classList.remove("hidden");
    out.classList.toggle("is-error", !!rec.failed);
    out.textContent = rec.text || "";
    window.DeltaCellOutputs?.render(out, Array.isArray(rec.outputs) ? rec.outputs : []);
    node.classList.add("has-run");
    node.classList.toggle("has-failed", !!rec.failed);
    node.classList.toggle("is-stale", !!rec.stale || !!stale);
    const count = node.querySelector(".nbv-count");
    if (count && rec.seq) count.textContent = `[${rec.seq}]`;
    return true;
  };

  /* Paint every record this notebook has back onto the cells that produced
     them, and report what was restored. `cellIdOf` is the VIEW's function, not
     a copy of it: the id a record is filed under is one expression owned by
     arena-notebook.js (`_cellIdOf`), and a second copy here would drift into
     restoring outputs under the wrong cells with nothing throwing. */
  const restoreInto = (root, slug, cellIdOf, sourceOf) => {
    const value = root && slug && load(slug);
    if (!value) return { restored: [], highest: 0 };
    const restored = [];
    let highest = 0;
    root.querySelectorAll(".nbv-cell.nbv-code").forEach((node) => {
      const rec = value.cells[cellIdOf(node)];
      if (!rec) return;
      /* The record was produced from code this cell no longer holds — a
         recompile, or an edit made in another tab. Show it, but never let it
         count as current. Derived on every restore rather than written back:
         it is a fact about the notebook in front of the learner NOW. */
      const moved = !!(rec.src && sourceOf && _fingerprint(sourceOf(node)) !== rec.src);
      if (!paint(node, rec, moved)) return;
      /* 🔴 THE NODES, NOT A COUNT. The kernel reconcile runs after an await and
         must only ever touch the cells it restored — see the caller. */
      node._ddRestored = true;
      restored.push(node);
      highest = Math.max(highest, rec.seq || 0);
    });
    return { restored, highest };
  };

  /* 🔴 A RESTORED OUTPUT IS A PHOTOGRAPH, AND THE LEARNER HAS TO BE TOLD WHEN
     IT IS ONLY THAT. The Modal sandbox behind an ARENA section's context idles
     out after 30 minutes; these records survive in localStorage and the NAMES
     DO NOT. A notebook that came back looking fully run over a dead kernel is a
     worse answer than either honest one, so the server is asked which session
     it is actually holding — `sessions[].context` + `alive`, practice/kernel.js.

     `alive === null` means the question could not be asked at all (signed out,
     an older backend, one bad request) and gets NO banner and NO stale marks: a
     warning nobody can act on is worse than silence, and staling on it throws
     away the contents rail's completion for nothing.

     🔴 ONLY THE CELLS THIS RENDER RESTORED, AND ONLY THE ONES NOT RUN SINCE.
     Sweeping every `.has-run` cell on the page marked a cell the learner ran
     WHILE this request was in flight — against a kernel that is by then
     perfectly alive — as stale, and persisted it that way. Codex, 2026-09-06. */
  const reconcile = async ({
    slug,
    context,
    restored = [],
    cellIdOf,
    stillOpen = () => true,
    banner = () => {},
    onChanged = () => {},
  } = {}) => {
    if (refused) {
      banner(
        "This browser is out of storage, so cell outputs will not be remembered " +
          "for next time. Your cell edits still are.",
        "warn",
      );
    }
    if (!restored.length) return;
    const alive = await window.DeltaKernel?.contextAlive?.(context);
    if (!stillOpen() || alive !== false) return;
    let marked = 0;
    for (const node of restored) {
      /* 🔴 `isConnected` as well as the flag: a restored cell the learner
         DELETED is still in this array, and counting it would announce that
         the session expired over a notebook where nothing visible changed.
         Codex, 2026-09-06. */
      if (node._ddRestored !== true || !node.isConnected) continue;
      node.classList.add("is-stale");
      markCellStale(slug, cellIdOf ? cellIdOf(node) : "");
      marked += 1;
    }
    if (!marked) return;
    onChanged();
    banner(
      "These outputs are from your last visit — the Python session has expired " +
        "since, so nothing is defined yet. Re-run the imports before the cells " +
        "that use them.",
      "info",
    );
  };

  // A close or a reload is the one exit with no later chance to flush.
  window.addEventListener("pagehide", () => _flush(slugLoaded));

  return {
    load,
    get,
    record,
    markCellStale,
    markStale,
    forget,
    clear,
    paint,
    restoreInto,
    reconcile,
  };
})();

window.ArenaNotebookOutputs = ArenaNotebookOutputs;
