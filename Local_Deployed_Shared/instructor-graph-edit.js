/* INSTRUCTOR GRAPH EDIT — direct manipulation of the lesson graph.
 *
 * Seth, 2026-08-27: "you should be able to click on an edge without deleting
 * it, and whenever you click on it, it should show the information about it on
 * the right, and you should be able to … either delete it or change its
 * direction… you should be able to click the plus button on a node for the top
 * edge of it and when you do that it should create a new node right there
 * connected to it. Or… drag to another node."
 *
 * WHAT AN EDIT IS. Nothing here writes the curriculum. The graph is built from
 * lessons/kc_registry.json and its edges are the unlock lattice the tutor
 * serves from — a browser is the wrong place to reshape that, and the endpoint
 * behind this deliberately has no repair runner. So every gesture is a
 * PROPOSAL: it is drawn on the live graph immediately, so the instructor is
 * arguing with a picture of what they mean instead of a form, and it is queued
 * as a /api/practice/graph-feedback entry for a human to act on. Leaving the
 * screen puts the graph back exactly as it was found.
 *
 * WHOSE GRAPH THIS IS. `cy` is not ours — instructor-review.js borrows the
 * Knowledge Graph tab's live Cytoscape and hands it here. That is why every
 * edit is expressed as one primitive pair, `remove` + `add`, and why the
 * ledger keeps the collection each removal returned: `.restore()` on that
 * collection is Cytoscape's own exact inverse, so detach() can hand back the
 * element the learner tab is about to render. An edit that could not be
 * undone by that pair does not belong in this file.
 *
 * The ledger OUTLIVES detach on purpose. An instructor who tabs away mid-review
 * and comes back should find their proposals still on the map, so attach()
 * replays the ledger in order and detach() reverts it in reverse.
 *
 * Styling is INLINE on the proposed elements, never a stylesheet rule: the
 * stylesheet belongs to lesson-graph.js and a selector added here would outlive
 * the visit. lesson-graph.js's recolor() writes node background on its own
 * schedule, so proposals are marked by BORDER — the one channel it does not
 * repaint — and re-marked when it announces a repaint anyway.
 */
(() => {
  "use strict";

  const PROPOSED_EDGE = { "line-color": "#3ddc84", "target-arrow-color": "#3ddc84", "line-style": "dashed", "width": 3, "opacity": 1 };
  const PROPOSED_NODE = { "border-color": "#3ddc84", "border-width": 4, "border-style": "dashed" };
  /* Roughly one dagre rank (rankSep 150) above the node the ✛ was pressed on,
     which is where the layout would have put it. Model units. */
  const RANK_UP = 130;
  const DRAG_SLOP = 6;   // px of movement below which a press is a click

  const EDGE_TAGS = [
    ["wrong_direction", "Points the wrong way"],
    ["should_not_exist", "Not really a prerequisite"],
    ["good", "👍 Good edge"],
  ];
  const NODE_TAGS = [
    ["mislabeled", "Mislabeled"],
    ["wrong_topic", "Filed under the wrong lesson"],
    ["should_not_exist", "Shouldn't exist"],
    ["good", "👍 Good concept"],
  ];

  const esc = (s) =>
    String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");

  /* ── module state ─────────────────────────────────────────────────
     `edits` is the ledger and survives detach; everything else is per-visit. */
  const edits = [];
  let newSeq = 0;
  let cy = null;
  let frame = null;      // #ir-kg-frame — where the panel lives
  let postFn = null;
  let panel = null, handle = null, band = null, bandLine = null;
  let onFocusCb = null;  // instructor-review.js names the ✛'d concept on its toolbar button
  let selection = null;  // {type:"edge"|"node"|"new", ele, edit?}
  let focusNode = null;  // the node the ✛ handle is attached to
  let armed = false;     // two-tap "propose a missing edge" path
  let armedFrom = null;
  /* A ✛ drag binds its move/up on WINDOW, so it outlives anything that
     happens on the page underneath — including the instructor navigating away
     mid-drag, which detaches the editor and nulls `cy`. One cleanup, called
     from the drop AND from detach, or the next pointer event asks a null
     cytoscape where a node is. */
  let dragCleanup = null;
  let draftNote = "";    // typed before anything was staged; see stage()
  let repaintPending = false;

  const label = (id) => {
    if (!cy) return id;
    const n = cy.getElementById(id);
    return (n && n.length && n.data("label")) || id;
  };

  /* ── the one primitive ────────────────────────────────────────────
     Every structural edit is "take these elements out, put those in". apply()
     records what it took so revert() can put it back; a pure opinion (a tag
     with no structural claim) supplies neither and is a no-op both ways. */
  const applyEdit = (e) => {
    if (e.applied || !cy) return;
    if (e.remove) e.removed = e.remove(cy);
    if (e.add) {
      const defs = e.add(cy);
      if (defs && defs.length) {
        e.added = cy.add(defs);
        e.added.nodes().style(PROPOSED_NODE);
        e.added.edges().style(PROPOSED_EDGE);
      }
    }
    e.applied = true;
  };

  const revertEdit = (e) => {
    if (!e.applied) return;
    if (e.added) { e.added.remove(); e.added = null; }
    // Restore AFTER the additions are gone: a reversed edge's replacement and
    // the original share a source/target pair, never an id, so the order is
    // about intent rather than collision — the graph is only ever momentarily
    // in a state with both.
    if (e.removed) { e.removed.restore(); e.removed = null; }
    e.applied = false;
  };

  /* lesson-graph.js repaints node background and border-style whenever the
     learner model moves. It has no idea these elements are proposals, so the
     dashed green border goes with it — put it back rather than fight it. */
  const remark = () => {
    edits.forEach((e) => {
      if (!e.applied || !e.added) return;
      e.added.nodes().style(PROPOSED_NODE);
      e.added.edges().style(PROPOSED_EDGE);
    });
  };

  const stage = (edit) => {
    // Typing and staging are two gestures and either can come first. A note
    // written before the button was pressed belongs to the edit it explains.
    if (!edit.note && draftNote) { edit.note = draftNote; draftNote = ""; }
    edits.push(edit);
    applyEdit(edit);
    render();
    return edit;
  };

  const unstage = (edit, quiet) => {
    const i = edits.indexOf(edit);
    if (i < 0) return;
    /* Dependents FIRST, and only proposed nodes have any: an edge drawn from a
       concept that does not exist yet cannot outlive it. Without this, undoing
       the concept leaves an edge edit naming a missing source, which survives
       until the next attach() replays it and cytoscape throws on the add. */
    if (edit.nodeId) {
      edits
        .filter((e) => e !== edit && dependsOn(e, edit.nodeId))
        .forEach((e) => unstage(e, true));
    }
    const j = edits.indexOf(edit);
    if (j < 0) return;
    revertEdit(edit);
    edits.splice(j, 1);
    if (selection && selection.edit === edit) selection.edit = null;
    if (!quiet) render();
  };

  const dependsOn = (edit, nodeId) =>
    edit.entry.source === nodeId || edit.entry.target === nodeId;

  /* ── edit constructors ───────────────────────────────────────────── */
  const edgeKey = (s, t) => `${s} → ${t}`;
  const findEdge = (s, t) =>
    cy.edges().filter((e) => e.data("source") === s && e.data("target") === t);

  const deleteEdgeEdit = (s, t) => ({
    row: `Delete ${label(s)} → ${label(t)}`,
    key: `del:${edgeKey(s, t)}`,
    note: "",
    entry: { kind: "edge", source: s, target: t, edge_type: "prereq", tag: "should_not_exist" },
    remove: () => findEdge(s, t).remove(),
  });

  const addEdgeEdit = (s, t) => ({
    row: `Add ${label(s)} → ${label(t)}`,
    key: `add:${edgeKey(s, t)}`,
    note: "",
    entry: { kind: "missing_edge", source: s, target: t, edge_type: "prereq", tag: "proposed" },
    add: () => [{ data: { id: `ir-e-${++newSeq}`, source: s, target: t } }],
  });

  /* Direction is one edit, not two, because "reverse this arrow" is one claim
     and undoing it must put the original arrow back in one move. `both` keeps
     the original and proposes the return arrow, which is exactly what the log
     should say: the existing prerequisite stands, a second one is missing. */
  const directionEdit = (s, t, mode) => {
    const base = {
      key: `dir:${edgeKey(s, t)}`,
      note: "",
      add: () => [{ data: { id: `ir-e-${++newSeq}`, source: t, target: s } }],
    };
    if (mode === "reverse") {
      return {
        ...base,
        mode,
        row: `Reverse ${label(s)} → ${label(t)}`,
        entry: { kind: "edge", source: s, target: t, edge_type: "prereq", tag: "wrong_direction" },
        remove: () => findEdge(s, t).remove(),
      };
    }
    return {
      ...base,
      mode,
      row: `Make ${label(s)} ↔ ${label(t)} bidirectional`,
      /* Not `wrong_direction`: the drawn arrow is not wrong, the missing one
         is. Filed as the proposal it is, so a maintainer reading the log adds
         an edge instead of flipping one. */
      entry: { kind: "missing_edge", source: t, target: s, edge_type: "prereq", tag: "proposed" },
    };
  };

  const deleteNodeEdit = (id) => ({
    row: `Delete concept “${label(id)}”`,
    key: `del:${id}`,
    note: "",
    entry: { kind: "node", source: id, target: null, edge_type: null, tag: "should_not_exist" },
    // Cytoscape removes a node's edges with it and returns them in the same
    // collection, so restore() brings the whole neighbourhood back.
    remove: () => cy.getElementById(id).remove(),
  });

  const addNodeEdit = (id, title, from, pos) => ({
    row: `New concept “${title}”`,
    key: `new:${id}`,
    note: "",
    title,
    nodeId: id,
    /* `label` is a real field on the endpoint, not a note convention: the id
       here is a placeholder this browser minted, so the NAME is the only part
       of a proposed concept a maintainer can act on. */
    entry: { kind: "missing_node", source: id, target: from || null, edge_type: null, tag: "proposed", label: title },
    add: () => {
      const defs = [{ data: { id, label: title }, position: pos }];
      if (from) defs.push({ data: { id: `ir-e-${++newSeq}`, source: from, target: id } });
      return defs;
    },
  });

  const flagEdit = (sel, tag, tagLabel) => ({
    row: `${sel.type === "edge" ? `${label(sel.src)} → ${label(sel.tgt)}` : `“${label(sel.id)}”`}: ${tagLabel}`,
    key: `flag:${sel.type === "edge" ? edgeKey(sel.src, sel.tgt) : sel.id}`,
    note: "",
    entry:
      sel.type === "edge"
        ? { kind: "edge", source: sel.src, target: sel.tgt, edge_type: "prereq", tag }
        : { kind: "node", source: sel.id, target: null, edge_type: null, tag },
  });

  /* One structural claim per thing. Re-choosing a direction, or deleting an
     edge you had already reversed, replaces the earlier edit rather than
     stacking a contradiction into the log. */
  const replaceByKeyPrefix = (prefix) => {
    for (let i = edits.length - 1; i >= 0; i--) {
      if (edits[i].key.startsWith(prefix)) unstage(edits[i]);
    }
  };
  const structuralFor = (a, b) =>
    edits.find((e) => !e.key.startsWith("flag:") && (e.key.endsWith(edgeKey(a, b)) || e.key.endsWith(edgeKey(b, a))));

  /* ── the ✛ handle ────────────────────────────────────────────────
     Cytoscape draws to a canvas and has no DOM per node, so the handle is an
     HTML button positioned over the container from the node's RENDERED box —
     which is why it has to be re-placed on pan, zoom and node drags rather
     than parented to anything. It lives in `.kg2-graph`, the overlay layer the
     Fit control and the legend already sit in, so it inherits that stacking
     context instead of inventing one. */
  const cyContainer = () => (cy ? cy.container() : null);

  const placeHandle = () => {
    if (!handle) return;
    if (!focusNode || focusNode.removed() || !cy) { handle.classList.add("hidden"); return; }
    const host = handle.parentElement;
    const c = cyContainer();
    if (!host || !c) return;
    const bb = focusNode.renderedBoundingBox();
    const cr = c.getBoundingClientRect();
    const hr = host.getBoundingClientRect();
    const dx = cr.left - hr.left, dy = cr.top - hr.top;
    const x = dx + (bb.x1 + bb.x2) / 2;
    const y = dy + bb.y1;
    // Off-screen nodes would otherwise park the handle on the pane's edge,
    // where it looks like it belongs to whatever is under it.
    const inside = bb.x2 > 0 && bb.x1 < cr.width && bb.y2 > 0 && bb.y1 < cr.height;
    handle.classList.toggle("hidden", !inside);
    handle.style.left = `${x}px`;
    handle.style.top = `${y}px`;
  };

  const schedulePlace = () => {
    if (repaintPending) return;
    repaintPending = true;
    requestAnimationFrame(() => { repaintPending = false; placeHandle(); });
  };

  const setFocus = (node) => {
    focusNode = node && node.length ? node : null;
    placeHandle();
    /* The toolbar's "Flag this concept" button names its target, so an
       instructor can see WHICH concept a flag is about to be filed against —
       the lesson pane can be showing a different one if they tapped, read,
       then tapped again. */
    if (onFocusCb) onFocusCb(focusNode ? focusNode.data("label") || focusNode.id() : null);
  };

  /* ── drag from the handle ─────────────────────────────────────────
     One gesture, two meanings, told apart by where it ends: on a bubble it
     proposes an edge to that concept, on empty canvas it proposes a NEW
     concept there. A press that never really moved is the click Seth asked
     for — a new concept one rank up — so it is the same code path with the
     drop point supplied by the layout instead of the pointer. */
  const modelPos = (clientX, clientY) => {
    const c = cyContainer();
    const r = c.getBoundingClientRect();
    const pan = cy.pan(), zoom = cy.zoom();
    return { x: (clientX - r.left - pan.x) / zoom, y: (clientY - r.top - pan.y) / zoom };
  };

  const nodeAtClient = (clientX, clientY) => {
    const c = cyContainer();
    const r = c.getBoundingClientRect();
    const x = clientX - r.left, y = clientY - r.top;
    let hit = null;
    cy.nodes().forEach((n) => {
      if (hit) return;
      const bb = n.renderedBoundingBox();
      if (x >= bb.x1 && x <= bb.x2 && y >= bb.y1 && y <= bb.y2) hit = n;
    });
    return hit;
  };

  /* The drop-target highlight is REMOVED, never written back. Reading the old
     value and re-setting it would leave an inline border-color on a node that
     had none — permanently outranking lesson-graph.js's own stylesheet, which
     is where a bubble's border comes from and where a future change to it
     would be made. removeStyle drops the override and the node goes back to
     being styled by its owner. A proposed node is the exception and is
     re-marked, because its border IS an inline style of ours. */
  const clearHover = (node) => {
    node.removeStyle("border-color");
    if (node.id().startsWith("new:")) node.style(PROPOSED_NODE);
  };

  const showBand = (fromNode, clientX, clientY) => {
    if (!band) return;
    const c = cyContainer();
    const host = band.parentElement;
    if (!c || !host) return;
    const cr = c.getBoundingClientRect(), hr = host.getBoundingClientRect();
    const bb = fromNode.renderedBoundingBox();
    const x1 = cr.left - hr.left + (bb.x1 + bb.x2) / 2;
    const y1 = cr.top - hr.top + bb.y1;
    band.classList.remove("hidden");
    bandLine.setAttribute("x1", x1);
    bandLine.setAttribute("y1", y1);
    bandLine.setAttribute("x2", clientX - hr.left);
    bandLine.setAttribute("y2", clientY - hr.top);
  };

  const hideBand = () => { if (band) band.classList.add("hidden"); };

  const onHandleDown = (ev) => {
    if (!focusNode || !cy) return;
    ev.preventDefault();
    ev.stopPropagation();
    const from = focusNode;
    const x0 = ev.clientX, y0 = ev.clientY;
    let moved = false;
    let over = null;

    const finish = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      dragCleanup = null;
      hideBand();
      if (over && cy) clearHover(over);
      over = null;
    };

    const move = (e) => {
      if (!cy) { finish(); return; }
      if (!moved && Math.hypot(e.clientX - x0, e.clientY - y0) < DRAG_SLOP) return;
      moved = true;
      showBand(from, e.clientX, e.clientY);
      const n = nodeAtClient(e.clientX, e.clientY);
      const next = n && n.id() !== from.id() ? n : null;
      if (next !== over) {
        if (over) clearHover(over);
        if (next) next.style({ "border-color": "#3ddc84" });
        over = next;
      }
    };

    const up = (e) => {
      finish();
      if (!cy || e.type === "pointercancel") return;
      const target = moved ? nodeAtClient(e.clientX, e.clientY) : null;
      if (target && target.id() !== from.id()) {
        if (structuralFor(from.id(), target.id())) {
          setStatus("There is already a proposal for that pair — undo it first.");
          return;
        }
        if (findEdge(from.id(), target.id()).length) {
          setStatus("That prerequisite already exists.");
          return;
        }
        const edit = stage(addEdgeEdit(from.id(), target.id()));
        selectEdit(edit, { type: "edge", src: from.id(), tgt: target.id() });
        return;
      }
      const p = moved
        ? modelPos(e.clientX, e.clientY)
        : { x: from.position("x"), y: from.position("y") - RANK_UP };
      newConcept(from.id(), p);
    };

    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    dragCleanup = finish;
  };

  const newConcept = (fromId, pos) => {
    const id = `new:${Date.now().toString(36)}-${++newSeq}`;
    const edit = stage(addNodeEdit(id, "New concept", fromId, pos));
    selection = { type: "new", edit, id };
    render();
    const input = panel && panel.querySelector("#ir-new-name");
    if (input) { input.focus(); input.select(); }
  };

  /* ── selection ──────────────────────────────────────────────────── */
  const selectEdit = (edit, sel) => { draftNote = ""; selection = { ...sel, edit }; render(); };

  const selectEdge = (e) => {
    armed = false; armedFrom = null;
    draftNote = "";
    selection = { type: "edge", src: e.data("source"), tgt: e.data("target") };
    selection.edit = structuralFor(selection.src, selection.tgt) || null;
    render();
  };

  const selectNode = (id) => {
    draftNote = "";
    selection = { type: "node", id, edit: edits.find((e) => e.key === `del:${id}`) || null };
    render();
  };

  const setStatus = (t) => {
    const s = panel && panel.querySelector("#ir-insp-status");
    if (!s) return;
    s.textContent = t || "";
    s.classList.toggle("hidden", !t);
  };

  /* ── the inspector ────────────────────────────────────────────────
     Right-hand side, per Seth. It OVERLAYS the lesson pane rather than taking
     a column beside it: the pane is the reason a bubble tap is worth making,
     and an edge — which has no lesson — is the only selection that needs this
     much room. Closing it hands the pane straight back. */
  const tagRow = (tags, sel) =>
    tags
      .map(([tag, text]) => `<button type="button" class="ghost ir-flag" data-tag="${esc(tag)}">${esc(text)}</button>`)
      .join("");

  /* A sent proposal stays on the list and stays DRAWN — the instructor is
     still looking at the map they argued for — but it is struck through and no
     longer counted, because the log is append-only and pressing Send twice
     would file the same claim twice. Undo still works on it: it takes the
     drawing back off the graph, which is all this screen can take back. */
  const unsent = () => edits.filter((e) => !e.sent);

  const ledgerHtml = () => {
    if (!edits.length) return '<p class="ir-hint">Nothing proposed yet.</p>';
    return (
      `<ul class="ir-ledger">` +
      edits
        .map(
          (e, i) =>
            `<li${e.sent ? ' class="is-sent"' : ""}><span>${esc(e.row)}${e.sent ? " · sent" : ""}</span>` +
            `<button type="button" class="ir-undo" data-i="${i}" title="Undo">✕</button></li>`
        )
        .join("") +
      `</ul>`
    );
  };

  const dirHtml = () => {
    const s = selection.src, t = selection.tgt;
    const st = stagedHere();
    const cur = st && st.key.startsWith("dir:") ? st.mode : "forward";
    const btn = (mode, text) =>
      `<button type="button" class="ir-dir${cur === mode ? " ir-dir--on" : ""}" data-dir="${mode}">${text}</button>`;
    return (
      `<div class="ir-insp-label">Direction</div><div class="ir-dirs">` +
      btn("forward", `${esc(label(s))} → ${esc(label(t))}`) +
      btn("reverse", `${esc(label(t))} → ${esc(label(s))}`) +
      btn("both", `${esc(label(s))} ↔ ${esc(label(t))}`) +
      `</div>`
    );
  };

  /* What is STAGED for the current selection, read off the ledger rather than
     off `selection.edit`. Flagging an edge you had already deleted overwrites
     that slot, and the panel would then offer "Delete this edge" a second time
     for an edge that is already gone from the map. */
  const stagedHere = () => {
    if (!selection) return null;
    if (selection.type === "new") return selection.edit;
    if (selection.type === "node") return edits.find((e) => e.key === `del:${selection.id}`) || null;
    return structuralFor(selection.src, selection.tgt) || null;
  };

  const render = () => {
    if (!panel) return;
    const body = panel.querySelector("#ir-insp-body");
    const staged = stagedHere();
    const deleted = !!staged && staged.key.startsWith("del:");
    let html = "";

    if (!selection) {
      html =
        '<p class="ir-hint">Tap an arrow to inspect a prerequisite. Tap a bubble to read its ' +
        'lesson — then press <strong>✛</strong> above it to add a concept, or drag ✛ onto another ' +
        'bubble to connect them.</p>';
    } else if (selection.type === "new") {
      html =
        `<div class="ir-insp-title">New concept</div>` +
        `<label class="ir-insp-label" for="ir-new-name">Name</label>` +
        `<input id="ir-new-name" type="text" maxlength="120" value="${esc(selection.edit.title)}">` +
        `<p class="ir-hint">Proposed under ${esc(label(selection.edit.entry.target || ""))}. It is a suggestion on ` +
        `this screen only — nothing is added to the curriculum until a maintainer acts on it.</p>`;
    } else if (selection.type === "edge") {
      html =
        `<div class="ir-insp-title">${esc(label(selection.src))} <span class="ir-arrow">→</span> ${esc(label(selection.tgt))}</div>` +
        `<p class="ir-hint">Prerequisite edge — ${esc(label(selection.src))} must be learnable before ` +
        `${esc(label(selection.tgt))} unlocks.</p>` +
        (deleted ? '<p class="ir-hint ir-warn">Proposed for deletion.</p>' : dirHtml()) +
        `<div class="ir-insp-acts">` +
        (deleted
          ? `<button type="button" class="ghost" id="ir-restore">Undo delete</button>`
          : `<button type="button" class="ghost ir-danger" id="ir-del-edge">Delete this edge</button>`) +
        `</div>` +
        `<div class="ir-insp-label">Or just flag it</div><div class="ir-tags">${tagRow(EDGE_TAGS)}</div>`;
    } else {
      html =
        `<div class="ir-insp-title">${esc(label(selection.id))}</div>` +
        `<div class="ir-insp-acts">` +
        (deleted
          ? `<button type="button" class="ghost" id="ir-restore">Undo delete</button>`
          : `<button type="button" class="ghost ir-danger" id="ir-del-node">Delete this concept</button>`) +
        `</div>` +
        `<div class="ir-insp-label">Or just flag it</div><div class="ir-tags">${tagRow(NODE_TAGS)}</div>`;
    }

    if (selection) {
      html +=
        `<label class="ir-insp-label" for="ir-insp-note">Note</label>` +
        `<textarea id="ir-insp-note" maxlength="5000" data-autogrow ` +
        `placeholder="Why, or what it should be instead…"></textarea>`;
    }

    const n = unsent().length;
    html +=
      `<div class="ir-insp-label">Proposed changes <span class="ir-count">${edits.length}</span></div>` +
      ledgerHtml() +
      `<button type="button" class="primary" id="ir-insp-send"${n ? "" : " disabled"}>` +
      (n ? `Send ${n} ${n === 1 ? "proposal" : "proposals"}` : edits.length ? "All sent" : "Nothing to send") +
      `</button>` +
      `<span class="ir-insp-status hidden" id="ir-insp-status"></span>`;

    body.innerHTML = html;
    panel.classList.remove("hidden");

    const note = body.querySelector("#ir-insp-note");
    if (note) {
      note.value = (selection.edit && selection.edit.note) || draftNote || "";
      /* The note belongs to the EDIT, not to the panel: staging is a separate
         gesture from typing, and either can come first. Live-binding is what
         lets an instructor delete an edge and then explain why. */
      note.addEventListener("input", () => {
        if (selection && selection.edit) selection.edit.note = note.value;
        else draftNote = note.value;
      });
      // Assigning .value fires no input event, and the box was display:none a
      // line ago — both are cases autogrow.js cannot see for itself.
      if (window.DDAutoGrow) window.DDAutoGrow.grow(note);
    }
    placeHandle();
  };

  /* ── panel wiring (delegated: render() rewrites this subtree) ────── */
  const onPanelClick = (ev) => {
    const t = ev.target.closest("button, .ir-undo");
    if (!t) return;

    if (t.classList.contains("ir-undo")) {
      const e = edits[Number(t.dataset.i)];
      if (e) unstage(e);
      return;
    }
    if (t.id === "ir-insp-close") { selection = null; panel.classList.add("hidden"); return; }
    if (t.id === "ir-del-edge") {
      replaceByKeyPrefix(`dir:${edgeKey(selection.src, selection.tgt)}`);
      replaceByKeyPrefix(`add:${edgeKey(selection.src, selection.tgt)}`);
      selectEdit(stage(deleteEdgeEdit(selection.src, selection.tgt)), selection);
      return;
    }
    if (t.id === "ir-del-node") {
      selectEdit(stage(deleteNodeEdit(selection.id)), selection);
      setFocus(null);
      return;
    }
    if (t.id === "ir-restore") {
      const st = stagedHere();
      if (st) unstage(st);
      selection.edit = null;
      render();
      return;
    }
    if (t.dataset.dir) {
      replaceByKeyPrefix(`dir:${edgeKey(selection.src, selection.tgt)}`);
      if (t.dataset.dir === "forward") { selection.edit = null; render(); return; }
      selectEdit(stage(directionEdit(selection.src, selection.tgt, t.dataset.dir)), selection);
      return;
    }
    if (t.dataset.tag) {
      const tags = selection.type === "edge" ? EDGE_TAGS : NODE_TAGS;
      const text = (tags.find(([k]) => k === t.dataset.tag) || [])[1] || t.dataset.tag;
      replaceByKeyPrefix(`flag:${selection.type === "edge" ? edgeKey(selection.src, selection.tgt) : selection.id}`);
      const e = stage(flagEdit(selection, t.dataset.tag, text));
      selection.edit = e;
      render();
      return;
    }
    if (t.id === "ir-insp-send") send(t);
  };

  const onPanelInput = (ev) => {
    if (ev.target.id !== "ir-new-name" || !selection || !selection.edit) return;
    const v = ev.target.value.trim() || "New concept";
    selection.edit.title = v;
    selection.edit.entry.label = v;
    selection.edit.row = `New concept “${v}”`;
    const n = cy.getElementById(selection.edit.nodeId);
    if (n && n.length) n.data("label", v);
    const c = panel.querySelector(".ir-ledger li:nth-child(" + (edits.indexOf(selection.edit) + 1) + ") span");
    if (c) c.textContent = selection.edit.row;
  };

  const send = async (btn) => {
    const batch = unsent();
    if (!batch.length || !postFn) return;
    btn.disabled = true;
    setStatus("Sending…");
    let sent = 0, queued = 0;
    // Sequential on purpose: the endpoint appends to one JSON file per user
    // and a parallel burst would have several requests read the same list.
    for (const e of batch) {
      const res = await postFn({ ...e.entry, note: e.note || "", graph: "lesson-kc" });
      // Marked either way: the offline path queued it locally, and that queue
      // is what gets replayed — sending again would post it a second time.
      e.sent = true;
      if (res && res.sent) sent++; else queued++;
    }
    render();
    setStatus(queued ? `${sent} sent, ${queued} saved locally ✓` : `${sent} sent ✓`);
  };

  /* ── attach / detach ─────────────────────────────────────────────── */
  const buildChrome = () => {
    const graphPane = frame.querySelector(".kg2-graph");
    if (!graphPane) return false;

    if (!panel) {
      panel = document.createElement("aside");
      panel.className = "ir-insp hidden";
      panel.innerHTML =
        '<div class="ir-insp-head"><span>Inspector</span>' +
        '<button type="button" class="ir-insp-x" id="ir-insp-close" title="Close">✕</button></div>' +
        '<div class="ir-insp-body" id="ir-insp-body"></div>';
      panel.addEventListener("click", onPanelClick);
      panel.addEventListener("input", onPanelInput);
    }
    if (!handle) {
      handle = document.createElement("button");
      handle.type = "button";
      handle.className = "ir-handle hidden";
      handle.title = "Add a concept above this one — or drag onto another bubble to connect them";
      handle.textContent = "✛";
      handle.addEventListener("pointerdown", onHandleDown);
    }
    if (!band) {
      band = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      band.setAttribute("class", "ir-band hidden");
      bandLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
      bandLine.setAttribute("stroke", "#3ddc84");
      bandLine.setAttribute("stroke-width", "2.5");
      bandLine.setAttribute("stroke-dasharray", "6 5");
      band.appendChild(bandLine);
    }
    // Re-parented every attach: the graph is MOVED between pages, so the
    // overlay layer these hang off is a different box each visit.
    graphPane.appendChild(band);
    graphPane.appendChild(handle);
    frame.appendChild(panel);
    return true;
  };

  const onCyEvent = () => schedulePlace();

  const api = {
    /* cy is borrowed. Handlers are bound here and unbound in detach(), because
       leaving them on a graph the learner tab is showing would put a ✛ handle
       and an inspector on a screen that has no instructor on it. */
    attach(opts) {
      cy = opts.cy;
      frame = opts.frame;
      postFn = opts.post;
      onFocusCb = opts.onFocus || null;
      if (!cy || !frame || !buildChrome()) return false;
      edits.forEach(applyEdit);
      selection = null;
      setFocus(null);
      render();
      panel.classList.add("hidden");
      cy.on("tap", "edge", onEdgeTap);
      cy.on("tap", "node", onNodeTap);
      cy.on("tap", onBgTap);
      cy.on("pan zoom position", onCyEvent);
      window.addEventListener("resize", onCyEvent);
      window.addEventListener("delta:adaptive-state-changed", remark);
      return true;
    },

    detach() {
      if (!cy) return;
      cy.removeListener("tap", "edge", onEdgeTap);
      cy.removeListener("tap", "node", onNodeTap);
      cy.removeListener("tap", onBgTap);
      cy.removeListener("pan zoom position", onCyEvent);
      window.removeEventListener("resize", onCyEvent);
      window.removeEventListener("delta:adaptive-state-changed", remark);
      if (dragCleanup) dragCleanup();
      // Reverse order: an edit can depend on one staged before it (a new edge
      // hanging off a new node), and undoing the older one first would strand
      // the newer one's elements.
      for (let i = edits.length - 1; i >= 0; i--) revertEdit(edits[i]);
      hideBand();
      if (handle) handle.classList.add("hidden");
      if (panel) panel.classList.add("hidden");
      selection = null;
      setFocus(null);
      armed = false;
      armedFrom = null;
      cy = null;
    },

    /* The touch path for the same claim the ✛ drag makes: tap the
       prerequisite, then tap what it unlocks. Kept because a drag is a poor
       gesture on a phone, and because it is the only way to connect two
       bubbles that are not both on screen. */
    armMissingEdge() {
      armed = true;
      armedFrom = null;
      selection = null;
      render();
      setStatus("Tap the FIRST concept (the prerequisite).");
    },

    isActive: () => !!cy,
    pending: () => edits.length,
  };

  function onEdgeTap(evt) { if (cy) selectEdge(evt.target); }

  function onNodeTap(evt) {
    if (!cy) return;
    const n = evt.target;
    setFocus(n);
    if (armed && !armedFrom) {
      armedFrom = n.id();
      setStatus(`First concept: ${n.data("label") || n.id()}. Now tap what it unlocks.`);
      return;
    }
    if (armed && armedFrom && armedFrom !== n.id()) {
      const s = armedFrom;
      armed = false; armedFrom = null;
      if (findEdge(s, n.id()).length || structuralFor(s, n.id())) {
        setStatus("Those two are already connected or already proposed.");
        return;
      }
      selectEdit(stage(addEdgeEdit(s, n.id())), { type: "edge", src: s, tgt: n.id() });
      return;
    }
    /* A bubble tap belongs to lesson-graph.js — it opens the lesson, which is
       what an instructor is here to read. This runs beside it and only moves
       the ✛; the inspector opens from the toolbar button instead, so reading
       a lesson never costs you the pane it is in. */
  }

  function onBgTap(evt) {
    if (cy && evt.target === cy) { setFocus(null); selection = null; if (panel) panel.classList.add("hidden"); }
  }

  api.inspectFocused = () => {
    if (!focusNode) return false;
    selectNode(focusNode.id());
    return true;
  };
  api.focusedLabel = () => (focusNode ? focusNode.data("label") || focusNode.id() : null);

  window.DDGraphEdit = api;
})();
