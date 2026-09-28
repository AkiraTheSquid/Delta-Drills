/* ================================================================
   THE CHECKLIST DOCUMENT — everything about a day's list that does
   not need Tiptap loaded.

   A member row's right-hand column is a three-state checklist: each
   line is open → checked → X → open. This file owns the DOCUMENT that
   is; `groups_checklist.js` owns the editor that writes it.

   ── 🔴 IT IS DELTA NOTE'S DOCUMENT, NOT A LOOKALIKE ──────────────
   Ported from `shared/web-components/js/subgoals/` in Delta Note
   (`subgoal_content.js`, `subgoal_taskitem.js`'s `effectiveCompletion`,
   `subgoal_view.js`), because Seth asked for "the tiptap checkboxes
   with the three states" from that app's sub-goals, and the part worth
   reusing is the stored shape: a Tiptap `{v, doc}` blob whose task
   items carry a `completion` attr of 'open' | 'checked' | 'x'. Keep
   the wire format identical and a list can be moved between the two
   apps by copying a string; diverge and it cannot.

   ── 🔴 TWO WAYS TO DRAW ONE DOCUMENT, AND WHY ────────────────────
   Your own column is a live editor. Everybody else's is `renderDoc`,
   which walks the JSON and builds the DOM directly. That is not an
   optimisation talking: a group is up to twelve people, and mounting
   twelve Tiptap instances would load the bundle's node views, input
   rules and plugins twelve times for documents nobody in this tab may
   type into, and put eleven real contenteditables on screen that only
   refuse edits at the last moment.

   The markup `renderDoc` produces is the editor's, down to the
   wrapping `.ProseMirror` div, because ONE set of CSS rules in
   styles/groups.css paints both. If the editor's DOM changes this has
   to change with it — `groups_checklist.js` is the shape of record.

   Text only, and never `innerHTML`: a checklist is plain lines in
   every surface that writes one, and `textContent` is what makes
   reading somebody else's document injection-proof by construction.
   ================================================================ */

const DDChecklistDoc = (() => {
  /* The version stamp on a stored payload. It is written in exactly one
     place (`payloadFromDoc`) — a second spelling of the wrapper is a row
     the editor loads as empty, with no error. */
  const STORAGE_VERSION = 1;

  const MAX_HEADING = 6;

  /* The cycle, and the ONLY statement of it. `groups_checklist.js`'s
     checkbox imports this rather than restating it, so "what does the
     next press mean" has one answer on the page. */
  const nextCompletionState = (cur) =>
    cur === "open" ? "checked" : cur === "checked" ? "x" : "open";

  /* 🔴 The real state, derived from BOTH attrs — never from `checked`
     alone. Tiptap's own TaskItem is binary; the X lives in a custom
     `completion` attr beside it. Deriving rather than migrating is what
     heals a document written before the third state existed, and an item
     the `[x] ` input rule made (checked=true, completion defaulted). */
  const effectiveCompletion = (attrs) => {
    if (attrs && attrs.completion === "x") return "x";
    return attrs && attrs.checked ? "checked" : "open";
  };

  /* 🔴 THE THIRD STATE HAS TO REACH A SCREEN READER TOO. A native checkbox
     knows two states, so an X'd item announced as "not checked" — the same
     words as an untouched one, and the X is the whole point of the third
     state. `indeterminate` is the native tri-state and AT reads it as
     "mixed"; the explicit `aria-checked` says the same thing to anything
     that reads attributes rather than properties. The label matters just as
     much: the box is inside a <label> whose text lives in a contenteditable
     sibling, which is not an accessible name. */
  const STATE_WORD = { open: "not done", checked: "done", x: "won't do" };

  const paintCheckboxState = (input, state, text) => {
    input.checked = state === "checked";
    input.indeterminate = state === "x";
    input.setAttribute("aria-checked", state === "x" ? "mixed" : String(state === "checked"));
    const name = String(text || "").trim();
    input.setAttribute("aria-label", name ? `${name} — ${STATE_WORD[state]}` : STATE_WORD[state]);
  };

  /* A fresh list is one empty checkbox, so typing lands in a checkbox and
     Enter starts the next one. */
  const EMPTY_TASK_DOC =
    '<ul data-type="taskList"><li data-type="taskItem" data-checked="false"><p></p></li></ul>';

  /* ---- the JSON transforms ------------------------------------------ */

  /* Bullet lists → task lists at EVERY depth, each item given a coherent
     three-state attr. A person who typed `- ` got a bullet; this is what
     makes it a checkbox without a migration. */
  const taskify = (node) => {
    if (!node || typeof node !== "object") return node;
    const out = { ...node };
    if (out.type === "bulletList") out.type = "taskList";
    if (out.type === "listItem") out.type = "taskItem";
    if (out.type === "taskItem") {
      const state = effectiveCompletion(out.attrs);
      out.attrs = { ...(out.attrs || {}), checked: state === "checked", completion: state };
    }
    if (Array.isArray(out.content)) out.content = out.content.map(taskify);
    return out;
  };

  const hasTaskList = (node) => {
    if (!node || typeof node !== "object") return false;
    if (node.type === "taskList") return true;
    return (Array.isArray(node.content) ? node.content : []).some(hasTaskList);
  };

  const blockText = (node) =>
    (Array.isArray(node && node.content) ? node.content : [])
      .map((child) => (child && child.text) || "")
      .join("")
      .trim();

  /* A document with no checkbox list anywhere was written before this
     editor existed, or handed in as a seed: each of its written lines
     becomes an item. Blank lines and headings are left alone — they are
     structure, not un-migrated tasks. */
  const itemsFromLooseParagraphs = (doc) => {
    const out = [];
    (Array.isArray(doc.content) ? doc.content : []).forEach((node) => {
      if (!node || node.type !== "paragraph" || !blockText(node)) {
        out.push(node);
        return;
      }
      const item = {
        type: "taskItem",
        attrs: { checked: false, completion: "open" },
        content: [node],
      };
      const prev = out[out.length - 1];
      /* Consecutive lines join ONE list, so a migrated document reads as a
         list rather than as a stack of one-item lists a caret cannot move
         between. */
      if (prev && prev.type === "taskList") prev.content = [...prev.content, item];
      else out.push({ type: "taskList", content: [item] });
    });
    return { ...doc, content: out };
  };

  /* The stored shape, as one pure transform.

     🔴 It does NOT force everything into a checkbox. A checklist people
     actually keep has headings in it, blank lines between its sections and
     items nested under items — all three valid in this schema, and all
     three destroyed by the `selectAll().toggleTaskList()` shortcut Delta
     Note started with: wrapping a whole document rewrites a heading as an
     item, swallows the list after it INTO that item, and lifts every
     nested child back to the top level.

     🔴 And it is IDEMPOTENT. `normalizeEditor` decides "did anything
     change" by comparing against this, so a transform whose output is not
     its own fixed point makes every load look like an edit — which is a
     write per open of somebody's own column. */
  const normalizeDoc = (doc) => {
    if (!doc || typeof doc !== "object") return doc;
    const tasked = taskify(doc);
    return hasTaskList(tasked) ? tasked : itemsFromLooseParagraphs(tasked);
  };

  /* ---- reading a stored payload ------------------------------------- */

  const looseLines = (text) =>
    text
      .split("\n")
      .map((line) => line.replace(/^\s*[-*]\s+/, "").trim())
      .filter(Boolean);

  /* A stored payload as a DOCUMENT, always — never the HTML the editor is
     also willing to parse. The renderer below cannot parse HTML and must
     not start, so the legacy shape (a plain-text row) is turned into nodes
     here instead. */
  const docFromStored = (raw) => {
    const text = String(raw || "").trim();
    if (!text) return null;
    try {
      const parsed = JSON.parse(text);
      if (parsed && parsed.doc) return normalizeDoc(parsed.doc);
    } catch (_) {
      /* not JSON → the plain-text shape below */
    }
    const lines = looseLines(text);
    if (!lines.length) return null;
    return {
      type: "doc",
      content: [
        {
          type: "taskList",
          content: lines.map((line) => ({
            type: "taskItem",
            attrs: { checked: false, completion: "open" },
            content: [{ type: "paragraph", content: [{ type: "text", text: line }] }],
          })),
        },
      ],
    };
  };

  /* ---- counting ------------------------------------------------------ */

  const taskItemOwnText = (node) =>
    (Array.isArray(node.content) ? node.content : [])
      .filter((c) => c && c.type === "paragraph")
      .map((p) => (Array.isArray(p.content) ? p.content : []).map((t) => (t && t.text) || "").join(""))
      .join("")
      .trim();

  /* checked / total. An item with no text is NOT counted: the seeded blank
     checkbox would otherwise make every empty day read `0/1`. An X'd item
     counts toward the total but not the checked side — it is "won't do",
     not "done". */
  const countTaskItems = (node, acc) => {
    const out = acc || { checked: 0, total: 0 };
    if (!node || typeof node !== "object") return out;
    if (node.type === "taskItem" && taskItemOwnText(node)) {
      out.total += 1;
      if (effectiveCompletion(node.attrs) === "checked") out.checked += 1;
    }
    (Array.isArray(node.content) ? node.content : []).forEach((child) => countTaskItems(child, out));
    return out;
  };

  const countsFromStored = (raw) => {
    const doc = docFromStored(raw);
    return doc ? countTaskItems(doc) : { checked: 0, total: 0 };
  };

  /* ---- writing -------------------------------------------------------- */

  const payloadFromDoc = (doc) => JSON.stringify({ v: STORAGE_VERSION, doc });

  /* ---- the read-only renderer ----------------------------------------- */

  const inlineInto = (el, node) => {
    (Array.isArray(node && node.content) ? node.content : []).forEach((child) => {
      if (child && child.type === "hardBreak") {
        el.appendChild(document.createElement("br"));
        return;
      }
      if (child && typeof child.text === "string") {
        el.appendChild(document.createTextNode(child.text));
      }
    });
  };

  const appendChildren = (parent, node) => {
    (Array.isArray(node && node.content) ? node.content : []).forEach((child) => {
      const rendered = renderNode(child);
      if (rendered) parent.appendChild(rendered);
    });
  };

  function renderNode(node) {
    if (!node || typeof node !== "object") return null;
    switch (node.type) {
      case "taskList": {
        const ul = document.createElement("ul");
        ul.dataset.type = "taskList";
        appendChildren(ul, node);
        return ul;
      }
      case "taskItem": {
        const li = document.createElement("li");
        li.dataset.type = "taskItem";
        const state = effectiveCompletion(node.attrs);
        li.dataset.checked = state === "checked" ? "true" : "false";
        li.dataset.completion = state;
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "checkbox";
        paintCheckboxState(input, state, taskItemOwnText(node));
        /* 🔴 Both. `disabled` is what refuses the click; `tabIndex = -1` is
           what keeps eleven other people's checkboxes out of the tab order
           between a reader and their own list. */
        input.disabled = true;
        input.tabIndex = -1;
        label.append(input, document.createElement("span"));
        const content = document.createElement("div");
        appendChildren(content, node);
        li.append(label, content);
        return li;
      }
      case "bulletList":
      case "orderedList": {
        const list = document.createElement(node.type === "orderedList" ? "ol" : "ul");
        appendChildren(list, node);
        return list;
      }
      case "listItem": {
        const li = document.createElement("li");
        appendChildren(li, node);
        return li;
      }
      case "heading": {
        const level = Math.min(MAX_HEADING, Math.max(1, Number(node.attrs && node.attrs.level) || 1));
        const heading = document.createElement(`h${level}`);
        inlineInto(heading, node);
        return heading;
      }
      case "paragraph": {
        const p = document.createElement("p");
        inlineInto(p, node);
        /* An empty paragraph is a BLANK LINE somebody typed on purpose —
           the spacer between two sections. Rendered as nothing it
           collapses, and the list loses the shape its author gave it. */
        if (!p.childNodes.length) p.appendChild(document.createElement("br"));
        return p;
      }
      default: {
        if (!Array.isArray(node.content)) return null;
        const fragment = document.createDocumentFragment();
        appendChildren(fragment, node);
        return fragment;
      }
    }
  }

  /* Somebody else's checklist as an element. `.ProseMirror` is the class
     the stylesheet keys on, and NOT a contenteditable: this is the one
     place the two shapes are allowed to differ, because the whole point is
     that a caret cannot get in here. */
  const renderDoc = (doc) => {
    const root = document.createElement("div");
    root.className = "ProseMirror";
    if (doc) appendChildren(root, doc);
    return root;
  };

  const renderStored = (raw) => renderDoc(docFromStored(raw));

  return {
    STORAGE_VERSION,
    EMPTY_TASK_DOC,
    nextCompletionState,
    effectiveCompletion,
    paintCheckboxState,
    normalizeDoc,
    docFromStored,
    countTaskItems,
    countsFromStored,
    payloadFromDoc,
    renderDoc,
    renderStored,
  };
})();

window.DDChecklistDoc = DDChecklistDoc;
