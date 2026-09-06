/* ================================================================
   NOTEBOOK-CODE-EDIT.JS — Colab-shaped code cells on the ARENA notebook
   ================================================================

   WHAT THIS IS

   Seth, 2026-09-06, about the ARENA chapter notebooks: "make it such that the
   code cells don't have the gray background or whatever, but are just the
   regular rectangles, kind of like Colab does, and also make it such that the
   syntax has the coloring just like the practice does ... along with the
   autocomplete."

   The colour and the ghost completion already exist, in
   `practice/code-highlight.js` and `practice/code-complete.js`. What did NOT
   transfer is the mechanism they are built on: those two files paint a
   `<pre>` BEHIND a `<textarea>` and rely on `value` / `selectionStart`. The
   notebook surfaces do not have a textarea. A cell here is

       <pre class="nbv-src"><code contenteditable="plaintext-only">

   which is a live DOM tree, so the colour can be painted INTO it — no overlay,
   no alignment contract, no second copy of the text. What this file adds is
   everything that becomes hard the moment you rewrite the DOM under a caret.

   WHAT IS REUSED, AND WHAT IS NOT

     * `DeltaCodeHighlight.tokenize` — the Python tokeniser and its
       `cm-*` class names. The same tokens, so a cell here and a cell in the
       practice tab colour identically, and styles/practice/code-highlight.css
       is still the one place the palette lives.
     * `DeltaCodeComplete.suggest` — the candidate pools and every rule about
       what may never be suggested. It only ever reads `value`,
       `selectionStart` and `selectionEnd` off what it is handed, so it is
       called with a plain object standing in for a textarea.

   `DeltaCodeHighlight.paint` is deliberately NOT reused. It emits the ghost as
   a span holding real text and appends a trailing newline — both correct for
   an overlay nobody reads back, both wrong here, where `innerText` of this
   element IS the learner's source and would grow by a character on every
   repaint.

   🔴 THE GHOST IS GENERATED CONTENT, NOT TEXT. The suggestion lives in
   `data-ghost` on an EMPTY span and is drawn by a `::after` rule in
   arena-notebook.css. CSS-generated content is not part of `textContent` or
   `innerText`, so a suggestion on screen cannot leak into the cell's source,
   into the localStorage copy `_persistCells` writes, or into what Run sends
   to the kernel. A span holding the ghost as real text would do all three.

   🔴 EVERY NODE THIS FILE ADDS THAT IS NOT THE LEARNER'S TEXT CARRIES
   `data-nb-skip`. That is the ghost and the zero-width span that holds a line
   box open on a trailing newline. `readText` and the caret walker BOTH skip
   those nodes, and they have to agree: a node counted by one and not the other
   puts the caret one character out on every keystroke.

   🔴 REWRITING innerHTML DESTROYS THE BROWSER'S UNDO STACK. That is not a
   cosmetic loss on a page whose whole point is editing someone else's
   notebook, so Ctrl+Z / Ctrl+Shift+Z are implemented here over text
   snapshots. Typing inside 500ms coalesces into one entry, the way an
   editor's undo does — otherwise one keystroke is one undo.

   HOW IT IS WIRED

   Nothing calls this file. `arena-notebook.js` already announces
   `arena-notebook:rendered`, and a MutationObserver on the notebook host
   catches cells minted later by "+ Code" and prose re-rendered after a
   markdown edit. That keeps the enhancement entirely out of the renderer:
   if this script is absent the cells are exactly the plain contenteditable
   they were before it existed.
   ================================================================ */
const DeltaNotebookCode = (() => {
  "use strict";

  const INDENT = "    ";
  // Same rule as runner.js: nothing follows one of these at the same depth.
  const DEDENT_AFTER = /^\s*(return|pass|break|continue|raise)\b/;
  // Fences whose body is Python. Anything else (bash, json, plain) is left
  // uncoloured rather than coloured wrongly — a `#` comment rule applied to a
  // shell block paints half of it green.
  const PY_FENCE = /^(|py|python|python3|ipython|ipython3)$/i;
  const UNDO_COALESCE_MS = 500;

  const escHtml = (s) => s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  const escAttr = (s) => escHtml(s).replace(/"/g, "&quot;");

  /* ---------- the text model ------------------------------------------- */

  /* Elements that mark a break in the text even though they hold none. A
     contenteditable can produce these for one keystroke before the next
     repaint normalises everything back to text nodes. */
  const BLOCKISH = new Set(["DIV", "P", "LI", "TR"]);

  const isSkip = (el) => el.nodeType === 1 && el.hasAttribute("data-nb-skip");

  /* The nodes that CARRY the learner's text, in document order. The caret
     walker and `readText` both run off this list so the two cannot disagree
     about what a character is. A <br> counts as exactly one character. */
  const countedNodes = (root) => {
    const out = [];
    (function walk(node) {
      for (const child of node.childNodes) {
        if (child.nodeType === 3) out.push(child);
        else if (child.nodeType === 1 && !isSkip(child)) {
          if (child.tagName === "BR") out.push(child);
          else {
            // A block element opened mid-text is a line break the DOM is
            // spelling with structure rather than with "\n".
            if (BLOCKISH.has(child.tagName) && out.length) out.push(child);
            walk(child);
          }
        }
      }
    })(root);
    return out;
  };

  const lengthOf = (node) => (node.nodeType === 3 ? node.nodeValue.length : 1);

  /* 🔴 NEVER `innerText`. It is defined in terms of LAYOUT, so a cell inside a
     closed <details> — which ARENA is full of — reads back as the empty
     string. `arena-notebook.js` documents the same trap on its own read path.
     A contenteditable also substitutes NBSP for a space it thinks would
     collapse; `white-space: pre` means it never would, so they are turned back
     into the spaces the learner typed. */
  const readText = (root) => {
    let out = "";
    for (const node of countedNodes(root)) {
      out += node.nodeType === 3 ? node.nodeValue : "\n";
    }
    return out.replace(/\u00a0/g, " ");
  };

  /* Character offset of a DOM position, measured with exactly the rules
     above: clone everything before the position and read its length. */
  const offsetOf = (root, node, nodeOffset) => {
    if (!node || !root.contains(node)) return null;
    const range = document.createRange();
    range.selectNodeContents(root);
    try {
      range.setEnd(node, nodeOffset);
    } catch (_) {
      return null;
    }
    const holder = document.createElement("div");
    holder.appendChild(range.cloneContents());
    return readText(holder).length;
  };

  const selectionIn = (root) => {
    const sel = window.getSelection();
    if (!sel || !sel.rangeCount) return null;
    const range = sel.getRangeAt(0);
    const start = offsetOf(root, range.startContainer, range.startOffset);
    const end = offsetOf(root, range.endContainer, range.endOffset);
    if (start == null || end == null) return null;
    return start <= end ? { start, end } : { start: end, end: start };
  };

  const locate = (root, offset) => {
    const nodes = countedNodes(root);
    let seen = 0;
    for (const node of nodes) {
      const len = lengthOf(node);
      if (offset <= seen + len) {
        if (node.nodeType === 3) return { node, offset: offset - seen };
        const parent = node.parentNode;
        const index = Array.prototype.indexOf.call(parent.childNodes, node);
        return { node: parent, offset: index + (offset - seen) };
      }
      seen += len;
    }
    const last = nodes[nodes.length - 1];
    if (last && last.nodeType === 3) return { node: last, offset: last.nodeValue.length };
    return { node: root, offset: 0 };
  };

  const placeCaret = (root, start, end = start) => {
    const from = locate(root, start);
    const to = locate(root, end);
    const range = document.createRange();
    try {
      range.setStart(from.node, from.offset);
      range.setEnd(to.node, to.offset);
    } catch (_) {
      return;
    }
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  };

  /* ---------- painting -------------------------------------------------- */

  const tokensOf = (src) => {
    const tokenize = window.DeltaCodeHighlight?.tokenize;
    if (typeof tokenize !== "function") return [{ start: 0, end: src.length, cls: null }];
    return tokenize(src);
  };

  const ghostSpan = (ghost) =>
    '<span class="nb-code-ghost" data-nb-skip="1" contenteditable="false" ' +
    `data-ghost="${escAttr(ghost)}"></span>`;

  /* A <pre>-formatted element gives a trailing newline no line box, so the
     blank line the learner just made by pressing Enter at the end has nowhere
     to put the caret. A zero-width space in GENERATED content holds the box
     open.

     🔴 IT CANNOT BE A REAL CHARACTER. `arena-notebook.js`'s own input handler
     re-reads the cell with `innerText`, which would pick a literal U+200B up
     and write it into `_ddSource`, into the localStorage copy, and from there
     into the program the kernel runs — where Python answers a zero-width
     space with `SyntaxError: invalid non-printable character`. Generated
     content is in none of those reads. `contenteditable="false"` for the
     other half of the same rule: an empty EDITABLE span is somewhere the
     caret can land, and text typed into it is text `readText` skips. */
  const TAIL = '<span class="nb-code-tail" data-nb-skip="1" contenteditable="false"></span>';

  const paint = (src, ghost, ghostAt) => {
    const parts = [];
    const emit = (cls, text) => {
      if (!text) return;
      parts.push(cls ? `<span class="cm-${cls}">${escHtml(text)}</span>` : escHtml(text));
    };
    for (const tok of tokensOf(src)) {
      const text = src.slice(tok.start, tok.end);
      /* Splice the ghost INTO the tokenised stream rather than tokenising
         `before + ghost + after` — the same reason code-highlight.js gives:
         a triple-quoted string straddling the caret would come apart. */
      if (ghost && ghostAt >= tok.start && ghostAt < tok.end) {
        emit(tok.cls, text.slice(0, ghostAt - tok.start));
        parts.push(ghostSpan(ghost));
        emit(tok.cls, text.slice(ghostAt - tok.start));
      } else {
        emit(tok.cls, text);
      }
    }
    if (ghost && ghostAt >= src.length) parts.push(ghostSpan(ghost));
    parts.push(TAIL);
    return parts.join("");
  };

  /* ---------- the read-only case: a fenced block in prose --------------- */

  /* ARENA's markdown is full of ```python blocks, and they are the majority of
     the code on the page. They are not editable and never will be — a solution
     block is written to be read beside your own attempt — but there is no
     reason for them to be the one code on the page with no colour. */
  const paintFence = (code) => {
    if (!code || code.dataset.nbPaint === "1") return;
    const pre = code.parentElement;
    const fence = pre?.getAttribute?.("data-fence") ?? "";
    code.dataset.nbPaint = "1";
    if (!PY_FENCE.test(fence.trim())) return;
    const src = code.textContent || "";
    code.innerHTML = paint(src, "", -1);
  };

  /* ---------- the editable case: a runnable cell ------------------------ */

  const suggestFor = (text, at) => {
    const complete = window.DeltaCodeComplete?.suggest;
    if (typeof complete !== "function") return "";
    try {
      return complete({ value: text, selectionStart: at, selectionEnd: at }) || "";
    } catch (_) {
      return "";
    }
  };

  const render = (code, state, caret = null, caretEnd = null) => {
    const scroller = code.closest(".nbv-src") || code;
    const left = scroller.scrollLeft;
    const top = scroller.scrollTop;
    state.writing = true;
    code.innerHTML = paint(state.text, state.ghost, state.ghostAt);
    state.writing = false;
    // Only into a cell that still has the caret. `blur` dismisses the ghost,
    // which repaints — restoring a selection there would pull focus straight
    // back out of whatever the learner just clicked on.
    if (caret != null && document.activeElement === code) {
      placeCaret(code, caret, caretEnd == null ? caret : caretEnd);
    }
    scroller.scrollLeft = left;
    scroller.scrollTop = top;
  };

  /* The cell's source is carried ON THE NODE (`_ddSource`) — that is what Run
     and the localStorage copy read, precisely so neither depends on layout.
     Keep it true, then let the notebook's own delegated `input` listener do
     its bookkeeping (mark the cell stale, queue the autosave). */
  const publish = (code, state) => {
    const cell = code.closest(".nbv-cell");
    if (cell) cell._ddSource = state.text;
    state.selfInput = true;
    code.dispatchEvent(new Event("input", { bubbles: true }));
    state.selfInput = false;
  };

  /* `text` is the state BEFORE the edit — an undo entry is the thing to go
     back TO. `force` is for an edit that is a decision rather than a
     keystroke: Tab, Enter, accepting a suggestion.

     🔴 THE WINDOW RUNS FROM THE LAST ENTRY KEPT, NOT THE LAST KEYSTROKE. An
     earlier version pushed the clock forward on a coalesced keystroke too,
     which means someone typing steadily never opens a second entry and their
     one Ctrl+Z empties the cell back to whatever it held when they started.
     Caught driving the real page from the keyboard, not by reading it. */
  const pushUndo = (state, text, caret, force = false) => {
    const now = Date.now();
    state.redo.length = 0;
    const top = state.undo[state.undo.length - 1];
    if (top && top.text === text) return;
    if (!force && top && now - state.lastPush < UNDO_COALESCE_MS) return;
    state.undo.push({ text, caret });
    if (state.undo.length > 200) state.undo.shift();
    /* A forced entry CLOSES the run rather than starting one. Tab, Enter and
       accepting a suggestion each push the state from before the decision;
       leaving the clock running would let the next keystroke, 20ms later,
       coalesce into that same entry, and one Ctrl+Z would then undo the
       decision AND everything typed after it. Zero means the next edit always
       opens its own entry. Codex, 2026-09-06. */
    state.lastPush = force ? 0 : now;
  };

  const setGhost = (code, state, text, at) => {
    const next = text || "";
    if (next === state.ghost && at === state.ghostAt) return;
    state.ghost = next;
    state.ghostAt = at;
    render(code, state, at);
  };

  const dismissGhost = (code, state) => {
    if (!state.ghost) return;
    state.ghost = "";
    const where = selectionIn(code);
    state.ghostAt = -1;
    render(code, state, where ? where.start : null, where ? where.end : null);
  };

  /* Replace the range [start, end) with `text` and put the caret after it.
     One path for Tab, Shift+Tab, Enter and accepting a suggestion, so all
     four agree about undo and about announcing the change. */
  const splice = (code, state, start, end, text, caretAt = null) => {
    pushUndo(state, state.text, start, true);
    state.text = state.text.slice(0, start) + text + state.text.slice(end);
    state.ghost = "";
    state.ghostAt = -1;
    const caret = caretAt == null ? start + text.length : caretAt;
    state.caret = caret;
    render(code, state, caret);
    publish(code, state);
  };

  const lineStartOf = (text, at) => text.lastIndexOf("\n", at - 1) + 1;

  const handleTab = (code, state, event) => {
    const where = selectionIn(code);
    if (!where) return;
    const { text } = state;

    if (state.ghost && !event.shiftKey && where.start === where.end) {
      const ghost = state.ghost;
      const at = state.ghostAt;
      splice(code, state, at, at, ghost);
      return;
    }

    const multiline = text.slice(where.start, where.end).includes("\n");
    if (!event.shiftKey && !multiline) {
      splice(code, state, where.start, where.end, INDENT);
      return;
    }

    // Block (de)indent: every line the selection touches, selection kept.
    const from = lineStartOf(text, where.start);
    /* A selection dragged down to the START of a line stops before that line:
       nothing on it is selected, so indenting it would move a line the learner
       never touched. Every editor draws the boundary here. Codex, 2026-09-06. */
    const last = where.end > where.start && lineStartOf(text, where.end) === where.end
      ? where.end - 1
      : where.end;
    const toEnd = text.indexOf("\n", last);
    const to = toEnd < 0 ? text.length : toEnd;
    const lines = text.slice(from, to).split("\n");
    let firstDelta = 0;
    let totalDelta = 0;
    const next = lines.map((line, i) => {
      if (event.shiftKey) {
        const cut = (line.match(/^ {1,4}/) || [""])[0].length;
        if (i === 0) firstDelta = -cut;
        totalDelta -= cut;
        return line.slice(cut);
      }
      if (i === 0) firstDelta = INDENT.length;
      totalDelta += INDENT.length;
      return INDENT + line;
    }).join("\n");
    pushUndo(state, state.text, where.start, true);
    state.text = text.slice(0, from) + next + text.slice(to);
    state.ghost = "";
    state.ghostAt = -1;
    state.caret = Math.max(from, where.start + firstDelta);
    render(code, state, state.caret, Math.max(from, where.end + totalDelta));
    publish(code, state);
  };

  /* Enter keeps the indent you are already at — the same rule runner.js
     applies in the practice editor, so the two editors behave alike. */
  const handleEnter = (code, state) => {
    const where = selectionIn(code);
    if (!where || where.start !== where.end) return false;
    const at = where.start;
    const from = lineStartOf(state.text, at);
    const line = state.text.slice(from, at);
    let indent = (line.match(/^[ \t]*/) || [""])[0];
    const before = line.replace(/#.*$/, "").trimEnd();
    if (before.endsWith(":")) indent += INDENT;
    else if (DEDENT_AFTER.test(line) && indent.length >= INDENT.length) {
      indent = indent.slice(0, indent.length - INDENT.length);
    }
    splice(code, state, at, at, "\n" + indent);
    return true;
  };

  const history = (code, state, redo) => {
    const stack = redo ? state.redo : state.undo;
    if (!stack.length) return;
    const other = redo ? state.undo : state.redo;
    const where = selectionIn(code);
    other.push({ text: state.text, caret: where ? where.start : state.text.length });
    const entry = stack.pop();
    state.text = entry.text;
    state.ghost = "";
    state.ghostAt = -1;
    state.lastPush = 0;
    state.caret = Math.min(entry.caret, entry.text.length);
    render(code, state, state.caret);
    publish(code, state);
  };

  function attachEditor(code) {
    if (!code || code.dataset.nbCode === "1") return;
    code.dataset.nbCode = "1";

    const state = {
      text: readText(code),
      ghost: "",
      ghostAt: -1,
      composing: false,
      writing: false,
      selfInput: false,
      undo: [],
      redo: [],
      lastPush: 0,
      composeFrom: null,
      caret: 0,
    };
    code.__nbCodeState = state;
    render(code, state);

    /* 🔴 A COMPOSITION IS ONE EDIT, AND NOTHING ELSE SNAPSHOTS IT. The `input`
       handler is skipped while `composing` is true — repainting mid-compose
       would tear the caret out of the IME's own preedit — so without these two
       lines a composed word never enters the history at all and Ctrl+Z steps
       straight over it to whatever preceded it. Codex, 2026-09-06. */
    code.addEventListener("compositionstart", () => {
      state.composing = true;
      const where = selectionIn(code);
      state.composeFrom = { text: state.text, caret: where ? where.start : state.text.length };
    });
    code.addEventListener("compositionend", () => {
      state.composing = false;
      const before = state.composeFrom;
      state.composeFrom = null;
      const where = selectionIn(code);
      state.text = readText(code);
      state.caret = where ? where.start : state.text.length;
      if (before) pushUndo(state, before.text, before.caret, true);
      render(code, state, where ? where.start : null);
      publish(code, state);
    });

    code.addEventListener("input", () => {
      if (state.selfInput || state.writing || state.composing) return;
      const where = selectionIn(code);
      const at = where ? where.start : null;
      const text = readText(code);
      /* 🔴 AN `input` THAT CHANGED NOTHING IS NOT AN EDIT. A browser can emit
         one after `compositionend` — the composition is already committed and
         already in the history, so pushing here would put the CURRENT text on
         the undo stack and the learner's first Ctrl+Z would visibly do
         nothing. Codex, 2026-09-06. The ghost still gets a look: the caret can
         move on an event that leaves the text alone. */
      if (text === state.text) {
        if (at != null && where.start === where.end) {
          setGhost(code, state, suggestFor(text, at), at);
        }
        return;
      }
      // A line break is a natural undo boundary — the same place an editor
      // breaks a run of typing into two entries.
      const newline = text.split("\n").length !== state.text.split("\n").length;
      // 🔴 The snapshot is the text BEFORE this edit, so the caret stored
      // beside it has to be the caret from before it too — pairing old text
      // with the new offset lands the caret in the wrong place on undo.
      pushUndo(state, state.text, state.caret, newline);
      state.text = text;
      state.caret = at == null ? text.length : at;
      const ghost = at != null && where.start === where.end ? suggestFor(text, at) : "";
      state.ghost = ghost;
      state.ghostAt = ghost ? at : -1;
      render(code, state, at, where ? where.end : null);
      const cell = code.closest(".nbv-cell");
      if (cell) cell._ddSource = text;
    });

    code.addEventListener("keydown", (event) => {
      if (event.isComposing || state.composing) return;

      if (event.key === "Escape") {
        // A one-shot "let the next Tab leave the field", so the cell is not a
        // keyboard trap. Same affordance as the practice editor.
        state.tabEscapes = true;
        if (state.ghost) {
          event.preventDefault();
          dismissGhost(code, state);
        }
        return;
      }

      if ((event.ctrlKey || event.metaKey) && !event.altKey
          && event.key.toLowerCase() === "z") {
        event.preventDefault();
        history(code, state, event.shiftKey);
        return;
      }
      if ((event.ctrlKey || event.metaKey) && !event.altKey
          && event.key.toLowerCase() === "y") {
        event.preventDefault();
        history(code, state, true);
        return;
      }

      if (event.key === "Tab") {
        if (state.tabEscapes) {
          state.tabEscapes = false;
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        handleTab(code, state, event);
        return;
      }
      state.tabEscapes = false;

      if (event.key === "Enter" && !event.shiftKey && !event.ctrlKey
          && !event.metaKey && !event.altKey) {
        if (handleEnter(code, state)) event.preventDefault();
      }
    });

    /* Every way the caret can move without a keystroke this file saw. A ghost
       computed for one line and left standing after a click somewhere else
       would splice its text into an unrelated line on the next Tab — the bug
       codex found in the practice editor, and it is the same bug here. */
    ["click", "mouseup", "select", "dragend"].forEach((type) => {
      code.addEventListener(type, () => {
        if (state.composing) return;
        const where = selectionIn(code);
        if (!where) return;
        state.caret = where.start;
        if (where.start !== where.end) { dismissGhost(code, state); return; }
        setGhost(code, state, suggestFor(state.text, where.start), where.start);
      });
    });
    code.addEventListener("keyup", (event) => {
      if (!/^(Arrow|Home|End|Page)/.test(event.key)) return;
      const where = selectionIn(code);
      if (!where) return;
      state.caret = where.start;
      if (where.start !== where.end) { dismissGhost(code, state); return; }
      setGhost(code, state, suggestFor(state.text, where.start), where.start);
    });
    code.addEventListener("blur", () => dismissGhost(code, state));
  }

  /* ---------- wiring ---------------------------------------------------- */

  const scan = (root) => {
    if (!root || !root.querySelectorAll) return;
    root.querySelectorAll('.nbv-src code:not([data-nb-code="1"])').forEach(attachEditor);
    root.querySelectorAll('pre[data-fence] > code:not([data-nb-paint="1"])').forEach(paintFence);
  };

  const observe = (host) => {
    if (!host || host.__nbCodeObserved || typeof MutationObserver !== "function") return;
    host.__nbCodeObserved = true;
    let queued = false;
    const observer = new MutationObserver((records) => {
      /* Our own repaints are childList mutations too. Anything inside an
         element this file already owns is one of ours; a rescan on it would
         be pure cost on every keystroke of a 600-cell page. */
      const external = records.some((record) => {
        const target = record.target;
        const el = target.nodeType === 1 ? target : target.parentElement;
        return !el || !el.closest('[data-nb-code="1"], [data-nb-paint="1"]');
      });
      if (!external || queued) return;
      queued = true;
      /* 🔴 setTimeout, NOT requestAnimationFrame. A hidden tab does not paint,
         so rAF never runs there — and a notebook that gains a cell while the
         tab is in the background (a restore, a "+ Code" then a switch away)
         would sit unpainted until some later mutation happened to arrive while
         it was visible. Measured: a "+ Code" in a backgrounded tab attached 0
         of 1 new cells on rAF, 1 of 1 on a timeout. */
      setTimeout(() => {
        queued = false;
        scan(host);
      }, 0);
    });
    observer.observe(host, { childList: true, subtree: true });
  };

  document.addEventListener("arena-notebook:rendered", (event) => {
    const host = event.detail?.host || document.getElementById("arena-notebook-host");
    if (!host) return;
    scan(host);
    observe(host);
  });

  /* 🔴 THE EVENT IS NOT THE ONLY WAY IN. `?arena=<slug>` renders a notebook as
     soon as the page is ready, so whether `arena-notebook:rendered` has already
     fired by the time this script parses is a question about where its <script>
     tag sits — exactly the kind of ordering that works until someone moves a
     line. The host div is static markup in index.html, so claim it directly:
     `scan` catches anything already drawn and the observer catches everything
     drawn later, with or without the event. Both are idempotent, so the two
     paths cannot double-attach. Codex, 2026-09-06. */
  const bootstrap = () => {
    const host = document.getElementById("arena-notebook-host");
    if (!host) return;
    scan(host);
    observe(host);
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
  } else {
    bootstrap();
  }

  return { attachEditor, paintFence, scan, readText, paint };
})();

window.DeltaNotebookCode = DeltaNotebookCode;
