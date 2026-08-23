/* ================================================================
   CODE-HIGHLIGHT.JS — Python syntax colour inside the practice editors.

   WHAT THIS IS
   Every code cell in this app is a plain `<textarea class="code-editor">`
   (index.html's cell 1, plus every cell `notebook-editor.js` mints). A
   textarea paints ONE colour for its whole value — there is no way to colour
   a keyword inside it. Colab colours code because CodeMirror replaces the
   textarea with a contenteditable DOM tree.

   We do not do that, on purpose. `runner.js`, `events.js`, `ui.js`,
   `timer.js`, `lessons.js` and `notebook-editor.js` all read and write
   `editor.value`, and the Tab/Enter indent rules in `runner.js` are written
   against `selectionStart`/`selectionEnd`. Swapping in a contenteditable
   editor rewrites every one of those call sites and changes what "the
   learner's code" even means at submit time. So instead:

     .code-surface            position:relative wrapper (added by attach)
       └ pre.code-highlight   the SAME text, tokenised into <span>s   (z 0)
       └ textarea.code-editor transparent text, visible caret         (z 1)

   The textarea stays the single source of truth. It is still the element
   that has focus, that the browser scrolls, that owns the selection, and
   that every existing module talks to. All we do is paint a coloured copy
   of its own text directly behind it, pixel-aligned, and make its own text
   invisible. Nothing above this file has to know.

   THE ALIGNMENT CONTRACT — the whole reason this can go wrong
   The overlay only works while its glyphs land on exactly the same pixels
   as the textarea's. That needs, and `styles/practice/code-highlight.css`
   enforces:

     * identical font-family / font-size / line-height / letter-spacing /
       tab-size — the CSS lists `.code-highlight` in the SAME rule as
       `.code-editor` rather than restating the values, so the two cannot
       drift apart in a later edit;
     * identical box model — same padding, same border WIDTHS (the overlay's
       border is transparent so it does not double-draw the visible one);
     * identical wrapping — `white-space: pre-wrap` + `overflow-wrap:
       break-word`, which is what a textarea does by default;
     * identical text-column WIDTH even when a scrollbar appears. The cells
       auto-grow, so the textarea should never scroll — but if it does, its
       bar narrows that layer's column by ~10px and every wrapped line below
       drifts. `syncGutter` measures the real bar and pads the overlay to
       match, rather than reserving the column permanently on both.

   HOW IT HEARS ABOUT A PROGRAMMATIC WRITE
   Assigning `editor.value = …` fires no event, and five modules do it —
   `ui.js` prefills the starter code, `events.js` resets between questions,
   `timer.js` restores a draft, `lessons.js` loads example code,
   `notebook-editor.js` seeds a cell. Listening for `input` alone leaves the
   overlay showing the PREVIOUS question's code under the new question's
   text. `runner.js`'s `announceValueWrites` shadows `value` on the element
   and dispatches `delta-editor-value-set`; this file listens. The patch is
   deliberately NOT here: the cell's auto-height needs the same signal, and
   two modules patching the same property is a silent breakage (the second
   captures the prototype descriptor and bypasses the first). One owner,
   many listeners.

   Consequence: calling `attach()` directly, outside
   `installCodeEditorKeys`, gives you a highlighter that repaints on typing
   but not on programmatic writes. Go through `installCodeEditorKeys`.
   ================================================================ */
const DeltaCodeHighlight = (() => {
  "use strict";

  const KEYWORDS = new Set([
    "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from",
    "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or",
    "pass", "raise", "return", "try", "while", "with", "yield", "match",
    "case",
  ]);
  const CONSTANTS = new Set(["True", "False", "None", "NotImplemented", "Ellipsis"]);
  const SOFT = new Set(["self", "cls"]);
  const BUILTINS = new Set([
    "abs", "all", "any", "bool", "bytes", "callable", "chr", "complex",
    "dict", "dir", "divmod", "enumerate", "eval", "filter", "float",
    "format", "frozenset", "getattr", "hasattr", "hash", "hex", "id",
    "input", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
    "max", "min", "next", "object", "open", "ord", "pow", "print", "range",
    "repr", "reversed", "round", "set", "setattr", "slice", "sorted", "str",
    "sum", "super", "tuple", "type", "vars", "zip",
  ]);

  /* One pass, one regex. The alternation order IS the precedence:
     a `#` inside a string must not start a comment, so strings are tried
     before comments would ever see those characters — and a comment is
     tried before a bare `#` can fall through as an operator.

     The lookbehinds matter:
       * string prefixes (`f"…"`, `rb'…'`) must not swallow the tail of an
         identifier, so the prefix letters are only a prefix when nothing
         word-ish precedes them;
       * `@` is a decorator only at the head of a line. In `q @ k.T` it is
         the matmul operator, and coloured as one. */
  const TOKEN = new RegExp([
    "(?<comment>#[^\\n]*)",
    "(?<tstring>(?<![\\w])[rRbBuUfF]{0,2}(?:\"\"\"[\\s\\S]*?(?:\"\"\"|$)|'''[\\s\\S]*?(?:'''|$)))",
    "(?<string>(?<![\\w])[rRbBuUfF]{0,2}(?:\"(?:\\\\[\\s\\S]|[^\"\\\\\\n])*\"?|'(?:\\\\[\\s\\S]|[^'\\\\\\n])*'?))",
    "(?<decorator>(?<=(?:^|\\n)[ \\t]*)@[A-Za-z_][\\w.]*)",
    "(?<number>(?<![\\w.])(?:0[xXoObB][0-9a-fA-F_]+|(?:\\d[\\d_]*\\.?[\\d_]*|\\.\\d[\\d_]*)(?:[eE][+-]?\\d+)?[jJ]?))",
    "(?<name>[A-Za-z_]\\w*)",
    "(?<op>[-+*/%=<>!&|^~@:,.;]+)",
  ].join("|"), "g");

  const escapeHtml = (s) => s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  /* What an identifier IS depends on what surrounds it, which a regex
     cannot see: `shape` is an attribute in `x.shape` and a variable on its
     own, `solve` is a definition after `def` and a call before `(`. */
  const classifyName = (src, word, start, end, prevWord) => {
    if (KEYWORDS.has(word)) return "kw";
    if (CONSTANTS.has(word)) return "const";
    let back = start - 1;
    while (back >= 0 && (src[back] === " " || src[back] === "\t")) back -= 1;
    const afterDot = back >= 0 && src[back] === "." && src[back - 1] !== ".";
    let fwd = end;
    while (fwd < src.length && (src[fwd] === " " || src[fwd] === "\t")) fwd += 1;
    const called = src[fwd] === "(";
    if (prevWord === "def") return "fn";
    if (prevWord === "class") return "cls";
    if (called) return "fn";
    if (afterDot) return "attr";
    if (BUILTINS.has(word)) return "builtin";
    if (SOFT.has(word)) return "self";
    // A bare Capitalised name is a class or a dtype far more often than not
    // (`nn.Linear`, `Tensor`, `MyModule`) — colouring it as one is what makes
    // a torch snippet read the way it does in Colab.
    if (/^[A-Z]/.test(word)) return "cls";
    return "var";
  };

  /* Tokenise `src` into a flat list of {start, end, cls}. Plain runs
     (whitespace, brackets, anything the regex did not claim) are emitted as
     `null` class so the caller can still splice the ghost into them. */
  const tokenize = (src) => {
    const out = [];
    let at = 0;
    let prevWord = "";
    TOKEN.lastIndex = 0;
    let m;
    while ((m = TOKEN.exec(src)) !== null) {
      if (m.index > at) out.push({ start: at, end: m.index, cls: null });
      const g = m.groups;
      let cls = null;
      if (g.comment !== undefined) cls = "com";
      else if (g.tstring !== undefined || g.string !== undefined) cls = "str";
      else if (g.decorator !== undefined) cls = "dec";
      else if (g.number !== undefined) cls = "num";
      else if (g.op !== undefined) cls = "op";
      else if (g.name !== undefined) {
        cls = classifyName(src, g.name, m.index, m.index + m[0].length, prevWord);
      }
      out.push({ start: m.index, end: m.index + m[0].length, cls });
      prevWord = g.name !== undefined ? g.name : "";
      at = m.index + m[0].length;
      if (m[0].length === 0) TOKEN.lastIndex += 1; // paranoia: never spin
    }
    if (at < src.length) out.push({ start: at, end: src.length, cls: null });
    return out;
  };

  const span = (cls, text) => (cls ? `<span class="cm-${cls}">${escapeHtml(text)}</span>` : escapeHtml(text));

  /* Paint `src`, optionally splicing a ghost-suggestion span in at offset
     `ghostAt`. The ghost is spliced INTO the already-tokenised stream rather
     than by highlighting `before + ghost + after`, because a triple-quoted
     string straddling the caret would be two broken halves that way. */
  const paint = (src, ghost, ghostAt) => {
    const parts = [];
    for (const tok of tokenize(src)) {
      const text = src.slice(tok.start, tok.end);
      if (ghost && ghostAt >= tok.start && ghostAt < tok.end) {
        parts.push(span(tok.cls, text.slice(0, ghostAt - tok.start)));
        parts.push(`<span class="code-ghost">${escapeHtml(ghost)}</span>`);
        parts.push(span(tok.cls, text.slice(ghostAt - tok.start)));
      } else {
        parts.push(span(tok.cls, text));
      }
    }
    if (ghost && ghostAt >= src.length) {
      parts.push(`<span class="code-ghost">${escapeHtml(ghost)}</span>`);
    }
    // A <pre> drops one trailing newline. Without this the last blank line
    // the learner is standing on has no line box, and the overlay is one
    // line shorter than the textarea it is supposed to be a copy of.
    parts.push("\n");
    return parts.join("");
  };

  const overlayOf = (editor) => editor?.__deltaHighlight || null;

  /* A scrollbar on the textarea and none on the overlay narrows the text
     column by ~10px on one layer only, and every wrapped line below drifts.
     `scrollbar-gutter: stable` on both would fix it by reserving that column
     permanently — but the cells auto-grow now (notebook-editor.js), so the
     textarea essentially never scrolls and that would be 10px of dead space
     on every cell forever. Measure the real bar instead and pad the overlay
     by exactly as much: normally zero, exact when it is not. Measured off
     the TEXTAREA only, never off the overlay, which would feed its own
     compensation back into the next measurement. */
  const syncGutter = (editor, overlay) => {
    const bar = Math.max(0, editor.offsetWidth - editor.clientWidth - editor.__deltaBorderLR);
    const want = `${editor.__deltaPadRight + bar}px`;
    if (overlay.style.paddingRight !== want) overlay.style.paddingRight = want;
  };

  const render = (editor) => {
    const overlay = overlayOf(editor);
    if (!overlay) return;
    const ghost = editor.__deltaGhost || "";
    overlay.innerHTML = paint(editor.value, ghost, editor.selectionStart);
    syncGutter(editor, overlay);
    overlay.scrollTop = editor.scrollTop;
    overlay.scrollLeft = editor.scrollLeft;
  };

  const setGhost = (editor, text) => {
    const next = text || "";
    if ((editor.__deltaGhost || "") === next) return;
    editor.__deltaGhost = next;
    render(editor);
  };

  const surfaceOf = (editor) => editor?.closest?.(".code-surface") || null;

  function attach(editor) {
    if (!editor || editor.__deltaHighlight) return null;

    const surface = document.createElement("div");
    surface.className = "code-surface";
    editor.parentNode.insertBefore(surface, editor);
    const overlay = document.createElement("pre");
    overlay.className = "code-highlight";
    overlay.setAttribute("aria-hidden", "true");
    surface.appendChild(overlay);

    /* NOTHING WORKS WITHOUT THE STYLESHEET, and the failure mode is ugly
       rather than invisible: an un-positioned <pre> is a second, full copy
       of the learner's code stacked above the editor. This file and
       styles/practice/code-highlight.css are two <link>/<script> lines in
       index.html that can be added, cache-busted or removed independently,
       so "the JS shipped and the CSS did not" is a real state. Check for
       the one property the whole overlay technique rests on, and if it is
       not there, undo the wrap and leave a plain textarea behind. */
    if (getComputedStyle(overlay).position !== "absolute") {
      surface.parentNode.insertBefore(editor, surface);
      surface.remove();
      return null;
    }

    surface.appendChild(editor);
    editor.__deltaHighlight = overlay;
    const metrics = getComputedStyle(editor);
    editor.__deltaPadRight = parseFloat(metrics.paddingRight) || 0;
    editor.__deltaBorderLR =
      (parseFloat(metrics.borderLeftWidth) || 0) + (parseFloat(metrics.borderRightWidth) || 0);

    editor.addEventListener("delta-editor-value-set", () => {
      // A programmatic rewrite is a NEW buffer; whatever the ghost was
      // suggesting belonged to the old one.
      editor.__deltaGhost = "";
      render(editor);
    });

    // `selectionStart` moves the ghost's insertion point, and nothing fires
    // an event for a caret move on its own — keyup and click are what a
    // caret move actually looks like.
    ["input", "keyup", "click", "focus", "blur"].forEach((type) => {
      editor.addEventListener(type, () => render(editor));
    });
    editor.addEventListener("scroll", () => {
      overlay.scrollTop = editor.scrollTop;
      overlay.scrollLeft = editor.scrollLeft;
    });
    /* The scrollbar can appear or disappear without a repaint: the cell
       auto-grows AFTER attach has already painted once (notebook-editor.js
       resizes at the end of bindCell), so a first paint made at the 96px
       floor sees a bar that is gone a moment later and the overlay keeps
       padding for it. Cheap — `syncGutter` only touches the DOM when the
       measurement actually changed, and padding the overlay cannot resize
       the textarea, so this cannot feed itself. */
    if (typeof ResizeObserver === "function") {
      new ResizeObserver(() => syncGutter(editor, overlay)).observe(editor);
    }

    render(editor);
    return overlay;
  }

  return { attach, render, setGhost, surfaceOf, tokenize, paint };
})();

window.DeltaCodeHighlight = DeltaCodeHighlight;
