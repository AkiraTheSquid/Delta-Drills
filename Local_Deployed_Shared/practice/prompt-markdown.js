/* ================================================================
   PROMPT MARKDOWN — the drill prompt's prose, as markdown

   Seth, 2026-09-25: "on the left it doesn't display the headers or bold
   properly when there is text associated with that. markdown needs to
   function, since that's a lot of what's in here". The LeetCode Patterns
   prompts are markdown throughout (`### Example 1`, `**Input:** nums = …`,
   `- constraint` lists), and renderQuestionBody (practice/ui.js) knew only
   paragraphs and inline `code`, so every `**` and `#` showed as typed.

   Block level: ATX headings (`#`…`######`), `-`/`*`/`+` and `1.` lists,
   paragraphs split on blank lines. A single newline inside a paragraph is a
   LINE BREAK, not a space: prompts put `Input:` and `Output:` on consecutive
   lines and mean them as two lines.
   Inline: `code`, **bold** / __bold__, *italic*. Emphasis never opens next to
   a word character or a space, so `a*b*c`, `n * m` and snake_case stay text.

   🔴 MATH IS HELD, NOT PARSED. `$…$` / `$$…$$` spans go through untouched
   (escaped, never emphasised) so KaTeX auto-render, which runs on the DOM
   after this, still finds them. `\(…\)` has no `*` in practice and is left
   to the same pass.

   Fenced ``` blocks are NOT this file's: ui.js cuts them out first and
   renders them as .question-code-block, then hands each prose chunk here.
   Output is escaped HTML; nothing from the prompt reaches innerHTML raw.
   ================================================================ */

(function initPromptMarkdown() {
  const esc = (s) =>
    String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const inline = (text) => {
    const held = [];
    const hold = (html) => `\u0000${held.push(html) - 1}\u0000`;
    let s = esc(text);
    s = s.replace(/`([^`]+)`/g, (_, c) => hold(`<code>${c}</code>`));
    s = s.replace(/\$\$[\s\S]+?\$\$|\$[^$\n]+?\$/g, (m) => hold(m));
    s = s.replace(/\*\*(?=\S)([^\n]*?\S)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^\w])__(?=\S)([^\n]*?\S)__(?!\w)/g, "$1<strong>$2</strong>");
    s = s.replace(/(^|[^\w*])\*(?=[^\s*])([^*\n]*?[^\s*])?\*(?![\w*])/g,
      (m, pre, body, off, all) => (body === undefined ? m : `${pre}<em>${body}</em>`));
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => held[Number(i)]);
  };

  /* One chunk of prose (no fences) → HTML blocks. Headings map `#` → h3 so a
     prompt's top heading sits under the page's own h1/h2. */
  const render = (text) => {
    /* A display-math span may cross lines; the line parser below would cut it
       into <br>-separated pieces KaTeX can no longer pair up. Held whole first,
       put back (escaped, untouched) once the blocks are built. Codex, 2026-09-25. */
    const maths = [];
    const src = String(text == null ? "" : text).replace(/\$\$[\s\S]+?\$\$/g,
      (m) => `\u0001${maths.push(m) - 1}\u0001`);
    const out = [];
    let para = [];
    let list = null;
    const flushPara = () => {
      if (para.length) out.push(`<div class="question-prose">${para.map(inline).join("<br>")}</div>`);
      para = [];
    };
    const flushList = () => {
      if (list) {
        const items = list.items.map((i) => `<li>${inline(i)}</li>`).join("");
        const start = list.start > 1 ? ` start="${list.start}"` : "";
        out.push(`<${list.tag} class="question-list"${start}>${items}</${list.tag}>`);
      }
      list = null;
    };
    for (const raw of src.replace(/\r\n?/g, "\n").split("\n")) {
      const line = raw.replace(/\s+$/, "");
      if (!line.trim()) {
        flushPara();
        flushList();
        continue;
      }
      const h = /^ {0,3}(#{1,6})\s+(.*?)(?:\s+#+)?$/.exec(line);
      if (h) {
        flushPara();
        flushList();
        const level = Math.min(6, h[1].length + 2);
        out.push(`<h${level} class="question-heading">${inline(h[2])}</h${level}>`);
        continue;
      }
      const item = /^\s*(?:([-*+])|(\d+)[.)])\s+(.*)$/.exec(line);
      if (item) {
        flushPara();
        const tag = item[2] ? "ol" : "ul";
        if (!list || list.tag !== tag) {
          flushList();
          list = { tag, items: [], start: item[2] ? Number(item[2]) : 1 };
        }
        list.items.push(item[3]);
        continue;
      }
      // An indented line under an item continues that item.
      if (list && /^\s{2,}\S/.test(raw)) {
        list.items[list.items.length - 1] += " " + line.trim();
        continue;
      }
      flushList();
      para.push(line.trim());
    }
    flushPara();
    flushList();
    return out.join("").replace(/\u0001(\d+)\u0001/g, (_, i) => esc(maths[Number(i)]));
  };

  window.PromptMarkdown = { render, inline };
})();
