/* ================================================================
   THE REFERENCE RENDERER

   Loads one compiled ARENA notebook — the same JSON the app loads — and paints
   it with the app's class names into a page wearing LessWrong's measured
   design. Then builds the same contents rail, with the same rules the real one
   uses (a row's share of the document decides its height; the current section
   is the last heading above the one-third mark), so that when the diff finds a
   difference it is a difference of STYLE and not of behaviour.

   ?nb=arena-0-1 picks the notebook.
   ================================================================ */

(() => {
  const params = new URLSearchParams(location.search);
  const slug = params.get("nb") || "arena-0-1";
  const cells = document.getElementById("cells");

  const esc = (text) =>
    String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  /* The app's renderer when it loaded, ours when it did not. The fallback is
     deliberately small: a design comparison needs the same BLOCKS (headings,
     paragraphs, code, images, lists), not a complete markdown implementation. */
  const appRenderer = window.LessonGate && window.LessonGate.renderMarkdown;
  const mini = (src) => {
    const lines = String(src).split("\n");
    const out = [];
    let para = [];
    let code = null;
    let bullets = null;
    const flush = () => {
      if (para.length) out.push(`<p>${inline(para.join(" "))}</p>`);
      para = [];
      if (bullets && bullets.length) out.push(`<ul>${bullets.join("")}</ul>`);
      bullets = null;
    };
    const inline = (text) =>
      esc(text)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img alt="$1" src="$2">')
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/(^|\W)\*([^*]+)\*/g, "$1<em>$2</em>");
    lines.forEach((line) => {
      const fence = line.match(/^```/);
      if (fence) {
        if (code === null) {
          flush();
          code = [];
        } else {
          out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
          code = null;
        }
        return;
      }
      if (code !== null) return code.push(line);
      const heading = line.match(/^(#{1,4})\s+(.*)$/);
      if (heading) {
        flush();
        return out.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`);
      }
      /* A run of bullets is ONE list. Emitting a <ul> per line gave every
         bullet a list's top and bottom margin, which changes the height of the
         section it is in — and section heights are exactly what the rail
         measures. Found by codex, 2026-09-02. */
      const bullet = line.match(/^\s*[-*]\s+(.*)$/);
      if (bullet) {
        if (bullets === null) {
          flush();
          bullets = [];
        }
        return bullets.push(`<li>${inline(bullet[1])}</li>`);
      }
      if (!line.trim()) return flush();
      para.push(line);
    });
    flush();
    if (code) out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
    return out.join("\n");
  };
  const md = (src) => (appRenderer ? appRenderer(src, { headingLevels: true }) : mini(src));

  const node = (cell) => {
    if (cell.role === "details") {
      const el = document.createElement("details");
      el.className = "nbv-cell nbv-hints arena-nb-details";
      el.id = `arena-${cell.id}`;
      el.innerHTML = `<summary>${esc(cell.summary || "Show")}</summary><div class="nbv-md">${md(cell.src)}</div>`;
      return el;
    }
    const el = document.createElement("section");
    el.id = `arena-${cell.id}`;
    if (cell.role === "code" || cell.role === "magic") {
      el.className = "nbv-cell nbv-code";
      el.innerHTML =
        '<div class="nbv-gutter"><button type="button" class="nbv-run">▶</button></div>' +
        `<div class="nbv-body"><pre class="nbv-src"><code>${esc(String(cell.src || "").trimEnd())}</code></pre></div>`;
      return el;
    }
    el.className = "nbv-cell nbv-md";
    el.innerHTML = md(cell.src);
    return el;
  };

  /* ---- the rail -------------------------------------------------------

     🔴 THE APP'S RAIL, NOT A COPY OF IT. This page used to build its own rows,
     compute its own shares and run its own highlight loop — a second
     implementation whose only job was to look like the first one. It stopped
     looking like it the moment the app's rail was restructured, and every
     comparison taken through this page in between was measuring a design
     nobody was shipping. So `arena-notebook-nav.js` is loaded from the app and
     mounted here directly. */
  const mountRail = (title) => {
    if (!window.ArenaNotebookNav) return;
    window.ArenaNotebookNav.mount(document.body, document.querySelector(".mock-column"), title);
  };

  fetch(`../../../Local_Deployed_Shared/lessons/notebooks/${slug}.json`, { cache: "no-cache" })
    .then((res) => res.json())
    .then((nb) => {
      document.title = `${nb.number || ""} ${nb.title || slug} · LessWrong design reference`.trim();
      (nb.cells || []).forEach((cell) => cells.appendChild(node(cell)));
      mountRail(`${nb.number || ""} ${nb.title || slug}`.trim());
      window.__mockup = { renderer: appRenderer ? "LessonGate" : "fallback", cells: (nb.cells || []).length };
    })
    .catch((err) => {
      cells.innerHTML = `<pre class="nbv-src">could not load ${esc(slug)}: ${esc(err.message)}</pre>`;
    });
})();
