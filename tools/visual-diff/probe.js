/* ================================================================
   THE PROBE — what a page's design actually IS, as numbers.
   ================================================================

   Injected by capture.py into whatever page is being measured. It reads the
   page's OWN COMPUTED STYLES: what the browser decided after every stylesheet,
   media query and inherited rule had its say. That is the only honest way to
   compare two designs built on two different stacks — LessWrong's typography
   lives in JSS objects compiled at runtime and ours lives in hand-written CSS
   files, and neither text form can be diffed against the other.

   🔴 THIS READS RENDERED FACTS, NOT SOURCE. Nothing here parses, copies or
   redistributes any of LessWrong's code — see README.md. What comes back is a
   measurement of a public web page: font sizes, gaps, colours, box geometry.

   The vocabulary is the point. Both sides describe themselves with the SAME
   role names (`prose_p`, `h2`, `toc_row`, ...) and different selectors, so the
   diff can line up a paragraph on one page with a paragraph on the other.
   ================================================================ */

(function (cfg) {
  const R = (n, places) => {
    const f = Math.pow(10, places == null ? 1 : places);
    return Number.isFinite(n) ? Math.round(n * f) / f : null;
  };
  const px = (value) => {
    const n = parseFloat(value);
    return Number.isFinite(n) ? R(n) : null;
  };

  /* rgb()/rgba() -> #rrggbb + alpha, so the diff can measure a colour distance
     instead of comparing two strings that disagree about formatting. */
  const color = (value) => {
    const m = String(value || "").match(/rgba?\(([^)]+)\)/);
    if (!m) return { hex: null, a: null, raw: value || null };
    const parts = m[1].split(/[,\s/]+/).filter(Boolean).map(Number);
    const [r, g, b] = parts;
    const a = parts.length > 3 ? parts[3] : 1;
    const hex =
      "#" + [r, g, b].map((c) => Math.max(0, Math.min(255, c | 0)).toString(16).padStart(2, "0")).join("");
    return { hex, a: R(a, 2), raw: value };
  };

  const laidOut = (el) => el.getClientRects().length > 0;

  const measure = (el) => {
    const cs = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    const size = px(cs.fontSize);
    const lh = cs.lineHeight === "normal" ? null : px(cs.lineHeight);
    return {
      box: {
        x: R(rect.left, 0),
        y: R(rect.top + window.scrollY, 0),
        w: R(rect.width, 0),
        h: R(rect.height, 0),
      },
      font: {
        family: (cs.fontFamily || "").split(",")[0].replace(/["']/g, "").trim() || null,
        size: size,
        weight: cs.fontWeight,
        style: cs.fontStyle,
        lineHeight: lh,
        /* The number that actually reads as "spacing" to a human. A 16px font
           at 1.5 and a 16px font at 1.2 are the same text at two densities. */
        ratio: lh && size ? R(lh / size, 2) : null,
        letterSpacing: cs.letterSpacing === "normal" ? 0 : px(cs.letterSpacing),
        transform: cs.textTransform,
      },
      color: color(cs.color),
      background: color(cs.backgroundColor),
      margin: [px(cs.marginTop), px(cs.marginRight), px(cs.marginBottom), px(cs.marginLeft)],
      padding: [px(cs.paddingTop), px(cs.paddingRight), px(cs.paddingBottom), px(cs.paddingLeft)],
      border: { width: px(cs.borderTopWidth), color: color(cs.borderTopColor).hex, radius: px(cs.borderTopLeftRadius) },
      opacity: R(parseFloat(cs.opacity), 2),
      text: (el.textContent || "").replace(/\s+/g, " ").trim().slice(0, 70) || null,
    };
  };

  const pick = (selectors) => {
    for (const sel of selectors) {
      let nodes;
      try {
        nodes = Array.from(document.querySelectorAll(sel));
      } catch (err) {
        continue;
      }
      /* A laid-out match beats an earlier one that is display:none — a hidden
         element measures 0 on every axis and would report as a design of
         zeroes rather than as "not found". */
      const hit = nodes.find((el) => laidOut(el) && el.getBoundingClientRect().width > 0) || null;
      if (hit) return { el: hit, selector: sel };
    }
    return null;
  };

  const median = (values) => {
    if (!values.length) return null;
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    return R(sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2);
  };

  /* A run of like elements — the ToC rows, the h2s, the paragraphs. What
     matters about a run is its RHYTHM, so what comes back is the spacing
     between consecutive members rather than a list of boxes. */
  const series = (selectors) => {
    let nodes = [];
    let used = null;
    for (const sel of selectors) {
      try {
        const found = Array.from(document.querySelectorAll(sel)).filter(laidOut);
        if (found.length) {
          nodes = found;
          used = sel;
          break;
        }
      } catch (err) {
        /* an invalid selector for this target is not an error for the run */
      }
    }
    if (!nodes.length) return { found: false, count: 0 };
    const rects = nodes.map((el) => el.getBoundingClientRect());
    const gaps = [];
    for (let i = 1; i < rects.length; i += 1) gaps.push(rects[i].top - rects[i - 1].top);
    const styles = nodes.map((el) => getComputedStyle(el));
    return {
      found: true,
      selector: used,
      count: nodes.length,
      gap: median(gaps.filter((g) => g > 0)),
      height: median(rects.map((r) => r.height)),
      width: median(rects.map((r) => r.width)),
      left: median(rects.map((r) => r.left)),
      indents: Array.from(new Set(rects.map((r) => Math.round(r.left)))).sort((a, b) => a - b).slice(0, 8),
      sizes: Array.from(new Set(styles.map((cs) => px(cs.fontSize)))).sort((a, b) => a - b),
      first: measure(nodes[0]),
    };
  };

  const roles = {};
  Object.keys(cfg.roles || {}).forEach((name) => {
    const hit = pick(cfg.roles[name]);
    roles[name] = hit
      ? Object.assign({ found: true, selector: hit.selector }, measure(hit.el))
      : { found: false };
  });

  const runs = {};
  Object.keys(cfg.series || {}).forEach((name) => {
    runs[name] = series(cfg.series[name]);
  });

  /* The reading column, derived rather than selected: the widest thing a
     paragraph is allowed to be, and where it sits. This is the single number
     that decides whether two pages "feel" the same before any type is read. */
  const p = roles.prose_p;
  const column = p && p.found
    ? {
        left: p.box.x,
        right: R(p.box.x + p.box.w, 0),
        width: p.box.w,
        centeredOffset: R(p.box.x - (window.innerWidth - (p.box.x + p.box.w)), 0),
      }
    : null;

  const root = getComputedStyle(document.documentElement);
  return {
    url: location.href,
    title: document.title,
    viewport: { w: window.innerWidth, h: window.innerHeight },
    document: { height: R(document.documentElement.scrollHeight, 0) },
    page: {
      background: color(getComputedStyle(document.body).backgroundColor).hex,
      rootFontSize: px(root.fontSize),
      colorScheme: root.colorScheme || null,
    },
    column: column,
    roles: roles,
    series: runs,
  };
});
