/* DOM CLONE — read one rail's rendered structure AND its resolved styling.

   probe.js answers "how big is the paragraph". This answers the different
   question Seth asked: what markup is actually there, and what does the
   browser think every node's style IS after the framework, the theme, the
   media queries and the cascade have all had their turn.

   🔴 THIS IS WHY "32 PROPERTIES IDENTICAL" AND "IT LOOKS WRONG" CAN BOTH BE
   TRUE. source_diff.py reads the declarations LessWrong's own components
   write. It cannot see anything they inherit, anything their theme injects,
   anything a parent lays out for them, or the fact that a node exists at all.
   getComputedStyle sees the finished job. When the two disagree, this one is
   right.

   Takes {root, roles} and returns a flat list of nodes in document order, each
   with its box, its own text, its computed style over a fixed property list,
   and the computed style of any ::before / ::after that actually renders. The
   baseline block is the same property list read off a bare element of each tag
   appended to this very page, so the emitter can tell "they set this" from
   "that is just what a <div> is here".
*/
(function domClone(spec) {
  var PROPS = [
    "display", "position", "top", "right", "bottom", "left", "float", "clear",
    "box-sizing", "width", "height", "min-width", "min-height", "max-width", "max-height",
    "margin-top", "margin-right", "margin-bottom", "margin-left",
    "padding-top", "padding-right", "padding-bottom", "padding-left",
    "flex-direction", "flex-wrap", "flex-grow", "flex-shrink", "flex-basis",
    "justify-content", "align-items", "align-self", "align-content", "gap", "order",
    "grid-template-columns", "grid-template-rows", "grid-column", "grid-row",
    "font-family", "font-size", "font-weight", "font-style", "font-variant",
    "line-height", "letter-spacing", "word-spacing", "text-align", "text-decoration-line",
    "text-transform", "text-indent", "text-overflow", "white-space", "word-break",
    "overflow-wrap", "vertical-align",
    "color", "background-color", "background-image", "opacity",
    "border-top-width", "border-right-width", "border-bottom-width", "border-left-width",
    "border-top-style", "border-right-style", "border-bottom-style", "border-left-style",
    "border-top-color", "border-right-color", "border-bottom-color", "border-left-color",
    "border-top-left-radius", "border-top-right-radius",
    "border-bottom-left-radius", "border-bottom-right-radius",
    "overflow-x", "overflow-y", "overscroll-behavior-x", "overscroll-behavior-y",
    "scrollbar-width", "visibility", "z-index", "pointer-events", "cursor",
    "transform", "transition-property", "transition-duration", "transition-timing-function",
    "list-style-type", "list-style-position", "content", "box-shadow", "filter",
  ];
  var MAX_NODES = 500;
  var MAX_DEPTH = 14;

  var root = document.querySelector(spec.root);
  if (!root) return { error: "root selector matched nothing: " + spec.root };

  var read = function (el, pseudo) {
    var cs = getComputedStyle(el, pseudo || null);
    var out = {};
    for (var i = 0; i < PROPS.length; i++) out[PROPS[i]] = cs.getPropertyValue(PROPS[i]);
    return out;
  };

  /* What a bare element of this tag computes to ON THIS PAGE — so inherited
     body typography and the browser's own defaults do not read as design. */
  var baselines = {};
  var baseFor = function (tag) {
    if (baselines[tag]) return baselines[tag];
    var probe = document.createElement(tag);
    probe.style.setProperty("position", "absolute");
    probe.style.setProperty("left", "-99999px");
    document.body.appendChild(probe);
    baselines[tag] = read(probe);
    probe.remove();
    return baselines[tag];
  };

  /* Only the text this element holds itself. A rail row's textContent is its
     label's text too, and printing it at every level makes the dump unreadable. */
  var ownText = function (el) {
    var parts = [];
    for (var i = 0; i < el.childNodes.length; i++) {
      var node = el.childNodes[i];
      if (node.nodeType === 3 && node.nodeValue.trim()) parts.push(node.nodeValue.trim());
    }
    return parts.join(" ").slice(0, 120);
  };

  var nodes = [];
  var index = new Map();
  var walk = function (el, path, depth) {
    if (nodes.length >= MAX_NODES || depth > MAX_DEPTH) return;
    var rect = el.getBoundingClientRect();
    var pseudo = {};
    ["::before", "::after"].forEach(function (name) {
      var got = read(el, name);
      /* `content: none` is the browser saying the pseudo-element does not
         exist. An empty string does not — `content: ""` is exactly how both
         rails draw their marks. */
      if (got.content && got.content !== "none") pseudo[name] = got;
    });
    index.set(el, nodes.length);
    nodes.push({
      i: nodes.length,
      path: path,
      depth: depth,
      tag: el.tagName.toLowerCase(),
      cls: el.getAttribute("class") || "",
      id: el.id || null,
      text: ownText(el),
      box: {
        x: Math.round(rect.left * 10) / 10,
        y: Math.round(rect.top * 10) / 10,
        w: Math.round(rect.width * 10) / 10,
        h: Math.round(rect.height * 10) / 10,
      },
      style: read(el),
      pseudo: pseudo,
    });
    baseFor(el.tagName.toLowerCase());
    var kids = el.children;
    for (var k = 0; k < kids.length; k++) walk(kids[k], path + "/" + k, depth + 1);
  };
  walk(root, "0", 0);

  /* Which captured node each shared role landed on, so a diff between two
     unrelated markups can line them up without guessing at the tree shape.
     `index` was filled by the walk itself rather than by a second traversal —
     a node the walk stopped short of has no style to compare anyway, and a
     separately-numbered second pass would quietly point at the wrong one. */
  var roles = {};
  Object.keys(spec.roles || {}).forEach(function (role) {
    var list = spec.roles[role];
    for (var i = 0; i < list.length; i++) {
      var found = root.matches(list[i]) ? root : root.querySelector(list[i]);
      if (found && index.has(found)) {
        roles[role] = { selector: list[i], node: index.get(found) };
        return;
      }
    }
    roles[role] = null;
  });

  return {
    root: spec.root,
    url: location.href,
    scrollY: window.scrollY,
    viewport: { w: window.innerWidth, h: window.innerHeight },
    rootFontSize: parseFloat(getComputedStyle(document.documentElement).fontSize),
    pageBackground: getComputedStyle(document.body).backgroundColor,
    props: PROPS,
    baselines: baselines,
    roles: roles,
    nodes: nodes,
  };
});
