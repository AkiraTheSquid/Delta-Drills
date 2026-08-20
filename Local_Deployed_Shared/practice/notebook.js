/* ================================================================
   LESSON NOTEBOOK — runnable code cells inside the explanation

   The lesson screen used to be the practice split: prose on the left, one
   editor on the right holding the whole worked example. That layout asks the
   learner to hold a mapping in their head — "the paragraph I am reading
   corresponds to *some* part of the block over there" — and the mapping gets
   harder exactly as the example gets longer.

   A notebook removes the mapping. Each code block sits where the prose that
   explains it sits, and each one runs on its own, so "what does this line
   actually do" is answered by pressing Run beside the line rather than by
   scrolling a second panel and deleting the parts you are not asking about.

   WHICH BLOCKS BECOME CELLS

   Every fenced block on the page except the ones the author marked `no-run`.

   That marker is not new and it is not a heuristic about the text. The lesson
   authoring format already draws exactly this line: `validate_lessons.py`
   executes every plain ```python fence in a KP — explanation and worked
   example alike — against one shared namespace, in document order, and skips
   ```python no-run fences, which exist precisely for pseudocode and blocks
   written to raise. So every cell this file mounts is a block CI already
   proves runs, in the order CI already proves it runs in. `md()` in
   practice/lessons.js carries the info string through as `data-fence` so it
   can be read here.

   This used to be scoped to `.lesson-worked`, on the reasoning that
   explanation fences were fragments that would only produce tracebacks. The
   marker makes that unnecessary, and the old scoping had a cost that showed
   up the first time anyone read a lesson: a page with one runnable block at
   the bottom, and the four places in the prose where the learner most wants
   to try something — "which is why `.contiguous()` exists" — left inert.

   HOW STATE WORKS ACROSS CELLS

   There are two answers, and which one a learner gets depends on whether the
   backend will give them a kernel.

   WITH A KERNEL (signed in). `DeltaKernel` holds one live Python process per
   learner on the backend, so cells share a namespace exactly as they do in
   Colab: run cell 6, run cell 8, and what cell 6 bound is still bound. Only
   the clicked cell is sent. That is what makes a long notebook usable at all —
   the fallback below costs O(cells) per click, which a six-cell lesson can
   afford and a 656-cell chapter cannot.

   The kernel can vanish between two clicks: it idles out, it is evicted for
   another learner, the box is redeployed. The server says so with `fresh` on
   the reply, and the answer to a fresh kernel is not to tell the learner their
   variables are gone — it is to replay the cells above this one into it, once,
   and hand back the clicked cell's own output. Rebuilding costs a round trip
   the learner did not ask for; being told to click nine buttons again costs
   more.

   WITHOUT ONE (guest, older backend, backend unreachable). Neither remaining
   runtime keeps a session: `/run-code` forks per request and the Pyodide
   preamble resets stdout each time. So a cell that says `flat = t.arange(9)`
   and a later one that says `flat.reshape(3, 3)` would fail on its own.

   Running a cell therefore executes every cell above it as well, and shows
   only the clicked cell's own output — the prefix is a way of rebuilding
   state, and its prints already sit under its own Run buttons. That gives
   ordinary notebook semantics on a stateless runner. The cost is honest and
   worth naming: editing an early cell changes what the later ones see, exactly
   as it would in Jupyter, and a slow early cell is paid for again by every
   cell below it.

   Both paths run the SAME cells through the SAME harness, so the two differ in
   how state got there and in nothing else.

   WHY THE CELLS ARE EXECUTED THROUGH A HARNESS

   Concatenating the prefix and handing the runner one string would work, but
   it loses two things a notebook is expected to have.

   The first is the echo. In Jupyter a cell ending in a bare expression prints
   its value; that is why `a.shape` is a useful thing to type. Plain `exec` of
   a script prints nothing, so before this the honest report for most cells was
   "no printed output" — a Run button whose whole reward was being told nothing
   happened. `_delta_cell` parses the cell, and when its last statement is an
   expression it evaluates that separately and prints the repr, skipping None
   so `print(...)` and in-place calls do not echo a spurious "None".

   The second is line numbers. A traceback from a concatenated program counts
   lines from the top of the PROGRAM, so an error in the cell the learner
   clicked is reported at a line they cannot see. Compiling each cell with its
   own `<cell N>` filename makes the reported line the line in that cell.

   Both are done with `ast` in the runtime rather than by pattern-matching the
   source in JavaScript. A regex that tries to decide "is this last line an
   expression?" gets multi-line calls, decorators and trailing comments wrong,
   and gets them wrong silently.

   WHAT "IT RAN" LOOKS LIKE

   A cell that has run keeps an execution counter (`In [3]`), the way a
   notebook does, so the page shows at a glance what has been run and in what
   order. The counter is shared across the page and increments per run, so
   re-running a cell moves it to the front.

   When a run prints nothing at all, saying "no printed output" is true but
   useless. If the cell contained assertions, the count of them is reported
   instead — those are checks that just passed, which is the actual result.

   WHAT RUNS, AND FOR WHOM

   Delegated wholesale to `DeltaRunner.runSnippet`, which owns the
   backend-vs-Pyodide decision for the whole app. The consequence worth
   knowing: the lessons are torch since the July dialect conversion, and
   Pyodide cannot import torch, so cells genuinely execute only for signed-in
   learners. Guests get the runner's plain explanation rather than a
   ModuleNotFoundError, and the lesson still reads correctly — the code is
   printed with its expected results either way.

   That held only once the runner was TOLD the code is torch. Two signals
   agreed it was not: `lessons.js` still advertised the lesson's library as
   "numpy" from before the conversion, and the sniff on the source cannot see
   an import that this file has already packed into a Python string literal.
   So every Run button on every lesson answered with a ModuleNotFoundError,
   for signed-in learners too. Hence `source:` on the call below — the runner
   decides on what the learner wrote, never on the harness around it.
   ================================================================ */

const LessonNotebook = (() => {
  "use strict";

  const esc = (value) =>
    String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

  /* The exact text `DeltaRunner.runSnippet` substitutes when a run succeeded
     and wrote nothing to stdout or stderr. Matched rather than imported
     because the runner returns one already-collapsed string; the alternative
     is a second return field threaded through both of its branches for this
     one caller. styles/practice/watch.py asserts runner.js still contains it,
     so the coupling fails loudly instead of silently reporting every quiet
     cell as an ordinary empty run. */
  const NO_OUTPUT = "✓ Ran successfully (no printed output)";
  // A cell that stopped the session without raising — `sys.exit(1)` is the one
  // that reaches here. It printed nothing, so there is nothing to show but the
  // fact that it did not finish.
  const SILENT_FAILURE = "✗ The cell stopped early (no error message)";

  /* Longest repr shown per name in the fallback summary. A tensor's repr runs
     to many lines, and the summary is a reminder of what the cell bound, not a
     substitute for printing the thing. */
  const SUMMARY_WIDTH = 160;
  const SUMMARY_MAX = 6;

  /* Executed ahead of the cells on every run. Kept to one definition with
     reserved names rather than inlined per cell, so a learner reading a
     traceback sees their own code and one frame they can ignore.

     Every name here is `_delta_`-prefixed because this shares a namespace with
     the learner's code: a helper called `names` would be clobbered by, or
     would clobber, a cell that used that word. */
  const HARNESS = [
    "import ast as _delta_ast",
    "import contextlib as _delta_ctx",
    "import io as _delta_io",
    "import sys as _delta_sys",
    "",
    "",
    // Top-level names a cell binds, in source order. Used only to say what a
    // silent cell did; nested and conditional bindings are deliberately not
    // chased, because the summary describes the cell as written.
    "def _delta_bound(_delta_body):",
    "    _delta_out = []",
    "    for _delta_node in _delta_body:",
    "        if isinstance(_delta_node, _delta_ast.Assign):",
    "            for _delta_tgt in _delta_node.targets:",
    "                if isinstance(_delta_tgt, _delta_ast.Name):",
    "                    _delta_out.append(_delta_tgt.id)",
    "        elif isinstance(_delta_node, (_delta_ast.AnnAssign, _delta_ast.AugAssign)):",
    "            if isinstance(_delta_node.target, _delta_ast.Name):",
    "                _delta_out.append(_delta_node.target.id)",
    "        elif isinstance(_delta_node, (_delta_ast.FunctionDef, _delta_ast.ClassDef)):",
    "            _delta_out.append(_delta_node.name)",
    "    return _delta_out",
    "",
    "",
    "def _delta_show(_delta_value):",
    "    try:",
    "        _delta_text = repr(_delta_value)",
    "    except Exception:",
    "        return '<unprintable>'",
    "    _delta_text = ' '.join(_delta_text.split())",
    `    if len(_delta_text) > ${SUMMARY_WIDTH}:`,
    `        _delta_text = _delta_text[:${SUMMARY_WIDTH - 1}] + '…'`,
    "    return _delta_text",
    "",
    "",
    "def _delta_cell(_delta_src, _delta_name, _delta_echo):",
    "    _delta_tree = _delta_ast.parse(_delta_src, _delta_name)",
    "    _delta_ns = globals()",
    "    _delta_body = _delta_tree.body",
    // Captured rather than written straight through, for two reasons. The cell
    // can be asked afterwards whether it said anything; and the prefix cells'
    // output is DISCARDED, because their output already sits under their own
    // Run buttons — replaying it into the clicked cell's pane would mean cell 4
    // showed cells 1-3's prints above its own, growing with every cell down the
    // page. A prefix that RAISED is the exception: whatever it printed before
    // failing is the context for the traceback, so that gets written out.
    "    _delta_buf = _delta_io.StringIO()",
    "    _delta_ok = False",
    "    try:",
    "        with _delta_ctx.redirect_stdout(_delta_buf):",
    "            if _delta_echo and _delta_body and isinstance(_delta_body[-1], _delta_ast.Expr):",
    "                _delta_head = _delta_ast.Module(body=_delta_body[:-1], type_ignores=[])",
    "                exec(compile(_delta_head, _delta_name, 'exec'), _delta_ns)",
    "                _delta_tail = _delta_ast.Expression(_delta_body[-1].value)",
    "                _delta_val = eval(compile(_delta_tail, _delta_name, 'eval'), _delta_ns)",
    // Jupyter does not echo None either: it is what a call with no return
    // value gives back, and printing it would put "None" under every cell
    // that ends in print() or an in-place operation.
    "                if _delta_val is not None:",
    "                    print(repr(_delta_val))",
    "            else:",
    "                exec(compile(_delta_tree, _delta_name, 'exec'), _delta_ns)",
    "        _delta_ok = True",
    "    finally:",
    "        if _delta_echo or not _delta_ok:",
    "            _delta_sys.stdout.write(_delta_buf.getvalue())",
    // Nothing printed and nothing to echo — the cell built something. Say what
    // it built, rather than reporting the run as empty.
    "    if _delta_echo and not _delta_buf.getvalue().strip():",
    "        _delta_seen = []",
    "        for _delta_key in _delta_bound(_delta_body):",
    "            if _delta_key in _delta_ns and _delta_key not in _delta_seen:",
    "                _delta_seen.append(_delta_key)",
    `        for _delta_key in _delta_seen[:${SUMMARY_MAX}]:`,
    "            print(_delta_key, '=', _delta_show(_delta_ns[_delta_key]))",
    "",
  ].join("\n");

  /* JSON string syntax is a subset of Python string syntax — the escapes
     JSON.stringify emits (\", \\, \n, \t, \uXXXX) all mean the same thing in a
     Python literal, and it leaves other non-ASCII as literal UTF-8, which is
     what a Python 3 source file is read as. So this is a safe way to hand a
     cell's source to the runtime as data rather than as code to be spliced. */
  const _pyLiteral = (text) => JSON.stringify(String(text == null ? "" : text));

  /* One cell's markup. The code is rendered as plain text rather than an
     editable field: these are the lesson's examples, and an edit box invites
     the learner to lose the example they were given. `contenteditable` on the
     <code> keeps experimentation possible without turning the page into a
     form — the text is read back at Run time, so an edit is honoured. */
  const cellHtml = (code, index) =>
    '<div class="nb-cell" data-nb-index="' + index + '">' +
    '<div class="nb-cell-code">' +
    '<pre><code contenteditable="plaintext-only" spellcheck="false">' +
    esc(code) +
    "</code></pre>" +
    "</div>" +
    '<div class="nb-cell-bar">' +
    '<button type="button" class="nb-run">Run</button>' +
    // Empty until the cell has run, then it keeps its execution number the way
    // a notebook's In[] prompt does, so the page shows what has been run.
    '<span class="nb-count" aria-hidden="true"></span>' +
    '<span class="nb-status" aria-live="polite"></span>' +
    "</div>" +
    '<pre class="nb-out hidden"></pre>' +
    "</div>";

  const _codeOf = (cell) => {
    const node = cell.querySelector(".nb-cell-code code");
    return node ? node.innerText.replace(/ /g, " ") : "";
  };

  /* Every cell up to and including this one, each handed to the harness as
     data. See the note above on why the whole prefix is re-run rather than
     only the cell that was clicked, and why the harness exists at all.

     Only the clicked cell echoes. Re-running the prefix is a means of
     rebuilding state, not something the learner asked to see the value of. */
  const _programUpTo = (cells, index) =>
    HARNESS +
    cells
      .slice(0, index + 1)
      .map(
        (cell, i) =>
          `_delta_cell(${_pyLiteral(_codeOf(cell))}, ` +
          `${_pyLiteral(`<cell ${i + 1}>`)}, ${i === index ? "True" : "False"})`,
      )
      .join("\n");

  /* The clicked cell alone, for the kernel path. Same harness call as the
     prefix builds, so a cell behaves identically whichever path ran it. */
  const _cellProgram = (cells, index) =>
    HARNESS +
    `_delta_cell(${_pyLiteral(_codeOf(cells[index]))}, ` +
    `${_pyLiteral(`<cell ${index + 1}>`)}, True)`;

  /* Which notebook the kernel is currently holding. Passed to the server so a
     learner who moves to a different concept gets a clean namespace instead of
     the last page's names — the cells on this page start from cell 1 either
     way, and a leftover `a` from another lesson is a silent wrong answer. */
  let mountContext = "";

  /* Kernel output, collapsed the way `DeltaRunner.runSnippet` collapses its
     own: one string worth showing, plus whether it failed. stderr is appended
     rather than replacing stdout — an interrupted cell has both, and what it
     printed before it was stopped is the context for why.

     FAILURE IS `ok`, NOT "there is something on stderr". The two are not the
     same in either direction: a cell that raises a DeprecationWarning writes to
     stderr and succeeded, and a cell that calls `sys.exit(1)` writes nothing
     and did not. Reading stderr alone paints the first one red and hands the
     second one "✓ Ran successfully (no printed output)". */
  const _kernelText = (reply) => {
    const stdout = (reply.stdout || "").replace(/\r\n/g, "\n").trim();
    const stderr = (reply.stderr || "").replace(/\r\n/g, "\n").trim();
    const failed = reply.ok === false;
    const text = [stdout, stderr].filter(Boolean).join("\n");
    if (text) return { text, failed };
    return { text: failed ? SILENT_FAILURE : NO_OUTPUT, failed };
  };

  /* Run one cell against the learner's live session.

     Returns null when there is no kernel to be had — guest, older backend,
     network — which is the caller's signal to take the stateless path. Null is
     deliberately not an error: a fallback the learner never notices is the
     whole point of keeping the prefix path alive.  */
  const _runOnKernel = async (cells, index, onStatus) => {
    const kernel = window.DeltaKernel;
    if (!kernel || !kernel.available()) return null;
    const request = { bootstrap: HARNESS, filename: "<harness>", context: mountContext };
    // The answer to a fresh kernel is to replay cells 1..N, and that replay
    // ENDS with the cell that was clicked. So on any cell but the first, tell
    // the server not to run it if it had to build the kernel — otherwise the
    // clicked cell runs twice, and a cell that appends to a list, writes a
    // file or bumps a counter is not the same run twice.
    let reply = await kernel.runCell({
      ...request,
      code: _cellProgram(cells, index),
      skipOnFresh: index > 0,
    });
    if (!reply || reply.unavailable) return null;
    if (reply.busy) {
      return { text: "The kernel is still running a cell — wait for it to finish.", failed: true };
    }
    // A kernel that did not exist a moment ago has never seen the cells above
    // this one. Replay them in one call and use THAT result, so the learner
    // sees the output of the cell they clicked rather than a NameError.
    if (reply.fresh && index > 0) {
      onStatus(`restoring cells 1–${index + 1}`);
      const rebuilt = await kernel.runCell({ ...request, code: _programUpTo(cells, index) });
      if (!rebuilt || rebuilt.unavailable) return null;
      if (rebuilt.busy) {
        return { text: "The kernel is still running a cell — wait for it to finish.", failed: true };
      }
      reply = rebuilt;
    }
    return _kernelText(reply);
  };

  /* Assertions that a silent run just passed.

     Counted from the source rather than reported by the runtime, which is a
     real limitation worth stating: an `assert` inside a function that is never
     called is counted here and never checked. Lesson cells assert at the top
     level, and the alternative — an audited counter in the harness — would
     mean rewriting the learner's code to instrument it. */
  const _checkCount = (code) => (String(code).match(/^[ \t]*assert\b/gm) || []).length;

  /* The whole page shares one counter, incremented per run, exactly like a
     notebook's In[] prompt: the numbers say what ran and in what order rather
     than how many times each cell has been clicked. */
  let runSeq = 0;

  const _runCell = async (cells, index) => {
    const cell = cells[index];
    const button = cell.querySelector(".nb-run");
    const status = cell.querySelector(".nb-status");
    const count = cell.querySelector(".nb-count");
    const out = cell.querySelector(".nb-out");
    if (!button || !out) return;

    button.disabled = true;
    button.textContent = "Running…";
    // Only the stateless path pays for the cells above; with a kernel this
    // cell is the only one that runs, and saying otherwise would be a lie the
    // learner can time.
    const usingKernel = !!(window.DeltaKernel && window.DeltaKernel.available());
    status.textContent = !usingKernel && index > 0 ? `running cells 1–${index + 1}` : "";
    if (count) count.textContent = "In [*]";
    cell.classList.add("is-running");
    out.classList.remove("hidden", "is-error");
    out.textContent = "";

    let failed = false;
    try {
      let result = await _runOnKernel(cells, index, (message) => {
        status.textContent = message;
      });
      if (!result) {
        // No kernel — rebuild state by re-running the prefix, as before.
        status.textContent = index > 0 ? `running cells 1–${index + 1}` : "";
        result = await window.DeltaRunner.runSnippet(_programUpTo(cells, index), {
          question: window.LessonGate?.activeQuestion || null,
          // What the learner actually wrote, for the runner's torch sniff. The
          // program above wraps every cell in a string literal, where an import
          // line is invisible to a line-anchored regex.
          source: cells.slice(0, index + 1).map(_codeOf).join("\n"),
          onStatus: (message) => {
            if (message) out.textContent = message;
          },
        });
      }
      failed = !!result.failed;
      const checks = _checkCount(_codeOf(cell));
      out.textContent =
        !failed && result.text === NO_OUTPUT && checks
          ? `✓ ${checks} check${checks === 1 ? "" : "s"} passed`
          : result.text;
      out.classList.toggle("is-error", failed);
    } catch (err) {
      failed = true;
      out.textContent = "Error: " + err.message;
      out.classList.add("is-error");
    }

    runSeq += 1;
    cell.classList.remove("is-running");
    cell.classList.add("has-run");
    cell.classList.toggle("has-failed", failed);
    if (count) count.textContent = `In [${runSeq}]`;
    status.textContent = "";
    button.disabled = false;
    button.textContent = "Run";
  };

  /* Exactly the fences `validate_lessons.py` executes.

     It filters on the info string being the bare word `python`, so `python
     no-run` is skipped and so is anything else. Matching that exactly, rather
     than "anything not marked no-run", keeps one rule: a block gets a Run
     button if and only if CI runs it. A `text` fence or a `## Watch out`
     snippet that CI never touches would otherwise get a button whose failure
     nobody would find until a learner pressed it. */
  const _isRunnable = (pre) => (pre?.dataset?.fence || "") === "python";

  /* Turn the runnable fenced blocks inside `host` into cells.

     Scope comes from the caller, which marks the regions whose fences CI
     executes with `.nb-scope` — the concept body and the worked example. That
     keeps the two decisions apart: the lesson page knows which of its sections
     are programs, this file knows what makes a block runnable.

     Document order IS execution order, and it is the same order
     `validate_lessons.py` uses, so the state a cell inherits on screen is the
     state CI proved it inherits.

     Operates on rendered DOM rather than on markdown so there is exactly one
     markdown renderer in the app: the lesson body, the ladder's inline example
     and the standalone viewer all keep producing the same `<pre><code>`, and
     this decides which of those become interactive. */
  const mount = (host, context = "") => {
    if (!host || !window.DeltaRunner) return 0;
    runSeq = 0;
    // Identifies this page's namespace to the kernel. Mounting a different
    // page means a different context, which the server answers by restarting
    // the session — the cells below start from cell 1 regardless, so a name
    // surviving from the previous concept could only mislead.
    mountContext = String(context || "");
    const blocks = Array.from(host.querySelectorAll(".nb-scope pre > code")).filter(
      (node) => !node.closest(".nb-cell") && _isRunnable(node.parentElement),
    );
    blocks.forEach((node, index) => {
      const pre = node.parentElement;
      const wrapper = document.createElement("div");
      wrapper.innerHTML = cellHtml(node.textContent, index);
      pre.replaceWith(wrapper.firstElementChild);
    });

    const cells = Array.from(host.querySelectorAll(".nb-cell"));
    cells.forEach((cell, index) => {
      const button = cell.querySelector(".nb-run");
      if (button) button.onclick = () => _runCell(cells, index);
    });
    return cells.length;
  };

  /* Run ONE source string as a cell in a named session.

     This is the primitive `practice/notebook-view.js` is built on, and it
     exists so the app has exactly one `_delta_cell` harness. `mount` above
     serves a lesson PAGE: its cells are fences in rendered DOM, and a missing
     kernel is answered by replaying the prefix. The notebook view serves a
     whole LESSON: its cells come from a compiled JSON file, and replaying a
     prefix of up to 656 of them is the precise thing the kernel was built to
     abolish. The two surfaces differ in everything except what a cell IS —
     so that, and only that, is shared.

     Returns null when there is no kernel, and says nothing about it. The two
     callers give the learner different answers to that (one falls back
     silently, one explains why Run is off), and picking one here would put the
     wrong sentence on one of the two screens. */
  const runSource = async (source, { context = "", name = "<cell>", echo = true } = {}) => {
    const kernel = window.DeltaKernel;
    if (!kernel || !kernel.available()) return null;
    const reply = await kernel.runCell({
      bootstrap: HARNESS,
      filename: "<harness>",
      context: String(context || ""),
      code:
        HARNESS +
        `_delta_cell(${_pyLiteral(source)}, ${_pyLiteral(name)}, ${echo ? "True" : "False"})`,
    });
    if (!reply || reply.unavailable) return null;
    if (reply.busy) {
      return {
        text: "The kernel is still running a cell — wait for it to finish.",
        failed: true,
        busy: true,
        fresh: false,
      };
    }
    return {
      ..._kernelText(reply),
      fresh: !!reply.fresh,
      busy: false,
      execCount: reply.execCount || 0,
    };
  };

  return { mount, runSource, checkCount: _checkCount };
})();

window.LessonNotebook = LessonNotebook;
