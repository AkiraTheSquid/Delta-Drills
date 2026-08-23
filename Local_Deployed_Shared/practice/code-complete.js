/* ================================================================
   CODE-COMPLETE.JS — inline ghost completion in the practice editors.

   WHAT THE LEARNER SEES
   They type `t.man`, and `ual_seed` appears after the caret in dimmed text.
   Tab takes it; anything else ignores it. Escape dismisses it. This is the
   Colab/Copilot "ghost text" affordance, drawn by `code-highlight.js` —
   the overlay it paints behind the textarea is the only layer in this app
   that can show a character the textarea does not contain.

   WHAT IT WILL NEVER SUGGEST — read this before adding a dictionary
   A completion is a NAME and nothing else. Never an argument list, never a
   body, never a line. The whole app is a grader: the moment a suggestion
   can carry a value, the editor is telling the learner what to write, and
   `t.manual_seed(0)` as one suggestion has already leaked the seed the
   question wanted them to know about. The candidate pools below are
   therefore, exhaustively:

     * PUBLIC API NAMES — torch / tensor methods / nn / nn.functional /
       einops / numpy. This is the same thing an editor's index of the
       installed package gives you, and knowing that `argmax` exists is not
       knowing the answer to a question about it.
     * IDENTIFIERS ALREADY IN THE BUFFER — the learner's own variables and
       defs, plus whatever the starter code put there. Nothing comes out
       that the learner cannot already see on screen.

   `question.answer_code` and the solution manifests are NOT read here, and
   must not become inputs. If a future change wants smarter suggestions, the
   test is: could this string appear on screen before the learner has worked
   it out? If yes, it does not belong.

   WHY TAB, AND HOW IT COEXISTS WITH THE INDENT RULE
   `runner.js`'s `installCodeEditorKeys` owns Tab: it inserts four spaces,
   or (de)indents a selected block. That rule is load-bearing — the question
   bank is space-indented, and a literal tab makes Python raise TabError.
   So this file does not touch the textarea's own Tab handler. It listens on
   the `.code-surface` WRAPPER in the CAPTURE phase, which runs before any
   listener on the textarea itself, and calls `stopPropagation()` ONLY on the
   two keys it actually consumes (Tab while a ghost is showing, Escape while
   a ghost is showing). With no ghost on screen it is transparent: Tab
   indents exactly as it did before this file existed.

   (Registration order would not have worked. For a listener on the target
   element the capture flag is ignored and listeners fire in the order they
   were added, so "add ours first" would depend on script order between two
   modules that have no other reason to care about each other.)
   ================================================================ */
const DeltaCodeComplete = (() => {
  "use strict";

  const TORCH = [
    "abs", "allclose", "arange", "argmax", "argmin", "argsort", "as_tensor",
    "bmm", "bool", "broadcast_to", "cat", "cdist", "chunk", "clamp",
    "clone", "column_stack", "cos", "cross", "cumprod", "cumsum", "device",
    "diag", "diagonal", "div", "dot", "einsum", "empty", "equal", "exp",
    "eye", "flatten", "flip", "float32", "float64", "from_numpy", "full",
    "gather", "int32", "int64", "isclose", "isnan", "linspace", "log",
    "log2", "log10", "log_softmax", "logical_and", "logical_not",
    "logical_or", "long", "manual_seed", "masked_select", "matmul", "max",
    "maximum", "mean", "median", "meshgrid", "min", "minimum", "mm", "mul",
    "nn", "no_grad", "nonzero", "norm", "ones", "ones_like", "optim",
    "outer", "permute", "pow", "prod", "rand", "randint", "randn",
    "randperm", "repeat_interleave", "reshape", "roll", "round", "save",
    "scatter", "sigmoid", "sign", "sin", "softmax", "sort", "split", "sqrt",
    "squeeze", "stack", "std", "sum", "take", "tanh", "tensor", "tile",
    "topk", "transpose", "tril", "triu", "unbind", "unique", "unsqueeze",
    "var", "vstack", "where", "zeros", "zeros_like",
  ];

  /* Methods and attributes on a Tensor. Also the pool used for a receiver
     this file cannot identify (`out.`, `logits.`) — in a torch question bank
     an unknown local is a tensor far more often than it is anything else,
     and a wrong pool costs a suggestion that simply never matches. */
  const TENSOR = [
    "T", "abs", "all", "any", "argmax", "argmin", "argsort", "backward",
    "bool", "clamp", "clone", "contiguous", "cos", "count_nonzero", "cpu",
    "cuda", "cumsum", "detach", "device", "dim", "dtype", "exp", "expand",
    "expand_as", "fill_", "flatten", "flip", "float", "gather", "grad",
    "index_select", "int", "item", "long", "masked_fill", "masked_fill_",
    "matmul", "max", "mean", "median", "min", "mT", "ndim", "norm",
    "numpy", "permute", "pow", "prod", "repeat", "requires_grad",
    "requires_grad_", "reshape", "roll", "scatter_", "shape", "sigmoid",
    "sin", "size", "softmax", "sort", "split", "sqrt", "squeeze", "std",
    "sum", "tanh", "to", "tolist", "topk", "transpose", "tril", "triu",
    "unbind", "unique", "unsqueeze", "var", "view", "view_as", "where",
    "zero_",
  ];

  const NN = [
    "AvgPool2d", "BatchNorm1d", "BatchNorm2d", "BCELoss", "Conv1d",
    "Conv2d", "CrossEntropyLoss", "Dropout", "Embedding", "Flatten", "GELU",
    "Identity", "LayerNorm", "Linear", "LogSoftmax", "MaxPool2d",
    "Module", "ModuleDict", "ModuleList", "MSELoss", "NLLLoss", "Parameter",
    "ReLU", "Sequential", "Sigmoid", "Softmax", "Tanh", "functional",
  ];

  const FUNCTIONAL = [
    "conv1d", "conv2d", "cross_entropy", "dropout", "gelu", "linear",
    "log_softmax", "max_pool2d", "mse_loss", "nll_loss", "normalize",
    "one_hot", "pad", "relu", "scaled_dot_product_attention", "sigmoid",
    "softmax", "tanh",
  ];

  const EINOPS = ["einsum", "pack", "parse_shape", "rearrange", "reduce", "repeat", "unpack"];

  const NUMPY = [
    "allclose", "arange", "argmax", "argmin", "argsort", "array", "concatenate",
    "cumsum", "dot", "einsum", "expand_dims", "eye", "float32", "int64",
    "linspace", "matmul", "max", "mean", "min", "ndarray", "ones", "prod",
    "random", "reshape", "shape", "sqrt", "squeeze", "stack", "std", "sum",
    "transpose", "unique", "where", "zeros",
  ];

  /* NOT the same list as code-highlight.js's, and deliberately so — they
     answer different questions. The highlighter colours everything Python
     HAS, so it carries `nonlocal`, `async`, `await`, `del`, `match`, `case`.
     This list is what the editor should put in front of a learner working
     through a torch drill, so it drops those and adds the three constants
     the highlighter keeps separate as CONSTANTS. Kept apart on purpose: a
     shared constant would mean every keyword Python gains also becomes a
     suggestion. */
  const KEYWORDS = [
    "and", "assert", "break", "class", "continue", "def", "elif", "else",
    "except", "False", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "None", "not", "or", "pass", "raise", "return",
    "True", "try", "while", "with", "yield",
  ];

  const BUILTINS = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "getattr", "hasattr", "int", "isinstance", "len", "list", "map", "max",
    "min", "print", "range", "repr", "reversed", "round", "set", "sorted",
    "str", "sum", "tuple", "type", "zip",
  ];

  const POOLS = {
    torch: TORCH, tensor: TENSOR, nn: NN, functional: FUNCTIONAL,
    einops: EINOPS, numpy: NUMPY,
  };

  /* Which pool you land in after walking one more dot. `nn` is reachable
     both as an import (`import torch.nn as nn`) and as an attribute of
     torch (`t.nn.Linear`), and the bank's own starters use both spellings —
     without this, `t.nn.Li` resolves its receiver to the string "t.nn",
     misses the alias table, and silently falls back to tensor methods, so
     `Linear` is never offered on the one expression a learner is most
     likely to be typing. */
  const SUBMODULES = {
    torch: { nn: "nn" },
    nn: { functional: "functional" },
  };

  const poolFor = (receiver, aliases) => {
    const parts = receiver.split(".");
    let ns = aliases[parts[0]];
    for (let i = 1; ns && i < parts.length; i += 1) ns = SUBMODULES[ns]?.[parts[i]] || null;
    return ns ? POOLS[ns] : null;
  };

  /* What a bare name binds to before any import line is read. These match
     the bank's own dialect (`DEFAULT_EDITOR_CODE` is `import torch as t`),
     so completion works on the very first keystroke of a fresh cell — the
     import scan below then corrects them for whatever the learner wrote. */
  const DEFAULT_ALIASES = {
    t: "torch", torch: "torch", nn: "nn", F: "functional",
    np: "numpy", numpy: "numpy", einops: "einops",
  };

  /* `import x as y` / `import a.b.c as y` / `from a import b, c` — enough to
     follow the four import shapes the bank actually uses. Anything else just
     falls back to DEFAULT_ALIASES, which costs a suggestion, never a wrong
     edit. */
  const readImports = (src) => {
    const aliases = Object.assign(Object.create(null), DEFAULT_ALIASES);
    const symbols = [];
    const nsOfImport = (dotted) => {
      if (dotted === "torch.nn.functional") return "functional";
      if (dotted === "torch.nn") return "nn";
      if (dotted === "torch") return "torch";
      if (dotted === "numpy") return "numpy";
      if (dotted === "einops") return "einops";
      return null;
    };
    src.split("\n").forEach((line) => {
      let m = line.match(/^\s*import\s+([\w.]+)(?:\s+as\s+(\w+))?/);
      if (m) {
        const pool = nsOfImport(m[1]);
        const name = m[2] || m[1].split(".")[0];
        if (pool) aliases[name] = pool;
        return;
      }
      m = line.match(/^\s*from\s+([\w.]+)\s+import\s+(.+)$/);
      if (!m) return;
      const from = m[1];
      m[2].split(",").forEach((piece) => {
        const parts = piece.trim().match(/^(\w+)(?:\s+as\s+(\w+))?/);
        if (!parts) return;
        const local = parts[2] || parts[1];
        const pool = nsOfImport(`${from}.${parts[1]}`);
        if (pool) aliases[local] = pool;
        else symbols.push(local);
      });
    });
    return { aliases, symbols };
  };

  const IDENT = /[A-Za-z_]\w*/g;

  /* Identifiers the learner can already see, by how often they occur. This
     is what makes completing a long variable name feel like an editor rather
     than a dictionary lookup, and it is the only source here that is not a
     fixed list. */
  const localNames = (src) => {
    const freq = new Map();
    IDENT.lastIndex = 0;
    let m;
    while ((m = IDENT.exec(src)) !== null) {
      freq.set(m[0], (freq.get(m[0]) || 0) + 1);
    }
    return freq;
  };

  const attrsOf = (src, receiver) => {
    const out = [];
    const re = new RegExp(`\\b${receiver.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\.(\\w+)`, "g");
    let m;
    while ((m = re.exec(src)) !== null) out.push(m[1]);
    return out;
  };

  // A completion inside a comment or a string literal is noise, and inside a
  // docstring it is noise that spans lines. The highlighter has already had
  // to answer "what is this character part of", so ask it rather than
  // writing a second, differently-wrong scanner.
  const inLiteral = (src, at) => {
    const tokens = window.DeltaCodeHighlight?.tokenize?.(src) || [];
    for (const tok of tokens) {
      if (at > tok.start && at <= tok.end) return tok.cls === "com" || tok.cls === "str";
    }
    return false;
  };

  const rank = (name, source, freq) => [
    source,
    -(freq.get(name) || 0),
    name.length,
    name,
  ];

  const better = (a, b) => {
    for (let i = 0; i < a.length; i += 1) {
      if (a[i] < b[i]) return true;
      if (a[i] > b[i]) return false;
    }
    return false;
  };

  /* The suggestion for wherever the caret currently is, or "" for none.

     Deliberately conservative — every guard below is a case where a ghost
     would be in the learner's way rather than ahead of them:
       * a selection means they are about to replace something;
       * text after the caret means the ghost would be spliced INTO a line,
         and the overlay's copy of that line would no longer sit under the
         textarea's (the alignment contract in code-highlight.js);
       * one character is not yet an intent, except after a dot, where the
         receiver has already narrowed it to one package. */
  const suggest = (editor) => {
    const value = editor.value;
    const at = editor.selectionStart;
    if (at !== editor.selectionEnd) return "";
    const lineEnd = value.indexOf("\n", at);
    if (value.slice(at, lineEnd < 0 ? value.length : lineEnd).trim()) return "";

    const before = value.slice(0, at);
    const chain = (before.match(/([A-Za-z_][\w.]*)$/) || [])[1];
    if (!chain) return "";
    const dot = chain.lastIndexOf(".");
    const prefix = dot < 0 ? chain : chain.slice(dot + 1);
    const receiver = dot < 0 ? "" : chain.slice(0, dot);
    if (!prefix || prefix.length < (receiver ? 1 : 2)) return "";
    if (inLiteral(value, at)) return "";

    const freq = localNames(value);
    const { aliases, symbols } = readImports(value);
    const groups = [];
    if (receiver) {
      const pool = poolFor(receiver, aliases) || TENSOR;
      groups.push([attrsOf(value, receiver), 0], [pool, 1]);
    } else {
      groups.push(
        [Array.from(freq.keys()), 0],
        [KEYWORDS, 1],
        [BUILTINS, 2],
        [symbols.concat(Object.keys(aliases)), 3],
      );
    }

    let best = null;
    let bestKey = null;
    groups.forEach(([names, source]) => {
      names.forEach((name) => {
        if (name === prefix || !name.startsWith(prefix)) return;
        const key = rank(name, source, freq);
        if (!best || better(key, bestKey)) { best = name; bestKey = key; }
      });
    });
    return best ? best.slice(prefix.length) : "";
  };

  const show = (editor) => {
    window.DeltaCodeHighlight?.setGhost(editor, suggest(editor));
  };

  const dismiss = (editor) => {
    window.DeltaCodeHighlight?.setGhost(editor, "");
  };

  const accept = (editor) => {
    const ghost = editor.__deltaGhost || "";
    if (!ghost) return;
    const at = editor.selectionStart;
    dismiss(editor);
    editor.value = editor.value.slice(0, at) + ghost + editor.value.slice(at);
    editor.selectionStart = editor.selectionEnd = at + ghost.length;
    // Every other writer in the app announces itself this way; the notebook's
    // draft autosave in timer.js is listening.
    editor.dispatchEvent(new Event("input", { bubbles: true }));
  };

  // A bare modifier press is not a decision. Dismissing on Shift would kill
  // the ghost the moment someone reached for an underscore.
  const MODIFIERS = new Set(["Shift", "Control", "Alt", "Meta", "CapsLock", "Dead"]);

  function attach(editor) {
    if (!editor || editor.dataset.deltaComplete === "1") return;
    const surface = window.DeltaCodeHighlight?.surfaceOf(editor);
    if (!surface) return;
    editor.dataset.deltaComplete = "1";

    surface.addEventListener("keydown", (event) => {
      if (!editor.__deltaGhost) return;
      if (event.key === "Tab" && !event.shiftKey) {
        event.preventDefault();
        event.stopPropagation();
        accept(editor);
        return;
      }
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        dismiss(editor);
        return;
      }
      if (MODIFIERS.has(event.key)) return;
      // Every other key either types over the suggestion or moves away from
      // it. Drop it and let the key through untouched; `input`/`keyup` below
      // will offer a fresh one if there is still something to offer.
      dismiss(editor);
    }, true);

    editor.addEventListener("input", () => show(editor));
    /* Every way the caret can move WITHOUT the keydown handler above seeing
       it. Missing these is not a cosmetic bug: the highlighter re-renders on
       click, so a ghost computed for `de` follows the caret to wherever the
       learner just clicked, still looks live, and Tab then splices `f` into
       the middle of an unrelated line. `show` recomputes from scratch and
       returns "" wherever a suggestion no longer makes sense, so one call
       covers both "offer a new one" and "drop the stale one". */
    ["click", "select", "mouseup", "dragend"].forEach((type) => {
      editor.addEventListener(type, () => show(editor));
    });
    editor.addEventListener("keyup", (event) => {
      if (event.key.startsWith("Arrow") || event.key === "Home" || event.key === "End"
          || event.key === "PageUp" || event.key === "PageDown") show(editor);
    });
    editor.addEventListener("blur", () => dismiss(editor));
  }

  return { attach, suggest, readImports };
})();

window.DeltaCodeComplete = DeltaCodeComplete;
