/* ================================================================
   JARGON GLOSSARY — every course term that earns a hover definition,
   and the concept-graph node that teaches it.

   WHAT THIS IS
     The lessons are written for someone who has never seen a tensor.
     They still lean on ~75 words that only mean something once you
     have met them — "view", "broadcasting", "contracted axis",
     "keepdims". A learner meeting one of those on page 3 of np-3 had
     no way back to the page that defined it except to remember which
     lesson it was in. jargon.js turns each of them into a hover link;
     this file is the copy plus the destination.

   THE SHAPE OF AN ENTRY
     term     the canonical spelling, lower case. What the popup titles.
     aliases  every other surface form that should light up: plurals,
              hyphenations, the spelled-out phrase. Matched case-
              insensitively on WORD BOUNDARIES, longest form first, so
              "boolean mask" wins over "mask" where both could match.
     kc       the knowledge-component id that TEACHES this term — the
              node "Take me to the lesson" opens, maximized. It must
              exist in lessons/kc_registry.json; watch_jargon.py
              asserts every one of them does, because a dead id sends
              the learner to a graph that quietly focuses nothing.
     def      one or two short sentences. This is a SIGNPOST, not the
              lesson: enough to keep reading, and a reason to click
              through if it isn't. Plain text — jargon.js escapes it.

   HOUSE RULES THAT ARE NOT OBVIOUS
     - A term is never linked inside the lesson that teaches it.
       jargon.js suppresses that at render time (see `_selfKc`), so
       entries here don't have to dodge their own prose.
     - Prefer PROSE spellings over API spellings. `t.where` and
       `topk` live inside <code> in almost every sentence that uses
       them, and jargon.js never decorates code — an entry spelled
       that way would simply never match anything.
     - Common English words that are also jargon here ("view",
       "axis", "shape") ARE included on purpose. The
       first-occurrence-per-section rule is what keeps them from
       becoming a sea of underlines; dropping them would leave the
       three words a beginner most needs unexplained.
   ================================================================ */

window.DD_GLOSSARY = {
  version: 1,
  terms: [
    /* ---- np-1: arrays from the ground up ------------------------- */
    {
      term: "tensor",
      aliases: ["tensors", "ndarray", "ndarrays", "n-dimensional array"],
      kc: "numpy.ndarray-model",
      def: "One block of memory holding elements that are all the same type, plus the metadata — shape and dtype — that says how to read that block. Everything else in the course manipulates this one object.",
    },
    {
      term: "shape",
      aliases: ["shapes"],
      kc: "numpy.ndarray-model",
      def: "The tuple of axis lengths describing how the flat block of memory is laid out — (2, 3) is two rows of three. It is metadata, not data, which is why reshaping is usually free.",
    },
    {
      term: "contiguous",
      aliases: ["contiguity", "non-contiguous", "reading order"],
      kc: "numpy.ndarray-model",
      def: "A tensor is contiguous when its elements sit in memory in the order you read them. Transposes and stepped slices break that, which is what .contiguous() exists to repair by copying.",
    },
    {
      term: "constructor",
      aliases: ["constructors"],
      kc: "numpy.constructors",
      def: "A function that builds a tensor from scratch rather than converting data you already have — zeros, ones, full, eye. They all share one calling convention: say how big, optionally say what type.",
    },
    {
      term: "slice",
      aliases: ["slices", "slicing", "stepped slice", "strided slice"],
      kc: "numpy.slicing-views",
      def: "Naming a rectangular piece of a tensor with start:stop:step, one slice per axis. Stop is exclusive, and the result shares memory with the original rather than copying it.",
    },
    {
      term: "view",
      aliases: ["views"],
      kc: "numpy.slicing-views",
      def: "A tensor that re-describes another tensor's memory instead of owning its own. Writing into a view writes into the original — the single most common source of surprise in tensor code.",
    },
    {
      term: "slice assignment",
      aliases: ["assigning into a slice"],
      kc: "numpy.slicing-views",
      def: "Writing to a slice — x[2:5] = 0 — mutates the original tensor in place, because the slice was never a copy.",
    },
    {
      term: "arange",
      aliases: [],
      kc: "numpy.ranges",
      def: "Builds a range when you know the STEP. Like Python's range it stops BEFORE the endpoint, which is where most range bugs come from.",
    },
    {
      term: "linspace",
      aliases: [],
      kc: "numpy.ranges",
      def: "Builds a range when you know how many POINTS you want. Unlike arange, both endpoints are included.",
    },
    {
      term: "dtype",
      aliases: ["dtypes", "data type", "type promotion"],
      kc: "numpy.dtype-astype",
      def: "The one element type shared by every entry of a tensor — int64, float32, bool. Mixing dtypes in an expression silently promotes to the common type.",
    },
    {
      term: "reshape",
      aliases: ["reshaping", "reshaped"],
      kc: "numpy.reshape-flatten",
      def: "Reinterpreting the same flat block under a new shape. The element count must match; the data is not touched, so it is usually free.",
    },
    {
      term: "flatten",
      aliases: ["flattening", "flattened", "ravel"],
      kc: "numpy.reshape-flatten",
      def: "Collapsing every axis into one, giving the tensor's elements in memory order.",
    },
    {
      term: "row-major",
      aliases: ["element order", "reading order of the elements", "c order"],
      kc: "numpy.reshape-flatten",
      def: "The order tensors are laid out and re-read in: the LAST axis varies fastest. It is why (h w) merges walk a full row before moving down.",
    },
    {
      term: "elementwise",
      aliases: ["element-wise", "element by element"],
      kc: "numpy.elementwise-ufuncs",
      def: "An operation applied independently to every entry — write the formula once and it lands on the whole tensor, no loop. Note that a * b is elementwise; matrix multiplication is @.",
    },
    {
      term: "aggregation",
      aliases: ["aggregations", "reduction", "reductions", "reduce over"],
      kc: "numpy.aggregations",
      def: "An operation that collapses a tensor to fewer numbers — sum, mean, min, max, std. With no arguments it reduces over everything regardless of shape.",
    },
    {
      term: "generator",
      aliases: ["seeded generator", "deterministic stream"],
      kc: "numpy.random-generator",
      def: "The object that carries reproducible randomness. In PyTorch you make one and pass it as an argument to a sampling function, rather than calling methods on it.",
    },
    {
      term: "matrix multiplication",
      aliases: ["matrix multiply", "matmul", "matrix product"],
      kc: "numpy.linalg-basics",
      def: "Row-times-column with a sum inside, spelled @. For (m, k) and (k, n) the result is (m, n), and entry [i, j] is the dot product of row i with column j.",
    },

    {
      term: "sort",
      aliases: ["sorts", "sorted", "sorting"],
      kc: "numpy.sorting",
      def: "Sorting does not hand back a sorted tensor — it hands back a PAIR, the sorted values and the indices that produced them. Forgetting that is the mistake the lesson exists to prevent.",
    },
    {
      term: "tiling",
      aliases: ["tile", "repetition", "repeat_interleave", "meshgrid"],
      kc: "numpy.tile-repeat-meshgrid",
      def: "Two different repetitions: interleaving copies each element before moving on, while tiling repeats the whole tensor as one block. Same numbers, different order.",
    },
    {
      term: "coordinate grid",
      aliases: ["checkerboard", "index grid", "coordinate mask", "distance-from-center"],
      kc: "numpy.index-grids",
      def: "Building or masking a matrix from each cell's own coordinates. A fixed period is a strided slice; anything else comes from index grids compared against a formula in i and j.",
    },

    /* ---- np-2: indexing and selection ---------------------------- */
    {
      term: "boolean mask",
      aliases: ["mask", "masks", "masked", "boolean array", "boolean tensor"],
      kc: "numpy.boolean-masking",
      def: "The same-shaped tensor of True/False that a comparison produces. The comparison IS the mask — there is no separate 'test each element' step to write.",
    },
    {
      term: "vectorized if/else",
      aliases: ["conditional selection", "choosing per position"],
      kc: "numpy.where-select",
      def: "Building a NEW tensor by picking one of two values at every position, according to a mask. Where the condition holds you take one value, everywhere else the other.",
    },
    {
      term: "nonzero",
      aliases: ["argwhere", "positions of the true entries"],
      kc: "numpy.nonzero-argwhere",
      def: "Where masks answer 'which values', nonzero answers 'at which POSITIONS' — it returns coordinates, in either a tuple-per-dimension or one-row-per-hit layout.",
    },
    {
      term: "argmax",
      aliases: ["argmin", "the arg twins"],
      kc: "numpy.argmin-argmax",
      def: "min and max give the extreme VALUE; argmin and argmax give WHERE it lives. Ties break to the first occurrence, deterministically.",
    },
    {
      term: "fancy indexing",
      aliases: ["index array", "index arrays", "lookup table", "lookup-table read"],
      kc: "numpy.fancy-indexing",
      def: "Indexing with a tensor of integers instead of a slice: you get exactly those positions, in the order you listed them, repetitions allowed. Reordering is indexing.",
    },
    {
      term: "distinct values",
      aliases: ["deduplication", "deduplicate"],
      kc: "numpy.unique",
      def: "unique returns a tensor's distinct values sorted ascending, and optionally the counts, inverse map, or first indices describing the duplicates it collapsed.",
    },
    {
      term: "diagonal",
      aliases: ["main diagonal", "superdiagonal", "subdiagonal", "trace", "offset k"],
      kc: "numpy.diag-triangles",
      def: "The entries where the two indices meet, selected by the offset k — 0 is the main diagonal, positive k sits above it, negative k below. The trace is their sum.",
    },
    {
      term: "upper triangle",
      aliases: ["lower triangle", "triangular part", "triangular mask"],
      kc: "numpy.diag-triangles",
      def: "The half of a matrix above (or below) a chosen diagonal — the shape behind causal masks and symmetric-matrix bookkeeping.",
    },
    {
      term: "argsort",
      aliases: ["rank", "ranks", "sort-by", "order statistics"],
      kc: "numpy.argsort-ranking",
      def: "The indices that WOULD sort a tensor. That indirection is the tool: argsort a key column, then fancy-index whole rows by it to sort one thing by another.",
    },
    {
      term: "NaN",
      aliases: ["nans", "not a number", "inf", "infinity"],
      kc: "numpy.nan-handling",
      def: "The float value marking a missing or undefined result. It compares unequal to everything including itself, so == never finds it — the isnan predicate does.",
    },
    {
      term: "padding",
      aliases: ["pad", "padded", "border", "borders"],
      kc: "numpy.pad-borders",
      def: "Growing a tensor by wrapping a frame around it. The pad tuple runs LAST DIMENSION FIRST in (before, after) pairs, which is the one real trap on the page.",
    },

    /* ---- np-3: vectorization and broadcasting -------------------- */
    {
      term: "broadcasting",
      aliases: ["broadcast", "broadcasts", "broadcasting rules", "stretched to fit"],
      kc: "numpy.broadcasting-rules",
      def: "The rule that lets mismatched shapes work together: align them from the RIGHT, and each axis pair is compatible if the sizes are equal or one of them is 1, which then stretches.",
    },
    {
      term: "axis",
      aliases: ["axes", "dimension", "dimensions"],
      kc: "numpy.axis-reductions",
      def: "One numbered direction through a tensor — axis 0 runs down the rows, axis 1 across the columns. The axis you name in a reduction is the axis that DISAPPEARS.",
    },
    {
      term: "keepdims",
      aliases: ["keepdim"],
      kc: "numpy.axis-reductions",
      def: "Keeps the reduced axis as length 1 instead of dropping it, so the statistic still lines up against the original tensor under broadcasting.",
    },
    {
      term: "centering",
      aliases: ["center", "centered", "standardize", "standardizing", "standardization"],
      kc: "numpy.centering",
      def: "Compute a statistic, then subtract it back into the data. The reduction produces the statistic and broadcasting spreads it back — the template behind most preprocessing.",
    },
    {
      term: "rescaling",
      aliases: ["rescale", "normalize", "normalizing", "min-max", "unit norm"],
      kc: "numpy.rescaling",
      def: "Mapping data into a target range or size — [0, 1] by min-max, unit Euclidean length, or rows that sum to 1. Each is defined by what the result must satisfy.",
    },
    {
      term: "cumulative sum",
      aliases: ["cumsum", "running total", "discrete difference", "diff"],
      kc: "numpy.cumulative-diff",
      def: "The vectorized replacement for the 'total so far' loop: entry i is the sum of everything up to i. Its inverse, the discrete difference, gives step-to-step changes.",
    },
    {
      term: "dot product",
      aliases: ["dot products", "inner product"],
      kc: "numpy.dot-matmul-patterns",
      def: "Multiply corresponding entries of two equal-length vectors, then add them up. Every product in linear algebra is built from this one atom: elementwise multiply plus a reduction.",
    },
    {
      term: "concatenate",
      aliases: ["concatenation", "stacking", "stack", "interleave", "interleaving"],
      kc: "numpy.stack-concat-interleave",
      def: "Combining tensors splits on one question: does the result get a NEW axis (stack), or grow an EXISTING one (cat)?",
    },
    {
      term: "one-hot encoding",
      aliases: ["one-hot", "class label", "class labels", "bincount"],
      kc: "numpy.onehot-bincount",
      def: "Turning integer labels into rows of the identity matrix — a one-hot row for class c is row c of eye(k), so encoding a whole label vector is one lookup-table read.",
    },
    {
      term: "top-k",
      aliases: ["topk", "k largest"],
      kc: "numpy.topk-selection",
      def: "The k largest entries and their positions in one call, largest first — without paying to sort everything.",
    },
    {
      term: "in-place",
      aliases: ["in place", "trailing underscore"],
      kc: "numpy.inplace-out",
      def: "An operation that writes into an existing buffer instead of allocating a new tensor. PyTorch marks them with a trailing underscore: add_, mul_, clamp_.",
    },
    {
      term: "sliding window",
      aliases: ["sliding windows", "moving average", "unfold"],
      kc: "numpy.sliding-windows",
      def: "Every contiguous window of length w, materialized as the rows of a matrix without copying — the substrate of moving averages, local maxima, and convolutions.",
    },

    /* ---- np-4: applied patterns ---------------------------------- */
    {
      term: "pairwise distance",
      aliases: ["distance matrix", "similarity matrix", "pairwise distances"],
      kc: "numpy.pairwise-metrics",
      def: "The (n, m) matrix comparing every row of X with every row of Y. Built in three broadcasting steps: insert axes so the rows meet, combine, then reduce the feature axis away.",
    },
    {
      term: "scatter",
      aliases: ["scatter-add", "gather", "grouped aggregation"],
      kc: "numpy.scatter-gather",
      def: "Pushing values INTO positions and accumulating when positions repeat — the reverse of fancy indexing. out[idx] += x does NOT accumulate repeats; the scatter tools do.",
    },
    {
      term: "storage",
      aliases: ["memory buffer", "stride", "strides", "reinterpret the bytes"],
      kc: "numpy.memory-model",
      def: "A tensor is one memory buffer plus an interpretation — dtype, shape, strides. Two tensors can read the same buffer differently, which is what makes a view cheap.",
    },
    {
      term: "run-length encoding",
      aliases: ["rle", "cartesian product", "set operations"],
      kc: "numpy.set-combinatorics",
      def: "Compressing a sequence into its values plus their run counts, derived vectorized: runs start wherever an element differs from its predecessor.",
    },
    {
      term: "stencil",
      aliases: ["stencils", "block sum", "block sums", "2-d window"],
      kc: "numpy.window-stencil",
      def: "The 2-D generalization of the sliding window — non-overlapping blocks, or every overlapping patch of a grid, used for pooling, correlation, and cellular automata.",
    },
    {
      term: "bootstrap",
      aliases: ["resample", "resampling", "inverse-cdf", "confidence interval"],
      kc: "numpy.random-sampling",
      def: "Drawing random INDICES and fancy-indexing with them. A whole matrix of indices gives every resample as a row, so per-resample statistics are one reduction.",
    },
    {
      term: "homogeneous coordinates",
      aliases: ["polar coordinates", "transform matrix", "arctan2"],
      kc: "numpy.geometry-transforms",
      def: "Geometry on point sets is column bookkeeping plus linear algebra, under one convention: points live in ROWS, so column 0 is every x and column 1 every y.",
    },
    {
      term: "right-hand side",
      aliases: ["multi-rhs", "batched solve", "block matrix"],
      kc: "numpy.linalg-applied",
      def: "Solving many systems at once: each COLUMN of B is an independent right-hand side, all solved from one factorization — which is why you should never loop over them.",
    },

    /* ---- es-1: einsum notation ----------------------------------- */
    {
      term: "einsum",
      aliases: ["spec string", "spec", "subscripts"],
      kc: "einsum.notation-model",
      def: "Name the axes of your tensors, then describe the result by those names. The whole operation lives in one spec string with inputs before the arrow and the output after it.",
    },
    {
      term: "contracted axis",
      aliases: ["contraction", "contract", "contracted", "summed away", "summed over", "index removal"],
      kc: "einsum.reductions",
      def: "A letter missing from an einsum's output side is summed over — it disappears, and every surviving position receives the total across it.",
    },
    {
      term: "shared letter",
      aliases: ["shared index", "shared axis"],
      kc: "einsum.dot-frobenius",
      def: "A letter appearing in two inputs pairs those axes elementwise. Shared and then dropped means multiply corresponding entries and add them up — which is exactly the dot product.",
    },
    {
      term: "Frobenius inner product",
      aliases: ["frobenius"],
      kc: "einsum.dot-frobenius",
      def: "The dot product's matrix twin: multiply two matrices entrywise and sum the whole thing to a scalar.",
    },
    {
      term: "outer product",
      aliases: ["outer products", "free index", "free indices"],
      kc: "einsum.outer-products",
      def: "Give each input its OWN letter and keep both: nothing is shared and nothing is dropped, so every element of one meets every element of the other. Unshared kept letters multiply combinatorially.",
    },
    {
      term: "matrix-vector product",
      aliases: ["matrix-vector", "matvec"],
      kc: "einsum.matvec-matmul",
      def: "The dot product done once per row: line the matrix's column axis up with the vector on a shared letter, and sum that letter away.",
    },
    {
      term: "repeated index",
      aliases: ["repeated letter", "repeated indices"],
      kc: "einsum.diag-trace",
      def: "A letter repeated WITHIN one operand makes those two axes move together, so einsum visits only the entries where their indices are equal — the diagonal.",
    },

    /* ---- es-2: batch dimensions and applied einsum --------------- */
    {
      term: "batch axis",
      aliases: ["batch dimension", "batch dimensions", "batch dims", "batched"],
      kc: "einsum.batch-dims",
      def: "A letter appearing in both inputs AND the output, never contracted. einsum cannot merge anything across it, so it simply iterates — that is what batching is.",
    },
    {
      term: "per-axis scaling",
      aliases: ["weighted sum", "weighted sums", "per-channel scale"],
      kc: "einsum.broadcast-scaling",
      def: "One small vector of weights applied along one axis of a big tensor. The vector declares WHICH axis it rides on by using that axis's letter; the output side picks collapse or scale.",
    },
    {
      term: "attention",
      aliases: ["queries", "keys and values", "attention scores"],
      kc: "einsum.attention-patterns",
      def: "Two einsums with a softmax between them: every query against every key, then the resulting weights against the values. Both are contractions you already know, at scale.",
    },
    {
      term: "quadratic form",
      aliases: ["gram matrix", "covariance matrix"],
      kc: "einsum.matrix-forms",
      def: "Classical matrix expressions written as short specs — the spec is literally the double sum with the sigmas removed.",
    },

    /* ---- eo-1: rearrange ----------------------------------------- */
    {
      term: "einops",
      aliases: ["rearrange", "einops pattern", "pattern language"],
      kc: "einops.pattern-language",
      def: "Reshape and transpose with the axes spelled out in words: name each input axis on the left, list the same names in the output's order on the right.",
    },
    {
      term: "merging axes",
      aliases: ["merge axes", "merged axes", "fuse axes"],
      kc: "einops.merge-axes",
      def: "Parentheses on the OUTPUT side fuse axes into one. Order inside the parens is nesting order — the leftmost name varies slowest, exactly like a row-major reshape.",
    },
    {
      term: "splitting axes",
      aliases: ["split axes", "named factor", "named factors", "factored axes"],
      kc: "einops.split-axes",
      def: "Parentheses on the INPUT side unpack one axis into factors. einops cannot guess the split, so the sizes it can't infer arrive as keyword arguments.",
    },
    {
      term: "singleton axis",
      aliases: ["singleton axes", "length-1 axis", "squeeze"],
      kc: "einops.singleton-and-lists",
      def: "A literal 1 in a pattern inserts a length-1 axis on the right, or consumes one on the left — einops' spelling of adding or squeezing a dimension, with verification.",
    },
    {
      term: "montage",
      aliases: ["image grid", "grid layout"],
      kc: "einops.grids-montage",
      def: "The flagship split-then-merge pattern: split the batch into grid coordinates, then merge each grid coordinate with its image axis.",
    },
    {
      term: "patch",
      aliases: ["patches", "patch extraction", "space-to-depth", "depth-to-space", "pixel-shuffle"],
      kc: "einops.patches-space-depth",
      def: "Trading SPACE for DEPTH: split each spatial axis into blocks and within-block position, then move the block coordinates wherever you need them — tokens, channels, or back again.",
    },

    /* ---- eo-2: reduce -------------------------------------------- */
    {
      term: "einops.reduce",
      aliases: ["reduce pattern"],
      kc: "einops.reduce-model",
      def: "Where rearrange must keep every axis, reduce may drop them — and the third argument names HOW the dropped values collapse: mean, max, min, sum, prod.",
    },
    {
      term: "pooling",
      aliases: ["pool", "pooled", "downsampling"],
      kc: "einops.pooling",
      def: "Window-wise downsampling as pure notation: factor each spatial axis into blocks and within-block position, keep the block coordinates, drop the rest with an aggregation.",
    },

    /* ---- eo-3: repeat and deep-learning patterns ----------------- */
    {
      term: "einops.repeat",
      aliases: ["repeat pattern", "stretched axis"],
      kc: "einops.repeat-model",
      def: "The mirror image of reduce: the output may contain names the input does NOT have, and the data is copied to fill them.",
    },
    {
      term: "attention heads",
      aliases: ["attention head", "head dimension", "classifier flatten"],
      kc: "einops.dl-flatten-heads",
      def: "Splitting one feature axis into (heads x per-head features) and moving heads next to batch — the shape manoeuvre in virtually every model's forward pass.",
    },
    {
      term: "channel groups",
      aliases: ["grouped channels", "temporal window", "temporal windows"],
      kc: "einops.channel-groups-temporal",
      def: "A channel axis of size g x c that really means g groups of c channels. The single load-bearing question is which index is SLOW — groups or channels.",
    },
  ],
};

/* Where each concept is taught, for the popup's footer line: kc -> [lesson
   title, concept title]. A COPY of what lessons/kc_registry.json and
   lessons_structured.json already say — held here so a hover costs no network
   request. 🔴 The runtime-fetch trap is exactly why: a file this app fetches at
   runtime can be dropped by .vercelignore and the SPA rewrite answers the 404
   with 200 text/html, so the loss is silent in production. A <script> tag
   cannot fail that way. watch_jargon.py re-derives this map from the registry
   on every run, so the copy cannot drift without failing the guard. */
window.DD_GLOSSARY.kcLesson = {
    "numpy.ndarray-model": ["Arrays from the ground up", "What a tensor is — data + shape + dtype"],
    "numpy.constructors": ["Arrays from the ground up", "Tensor constructors — zeros, ones, full, eye, *_like"],
    "numpy.slicing-views": ["Arrays from the ground up", "Slicing, views, and slice assignment"],
    "numpy.ranges": ["Arrays from the ground up", "Numeric ranges — arange and linspace"],
    "numpy.dtype-astype": ["Arrays from the ground up", "Dtypes, .to(), and memory size"],
    "numpy.reshape-flatten": ["Arrays from the ground up", "Reshape, flatten, and element order"],
    "numpy.elementwise-ufuncs": ["Arrays from the ground up", "Elementwise math"],
    "numpy.aggregations": ["Arrays from the ground up", "Whole-tensor aggregations and Python scalars"],
    "numpy.sorting": ["Arrays from the ground up", "Sorting tensors"],
    "numpy.tile-repeat-meshgrid": ["Arrays from the ground up", "Tiling and repetition — repeat, repeat_interleave, meshgrid"],
    "numpy.random-generator": ["Arrays from the ground up", "Random numbers with a Generator"],
    "numpy.linalg-basics": ["Arrays from the ground up", "Matrix multiply and t.linalg basics"],
    "numpy.boolean-masking": ["Indexing and selection", "Boolean masks — compare, count, filter"],
    "numpy.where-select": ["Indexing and selection", "Conditional values — t.where and where= arguments"],
    "numpy.nonzero-argwhere": ["Indexing and selection", "Finding positions — nonzero and argwhere"],
    "numpy.argmin-argmax": ["Indexing and selection", "Locating extremes — argmin and argmax"],
    "numpy.fancy-indexing": ["Indexing and selection", "Fancy indexing — index arrays and lookup tables"],
    "numpy.unique": ["Indexing and selection", "Distinct values — t.unique and friends"],
    "numpy.diag-triangles": ["Indexing and selection", "Diagonals, triangles, and trace"],
    "numpy.argsort-ranking": ["Indexing and selection", "Order statistics — argsort, ranks, sort-by"],
    "numpy.nan-handling": ["Indexing and selection", "NaN and Inf — detecting and repairing"],
    "numpy.pad-borders": ["Indexing and selection", "Borders and padding"],
    "numpy.index-grids": ["Indexing and selection", "Index-pattern grids — checkerboards and coordinate masks"],
    "numpy.broadcasting-rules": ["Vectorization and broadcasting", "Broadcasting rules"],
    "numpy.axis-reductions": ["Vectorization and broadcasting", "Reductions along an axis — and keepdims"],
    "numpy.centering": ["Vectorization and broadcasting", "Centering and standardizing rows/columns"],
    "numpy.rescaling": ["Vectorization and broadcasting", "Rescaling — min-max, unit norm, probability rows"],
    "numpy.cumulative-diff": ["Vectorization and broadcasting", "Cumulative ops and discrete differences"],
    "numpy.dot-matmul-patterns": ["Vectorization and broadcasting", "Dot products and matrix-multiply patterns"],
    "numpy.stack-concat-interleave": ["Vectorization and broadcasting", "Stacking, concatenating, interleaving"],
    "numpy.onehot-bincount": ["Vectorization and broadcasting", "Labels — one-hot encoding and bincount"],
    "numpy.topk-selection": ["Vectorization and broadcasting", "Top-k selection — topk vs sort"],
    "numpy.inplace-out": ["Vectorization and broadcasting", "In-place operations and the trailing underscore"],
    "numpy.sliding-windows": ["Vectorization and broadcasting", "Sliding windows and moving averages"],
    "numpy.pairwise-metrics": ["Applied patterns", "Pairwise distances and similarities"],
    "numpy.scatter-gather": ["Applied patterns", "Scatter and grouped aggregation — bincount weights and add.at"],
    "numpy.memory-model": ["Applied patterns", "Memory model — views, reinterpreting bytes, bit unpacking"],
    "numpy.set-combinatorics": ["Applied patterns", "Set operations, cartesian products, run-length encoding"],
    "numpy.window-stencil": ["Applied patterns", "2-D windows and stencils — block sums, correlation, Life"],
    "numpy.random-sampling": ["Applied patterns", "Sampling — bootstrap, choice, inverse-CDF"],
    "numpy.geometry-transforms": ["Applied patterns", "Geometry — coordinates and transforms"],
    "numpy.linalg-applied": ["Applied patterns", "Applied linear algebra — multi-RHS, batched solve, block matrices"],
    "einsum.notation-model": ["Einsum notation", "Reading an einsum spec string"],
    "einsum.reductions": ["Einsum notation", "Sums as index removal"],
    "einsum.dot-frobenius": ["Einsum notation", "Dot and Frobenius inner products"],
    "einsum.outer-products": ["Einsum notation", "Outer products — new axes from free indices"],
    "einsum.matvec-matmul": ["Einsum notation", "Matrix-vector and matrix-matrix products"],
    "einsum.diag-trace": ["Einsum notation", "Repeated indices on one operand — diagonal and trace"],
    "einsum.batch-dims": ["Batch dimensions and applied einsum", "Batch dimensions — carrying axes through"],
    "einsum.broadcast-scaling": ["Batch dimensions and applied einsum", "Weighted sums and per-axis scaling"],
    "einsum.attention-patterns": ["Batch dimensions and applied einsum", "Attention-shaped contractions"],
    "einsum.matrix-forms": ["Batch dimensions and applied einsum", "Matrix forms — quadratic, Gram, covariance"],
    "einops.pattern-language": ["Rearrange", "The einops pattern language — naming and permuting axes"],
    "einops.merge-axes": ["Rearrange", "Merging axes with (parentheses)"],
    "einops.split-axes": ["Rearrange", "Splitting axes with named factors"],
    "einops.singleton-and-lists": ["Rearrange", "Singleton axes and lists as an axis"],
    "einops.grids-montage": ["Rearrange", "Laying batches out as grids"],
    "einops.patches-space-depth": ["Rearrange", "Patches, space-to-depth, depth-to-space"],
    "einops.reduce-model": ["Reduce", "einops.reduce — dropping axes with an aggregation"],
    "einops.pooling": ["Reduce", "Pooling with factored axes"],
    "einops.repeat-model": ["Repeat and deep-learning patterns", "einops.repeat — new axes and stretched axes"],
    "einops.dl-flatten-heads": ["Repeat and deep-learning patterns", "Deep-learning shapes — flattening and attention heads"],
    "einops.channel-groups-temporal": ["Repeat and deep-learning patterns", "Channel groups and temporal windows"],
};
