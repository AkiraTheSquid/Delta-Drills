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
      term: "transpose",
      aliases: ["transposes", "transposed", "transposing", "transposition"],
      kc: "numpy.transpose-axes",
      def: "Turning a grid on its side: row i of the result is column i of the original, and the shape reverses. Spelled a.T, with no parentheses \u2014 it is an attribute, not a method.",
    },
    {
      term: "block",
      aliases: ["memory block", "block of memory", "buffer", "the same block"],
      kc: "numpy.views-and-copies",
      def: "The one long strip of numbers a tensor actually occupies in memory. Two tensors share a block when both describe the same strip; shape and dtype are just the note saying how to read it. data_ptr() compares where each STARTS, so a slice can share the block and still answer a different address.",
    },
    {
      term: "contiguous",
      aliases: ["contiguity", "non-contiguous", "reading order", "in reading order"],
      kc: "numpy.views-and-copies",
      def: "A tensor is contiguous when its numbers sit on the strip in the same order you read them off the grid \u2014 no hopping. A freshly built tensor always is; a transpose usually is not.",
    },
    {
      term: "packed copy",
      aliases: ["packed", "pack", "packing"],
      kc: "numpy.views-and-copies",
      def: "What .contiguous() gives you: the same numbers written out in reading order. That normally means a new strip of its own \u2014 unless the tensor was already in order, in which case it hands the same one back.",
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
      kc: "numpy.views-and-copies",
      def: "A tensor that re-describes another tensor's block instead of owning one. Writing into a view writes into the original — the single most common source of surprise in tensor code.",
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

    /* ---- np-2: indexing and selection ---------------------------- */
    {
      term: "boolean mask",
      aliases: ["mask", "masks", "masked", "boolean array", "boolean tensor"],
      kc: "numpy.boolean-masking",
      def: "The same-shaped tensor of True/False that a comparison produces. The comparison IS the mask — there is no separate 'test each element' step to write.",
    },
    {
      term: "argmax",
      aliases: ["argmin", "the arg twins"],
      kc: "numpy.argmin-argmax",
      def: "min and max give the extreme VALUE; argmin and argmax give WHERE it lives. Ties break to the first occurrence, deterministically.",
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

    /* ---- np-4: applied patterns ---------------------------------- */

    /* ---- es-1: einsum notation ----------------------------------- */

    /* ---- es-2: batch dimensions and applied einsum --------------- */

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
    "python.values-and-names": ["Python you need first", "Values and names — what = really does"],
    "python.types-and-conversion": ["Python you need first", "The everyday types, and converting between them"],
    "python.lists-and-tuples": ["Python you need first", "Lists and tuples — holding more than one value"],
    "python.indexing": ["Python you need first", "Indexing — pulling one item out, counting from zero"],
    "python.calling-functions": ["Python you need first", "Calling a function — arguments in, one value out"],
    "python.defining-functions": ["Python you need first", "Writing your own function — def and return"],
    "python.dots-and-imports": ["Python you need first", "Dots — importing a library, attributes, and methods"],
    "numpy.ndarray-model": ["Arrays from the ground up", "What a tensor is — data + shape + dtype"],
    "numpy.transpose-axes": ["Arrays from the ground up", "Transpose — swapping which axis is which"],
    "numpy.views-and-copies": ["Arrays from the ground up", "Views and copies — the same numbers, read a different way"],
    "numpy.constructors": ["Arrays from the ground up", "Tensor constructors — zeros, ones, full, eye, *_like"],
    "numpy.slicing-views": ["Arrays from the ground up", "Slicing, views, and slice assignment"],
    "numpy.ranges": ["Arrays from the ground up", "Numeric ranges — arange and linspace"],
    "numpy.dtype-astype": ["Arrays from the ground up", "Dtypes, .to(), and memory size"],
    "numpy.reshape-flatten": ["Arrays from the ground up", "Reshape, flatten, and element order"],
    "numpy.elementwise-ufuncs": ["Arrays from the ground up", "Elementwise math"],
    "numpy.aggregations": ["Arrays from the ground up", "Whole-tensor aggregations and Python scalars"],
    "numpy.sorting": ["Arrays from the ground up", "Sorting tensors"],
    "numpy.random-generator": ["Arrays from the ground up", "Random numbers with a Generator"],
    "numpy.linalg-basics": ["Arrays from the ground up", "Matrix multiply and t.linalg basics"],
    "numpy.boolean-masking": ["Indexing and selection", "Boolean masks — compare, count, filter"],
    "numpy.argmin-argmax": ["Indexing and selection", "Locating extremes — argmin and argmax"],
    "numpy.broadcasting-rules": ["Vectorization and broadcasting", "Broadcasting rules"],
    "numpy.axis-reductions": ["Vectorization and broadcasting", "Reductions along an axis — and keepdims"],
    "numpy.dot-matmul-patterns": ["Vectorization and broadcasting", "Dot products and matrix-multiply patterns"],
    "numpy.stack-concat-interleave": ["Vectorization and broadcasting", "Stacking, concatenating, interleaving"],
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
