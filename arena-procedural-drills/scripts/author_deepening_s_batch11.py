#!/usr/bin/env python3
"""Author 8 deepening ex2 drills for prereqs_distributed + prereqs_generative.

Each ex2 probes a DISTINCT facet from the existing ex1 — different cognitive
operation, different surface context. ONE LO + ONE Bloom + <=2 KCs per drill.

Distributed atoms use a thread-based _FakeWorld mock (no torch.distributed
process group, no real cuda) so the test runs on a single CPU Python.

Verification re-runs each spec's solution against its test_body inside the
build venv (torch 2.12.0+cpu) before any notebook is emitted.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC_DIST = "prereqs_distributed"
TOPIC_GEN = "prereqs_generative"


# ---------------------------------------------------------------------------
# Fake-distributed harness — thread mock with all_reduce / reduce / broadcast
# / gather / all_gather / barrier. Prepended to every distributed test_body.
# ---------------------------------------------------------------------------

_FAKE_DIST_HARNESS = r'''
import threading
import types as _types
import torch as _t_for_fake


class _FakeReduceOp:
    SUM = 'SUM'
    MAX = 'MAX'
    MIN = 'MIN'
    PRODUCT = 'PROD'


class _FakeWorld:
    """Shared state across `world_size` rank-threads."""
    def __init__(self, world_size):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size)
        self.lock = threading.Lock()
        self.scratch = {}
        self.tls = threading.local()
        self.results = [None] * world_size
        # rank-0-only side-effect log (per-call log lines for tests to inspect)
        self.side_effects = []

    def _reduce_op(self, bag, op):
        if op == 'SUM':
            out = bag[0].clone()
            for x in bag[1:]:
                out = out + x
            return out
        if op == 'MAX':
            out = bag[0].clone()
            for x in bag[1:]:
                out = _t_for_fake.maximum(out, x)
            return out
        if op == 'MIN':
            out = bag[0].clone()
            for x in bag[1:]:
                out = _t_for_fake.minimum(out, x)
            return out
        if op == 'PROD':
            out = bag[0].clone()
            for x in bag[1:]:
                out = out * x
            return out
        raise ValueError(f'unknown fake op {op!r}')

    def all_reduce(self, tensor, op='SUM'):
        rank = self.tls.rank
        self.barrier.wait()
        with self.lock:
            self.scratch.setdefault('ar', [None] * self.world_size)
            self.scratch['ar'][rank] = tensor.detach().clone()
        self.barrier.wait()
        bag = self.scratch['ar']
        reduced = self._reduce_op(bag, op)
        tensor.copy_(reduced)
        self.barrier.wait()
        if rank == 0:
            self.scratch.pop('ar', None)
        self.barrier.wait()

    def reduce(self, tensor, dst, op='SUM'):
        rank = self.tls.rank
        self.barrier.wait()
        with self.lock:
            self.scratch.setdefault('rd', [None] * self.world_size)
            self.scratch['rd'][rank] = tensor.detach().clone()
        self.barrier.wait()
        if rank == dst:
            bag = self.scratch['rd']
            tensor.copy_(self._reduce_op(bag, op))
        self.barrier.wait()
        if rank == 0:
            self.scratch.pop('rd', None)
        self.barrier.wait()

    def broadcast(self, tensor, src):
        rank = self.tls.rank
        self.barrier.wait()
        if rank == src:
            with self.lock:
                self.scratch['bc'] = tensor.detach().clone()
        self.barrier.wait()
        if rank != src:
            tensor.copy_(self.scratch['bc'])
        self.barrier.wait()
        if rank == 0:
            self.scratch.pop('bc', None)
        self.barrier.wait()

    def gather(self, tensor, gather_list, dst):
        """Mock dist.gather — only dst's gather_list is populated."""
        rank = self.tls.rank
        self.barrier.wait()
        with self.lock:
            self.scratch.setdefault('gth', [None] * self.world_size)
            self.scratch['gth'][rank] = tensor.detach().clone()
        self.barrier.wait()
        if rank == dst:
            bag = self.scratch['gth']
            for i, src_tensor in enumerate(bag):
                gather_list[i].copy_(src_tensor)
        self.barrier.wait()
        if rank == 0:
            self.scratch.pop('gth', None)
        self.barrier.wait()

    def all_gather(self, gather_list, tensor):
        """Mock dist.all_gather — every rank's gather_list is populated."""
        rank = self.tls.rank
        self.barrier.wait()
        with self.lock:
            self.scratch.setdefault('agth', [None] * self.world_size)
            self.scratch['agth'][rank] = tensor.detach().clone()
        self.barrier.wait()
        bag = self.scratch['agth']
        for i, src_tensor in enumerate(bag):
            gather_list[i].copy_(src_tensor)
        self.barrier.wait()
        if rank == 0:
            self.scratch.pop('agth', None)
        self.barrier.wait()

    def barrier_op(self):
        self.barrier.wait()


def _run_fake_world(worker_fn, world_size, *extra_args, timeout=30):
    world = _FakeWorld(world_size)
    errors = [None] * world_size

    def _runner(rank):
        world.tls.rank = rank
        fake_dist = _types.SimpleNamespace()
        fake_dist.ReduceOp = _FakeReduceOp
        fake_dist.all_reduce = lambda tensor, op='SUM': world.all_reduce(tensor, op)
        fake_dist.reduce = lambda tensor, dst, op='SUM': world.reduce(tensor, dst, op)
        fake_dist.broadcast = lambda tensor, src: world.broadcast(tensor, src)
        fake_dist.gather = lambda tensor, gather_list, dst: world.gather(tensor, gather_list, dst)
        fake_dist.all_gather = lambda gather_list, tensor: world.all_gather(gather_list, tensor)
        fake_dist.barrier = world.barrier_op
        fake_dist.get_rank = lambda: rank
        fake_dist.get_world_size = lambda: world_size
        fake_dist.init_process_group = lambda **kw: None
        fake_dist.destroy_process_group = lambda: None
        try:
            worker_fn(rank, world_size, fake_dist, world, *extra_args)
        except BaseException as e:
            import traceback as _tb
            errors[rank] = (e, _tb.format_exc())

    threads = [threading.Thread(target=_runner, args=(r,), daemon=True) for r in range(world_size)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=timeout)
    for r, err in enumerate(errors):
        if err is not None:
            raise RuntimeError(f'rank {r} failed: {err[0]!r}\n{err[1]}')
    return world.results
'''


# ---------------------------------------------------------------------------
# Recap blocks
# ---------------------------------------------------------------------------

RECAP_RANK0_BARRIER = (
    "## Rank-0 side effects with `dist.barrier()` ordering — quick refresher\n"
    "\n"
    "Pure `if rank == 0:` is enough for *fire-and-forget* writes (logs, "
    "checkpoints other ranks never read). When rank 0 produces something "
    "OTHER ranks must read (downloaded dataset, generated split file, "
    "tokenizer cache), you need a barrier so the readers don't race the "
    "writer:\n"
    "\n"
    "```python\n"
    "if rank == 0:\n"
    "    download_dataset(url, target)     # rank-0-only side effect\n"
    "dist.barrier()                        # everyone waits here\n"
    "data = read_dataset(target)           # every rank can now read safely\n"
    "```\n"
    "\n"
    "**Why barrier on every rank.** `dist.barrier()` is a collective — every "
    "rank must call it, or the call deadlocks. Rank > 0 hits the barrier "
    "immediately and blocks; rank 0 does the download first, THEN hits the "
    "barrier, which unblocks everyone.\n"
    "\n"
    "**Order matters at one place only.** The barrier goes BETWEEN the "
    "rank-0 side effect and the all-rank read. Pre-barrier, ranks > 0 wait. "
    "Post-barrier, the produced file is guaranteed to exist for every rank."
)

RECAP_ALL_GATHER_MEDIAN = (
    "## `all_gather` + manual aggregation — quick refresher\n"
    "\n"
    "`dist.all_gather(gather_list, tensor)` collects each rank's `tensor` "
    "into a length-`world_size` list of tensors on EVERY rank. Unlike "
    "`gather` (which populates only the dst rank), `all_gather` fans the "
    "result out — every rank ends with the same `gather_list`.\n"
    "\n"
    "**Why all_gather instead of all_reduce.** `all_reduce` collapses to a "
    "single aggregate (sum, max, min, product). For aggregations that AREN'T "
    "associative-binary — median, percentile, sorted top-k — there's no "
    "`ReduceOp` you can pass. You need the per-rank values, on every rank, "
    "then compute the aggregation locally.\n"
    "\n"
    "**Memory trade-off.** `all_gather` holds N tensors on every rank "
    "(N×memory). `all_reduce` holds 1 tensor on every rank. For scalar "
    "metrics across modest world sizes, the cost is negligible.\n"
    "\n"
    "**Same gather_list pre-allocation pattern as `gather`.** Callers "
    "pre-build the `[t.zeros_like(...) for _ in range(world_size)]` list "
    "and pass it as the destination. The collective fills the slots."
)

RECAP_HARMONIC = (
    "## Harmonic mean via sum-then-divide-then-invert — quick refresher\n"
    "\n"
    "`dist.ReduceOp` still has no `MEAN`, no `HARMONIC_MEAN`, no "
    "`GEOMETRIC_MEAN`. Every named mean is built from the same recipe: "
    "transform → `all_reduce(SUM)` → divide → inverse transform.\n"
    "\n"
    "For the **harmonic mean** of N rank-local values `x_r > 0`:\n"
    "```\n"
    "H = N / (sum_r 1/x_r)\n"
    "```\n"
    "Distributed implementation:\n"
    "```python\n"
    "tensor = t.tensor([1.0 / local_value], dtype=t.float32)    # transform\n"
    "dist.all_reduce(tensor, op=dist.ReduceOp.SUM)              # sum 1/x_r\n"
    "tensor /= world_size                                       # divide → mean of 1/x\n"
    "harmonic = 1.0 / tensor.item()                             # inverse transform\n"
    "```\n"
    "\n"
    "**The in-place divide still matters.** Same `/=` vs `=` distinction as "
    "the arithmetic-mean case — keeps caller-held references stable.\n"
    "\n"
    "**Why harmonic for rates / speeds.** Averaging samples/sec across "
    "ranks: arithmetic mean over-weights fast ranks (they finished more "
    "iterations). Harmonic mean weights by *time spent*, which is the "
    "throughput-correct aggregate."
)

RECAP_TIED_DECODE = (
    "## Tied-weight decoder — quick refresher\n"
    "\n"
    "A symmetric autoencoder bottleneck can share its weight matrix with "
    "the matching decoder layer:\n"
    "```python\n"
    "# encode: (B, input_dim) -> (B, latent_dim)\n"
    "z = flat @ W.T + b_enc                # W shape (latent_dim, input_dim)\n"
    "# decode: (B, latent_dim) -> (B, input_dim)\n"
    "x_hat = z @ W + b_dec                 # SAME W, no .T this time\n"
    "```\n"
    "\n"
    "**Why tied.** Halves the parameter count and biases the decoder toward "
    "the pseudoinverse of the encoder — useful regularizer for "
    "under-trained autoencoders. Hinton's original AE work used tied "
    "weights; modern VAEs usually untie.\n"
    "\n"
    "**Round-trip identity for orthonormal W.** If `W @ W.T == I` (rows of "
    "W are orthonormal), then `(flat @ W.T) @ W = flat @ (W.T @ W)` — and "
    "when `W` is square+orthogonal, `W.T @ W == I`, so the round-trip "
    "reproduces the input exactly (ignoring biases). This is the "
    "PCA-with-tied-weights connection: PCA's encoder/decoder pair is "
    "exactly this with W as the eigenbasis matrix."
)

RECAP_EMBEDDING_SPARSE = (
    "## `nn.Embedding` and sparse gradients — quick refresher\n"
    "\n"
    "```python\n"
    "embed = nn.Embedding(num_classes, D)\n"
    "out = embed(labels)        # labels: (B,) -> out: (B, D)\n"
    "```\n"
    "\n"
    "`nn.Embedding` is a thin wrapper: `weight` is a `(num_classes, D)` "
    "`Parameter`, forward is `F.embedding(labels, self.weight)` (the same "
    "integer-indexing as ex1 — `weight[labels]`). The Module form gets you "
    "`.weight` as a registered parameter, so the optimizer sees it "
    "automatically.\n"
    "\n"
    "**Sparse gradient property.** When you backprop through `embed(labels)`, "
    "the gradient is non-zero ONLY for the rows of `weight` that were "
    "actually indexed. Classes absent from the batch get exact zero — not "
    "a tiny floating-point value, but a structural zero from the autograd "
    "graph itself.\n"
    "\n"
    "**Why this matters.** With imbalanced label distributions, rare "
    "classes accumulate updates SLOWLY because they're indexed rarely. "
    "`optim.SparseAdam` is built for exactly this case (only updates the "
    "rows that received non-zero grad), saving compute when num_classes is "
    "huge."
)

RECAP_WRAPPER_TRAIN_EVAL = (
    "## DCGAN wrapper `train()` / `eval()` propagation — quick refresher\n"
    "\n"
    "`nn.Module.train(mode=True)` recursively switches `self.training = "
    "mode` for every submodule. For DCGAN that means `wrapper.train()` "
    "flips `netG`, `netD`, AND every BatchNorm/Dropout buried inside them "
    "— all in one call.\n"
    "\n"
    "**Why this matters.** `nn.BatchNorm2d` uses BATCH statistics in train "
    "mode and FROZEN running statistics in eval mode. If you only flip the "
    "wrapper to `eval()` and one of the BN layers stays in train mode, "
    "your D will see a different distribution on identical input — a "
    "silent correctness bug that's hard to debug.\n"
    "\n"
    "**`state_dict()` round-trips both subnets.** "
    "`wrapper.state_dict()` returns a flat dict keyed by qualified name "
    "(`netG.0.weight`, `netD.2.bias`, etc.). `wrapper.load_state_dict(sd)` "
    "restores both subnets at once. Saving a checkpoint = one call, not "
    "two."
)

RECAP_NO_GRAD = (
    "## `with torch.no_grad():` vs `.detach()` — quick refresher\n"
    "\n"
    "Two ways to keep gradients from flowing into G during the D-step:\n"
    "\n"
    "**`.detach()` (ex1):** `fake = G(z).detach()`. G's forward STILL builds "
    "the autograd graph; detach severs the link AFTER the fact. Cost: full "
    "forward graph allocation, then one node-level detach.\n"
    "\n"
    "**`torch.no_grad()` (this drill):** `with torch.no_grad(): fake = "
    "G(z)`. G's forward DOES NOT build the autograd graph at all — no "
    "intermediate activation buffers, no edges. Cheaper.\n"
    "\n"
    "Both produce a bit-identical `fake` tensor (same forward math) and "
    "leave D's gradient identical (D's path to the loss is the same). The "
    "difference is purely memory + compute on G's forward.\n"
    "\n"
    "**Why `.detach()` is still common.** Makes the stop-gradient point "
    "EXPLICIT at the use site. `no_grad()` scopes the whole block — if you "
    "later add another op inside the `with`, it silently won't get a grad "
    "either. `.detach()` only severs the one tensor you name."
)

RECAP_RANDOM_HOLDOUT = (
    "## Deterministic random per-class holdout — quick refresher\n"
    "\n"
    "ex1 picked the FIRST sample of each class — cheap, but biased toward "
    "whatever the data loader happens to surface first. A more "
    "representative gallery uses a RANDOM sample per class, sampled with a "
    "fixed seed so the gallery is stable across runs:\n"
    "```python\n"
    "g = t.Generator().manual_seed(seed)\n"
    "for c in range(num_classes):\n"
    "    idxs_for_c = (labels == c).nonzero(as_tuple=True)[0]\n"
    "    pick = idxs_for_c[t.randint(len(idxs_for_c), (1,), generator=g)]\n"
    "    holdout.append(data[pick.item()])\n"
    "```\n"
    "\n"
    "**Why a generator (not global seed).** The global RNG state changes "
    "every time any other code samples — your gallery would silently shift. "
    "A local `t.Generator()` keeps the selection reproducible regardless of "
    "surrounding training noise.\n"
    "\n"
    "**`nonzero(as_tuple=True)[0]`** is the canonical recipe for 'integer "
    "indices where mask is True'. Equivalent to `t.where(mask)[0]`. The "
    "result is a 1-D `int64` tensor of positions you can sample from."
)


# ===========================================================================
# SPEC 1 — rank0-only-side-effects ex2
# ===========================================================================

SPEC_RANK0_BARRIER = {
    "atom_id": "rank0-only-side-effects",
    "subtopic": "Distributed: rank-0-only side effects",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_RANK0_BARRIER,
    "exercise_index": 2,
    "exercise_title": "rank-0 download + barrier so every rank reads safely",
    "slug": "rank0-download-with-barrier-then-all-rank-read",
    "bloom_level": "Analyze",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["rank-0", "barrier", "ordering", "download", "shared-resource"],
    "kcs": ["rank0-side-effect-with-barrier", "barrier-orders-writer-before-readers"],
    "lo": (
        "Analyze the writer-readers ordering pattern by combining the "
        "`if rank == 0:` guard for a download with a `dist.barrier()` so "
        "rank > 0's read is guaranteed to happen AFTER rank 0's write."
    ),
    "prompt_body": (
        "Implement `ex2_rank0_download_then_all_read(rank, world_size, "
        "dist_module, downloader, reader, log)`. The producer-consumer "
        "pattern that EVERY distributed training script needs once at "
        "startup:\n\n"
        "1. **Rank 0 only — download.** If `rank == 0`, call "
        "`downloader()` and then `log('rank0-downloaded')` so the test can "
        "verify the order.\n"
        "2. **Every rank — barrier.** Call `dist_module.barrier()` "
        "unconditionally. Ranks > 0 hit this first and block; rank 0 hits "
        "it after step 1 and unblocks everyone.\n"
        "3. **Every rank — read.** Call `value = reader()` (the file is "
        "now guaranteed to exist on shared storage), then "
        "`log(f'rank{rank}-read-{value}')`. Return `value`.\n\n"
        "Critical ordering invariants the test checks:\n"
        "- `rank0-downloaded` appears in the log BEFORE any "
        "`rank*-read-...` entry.\n"
        "- `downloader()` is called EXACTLY ONCE across all ranks (only "
        "rank 0 invokes it).\n"
        "- `reader()` is called once per rank (every rank reads).\n\n"
        "Input: `rank`, `world_size` ints; `dist_module` (mocked dist); "
        "`downloader`, `reader`, `log` callables.\n"
        "Output: the value returned by `reader()` — same on every rank "
        "(reader returns a constant in the mock)."
    ),
    "stub": (
        "def ex2_rank0_download_then_all_read(rank, world_size, dist_module,\n"
        "                                    downloader, reader, log):\n"
        '    """Rank-0 download, barrier, then every rank reads."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FAKE_DIST_HARNESS + "\n\n"
        "# Shared mutable state — guarded by the world.lock that the harness exposes.\n"
        "_log_lines = []\n"
        "_log_lock = threading.Lock()\n"
        "_download_calls = [0]\n"
        "_read_calls = [0]\n"
        "\n"
        "def _downloader():\n"
        "    with _log_lock:\n"
        "        _download_calls[0] += 1\n"
        "    return None\n"
        "\n"
        "def _reader():\n"
        "    with _log_lock:\n"
        "        _read_calls[0] += 1\n"
        "    return 'payload-42'\n"
        "\n"
        "def _log(msg):\n"
        "    with _log_lock:\n"
        "        _log_lines.append(msg)\n"
        "\n"
        "def _worker(rank, world_size, dist_module, world):\n"
        "    val = ex2_rank0_download_then_all_read(rank, world_size, dist_module,\n"
        "                                          _downloader, _reader, _log)\n"
        "    world.results[rank] = val\n"
        "\n"
        "results = _run_fake_world(_worker, 4)\n"
        "\n"
        "# Every rank got the same payload.\n"
        "for rank, r in enumerate(results):\n"
        "    assert r == 'payload-42', f'rank {rank}: got {r!r}, expected payload-42'\n"
        "\n"
        "# downloader called exactly once.\n"
        "assert _download_calls[0] == 1, (\n"
        "    f'downloader fired {_download_calls[0]} times — must be 1 (rank 0 only)'\n"
        ")\n"
        "# reader called once per rank.\n"
        "assert _read_calls[0] == 4, f'reader fired {_read_calls[0]} times — expected 4'\n"
        "\n"
        "# Ordering invariant: rank0-downloaded must appear BEFORE any rank*-read entry.\n"
        "download_idx = _log_lines.index('rank0-downloaded')\n"
        "read_idxs = [i for i, m in enumerate(_log_lines) if m.startswith('rank') and '-read-' in m]\n"
        "assert len(read_idxs) == 4, f'expected 4 read log entries, got {len(read_idxs)}'\n"
        "assert download_idx < min(read_idxs), (\n"
        "    f'rank0-downloaded must precede every read; got download@{download_idx}, '\n"
        "    f'min read@{min(read_idxs)}.  Did you forget the barrier?'\n"
        ")\n"
        "\n"
        "# Reset + re-run with world_size=1 — rank 0 IS everyone; barrier still works.\n"
        "_log_lines.clear(); _download_calls[0] = 0; _read_calls[0] = 0\n"
        "results1 = _run_fake_world(_worker, 1)\n"
        "assert results1[0] == 'payload-42'\n"
        "assert _download_calls[0] == 1 and _read_calls[0] == 1\n"
        "\n"
        "# Sanity: world_size=2 — only one downloader call, both ranks read.\n"
        "_log_lines.clear(); _download_calls[0] = 0; _read_calls[0] = 0\n"
        "results2 = _run_fake_world(_worker, 2)\n"
        "assert _download_calls[0] == 1, f'ws=2: downloader fired {_download_calls[0]}'\n"
        "assert _read_calls[0] == 2"
    ),
    "solution_body": (
        "def ex2_rank0_download_then_all_read(rank, world_size, dist_module,\n"
        "                                    downloader, reader, log):\n"
        "    if rank == 0:\n"
        "        downloader()\n"
        "        log('rank0-downloaded')\n"
        "    dist_module.barrier()\n"
        "    value = reader()\n"
        "    log(f'rank{rank}-read-{value}')\n"
        "    return value"
    ),
    "solution_notes": (
        "**Without the barrier, rank 1 races.** A bare `if rank == 0:` "
        "guarded write returns immediately on rank 0; meanwhile rank 1 "
        "tries to call `reader()` before rank 0 has finished writing. "
        "The file may be missing, half-written, or — worst — exist but "
        "with stale contents from a previous run. The barrier is the one "
        "line that prevents the race.\n\n"
        "**Every rank must call barrier.** A barrier on rank 0 alone "
        "doesn't help — barrier is a collective. If rank 1 skips it, rank "
        "0 hangs forever waiting for the world to catch up.\n\n"
        "**Real DDP recipe:** rank 0 calls `download + write to NFS / "
        "S3`, `dist.barrier()`, every rank reads from the shared store. "
        "Same shape as this drill — just bigger downloader."
    ),
    "extra_imports": [],
}


# ===========================================================================
# SPEC 2 — reduce-gather-sum ex2 (all_gather + median)
# ===========================================================================

SPEC_GATHER_MEDIAN = {
    "atom_id": "reduce-gather-sum",
    "subtopic": "Distributed: reduce.gather + sum",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_ALL_GATHER_MEDIAN,
    "exercise_index": 2,
    "exercise_title": "global median across ranks via all_gather + manual sort",
    "slug": "global-median-via-all-gather-and-manual-sort",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["all_gather", "median", "manual-aggregation", "non-associative"],
    "kcs": ["all-gather-pre-allocates-gather-list", "median-from-gathered-tensors"],
    "lo": (
        "Apply `dist.all_gather` followed by a local sort + median to "
        "compute a global median across ranks — an aggregation `ReduceOp` "
        "cannot express."
    ),
    "prompt_body": (
        "Implement `ex2_all_gather_median(rank, world_size, dist_module, "
        "local_value)`. Compute the global MEDIAN of `world_size` rank-"
        "local scalars on every rank.\n\n"
        "Steps:\n"
        "1. Wrap the local value: `tensor = t.tensor([local_value], "
        "dtype=t.float32)`.\n"
        "2. Pre-allocate the gather list: `gather_list = [t.zeros(1, "
        "dtype=t.float32) for _ in range(world_size)]`. Build this on "
        "EVERY rank (not just rank 0) — `all_gather` populates every "
        "rank's list.\n"
        "3. `dist_module.all_gather(gather_list, tensor)`. After this, "
        "every rank's `gather_list[r]` holds rank r's value.\n"
        "4. Stack into one tensor: `gathered = t.cat(gather_list)` "
        "(shape `(world_size,)`).\n"
        "5. Sort and take the median index — for even `world_size`, "
        "average the two middle values:\n"
        "   ```python\n"
        "   sorted_vals = t.sort(gathered).values\n"
        "   mid = world_size // 2\n"
        "   if world_size % 2 == 1:\n"
        "       median = sorted_vals[mid].item()\n"
        "   else:\n"
        "       median = ((sorted_vals[mid - 1] + sorted_vals[mid]) / 2).item()\n"
        "   ```\n"
        "6. Return `median` — a Python float, identical on every rank.\n\n"
        "**Why `all_gather`, not `reduce`.** `ReduceOp` only supports "
        "associative binary ops (sum, max, min, product). Median requires "
        "the FULL distribution to compute — every rank needs the whole "
        "set, hence `all_gather`."
    ),
    "stub": (
        "def ex2_all_gather_median(rank: int, world_size: int, dist_module, local_value: float) -> float:\n"
        '    """Compute global median across ranks via all_gather + sort."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FAKE_DIST_HARNESS + "\n\n"
        "# Odd world_size = 5, values [1, 5, 2, 9, 3] → sorted [1,2,3,5,9] → median 3.\n"
        "_vals5 = [1.0, 5.0, 2.0, 9.0, 3.0]\n"
        "\n"
        "def _worker5(rank, world_size, dist_module, world):\n"
        "    world.results[rank] = ex2_all_gather_median(rank, world_size, dist_module, _vals5[rank])\n"
        "\n"
        "results = _run_fake_world(_worker5, 5)\n"
        "for rank, r in enumerate(results):\n"
        "    assert r is not None, f'rank {rank} returned None'\n"
        "    assert abs(r - 3.0) < 1e-5, f'rank {rank}: got {r}, expected 3.0'\n"
        "\n"
        "# Even world_size = 4, values [1, 2, 8, 4] → sorted [1,2,4,8] → median (2+4)/2 = 3.\n"
        "_vals4 = [1.0, 2.0, 8.0, 4.0]\n"
        "\n"
        "def _worker4(rank, world_size, dist_module, world):\n"
        "    world.results[rank] = ex2_all_gather_median(rank, world_size, dist_module, _vals4[rank])\n"
        "\n"
        "results4 = _run_fake_world(_worker4, 4)\n"
        "for rank, r in enumerate(results4):\n"
        "    assert abs(r - 3.0) < 1e-5, f'even rank {rank}: got {r}, expected 3.0'\n"
        "\n"
        "# Negative + duplicate values — median must still be stable.\n"
        "_vals_neg = [-5.0, -1.0, 0.0, -1.0, 7.0]   # sorted [-5,-1,-1,0,7] → median -1\n"
        "\n"
        "def _worker_neg(rank, world_size, dist_module, world):\n"
        "    world.results[rank] = ex2_all_gather_median(rank, world_size, dist_module, _vals_neg[rank])\n"
        "\n"
        "results_neg = _run_fake_world(_worker_neg, 5)\n"
        "for rank, r in enumerate(results_neg):\n"
        "    assert abs(r - (-1.0)) < 1e-5, f'neg/dup rank {rank}: got {r}, expected -1.0'\n"
        "\n"
        "# Identical values — median is that value.\n"
        "def _worker_same(rank, world_size, dist_module, world):\n"
        "    world.results[rank] = ex2_all_gather_median(rank, world_size, dist_module, 7.5)\n"
        "\n"
        "results_same = _run_fake_world(_worker_same, 3)\n"
        "for r in results_same:\n"
        "    assert abs(r - 7.5) < 1e-5\n"
        "\n"
        "# Single-rank degenerate — median of one value is itself.\n"
        "def _worker1(rank, world_size, dist_module, world):\n"
        "    world.results[rank] = ex2_all_gather_median(rank, world_size, dist_module, 42.0)\n"
        "\n"
        "results1 = _run_fake_world(_worker1, 1)\n"
        "assert abs(results1[0] - 42.0) < 1e-5"
    ),
    "solution_body": (
        "def ex2_all_gather_median(rank: int, world_size: int, dist_module, local_value: float) -> float:\n"
        "    tensor = t.tensor([local_value], dtype=t.float32)\n"
        "    gather_list = [t.zeros(1, dtype=t.float32) for _ in range(world_size)]\n"
        "    dist_module.all_gather(gather_list, tensor)\n"
        "    gathered = t.cat(gather_list)\n"
        "    sorted_vals = t.sort(gathered).values\n"
        "    mid = world_size // 2\n"
        "    if world_size % 2 == 1:\n"
        "        return sorted_vals[mid].item()\n"
        "    return ((sorted_vals[mid - 1] + sorted_vals[mid]) / 2).item()"
    ),
    "solution_notes": (
        "**The list-building step is the easiest place to introduce a "
        "bug.** `[t.zeros(1)] * world_size` creates `world_size` "
        "references to the SAME tensor — all `world_size` 'slots' alias "
        "and the gather silently overwrites itself. Always use a list "
        "comprehension to allocate distinct tensors.\n\n"
        "**Median vs `kthvalue`.** `t.median` on an even-length tensor "
        "returns the LOWER of the two middle values, not their average. "
        "Above we compute the average explicitly so the answer matches "
        "numpy/scipy conventions. If you actually want torch's lower-mid "
        "behavior, `t.median(gathered).values` works for both odd and "
        "even.\n\n"
        "**`gather` (single dst) vs `all_gather` (everyone).** Use "
        "`gather` when only rank 0 needs the result (then broadcast). Use "
        "`all_gather` when every rank needs to act on the per-rank "
        "values. For a metric you only log on rank 0, gather is cheaper."
    ),
    "extra_imports": [],
}


# ===========================================================================
# SPEC 3 — reduce-op-mean-divide ex2 (harmonic mean)
# ===========================================================================

SPEC_HARMONIC = {
    "atom_id": "reduce-op-mean-divide",
    "subtopic": "Distributed: reduce-op mean divide",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_HARMONIC,
    "exercise_index": 2,
    "exercise_title": "harmonic mean across ranks via 1/x transform + all_reduce + divide + invert",
    "slug": "harmonic-mean-across-ranks-via-transform-and-all-reduce",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["harmonic-mean", "all_reduce", "transform-and-invert", "throughput"],
    "kcs": ["transform-then-sum-then-divide-then-invert", "throughput-aggregation"],
    "lo": (
        "Apply the transform-sum-divide-invert recipe (1/x → "
        "`all_reduce(SUM)` → `/= world_size` → 1/.) to compute the "
        "harmonic mean of per-rank values on every rank."
    ),
    "prompt_body": (
        "Implement `ex2_harmonic_mean(rank, world_size, dist_module, "
        "local_value)`. Compute the harmonic mean `H = N / sum(1/x_r)` "
        "across ranks.\n\n"
        "Steps inside the function:\n"
        "1. Validate `local_value > 0` (harmonic mean is undefined on "
        "zero/negative inputs). Raise `ValueError` if not.\n"
        "2. Build the reciprocal tensor: `tensor = t.tensor([1.0 / "
        "local_value], dtype=t.float32)`.\n"
        "3. `dist_module.all_reduce(tensor, op=dist_module.ReduceOp.SUM)` "
        "— now `tensor[0]` is `sum_r 1/x_r`.\n"
        "4. In-place divide: `tensor /= world_size` — now `tensor[0]` is "
        "the arithmetic mean of the reciprocals.\n"
        "5. Invert: `harmonic = 1.0 / tensor.item()`.\n"
        "6. Return `harmonic` — a Python float, identical on every rank.\n\n"
        "**Use case.** Per-rank training throughput in samples/sec. "
        "Rank 0 might do 100 samples/sec, rank 1 might do 50; the "
        "ARITHMETIC mean (75) over-weights the fast rank. The HARMONIC "
        "mean = 2 / (1/100 + 1/50) = 66.67 — the time-weighted average, "
        "which is the throughput a downstream consumer actually sees.\n\n"
        "Input: `rank`, `world_size` ints; `dist_module`; `local_value` "
        "positive float.\n"
        "Output: float — harmonic mean, same on every rank."
    ),
    "stub": (
        "def ex2_harmonic_mean(rank: int, world_size: int, dist_module, local_value: float) -> float:\n"
        '    """Harmonic mean across ranks via 1/x transform + all_reduce + divide + invert."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FAKE_DIST_HARNESS + "\n\n"
        "# Classic case: throughput rank 0=100, rank 1=50.\n"
        "# H = 2 / (1/100 + 1/50) = 2 / 0.03 = 66.6667.\n"
        "_vals2 = [100.0, 50.0]\n"
        "\n"
        "def _worker2(rank, world_size, dist_module, world):\n"
        "    world.results[rank] = ex2_harmonic_mean(rank, world_size, dist_module, _vals2[rank])\n"
        "\n"
        "results = _run_fake_world(_worker2, 2)\n"
        "expected = 2.0 / (1.0/100.0 + 1.0/50.0)   # = 200/3 ≈ 66.667\n"
        "for rank, r in enumerate(results):\n"
        "    assert r is not None\n"
        "    assert abs(r - expected) < 1e-3, (\n"
        "        f'rank {rank}: got {r}, expected {expected}. '\n"
        "        f'If you got 75.0, you computed the ARITHMETIC mean — '\n"
        "        f'remember to transform/invert AROUND the reduce.'\n"
        "    )\n"
        "\n"
        "# Identical values — harmonic mean degenerates to that value.\n"
        "def _worker_same(rank, world_size, dist_module, world):\n"
        "    world.results[rank] = ex2_harmonic_mean(rank, world_size, dist_module, 12.0)\n"
        "\n"
        "results_same = _run_fake_world(_worker_same, 4)\n"
        "for rank, r in enumerate(results_same):\n"
        "    assert abs(r - 12.0) < 1e-4, f'identical-values rank {rank}: got {r}'\n"
        "\n"
        "# Three ranks, asymmetric.\n"
        "_vals3 = [1.0, 2.0, 4.0]\n"
        "\n"
        "def _worker3(rank, world_size, dist_module, world):\n"
        "    world.results[rank] = ex2_harmonic_mean(rank, world_size, dist_module, _vals3[rank])\n"
        "\n"
        "results3 = _run_fake_world(_worker3, 3)\n"
        "expected3 = 3.0 / (1.0/1.0 + 1.0/2.0 + 1.0/4.0)   # = 3 / 1.75 = 12/7 ≈ 1.714\n"
        "for rank, r in enumerate(results3):\n"
        "    assert abs(r - expected3) < 1e-4, f'3-rank rank {rank}: got {r}, expected {expected3}'\n"
        "\n"
        "# Single-rank degenerate — harmonic mean of one value is itself.\n"
        "def _worker1(rank, world_size, dist_module, world):\n"
        "    world.results[rank] = ex2_harmonic_mean(rank, world_size, dist_module, 9.0)\n"
        "\n"
        "results1 = _run_fake_world(_worker1, 1)\n"
        "assert abs(results1[0] - 9.0) < 1e-5\n"
        "\n"
        "# Zero input must raise.\n"
        "def _worker_zero(rank, world_size, dist_module, world):\n"
        "    try:\n"
        "        ex2_harmonic_mean(rank, world_size, dist_module, 0.0)\n"
        "        world.results[rank] = 'no-raise'\n"
        "    except ValueError:\n"
        "        world.results[rank] = 'raised'\n"
        "    except Exception as e:\n"
        "        world.results[rank] = f'wrong-exc:{type(e).__name__}'\n"
        "\n"
        "# Single-rank world so the lack of a paired call doesn't matter.\n"
        "rz = _run_fake_world(_worker_zero, 1)\n"
        "assert rz[0] == 'raised', f'zero input must raise ValueError, got {rz[0]!r}'"
    ),
    "solution_body": (
        "def ex2_harmonic_mean(rank: int, world_size: int, dist_module, local_value: float) -> float:\n"
        "    if local_value <= 0:\n"
        "        raise ValueError(f'harmonic mean undefined for non-positive input: {local_value}')\n"
        "    tensor = t.tensor([1.0 / local_value], dtype=t.float32)\n"
        "    dist_module.all_reduce(tensor, op=dist_module.ReduceOp.SUM)\n"
        "    tensor /= world_size\n"
        "    return 1.0 / tensor.item()"
    ),
    "solution_notes": (
        "**The 'transform-aggregate-invert' shape is general.** "
        "Geometric mean = log → SUM → /= N → exp. Quadratic mean = "
        "square → SUM → /= N → sqrt. All built from `all_reduce(SUM)` + "
        "in-place divide + element-wise nonlinearities.\n\n"
        "**Why validate `> 0`.** `1.0 / 0.0` is `inf` in IEEE 754; `1.0 "
        "/ -2.0` flips sign and silently produces a finite-but-wrong "
        "answer. Both are footguns the test explicitly checks against.\n\n"
        "**ReduceOp.SUM works on the transformed values, not the "
        "originals.** This is the key insight ex1 hints at and ex2 makes "
        "concrete: the missing `ReduceOp.MEAN` is just sugar for SUM + "
        "divide. Any named mean is SUM + divide of the right transformed "
        "values."
    ),
    "extra_imports": [],
}


# ===========================================================================
# SPEC 4 — bottleneck-latent-projection ex2 (tied-weight round trip)
# ===========================================================================

SPEC_TIED_DECODE = {
    "atom_id": "bottleneck-latent-projection",
    "subtopic": "Generative: Bottleneck latent projection",
    "topic_folder": TOPIC_GEN,
    "atom_recap_md": RECAP_TIED_DECODE,
    "exercise_index": 2,
    "exercise_title": "tied-weight encode-decode round trip and identity for orthonormal W",
    "slug": "tied-weight-encode-decode-round-trip",
    "bloom_level": "Analyze",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["tied-weights", "autoencoder", "orthonormal", "round-trip"],
    "kcs": ["tied-decoder-uses-W-not-W.T", "orthonormal-W-yields-identity-roundtrip"],
    "lo": (
        "Analyze the tied-weight autoencoder by implementing a "
        "round-trip encode+decode using the SAME weight matrix (W.T for "
        "encode, W for decode) and verifying that an orthonormal W "
        "reconstructs the input."
    ),
    "prompt_body": (
        "Implement `ex2_tied_encode_decode(flat_batch, weight, b_enc, "
        "b_dec)`. The classic tied-weight symmetric autoencoder pass:\n\n"
        "1. `flat_batch` shape `(B, input_dim)`. `weight` shape "
        "`(latent_dim, input_dim)` — same convention as `nn.Linear` "
        "stores it (rows = outputs). `b_enc` shape `(latent_dim,)`. "
        "`b_dec` shape `(input_dim,)`.\n"
        "2. Encode: `z = flat_batch @ weight.T + b_enc`. Shape `(B, "
        "latent_dim)`. SAME formula as ex1.\n"
        "3. Decode using the SAME weight (NOT a separate `W_dec`): "
        "`x_hat = z @ weight + b_dec`. Shape `(B, input_dim)`. Note: NO "
        "`.T` on the decode — `weight` is already shaped `(latent_dim, "
        "input_dim)` which is exactly what `z @ weight` needs.\n"
        "4. Return `(z, x_hat)` — both tensors as a tuple.\n\n"
        "Critical detail: ONE weight matrix, used twice (`weight.T` on "
        "the way in, `weight` on the way out). The decoder has NO "
        "independent weight parameter.\n\n"
        "Input: `flat_batch` `(B, input_dim)`, `weight` `(latent_dim, "
        "input_dim)`, `b_enc` `(latent_dim,)`, `b_dec` `(input_dim,)`.\n"
        "Output: tuple `(z, x_hat)` with shapes `(B, latent_dim)`, `(B, "
        "input_dim)`.\n\n"
        "The visualization renders one input image, its latent code as a "
        "bar chart, and the round-trip reconstruction — when `W` is "
        "orthonormal and biases are zero, reconstruction is identical to "
        "the input."
    ),
    "stub": (
        "def ex2_tied_encode_decode(flat_batch: Tensor, weight: Tensor,\n"
        "                           b_enc: Tensor, b_dec: Tensor):\n"
        '    """Tied-weight encode-decode round trip. Return (z, x_hat)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Shape contract.\n"
        "B, input_dim, latent_dim = 5, 8, 8   # square so orthonormal works\n"
        "flat = t.randn(B, input_dim, generator=t.Generator().manual_seed(0))\n"
        "W = t.eye(input_dim)   # identity → orthonormal trivially\n"
        "b_enc = t.zeros(latent_dim)\n"
        "b_dec = t.zeros(input_dim)\n"
        "z, x_hat = ex2_tied_encode_decode(flat, W, b_enc, b_dec)\n"
        "assert z.shape == (B, latent_dim), f'z shape {tuple(z.shape)}'\n"
        "assert x_hat.shape == (B, input_dim), f'x_hat shape {tuple(x_hat.shape)}'\n"
        "\n"
        "# Identity W with zero biases must reconstruct exactly.\n"
        "assert t.allclose(x_hat, flat, atol=1e-5), 'identity W round-trip must recover input'\n"
        "assert t.allclose(z, flat, atol=1e-5), 'identity W encode must equal input'\n"
        "\n"
        "# A proper square orthonormal W must also round-trip exactly.\n"
        "rng = t.Generator().manual_seed(7)\n"
        "rand = t.randn(input_dim, input_dim, generator=rng)\n"
        "Q, _ = t.linalg.qr(rand)   # Q is orthonormal: Q @ Q.T = I\n"
        "z2, x_hat2 = ex2_tied_encode_decode(flat, Q, t.zeros(latent_dim), t.zeros(input_dim))\n"
        "assert t.allclose(x_hat2, flat, atol=1e-4), (\n"
        "    f'orthonormal W round-trip must be identity; max diff '\n"
        "    f'{(x_hat2 - flat).abs().max().item():.6f}'\n"
        ")\n"
        "\n"
        "# Non-square: latent_dim < input_dim. Round-trip is LOSSY but z must match ex1's projection.\n"
        "latent_dim_small = 3\n"
        "W_small = t.randn(latent_dim_small, input_dim, generator=t.Generator().manual_seed(1))\n"
        "b_enc_s = t.randn(latent_dim_small, generator=t.Generator().manual_seed(2))\n"
        "b_dec_s = t.randn(input_dim, generator=t.Generator().manual_seed(3))\n"
        "z3, x_hat3 = ex2_tied_encode_decode(flat, W_small, b_enc_s, b_dec_s)\n"
        "assert z3.shape == (B, latent_dim_small)\n"
        "assert x_hat3.shape == (B, input_dim)\n"
        "\n"
        "# Encode value match ex1's affine.\n"
        "expected_z = flat @ W_small.T + b_enc_s\n"
        "assert t.allclose(z3, expected_z, atol=1e-5), 'encode must equal flat @ W.T + b_enc'\n"
        "# Decode value match the analytical form (no .T on decode!).\n"
        "expected_xhat = z3 @ W_small + b_dec_s\n"
        "assert t.allclose(x_hat3, expected_xhat, atol=1e-5), (\n"
        "    'decode must equal z @ W + b_dec — did you put a .T on W during decode? '\n"
        "    '(The tied-weight pattern uses NO .T on the way out.)'\n"
        "  )\n"
        "\n"
        "# Catch the most common bug: decoding with `.T` on a non-square W blows up\n"
        "# the matmul. Confirm explicitly that the BUGGY path would raise.\n"
        "buggy_raised = False\n"
        "try:\n"
        "    _ = z3 @ W_small.T   # shape mismatch (5,3) @ (8,3)\n"
        "except RuntimeError:\n"
        "    buggy_raised = True\n"
        "assert buggy_raised, 'decoding with .T on non-square W should raise — this drill keys off that'\n"
        "\n"
        "# Bias contribution: zero W must produce x_hat = b_dec (broadcast over batch).\n"
        "W_zero = t.zeros(latent_dim_small, input_dim)\n"
        "b_enc_z = t.zeros(latent_dim_small)\n"
        "b_dec_fixed = t.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])\n"
        "z_z, xhat_z = ex2_tied_encode_decode(flat, W_zero, b_enc_z, b_dec_fixed)\n"
        "for b_idx in range(B):\n"
        "    assert t.allclose(xhat_z[b_idx], b_dec_fixed, atol=1e-6), (\n"
        "        'zero-W round-trip must produce just b_dec broadcast over batch'\n"
        "    )\n"
        "\n"
        "# --- Visualization: one sample, its latent, and the reconstruction ---\n"
        "fig, axes = plt.subplots(1, 3, figsize=(10, 3))\n"
        "sample_idx = 0\n"
        "axes[0].bar(range(input_dim), flat[sample_idx].numpy(), color='steelblue', edgecolor='black')\n"
        "axes[0].set_title('input')\n"
        "axes[1].bar(range(latent_dim), z[sample_idx].numpy(), color='seagreen', edgecolor='black')\n"
        "axes[1].set_title('latent z (identity W → z == input)')\n"
        "axes[2].bar(range(input_dim), x_hat[sample_idx].numpy(), color='coral', edgecolor='black')\n"
        "axes[2].set_title('reconstruction (orthonormal W → exact)')\n"
        "plt.suptitle('ex2 tied-weight round trip — orthonormal W reconstructs exactly')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex2_tied_encode_decode(flat_batch: Tensor, weight: Tensor,\n"
        "                           b_enc: Tensor, b_dec: Tensor):\n"
        "    z = flat_batch @ weight.T + b_enc\n"
        "    x_hat = z @ weight + b_dec\n"
        "    return z, x_hat"
    ),
    "solution_notes": (
        "**The `.T` asymmetry.** `weight` is stored as "
        "`(latent_dim, input_dim)` — rows are encoder outputs. To encode "
        "`(B, input_dim) -> (B, latent_dim)` you need the transpose. To "
        "decode `(B, latent_dim) -> (B, input_dim)` you use it as stored. "
        "ONE weight, two shapes-of-use.\n\n"
        "**Why orthonormal W reconstructs exactly.** "
        "`x_hat = (flat @ W.T) @ W = flat @ (W.T @ W)`. For an "
        "orthonormal square `W`, `W.T @ W = I`, so the round trip is the "
        "identity. For non-square `W` (latent_dim < input_dim), "
        "`W.T @ W` is a projection onto W's row space — reconstruction "
        "is lossy by the amount of variance W's rows don't capture. "
        "This is PCA, restated.\n\n"
        "**Tied vs untied in modern code.** Modern VAEs/AEs usually "
        "UNTIE — separate `W_enc` and `W_dec` parameters. Tying halves "
        "the parameter count and adds an implicit regularizer, but "
        "limits decoder expressivity. The tie is most useful when "
        "training data is scarce."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


# ===========================================================================
# SPEC 5 — broadcast-source-fanout ex2 (nn.Embedding sparse grad)
# ===========================================================================

SPEC_EMBED_SPARSE = {
    "atom_id": "broadcast-source-fanout",
    "subtopic": "Generative: Broadcast source fan-out",
    "topic_folder": TOPIC_GEN,
    "atom_recap_md": RECAP_EMBEDDING_SPARSE,
    "exercise_index": 2,
    "exercise_title": "nn.Embedding lookup with verified sparse gradient on absent classes",
    "slug": "nn-embedding-with-sparse-grad-on-absent-classes",
    "bloom_level": "Analyze",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["nn.Embedding", "sparse-grad", "absent-classes", "F.embedding"],
    "kcs": ["nn-embedding-equiv-to-index", "sparse-grad-on-unindexed-rows"],
    "lo": (
        "Analyze the embedding-lookup gradient structure by building an "
        "`nn.Embedding`, fanning out via labels, summing the result, "
        "calling backward, and verifying that ONLY rows for classes "
        "present in the batch receive non-zero gradient."
    ),
    "prompt_body": (
        "Implement `ex2_embed_with_sparse_grad(num_classes, D, labels)`. "
        "Build the Module form, do the fan-out, force a backward, then "
        "return the embedding's `weight.grad` so the caller can inspect "
        "the sparsity:\n\n"
        "1. Construct `embed = nn.Embedding(num_classes, D)`. Initialize "
        "`embed.weight.data` to a known matrix: `t.arange(num_classes * "
        "D, dtype=t.float32).reshape(num_classes, D)`. This gives every "
        "row a unique fingerprint.\n"
        "2. Fan out: `per_sample = embed(labels)` (shape `(B, D)`).\n"
        "3. Compute a scalar loss: `loss = per_sample.sum()`. (Sum of "
        "all elements — gives gradient 1 per element of `per_sample`.)\n"
        "4. Call `loss.backward()`.\n"
        "5. Return a tuple `(per_sample.detach(), embed.weight.grad.clone())`. "
        "Detach the forward so the caller can read it without "
        "complicating its own autograd graph; clone the grad so future "
        "backward calls don't mutate the returned tensor.\n\n"
        "What the gradient looks like. Each row of `weight.grad` equals "
        "the number of times that class appeared in `labels` (because "
        "`loss = sum` differentiates element-wise to 1, so each fan-out "
        "contributes 1 to each component of the row it indexed). A "
        "class absent from labels has `weight.grad[c]` exactly zero "
        "(structural zero from autograd, not just numerically zero)."
    ),
    "stub": (
        "def ex2_embed_with_sparse_grad(num_classes: int, D: int, labels: Tensor):\n"
        '    """Build Embedding, fan out, backward sum-loss, return (per_sample, weight.grad)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# 5 classes, D=3. Labels contain classes 0, 2, 4 only (1 and 3 absent).\n"
        "labels = t.tensor([0, 2, 2, 4, 0, 4])   # counts: cls0=2, cls2=2, cls4=2, cls1=0, cls3=0\n"
        "per_sample, grad = ex2_embed_with_sparse_grad(num_classes=5, D=3, labels=labels)\n"
        "\n"
        "# Forward shape + value.\n"
        "assert per_sample.shape == (6, 3), f'per_sample shape {tuple(per_sample.shape)}'\n"
        "# Row 0 = embed[0] = arange[0:3] = [0,1,2]; row 1 = embed[2] = [6,7,8]; ...\n"
        "expected_rows = t.tensor([\n"
        "    [0., 1., 2.],   # label 0\n"
        "    [6., 7., 8.],   # label 2\n"
        "    [6., 7., 8.],   # label 2 (repeat)\n"
        "    [12., 13., 14.],   # label 4\n"
        "    [0., 1., 2.],   # label 0 (repeat)\n"
        "    [12., 13., 14.],   # label 4 (repeat)\n"
        "])\n"
        "assert t.allclose(per_sample, expected_rows), f'per_sample mismatch:\\n{per_sample}'\n"
        "\n"
        "# Grad shape.\n"
        "assert grad.shape == (5, 3), f'grad shape {tuple(grad.shape)}, expected (5,3)'\n"
        "\n"
        "# Sparse rows — classes 1 and 3 must be exact zero.\n"
        "assert t.equal(grad[1], t.zeros(3)), f'class 1 absent → grad must be 0, got {grad[1]}'\n"
        "assert t.equal(grad[3], t.zeros(3)), f'class 3 absent → grad must be 0, got {grad[3]}'\n"
        "\n"
        "# Indexed rows — grad value = number of times the class appeared (sum-loss gives 1 per element).\n"
        "assert t.allclose(grad[0], t.full((3,), 2.0)), f'class 0 appeared 2x, grad row {grad[0]}'\n"
        "assert t.allclose(grad[2], t.full((3,), 2.0)), f'class 2 appeared 2x, grad row {grad[2]}'\n"
        "assert t.allclose(grad[4], t.full((3,), 2.0)), f'class 4 appeared 2x, grad row {grad[4]}'\n"
        "\n"
        "# All-class-present case — every grad row must be > 0.\n"
        "all_labels = t.tensor([0, 1, 2, 3, 4])\n"
        "_, grad_full = ex2_embed_with_sparse_grad(5, 3, all_labels)\n"
        "for c in range(5):\n"
        "    assert (grad_full[c] != 0).all(), f'class {c} present but grad row is zero: {grad_full[c]}'\n"
        "    assert t.allclose(grad_full[c], t.ones(3)), f'each class appeared 1x, expected 1.0 per element'\n"
        "\n"
        "# Imbalanced case — class 0 appearing 5 times, class 1 once.\n"
        "imb = t.tensor([0, 0, 0, 0, 0, 1])\n"
        "_, grad_imb = ex2_embed_with_sparse_grad(2, 3, imb)\n"
        "assert t.allclose(grad_imb[0], t.full((3,), 5.0))\n"
        "assert t.allclose(grad_imb[1], t.ones(3))"
    ),
    "solution_body": (
        "def ex2_embed_with_sparse_grad(num_classes: int, D: int, labels: Tensor):\n"
        "    import torch.nn as nn\n"
        "    embed = nn.Embedding(num_classes, D)\n"
        "    with t.no_grad():\n"
        "        embed.weight.copy_(t.arange(num_classes * D, dtype=t.float32).reshape(num_classes, D))\n"
        "    per_sample = embed(labels)\n"
        "    loss = per_sample.sum()\n"
        "    loss.backward()\n"
        "    return per_sample.detach(), embed.weight.grad.clone()"
    ),
    "solution_notes": (
        "**`nn.Embedding(labels) == weight[labels]`.** They produce "
        "bit-identical tensors and the same autograd graph. The Module "
        "form's only difference: `weight` is auto-registered as a "
        "`nn.Parameter`, so an optimizer over `embed.parameters()` "
        "picks it up.\n\n"
        "**Why the absent rows are structural zero.** Autograd builds "
        "the gradient as `sum over batch of (one-hot[labels[i]] · "
        "d(loss)/d(per_sample[i]))`. The one-hot is zero for every "
        "class that didn't appear, so no contribution accumulates "
        "there. This is genuinely zero — not 1e-12 epsilon noise.\n\n"
        "**SparseAdam exploits this.** For embedding tables with "
        "millions of classes (token embeddings in a large vocab), most "
        "rows have zero grad most steps. `optim.SparseAdam` reads "
        "`weight.grad` as a sparse tensor and skips the zero rows — "
        "linear-time-per-step independent of vocab size. The drill "
        "doesn't use SparseAdam, but the sparsity property is the same."
    ),
    "extra_imports": [],
}


# ===========================================================================
# SPEC 6 — dcgan-wrapper-netG-netD ex2 (train/eval + state_dict round trip)
# ===========================================================================

SPEC_WRAPPER_TRAIN_EVAL = {
    "atom_id": "dcgan-wrapper-netG-netD",
    "subtopic": "Generative: DCGAN netG+netD wrapper",
    "topic_folder": TOPIC_GEN,
    "atom_recap_md": RECAP_WRAPPER_TRAIN_EVAL,
    "exercise_index": 2,
    "exercise_title": "wrapper train()/eval() propagation and state_dict round trip",
    "slug": "wrapper-train-eval-and-state-dict-round-trip",
    "bloom_level": "Analyze",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["dcgan", "wrapper", "train", "eval", "state_dict", "batchnorm"],
    "kcs": ["wrapper-mode-propagates-to-both-subnets", "state-dict-round-trips-both-subnets"],
    "lo": (
        "Analyze the wrapper's compound semantics by building it, "
        "toggling `train()`/`eval()` and verifying BOTH subnets (incl. "
        "BatchNorm submodules) follow, then save+load `state_dict()` "
        "and verify a numerical round trip on both subnets."
    ),
    "prompt_body": (
        "Implement `ex2_wrapper_lifecycle(generator, discriminator)`. "
        "The drill exercises the OPERATIONAL semantics the wrapper "
        "module enables — beyond just holding attributes:\n\n"
        "1. Build the wrapper class (same shape as ex1: `nn.Module` "
        "subclass with `netG` and `netD` as submodules, no `forward`).\n"
        "2. Instantiate it as `wrapper = DCGAN(generator, "
        "discriminator)`.\n"
        "3. **Toggle to eval mode** on the wrapper: `wrapper.eval()`.\n"
        "4. Snapshot the state dict: `sd_before = wrapper.state_dict()`. "
        "(Don't clone keys; do clone values to insulate from later "
        "writes.) Implement as: `sd_before = {k: v.detach().clone() for "
        "k, v in wrapper.state_dict().items()}`.\n"
        "5. **Mutate every parameter in-place** (to prove the round "
        "trip recovers): `for p in wrapper.parameters(): p.data.add_("
        "1.0)`.\n"
        "6. **Load the snapshot back:** `wrapper.load_state_dict("
        "sd_before)`.\n"
        "7. Return the tuple `(wrapper, sd_before)`.\n\n"
        "The test checks four invariants on the returned wrapper:\n"
        "(a) `wrapper.training is False` (eval propagated to wrapper).\n"
        "(b) `wrapper.netG.training is False` AND `wrapper.netD.training "
        "is False` (eval propagated to BOTH subnets).\n"
        "(c) Every BatchNorm submodule inside either subnet has "
        "`m.training is False`.\n"
        "(d) After load_state_dict, every parameter is bit-identical to "
        "the snapshot."
    ),
    "stub": (
        "def ex2_wrapper_lifecycle(generator: nn.Module, discriminator: nn.Module):\n"
        '    """Build wrapper, eval(), snapshot, mutate, restore. Return (wrapper, snapshot)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# Build a generator with a BatchNorm so we can probe propagation.\n"
        "gen = nn.Sequential(\n"
        "    nn.Linear(8, 16),\n"
        "    nn.BatchNorm1d(16),\n"
        "    nn.ReLU(),\n"
        "    nn.Linear(16, 4),\n"
        ")\n"
        "disc = nn.Sequential(\n"
        "    nn.Linear(4, 16),\n"
        "    nn.BatchNorm1d(16),\n"
        "    nn.LeakyReLU(),\n"
        "    nn.Linear(16, 1),\n"
        ")\n"
        "\n"
        "wrapper, sd_before = ex2_wrapper_lifecycle(gen, disc)\n"
        "assert isinstance(wrapper, nn.Module), f'expected nn.Module, got {type(wrapper)}'\n"
        "assert hasattr(wrapper, 'netG') and hasattr(wrapper, 'netD'), 'wrapper must have netG/netD attrs'\n"
        "assert wrapper.netG is gen and wrapper.netD is disc, 'subnets must be the same instances'\n"
        "\n"
        "# (a) wrapper in eval mode.\n"
        "assert wrapper.training is False, f'wrapper.training={wrapper.training}, expected False'\n"
        "# (b) Both subnets in eval mode.\n"
        "assert wrapper.netG.training is False, f'netG.training={wrapper.netG.training}, expected False'\n"
        "assert wrapper.netD.training is False, f'netD.training={wrapper.netD.training}, expected False'\n"
        "\n"
        "# (c) Every BatchNorm inside either subnet in eval mode.\n"
        "bn_count = 0\n"
        "for m in wrapper.modules():\n"
        "    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):\n"
        "        bn_count += 1\n"
        "        assert m.training is False, f'BN {m} still in train mode after wrapper.eval()'\n"
        "assert bn_count >= 2, f'expected at least 2 BN layers (one per subnet), got {bn_count}'\n"
        "\n"
        "# (d) Snapshot round-trip — every param bit-identical to sd_before.\n"
        "sd_after = wrapper.state_dict()\n"
        "assert set(sd_after.keys()) == set(sd_before.keys()), 'state_dict keys must match'\n"
        "for k in sd_before:\n"
        "    assert t.equal(sd_after[k], sd_before[k]), (\n"
        "        f'state_dict round-trip failed for {k!r} — '\n"
        "        f'max diff {(sd_after[k] - sd_before[k]).abs().max().item():.6f}.  '\n"
        "        f'Did you call load_state_dict at the end?'\n"
        "    )\n"
        "\n"
        "# Snapshot keys must span BOTH subnets — confirms state_dict captured both.\n"
        "netG_keys = [k for k in sd_before if k.startswith('netG.')]\n"
        "netD_keys = [k for k in sd_before if k.startswith('netD.')]\n"
        "assert len(netG_keys) > 0, f'snapshot has no netG.* keys: {list(sd_before)}'\n"
        "assert len(netD_keys) > 0, f'snapshot has no netD.* keys: {list(sd_before)}'\n"
        "\n"
        "# Toggle back to train mode — both subnets must follow.\n"
        "wrapper.train()\n"
        "assert wrapper.training is True\n"
        "assert wrapper.netG.training is True and wrapper.netD.training is True\n"
        "for m in wrapper.modules():\n"
        "    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):\n"
        "        assert m.training is True"
    ),
    "solution_body": (
        "def ex2_wrapper_lifecycle(generator: nn.Module, discriminator: nn.Module):\n"
        "    class DCGAN(nn.Module):\n"
        "        def __init__(self, netG, netD):\n"
        "            super().__init__()\n"
        "            self.netG = netG\n"
        "            self.netD = netD\n"
        "    wrapper = DCGAN(generator, discriminator)\n"
        "    wrapper.eval()\n"
        "    sd_before = {k: v.detach().clone() for k, v in wrapper.state_dict().items()}\n"
        "    for p in wrapper.parameters():\n"
        "        p.data.add_(1.0)\n"
        "    wrapper.load_state_dict(sd_before)\n"
        "    return wrapper, sd_before"
    ),
    "solution_notes": (
        "**`eval()` is recursive by design.** `nn.Module.eval()` calls "
        "`self.train(False)`, which sets `self.training = False` AND "
        "iterates over `self.children()` calling `.train(False)` on each "
        "— recursively. Same for `.train(True)`. The wrapper inherits "
        "this for free; you do nothing special.\n\n"
        "**Clone values in the snapshot.** `wrapper.state_dict()` "
        "returns tensors that ALIAS the live parameters. If you mutate "
        "parameters AFTER taking the snapshot but BEFORE cloning, your "
        "snapshot mutates too — the round trip becomes a no-op. "
        "Cloning at snapshot time decouples them.\n\n"
        "**`load_state_dict` is in-place.** It copies values into the "
        "existing parameter tensors — it does NOT rebind the parameters. "
        "Any external references to `wrapper.netG[0].weight` still point "
        "at the same tensor object, now holding the restored values. "
        "Critical for optimizer state preservation."
    ),
    "extra_imports": ["import torch.nn as nn"],
}


# ===========================================================================
# SPEC 7 — detach-stop-gradient-trick ex2 (no_grad alternative)
# ===========================================================================

SPEC_NO_GRAD = {
    "atom_id": "detach-stop-gradient-trick",
    "subtopic": "GAN: detach stop-gradient trick",
    "topic_folder": TOPIC_GEN,
    "atom_recap_md": RECAP_NO_GRAD,
    "exercise_index": 2,
    "exercise_title": "no_grad equivalent for the GAN D-step",
    "slug": "no-grad-equivalent-for-gan-d-step",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["no_grad", "detach", "gan", "autograd-graph"],
    "kcs": ["no-grad-vs-detach-equivalence", "no-grad-skips-graph-construction"],
    "lo": (
        "Analyze the `torch.no_grad()` alternative to `.detach()` in the "
        "GAN D-step by implementing both and verifying identical fake "
        "values + identical D gradients + the autograd graph absence for "
        "the no_grad path."
    ),
    "prompt_body": (
        "Implement `ex2_d_loss_with_no_grad(G, D, z, x_real)`. The "
        "cheaper alternative to ex1's `.detach()`:\n\n"
        "1. Wrap G's forward in `with t.no_grad():` — `fake = G(z)` "
        "inside the block.\n"
        "2. Compute `loss = (D(fake) - D(x_real)).mean()`.\n"
        "3. `loss.backward()`.\n"
        "4. Return `(loss.item(), fake.requires_grad)` — the second "
        "element is the test's way of probing that `fake` was built "
        "WITHOUT an autograd graph (under no_grad, the produced tensor "
        "has `requires_grad=False` and `grad_fn=None`).\n\n"
        "The test confirms that:\n"
        "- Loss value is identical to ex1's `.detach()` version (within "
        "1e-5).\n"
        "- D's parameter grads are identical to the `.detach()` version "
        "(modulo cumulative-grad ordering, which we control).\n"
        "- G's parameter grads remain zero (same as `.detach()`).\n"
        "- `fake.requires_grad is False` (the autograd-graph absence "
        "signature).\n\n"
        "Input: `G`, `D` — modules; `z`, `x_real` — input tensors.\n"
        "Output: `(loss_value: float, fake_requires_grad: bool)`."
    ),
    "stub": (
        "def ex2_d_loss_with_no_grad(G, D, z, x_real):\n"
        '    """D-step loss with `with t.no_grad(): fake = G(z)`; return (loss, fake.requires_grad)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "def _zero_all(*modules):\n"
        "    for m in modules:\n"
        "        for p in m.parameters():\n"
        "            if p.grad is not None:\n"
        "                p.grad.detach_(); p.grad.zero_()\n"
        "\n"
        "def _grad_norm(module):\n"
        "    s = 0.0\n"
        "    for p in module.parameters():\n"
        "        if p.grad is not None:\n"
        "            s += p.grad.pow(2).sum().item()\n"
        "    return s ** 0.5\n"
        "\n"
        "t.manual_seed(0)\n"
        "z_dim, x_dim, B = 4, 6, 8\n"
        "G = nn.Sequential(nn.Linear(z_dim, 16), nn.ReLU(), nn.Linear(16, x_dim))\n"
        "D = nn.Sequential(nn.Linear(x_dim, 16), nn.ReLU(), nn.Linear(16, 1))\n"
        "z = t.randn(B, z_dim)\n"
        "x_real = t.randn(B, x_dim)\n"
        "\n"
        "# Reference: detach version (we re-derive ex1's logic inline for comparison).\n"
        "def _ref_d_loss_detach(G, D, z, x_real):\n"
        "    fake = G(z).detach()\n"
        "    loss = (D(fake) - D(x_real)).mean()\n"
        "    loss.backward()\n"
        "    return loss.item()\n"
        "\n"
        "# --- no_grad version ---\n"
        "_zero_all(G, D)\n"
        "loss_nograd, fake_req = ex2_d_loss_with_no_grad(G, D, z, x_real)\n"
        "assert isinstance(loss_nograd, float)\n"
        "assert fake_req is False, (\n"
        "    f'under `with t.no_grad():`, fake.requires_grad must be False; got {fake_req}. '\n"
        "    f'Did you put G(z) outside the with-block?'\n"
        "  )\n"
        "g_norm_nograd = _grad_norm(G)\n"
        "d_norm_nograd = _grad_norm(D)\n"
        "assert g_norm_nograd == 0.0, f'no_grad should keep G grads at 0, got {g_norm_nograd}'\n"
        "assert d_norm_nograd > 0.0, f'D should still receive gradient, got {d_norm_nograd}'\n"
        "d_grads_nograd = {name: p.grad.detach().clone() for name, p in D.named_parameters() if p.grad is not None}\n"
        "\n"
        "# --- detach reference ---\n"
        "_zero_all(G, D)\n"
        "loss_detach = _ref_d_loss_detach(G, D, z, x_real)\n"
        "d_grads_detach = {name: p.grad.detach().clone() for name, p in D.named_parameters() if p.grad is not None}\n"
        "\n"
        "# Loss values match.\n"
        "assert abs(loss_nograd - loss_detach) < 1e-5, (\n"
        "    f'losses differ — nograd={loss_nograd}, detach={loss_detach}'\n"
        "  )\n"
        "# D grads match between paths.\n"
        "assert set(d_grads_nograd.keys()) == set(d_grads_detach.keys())\n"
        "for k in d_grads_nograd:\n"
        "    assert t.allclose(d_grads_nograd[k], d_grads_detach[k], atol=1e-5), (\n"
        "        f'D grad for {k!r} differs between no_grad and detach paths'\n"
        "    )\n"
        "\n"
        "# Sanity: G(z) inside no_grad really has no grad_fn.\n"
        "with t.no_grad():\n"
        "    probe_fake = G(z)\n"
        "assert probe_fake.grad_fn is None, 'no_grad must produce a tensor with grad_fn=None'\n"
        "\n"
        "# Stress: run no_grad version 10 times, G must NEVER accumulate grad.\n"
        "for _ in range(10):\n"
        "    _zero_all(G, D)\n"
        "    ex2_d_loss_with_no_grad(G, D, t.randn(B, z_dim), t.randn(B, x_dim))\n"
        "    assert _grad_norm(G) == 0.0, 'G must never accumulate grad under no_grad'"
    ),
    "solution_body": (
        "def ex2_d_loss_with_no_grad(G, D, z, x_real):\n"
        "    with t.no_grad():\n"
        "        fake = G(z)\n"
        "    loss = (D(fake) - D(x_real)).mean()\n"
        "    loss.backward()\n"
        "    return loss.item(), fake.requires_grad"
    ),
    "solution_notes": (
        "**Memory savings.** `.detach()` runs G's forward inside the "
        "autograd-tracking machinery — every intermediate activation is "
        "kept alive in case the backward pass needs it. The detach only "
        "removes the OUTPUT edge; the internal tape is still built. "
        "`no_grad` skips graph construction entirely — no activation "
        "tape, no edges, just the value. For a deep G with many "
        "intermediate tensors, this is a measurable VRAM win during "
        "training.\n\n"
        "**Why D's grads match exactly.** D's backward path only uses "
        "the VALUE of `fake`, not its grad_fn. Whether `fake` carries "
        "an autograd graph back to G's parameters or not is irrelevant "
        "to D's gradient computation — the forward through D is "
        "identical, and its backward only differentiates D's own "
        "weights w.r.t. the loss.\n\n"
        "**When `.detach()` is still preferable.** When G's forward "
        "output is used in BOTH a no-grad-needed pass (D-step) and a "
        "grad-needed pass (G-step) in the same step. `no_grad` is a "
        "block scope; `.detach()` is per-tensor. Mix them: build the "
        "graph once for the G-step, detach a copy for the D-step."
    ),
    "extra_imports": [],
}


# ===========================================================================
# SPEC 8 — holdout-data-one-per-class ex2 (seeded random pick)
# ===========================================================================

SPEC_RANDOM_HOLDOUT = {
    "atom_id": "holdout-data-one-per-class",
    "subtopic": "Generative: Hold-out one-per-class data",
    "topic_folder": TOPIC_GEN,
    "atom_recap_md": RECAP_RANDOM_HOLDOUT,
    "exercise_index": 2,
    "exercise_title": "seeded random one-per-class holdout for a stable gallery",
    "slug": "seeded-random-one-per-class-holdout",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["holdout", "random-pick", "seeded-generator", "nonzero"],
    "kcs": ["nonzero-mask-to-indices", "seeded-generator-for-reproducibility"],
    "lo": (
        "Apply a SEEDED local `t.Generator` to pick a random index per "
        "class via `(labels == c).nonzero(as_tuple=True)[0]`, then stack "
        "into a reproducible holdout gallery."
    ),
    "prompt_body": (
        "Implement `ex2_seeded_random_per_class(data, labels, "
        "num_classes, seed)`. Like ex1, but instead of taking the FIRST "
        "sample of each class, pick a deterministic-random one using a "
        "local seeded RNG:\n\n"
        "1. Build a local generator: `g = "
        "t.Generator().manual_seed(seed)`. Use a LOCAL generator (not "
        "the global RNG) so the pick is stable regardless of "
        "surrounding `torch.manual_seed` calls.\n"
        "2. For each class `c` in `range(num_classes)`:\n"
        "   - Find indices where `labels == c`: `idxs = (labels == c)."
        "nonzero(as_tuple=True)[0]`. This is a 1-D `int64` tensor of "
        "positions in `labels` whose value is `c`.\n"
        "   - Sample one index from `idxs` using the generator: `pick = "
        "t.randint(len(idxs), (1,), generator=g).item()`. This gives an "
        "index INTO `idxs`, so the actual data index is `idxs[pick]"
        ".item()`.\n"
        "   - Append `data[idxs[pick].item()]` to the per-class list.\n"
        "3. Stack: `return t.stack(per_class, dim=0)`. Shape "
        "`(num_classes, *sample_shape)`.\n\n"
        "Assume every class has at least one sample (no need to handle "
        "empty `idxs`).\n\n"
        "Input: `data` `(N, *)`, `labels` `(N,)` int64, `num_classes` "
        "int, `seed` int.\n"
        "Output: `(num_classes, *)` tensor.\n\n"
        "The visualization renders the seeded gallery and confirms that "
        "two calls with the same seed produce the IDENTICAL gallery, "
        "while different seeds typically produce different ones."
    ),
    "stub": (
        "def ex2_seeded_random_per_class(data: Tensor, labels: Tensor,\n"
        "                                num_classes: int, seed: int) -> Tensor:\n"
        '    """One random sample per class, seeded for reproducibility."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Build a dataset with multiple samples per class.\n"
        "rng = t.Generator().manual_seed(0)\n"
        "N, K = 60, 3\n"
        "data = t.randn(N, 4, generator=rng)\n"
        "# Repeated round-robin so every class has 20 samples.\n"
        "labels = t.arange(N) % K\n"
        "\n"
        "# Same seed → same gallery (reproducibility).\n"
        "h1 = ex2_seeded_random_per_class(data, labels, num_classes=K, seed=42)\n"
        "h2 = ex2_seeded_random_per_class(data, labels, num_classes=K, seed=42)\n"
        "assert h1.shape == (K, 4), f'expected (K, 4), got {tuple(h1.shape)}'\n"
        "assert t.equal(h1, h2), 'same seed must give bit-identical gallery'\n"
        "\n"
        "# Each holdout row must be a sample of the corresponding class.\n"
        "for c in range(K):\n"
        "    cls_samples = data[labels == c]   # (Nc, 4)\n"
        "    matches = (cls_samples == h1[c]).all(dim=1)   # (Nc,)\n"
        "    assert matches.any(), f'h1[{c}] is not present in class-{c} samples'\n"
        "\n"
        "# Local generator must NOT depend on global RNG state.\n"
        "t.manual_seed(123); _ = t.randn(50)   # mutate global RNG\n"
        "h3 = ex2_seeded_random_per_class(data, labels, K, seed=42)\n"
        "assert t.equal(h1, h3), (\n"
        "    'gallery must not depend on global RNG state — '\n"
        "    'did you call t.randint without a `generator=` arg?'\n"
        ")\n"
        "\n"
        "# Different seeds → typically different gallery (sanity check the seed actually drives the pick).\n"
        "h_alt = ex2_seeded_random_per_class(data, labels, K, seed=999)\n"
        "different_classes = sum(not t.equal(h1[c], h_alt[c]) for c in range(K))\n"
        "assert different_classes >= 1, 'different seeds should usually pick a different sample for at least one class'\n"
        "\n"
        "# Imbalanced case — class 0 has many samples, class 1 has only one.\n"
        "imb_labels = t.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0])   # cls 1 only at idx 6\n"
        "imb_data = t.arange(12).float().unsqueeze(1)\n"
        "h_imb = ex2_seeded_random_per_class(imb_data, imb_labels, num_classes=3, seed=1)\n"
        "assert h_imb.shape == (3, 1)\n"
        "# class 1 must be data[6] = 6 (only sample).\n"
        "assert h_imb[1].item() == 6.0, f'class 1 has unique sample at idx 6, got {h_imb[1].item()}'\n"
        "# class 2 must be data[9] = 9 (only sample).\n"
        "assert h_imb[2].item() == 9.0, f'class 2 has unique sample at idx 9, got {h_imb[2].item()}'\n"
        "# class 0 must be one of the 9 class-0 samples.\n"
        "class_0_vals = set(imb_data[imb_labels == 0].flatten().tolist())\n"
        "assert h_imb[0].item() in class_0_vals, f'class 0 pick {h_imb[0].item()} not in {class_0_vals}'\n"
        "\n"
        "# --- Visualization: gallery for two different seeds, side by side ---\n"
        "vrng = t.Generator().manual_seed(0)\n"
        "Nv, Kv = 100, 10\n"
        "vimgs = t.rand(Nv, 1, 16, 16, generator=vrng)\n"
        "vlbl = t.arange(Nv) % Kv\n"
        "g_a = ex2_seeded_random_per_class(vimgs, vlbl, Kv, seed=11)\n"
        "g_b = ex2_seeded_random_per_class(vimgs, vlbl, Kv, seed=22)\n"
        "fig, axes = plt.subplots(2, Kv, figsize=(1.2 * Kv, 3))\n"
        "for c in range(Kv):\n"
        "    axes[0, c].imshow(g_a[c, 0].numpy(), cmap='gray', vmin=0, vmax=1)\n"
        "    axes[0, c].set_title(f'cls{c}', fontsize=8); axes[0, c].axis('off')\n"
        "    axes[1, c].imshow(g_b[c, 0].numpy(), cmap='gray', vmin=0, vmax=1)\n"
        "    axes[1, c].axis('off')\n"
        "axes[0, 0].set_ylabel('seed=11', fontsize=9)\n"
        "axes[1, 0].set_ylabel('seed=22', fontsize=9)\n"
        "plt.suptitle('ex2 seeded random one-per-class holdout — different seed → different gallery')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "solution_body": (
        "def ex2_seeded_random_per_class(data: Tensor, labels: Tensor,\n"
        "                                num_classes: int, seed: int) -> Tensor:\n"
        "    g = t.Generator().manual_seed(seed)\n"
        "    per_class = []\n"
        "    for c in range(num_classes):\n"
        "        idxs = (labels == c).nonzero(as_tuple=True)[0]\n"
        "        pick = t.randint(len(idxs), (1,), generator=g).item()\n"
        "        per_class.append(data[idxs[pick].item()])\n"
        "    return t.stack(per_class, dim=0)"
    ),
    "solution_notes": (
        "**Local generator is the only way to be reproducible.** The "
        "global RNG is shared with every other piece of torch code in "
        "your script — data augmentation, dropout, weight init. Any of "
        "those moving by a tick reorders the global state, which "
        "silently shifts your gallery. A local `t.Generator()` is "
        "insulated.\n\n"
        "**`nonzero(as_tuple=True)[0]` vs `nonzero()`.** Tuple form "
        "returns one 1-D tensor per axis of the input. For a 1-D mask "
        "the tuple has length 1, so `[0]` extracts it. The non-tuple "
        "form returns a `(K, 1)` 2-D tensor — usable, but awkward.\n\n"
        "**Why `t.randint(len(idxs), (1,))` not `t.randperm(len(idxs))"
        "[0]`.** Both correct, but `randint` is `O(1)` while `randperm` "
        "is `O(N)`. For a tight per-class loop on many classes, the "
        "difference is measurable.\n\n"
        "**Imbalanced robustness.** When a class has only one sample, "
        "`t.randint(1, ...)` returns `0` deterministically — the unique "
        "sample. The test's imbalanced case asserts this; no special "
        "case in the code."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


# ---------------------------------------------------------------------------
# All specs
# ---------------------------------------------------------------------------

SPECS = [
    SPEC_RANK0_BARRIER,
    SPEC_GATHER_MEDIAN,
    SPEC_HARMONIC,
    SPEC_TIED_DECODE,
    SPEC_EMBED_SPARSE,
    SPEC_WRAPPER_TRAIN_EVAL,
    SPEC_NO_GRAD,
    SPEC_RANDOM_HOLDOUT,
]


def _verify_all(specs):
    import torch as t
    import numpy as np
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torch import Tensor
    import einops
    from einops import rearrange, reduce, repeat
    import threading

    passed = 0
    failed = []

    for spec in specs:
        ex_id = f"ex{spec['exercise_index']}"
        tag = f"{spec['atom_id']}/{ex_id}"
        ns = {
            "t": t,
            "np": np,
            "nn": nn,
            "Tensor": Tensor,
            "plt": plt,
            "einops": einops,
            "rearrange": rearrange,
            "reduce": reduce,
            "repeat": repeat,
            "threading": threading,
            "_dd_passed": set(),
            "__name__": "__main__",
        }
        t.manual_seed(0)
        np.random.seed(0)

        try:
            exec(spec["stub"], ns)
        except Exception:
            pass

        try:
            exec(spec["solution_body"], ns)
            exec(spec["test_body"], ns)
        except Exception as e:
            failed.append((tag, repr(e), traceback.format_exc()))
            plt.close("all")
            continue
        plt.close("all")
        passed += 1
        print(f"  [verify] {tag}: ok")

    print(f"\n[verify] {passed}/{len(specs)} specs passed")
    if failed:
        for tag, err, tb in failed:
            print(f"\n--- FAILED: {tag} ---")
            print(err)
            print(tb)
        raise SystemExit(1)


def main():
    print(f"[deepening_s_batch11] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_s_batch11] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_s_batch11] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
