#!/usr/bin/env python3
"""Author 8 standalone Colab drills — DDP/distributed + geometry cleanup atoms (batch 7).

PS4 framing: each drill exercises ONE atomic skill plucked out of a larger
ARENA composite. The 4 distributed atoms underpin chap-0 part-3 (DistResNet
training loop); the 4 geometry/linalg atoms close gaps in chap-0 part-1
(ray tracing).

Distributed mocking strategy:
- Real ranks via `mp.get_context('fork').Process` + `gloo` backend where the
  test asserts collective-correct values across processes (broadcast,
  all_reduce eval metrics, model save, reduce-vs-gather).
- All tests run on backend venv torch 2.12.0+cpu (no CUDA needed).

Geometry / linalg drills are single-process pure torch — no mp harness.

Atoms covered (8 × single-exercise):
1. broadcast-initial-weights      — Distributed: broadcast initial weights
2. all-reduce-eval-metrics        — Distributed: all_reduce eval metrics
3. model-save-state-dict          — Distributed: model save state_dict rank-0
4. reduce-gather-sum              — Distributed: reduce.gather + sum
5. segment-line-intersect-2d      — Geometry: Segment-line intersect 2-D
6. rotation-matrix-3d             — Geometry: Rotation matrix 3-D (full)
7. try-except-solve               — LinAlg: try/except solve
8. cross-product-normal           — Geometry: Cross-product surface normal
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC_DIST = "prereqs_distributed"
TOPIC_GEOM = "prereqs_geometry_cnn"

# ---------------------------------------------------------------------------
# Distributed shared bits
# ---------------------------------------------------------------------------

_FORK_HARNESS = '''
import os as _os
import datetime as _dt
import torch.distributed as _dist
import torch.multiprocessing as _mp

def _dd_run_workers(worker_fn, world_size, port, *extra_args, timeout=30):
    """Spawn `world_size` fork-context procs, return list of exitcodes."""
    ctx = _mp.get_context('fork')
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=worker_fn, args=(rank, world_size, port, *extra_args))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=timeout)
    codes = [p.exitcode for p in procs]
    for p in procs:
        if p.is_alive():
            p.terminate()
    return codes
'''.strip()


RECAP_DIST_CORE = (
    "## torch.distributed quick refresher\n"
    "\n"
    "PyTorch's collective-communication library (`torch.distributed`, aliased "
    "`dist`) lets multiple processes coordinate over tensors. Each rank runs "
    "the same function in its own process; collectives operate in-place on "
    "tensors of identical shape across ranks.\n"
    "\n"
    "**Backends.** `'nccl'` for multi-GPU (ARENA's setup), `'gloo'` for CPU "
    "(what these drills use — Colab CPU runtimes have no GPUs).\n"
    "\n"
    "**Launch pattern.** Each test uses `mp.get_context('fork').Process` so "
    "worker fns defined in a notebook cell are picklable. Workers init the "
    "group, do their work, push results onto a `manager.Queue`, then "
    "destroy the group."
)


# ---------------------------------------------------------------------------
# 1. broadcast-initial-weights
# ---------------------------------------------------------------------------

SPEC_BROADCAST_WEIGHTS = {
    "atom_id": "broadcast-initial-weights",
    "subtopic": "Distributed: broadcast initial weights",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_DIST_CORE + (
        "\n\n### This drill's atom: broadcast initial weights at training start\n"
        "Every rank constructs its own `nn.Module` — but stochastic init means "
        "each rank ends up with DIFFERENT random weights. For data-parallel "
        "training to be valid, all ranks must start from the SAME parameters. "
        "The standard pattern:\n"
        "```python\n"
        "model = MyModel()  # each rank has random weights\n"
        "for p in model.parameters():\n"
        "    dist.broadcast(p.data, src=0)\n"
        "# now every rank's params == rank 0's params\n"
        "```\n"
        "After this, `loss.backward()` + grad-sync keep them in lockstep forever."
    ),
    "exercise_index": 1,
    "exercise_title": "broadcast rank-0 weights to all ranks at start",
    "slug": "broadcast-rank-0-weights-to-all-ranks-at-start",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["broadcast", "DDP-init", "parameter-sync", "rank-0-source"],
    "kcs": ["loop-parameters-broadcast", "broadcast-src-rank-0"],
    "lo": (
        "Apply `dist.broadcast(p.data, src=0)` over every parameter to "
        "synchronize all ranks to rank 0's initial weights at the start of "
        "data-parallel training."
    ),
    "prompt_body": (
        "Implement `ex1_broadcast_init_worker(rank, world_size, port, out_queue)`. "
        "Each rank:\n\n"
        "1. Inits `gloo`.\n"
        "2. Builds a fresh model whose params depend on `rank` (this is the "
        "test fixture — each rank gets *different* initial weights to expose "
        "the bug if broadcast is skipped):\n"
        "   ```python\n"
        "   model = t.nn.Linear(2, 2, bias=False)\n"
        "   with t.no_grad():\n"
        "       model.weight.fill_(float(rank + 1))\n"
        "   ```\n"
        "   So rank 0 has weights `[[1,1],[1,1]]`, rank 1 has `[[2,2],[2,2]]`.\n"
        "3. **Broadcast rank-0 weights to every other rank:**\n"
        "   ```python\n"
        "   for p in model.parameters():\n"
        "       dist.broadcast(p.data, src=0)\n"
        "   ```\n"
        "4. Pushes `(rank, model.weight.detach().flatten().tolist())` onto "
        "`out_queue`.\n"
        "5. Destroys process group.\n\n"
        "**Expected.** Pre-broadcast: rank 0 = `[1,1,1,1]`, rank 1 = `[2,2,2,2]`. "
        "Post-broadcast: BOTH ranks hold `[1,1,1,1]` (rank 0's values won)."
    ),
    "stub": (
        "import os, datetime\n"
        "import torch as t\n"
        "import torch.distributed as dist\n"
        "\n"
        "def ex1_broadcast_init_worker(rank, world_size, port, out_queue):\n"
        '    """Init gloo, broadcast rank-0 weights to all ranks, queue result."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FORK_HARNESS + "\n\n"
        "manager = _mp.Manager()\n"
        "q = manager.Queue()\n"
        "codes = _dd_run_workers(ex1_broadcast_init_worker, 2, 29610, q)\n"
        "assert codes == [0, 0], f'workers failed: {codes}'\n"
        "\n"
        "results = {}\n"
        "while not q.empty():\n"
        "    rank, weights = q.get()\n"
        "    results[rank] = weights\n"
        "assert set(results.keys()) == {0, 1}, f'expected ranks {{0,1}}, got {set(results.keys())}'\n"
        "# Both ranks should hold rank-0's original weights (all 1.0).\n"
        "for rank, w in results.items():\n"
        "    assert len(w) == 4, f'rank {rank}: expected 4 weights, got {len(w)}'\n"
        "    for v in w:\n"
        "        assert abs(v - 1.0) < 1e-6, f'rank {rank}: expected 1.0, got {v} (broadcast skipped?)'\n"
        "\n"
        "# 3-rank case — every rank should still converge to rank-0's [1,1,1,1].\n"
        "q3 = manager.Queue()\n"
        "codes3 = _dd_run_workers(ex1_broadcast_init_worker, 3, 29611, q3)\n"
        "assert codes3 == [0, 0, 0], f'3-rank workers failed: {codes3}'\n"
        "results3 = {}\n"
        "while not q3.empty():\n"
        "    rank, w = q3.get()\n"
        "    results3[rank] = w\n"
        "assert set(results3.keys()) == {0, 1, 2}\n"
        "for rank, w in results3.items():\n"
        "    for v in w:\n"
        "        assert abs(v - 1.0) < 1e-6, f'3-rank case, rank {rank}: got {v}'"
    ),
    "solution_body": (
        "def ex1_broadcast_init_worker(rank, world_size, port, out_queue):\n"
        "    os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        "    os.environ['MASTER_PORT'] = str(port)\n"
        "    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size,\n"
        "                            timeout=datetime.timedelta(seconds=20))\n"
        "    model = t.nn.Linear(2, 2, bias=False)\n"
        "    with t.no_grad():\n"
        "        model.weight.fill_(float(rank + 1))\n"
        "    # Sync initial weights — every rank now mirrors rank 0.\n"
        "    for p in model.parameters():\n"
        "        dist.broadcast(p.data, src=0)\n"
        "    out_queue.put((rank, model.weight.detach().flatten().tolist()))\n"
        "    dist.destroy_process_group()"
    ),
    "solution_notes": (
        "**Why `p.data` not `p`.** `dist.broadcast` operates on a `Tensor`, "
        "but `nn.Parameter` is a `Tensor` subclass. Using `p.data` strips the "
        "autograd machinery and gives you the raw storage — broadcast writes "
        "in-place into that storage. Using `p` directly works in modern PyTorch "
        "but is conventionally avoided to make the in-place mutation explicit.\n\n"
        "**Why broadcast not all_reduce.** `all_reduce` SUMS across ranks — "
        "you'd get `(1+2)/2 = 1.5` weights, not rank 0's `1.0`. Broadcast picks "
        "ONE source rank and copies its tensor to all others.\n\n"
        "**Alternative: seed every rank identically.** `t.manual_seed(0)` "
        "before the model constructor *would* give matching weights — but only "
        "if every rank uses the same code path. Broadcast is the robust pattern: "
        "it works even when ranks load different checkpoints, use different "
        "init strategies, or boot from non-deterministic ops."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# 2. all-reduce-eval-metrics
# ---------------------------------------------------------------------------

SPEC_EVAL_METRICS = {
    "atom_id": "all-reduce-eval-metrics",
    "subtopic": "Distributed: all_reduce eval metrics",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_DIST_CORE + (
        "\n\n### This drill's atom: average eval metrics across ranks\n"
        "During distributed evaluation, each rank computes a metric (loss, "
        "accuracy) on its own shard of the validation set. To get the global "
        "metric you must average across ranks:\n"
        "```python\n"
        "local_loss = compute_loss(local_batch)\n"
        "loss_t = t.tensor([local_loss])\n"
        "dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)\n"
        "loss_t /= world_size              # convert SUM → MEAN\n"
        "global_loss = loss_t.item()\n"
        "```\n"
        "**Why wrap in a tensor.** `dist.all_reduce` requires a tensor input — "
        "raw Python floats can't be reduced directly. Build a singleton "
        "tensor, reduce it, unwrap with `.item()`.\n\n"
        "**Why SUM-then-divide vs MEAN.** PyTorch's `ReduceOp` has no `MEAN` — "
        "the canonical idiom is sum then divide by `world_size`."
    ),
    "exercise_index": 1,
    "exercise_title": "average a per-rank eval loss across ranks",
    "slug": "average-a-per-rank-eval-loss-across-ranks",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["all_reduce", "eval", "metric", "mean", "wrap-scalar-tensor"],
    "kcs": ["wrap-float-in-tensor-for-reduce", "sum-then-divide-by-world-size"],
    "lo": (
        "Apply the `dist.all_reduce(SUM) / world_size` pattern to average a "
        "per-rank scalar eval loss into a global mean visible on every rank."
    ),
    "prompt_body": (
        "Implement `ex1_eval_metric_worker(rank, world_size, port, out_queue)`. "
        "Each rank:\n\n"
        "1. Inits `gloo`.\n"
        "2. Computes a fake per-rank loss: `local_loss = float(rank + 1)`. "
        "(Rank 0 → 1.0, rank 1 → 2.0, rank 2 → 3.0.)\n"
        "3. **Averages across ranks via all_reduce + divide:**\n"
        "   ```python\n"
        "   loss_t = t.tensor([local_loss])\n"
        "   dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)\n"
        "   loss_t /= world_size\n"
        "   global_loss = loss_t.item()\n"
        "   ```\n"
        "4. Pushes `(rank, global_loss)` onto `out_queue`.\n"
        "5. Destroys process group.\n\n"
        "**Expected.** With `world_size=3`, mean of `[1, 2, 3]` = `2.0` on "
        "every rank. With `world_size=2`, mean of `[1, 2]` = `1.5`."
    ),
    "stub": (
        "import os, datetime\n"
        "import torch as t\n"
        "import torch.distributed as dist\n"
        "\n"
        "def ex1_eval_metric_worker(rank, world_size, port, out_queue):\n"
        '    """Init gloo, average a fake per-rank loss across ranks, queue it."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FORK_HARNESS + "\n\n"
        "manager = _mp.Manager()\n"
        "q = manager.Queue()\n"
        "codes = _dd_run_workers(ex1_eval_metric_worker, 3, 29612, q)\n"
        "assert codes == [0, 0, 0], f'workers failed: {codes}'\n"
        "\n"
        "results = {}\n"
        "while not q.empty():\n"
        "    rank, val = q.get()\n"
        "    results[rank] = val\n"
        "assert set(results.keys()) == {0, 1, 2}\n"
        "expected = (1.0 + 2.0 + 3.0) / 3\n"
        "for rank, val in results.items():\n"
        "    assert abs(val - expected) < 1e-5, (\n"
        "        f'rank {rank}: got {val}, expected {expected} '\n"
        "        f'(SUM={val * 3:.3f}? — forgot to divide by world_size)'\n"
        "    )\n"
        "\n"
        "# 2-rank case: mean of [1, 2] = 1.5\n"
        "q2 = manager.Queue()\n"
        "codes2 = _dd_run_workers(ex1_eval_metric_worker, 2, 29613, q2)\n"
        "assert codes2 == [0, 0]\n"
        "results2 = {}\n"
        "while not q2.empty():\n"
        "    rank, val = q2.get()\n"
        "    results2[rank] = val\n"
        "for rank, val in results2.items():\n"
        "    assert abs(val - 1.5) < 1e-5, f'2-rank rank {rank}: got {val}, expected 1.5'"
    ),
    "solution_body": (
        "def ex1_eval_metric_worker(rank, world_size, port, out_queue):\n"
        "    os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        "    os.environ['MASTER_PORT'] = str(port)\n"
        "    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size,\n"
        "                            timeout=datetime.timedelta(seconds=20))\n"
        "    local_loss = float(rank + 1)\n"
        "    loss_t = t.tensor([local_loss])\n"
        "    dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)\n"
        "    loss_t /= world_size\n"
        "    out_queue.put((rank, loss_t.item()))\n"
        "    dist.destroy_process_group()"
    ),
    "solution_notes": (
        "**Trap: forgetting the divide.** The most common bug. You'll see "
        "training logs where 'eval loss' silently scales with `world_size`. "
        "The fix is one line; the bug is invisible until someone notices the "
        "loss curve looks weirdly stable across scaling experiments.\n\n"
        "**For batch-weighted metrics.** If ranks have unequal batch sizes "
        "(last batch problem), do a SUM all_reduce on numerator AND on count, "
        "then divide. Plain mean would over-weight ranks with smaller batches.\n\n"
        "**Why on every rank not just rank-0.** `all_reduce` puts the result "
        "on EVERY rank (that's the 'all' part vs `reduce` to one). This is "
        "useful when downstream logic — early stopping, LR scheduling, "
        "checkpointing — needs the global metric on every rank."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# 3. model-save-state-dict (rank-0 only + barrier)
# ---------------------------------------------------------------------------

SPEC_MODEL_SAVE = {
    "atom_id": "model-save-state-dict",
    "subtopic": "Distributed: model save state_dict rank-0",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_DIST_CORE + (
        "\n\n### This drill's atom: save state_dict from rank 0 only\n"
        "After grad-sync, every rank holds identical parameters. Saving a "
        "checkpoint from EVERY rank would (a) waste IO bandwidth, (b) race "
        "on the file path. The standard pattern:\n"
        "```python\n"
        "if rank == 0:\n"
        "    t.save(model.state_dict(), checkpoint_path)\n"
        "dist.barrier()  # other ranks wait for rank 0 to finish\n"
        "```\n"
        "**Why the barrier.** Without it, rank 1 might charge ahead into the "
        "next epoch before rank 0 finishes writing — fine on its own, but if "
        "code later does `if rank == 1: load(checkpoint)` you'll race. "
        "`dist.barrier()` makes EVERY rank stop until all ranks reach it."
    ),
    "exercise_index": 1,
    "exercise_title": "save state_dict from rank 0 with barrier",
    "slug": "save-state-dict-from-rank-0-with-barrier",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["state_dict", "checkpoint", "rank-0-only", "barrier", "side-effect"],
    "kcs": ["rank0-only-side-effects", "dist-barrier-synchronization"],
    "lo": (
        "Apply the `if rank == 0: save; dist.barrier()` pattern so a single "
        "process writes the checkpoint while all other ranks wait at the "
        "barrier before continuing."
    ),
    "prompt_body": (
        "Implement `ex1_save_checkpoint_worker(rank, world_size, port, "
        "ckpt_path, out_queue)`. Each rank:\n\n"
        "1. Inits `gloo`.\n"
        "2. Builds an identical model (rank-independent for this drill):\n"
        "   ```python\n"
        "   model = t.nn.Linear(2, 2, bias=False)\n"
        "   with t.no_grad():\n"
        "       model.weight.fill_(7.0)\n"
        "   ```\n"
        "3. **Saves the state_dict from rank 0 only, then barriers:**\n"
        "   ```python\n"
        "   if rank == 0:\n"
        "       t.save(model.state_dict(), ckpt_path)\n"
        "   dist.barrier()\n"
        "   ```\n"
        "4. After the barrier, EVERY rank reads back the file to confirm it "
        "exists and contains the expected tensor:\n"
        "   ```python\n"
        "   sd = t.load(ckpt_path, weights_only=True)\n"
        "   out_queue.put((rank, sd['weight'].sum().item()))\n"
        "   ```\n"
        "5. Destroys process group.\n\n"
        "**Expected.** After the barrier, every rank can load `ckpt_path` "
        "and the loaded weights sum to `7.0 * 4 = 28.0` (2×2 matrix filled "
        "with 7s)."
    ),
    "stub": (
        "import os, datetime\n"
        "import torch as t\n"
        "import torch.distributed as dist\n"
        "\n"
        "def ex1_save_checkpoint_worker(rank, world_size, port, ckpt_path, out_queue):\n"
        '    """Init gloo, rank-0 saves state_dict, barrier, every rank loads + reports."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FORK_HARNESS + "\n\n"
        "import tempfile, pathlib\n"
        "tmpdir = tempfile.mkdtemp(prefix='dd_ckpt_')\n"
        "ckpt = str(pathlib.Path(tmpdir) / 'model.pt')\n"
        "\n"
        "manager = _mp.Manager()\n"
        "q = manager.Queue()\n"
        "codes = _dd_run_workers(ex1_save_checkpoint_worker, 2, 29614, ckpt, q)\n"
        "assert codes == [0, 0], f'workers failed: {codes}'\n"
        "\n"
        "# File must exist on disk after workers finish.\n"
        "assert pathlib.Path(ckpt).exists(), f'checkpoint not written: {ckpt}'\n"
        "\n"
        "results = {}\n"
        "while not q.empty():\n"
        "    rank, weight_sum = q.get()\n"
        "    results[rank] = weight_sum\n"
        "assert set(results.keys()) == {0, 1}\n"
        "# Both ranks must have successfully loaded the file.\n"
        "for rank, val in results.items():\n"
        "    assert abs(val - 28.0) < 1e-5, (\n"
        "        f'rank {rank}: expected sum=28.0 (2x2 of 7s), got {val} '\n"
        "        f'(load failed? barrier missing?)'\n"
        "    )\n"
        "\n"
        "# Re-run with 3 ranks at a fresh path — same contract.\n"
        "ckpt2 = str(pathlib.Path(tmpdir) / 'model2.pt')\n"
        "q2 = manager.Queue()\n"
        "codes2 = _dd_run_workers(ex1_save_checkpoint_worker, 3, 29615, ckpt2, q2)\n"
        "assert codes2 == [0, 0, 0]\n"
        "assert pathlib.Path(ckpt2).exists()\n"
        "results2 = {}\n"
        "while not q2.empty():\n"
        "    rank, val = q2.get()\n"
        "    results2[rank] = val\n"
        "assert set(results2.keys()) == {0, 1, 2}\n"
        "for rank, val in results2.items():\n"
        "    assert abs(val - 28.0) < 1e-5, f'3-rank rank {rank}: got {val}'"
    ),
    "solution_body": (
        "def ex1_save_checkpoint_worker(rank, world_size, port, ckpt_path, out_queue):\n"
        "    os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        "    os.environ['MASTER_PORT'] = str(port)\n"
        "    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size,\n"
        "                            timeout=datetime.timedelta(seconds=20))\n"
        "    model = t.nn.Linear(2, 2, bias=False)\n"
        "    with t.no_grad():\n"
        "        model.weight.fill_(7.0)\n"
        "    if rank == 0:\n"
        "        t.save(model.state_dict(), ckpt_path)\n"
        "    dist.barrier()  # other ranks wait until rank-0 finishes the write\n"
        "    sd = t.load(ckpt_path, weights_only=True)\n"
        "    out_queue.put((rank, sd['weight'].sum().item()))\n"
        "    dist.destroy_process_group()"
    ),
    "solution_notes": (
        "**Why rank 0 only.** Disk IO from N ranks to the same path = N "
        "concurrent writes, undefined contents, possibly corrupted file. "
        "Even when writing to different paths, save time scales with N — "
        "wasteful when ranks 1..N-1 hold identical tensors.\n\n"
        "**Why the barrier.** Without `dist.barrier()`, rank 1 might reach "
        "the next collective op (say, `dist.all_reduce` in the next epoch) "
        "before rank 0 finishes saving. Real DDP usually survives this — "
        "but ANY code that depends on the file existing (e.g. immediate "
        "reload, an external evaluator) will race.\n\n"
        "**`weights_only=True` since PyTorch 2.6.** Mandatory for security: "
        "`t.load` used to allow arbitrary code execution via pickle, now "
        "the default refuses unless you opt in. For state_dicts, "
        "`weights_only=True` is always safe."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# 4. reduce-gather-sum  (reduce vs gather — KEY contrast)
# ---------------------------------------------------------------------------

SPEC_REDUCE_GATHER = {
    "atom_id": "reduce-gather-sum",
    "subtopic": "Distributed: reduce.gather + sum",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_DIST_CORE + (
        "\n\n### This drill's atom: `reduce` vs `gather`\n"
        "Both collect tensors from all ranks to ONE destination rank — but they "
        "do different things with them:\n"
        "\n"
        "**`dist.reduce(tensor, dst=0, op=SUM)`** — every rank contributes its "
        "tensor; the destination rank ends with the *combined* value (sum, "
        "max, etc.). On non-dst ranks, the tensor is left in an "
        "**implementation-defined state** (it may hold partial sums from the "
        "tree-reduction; do NOT use the value). Memory: dst holds ONE tensor.\n"
        "\n"
        "**`dist.gather(tensor, gather_list=[...], dst=0)`** — every rank "
        "contributes its tensor; the destination rank ends with a *list* of "
        "all N tensors (one per rank, in rank order). Memory: dst holds N "
        "tensors. From there you can `sum`, `mean`, take min, take median — "
        "anything `reduce` can't.\n"
        "\n"
        "**Rule of thumb.** Use `reduce` if you only need the aggregate (and "
        "only consume the value on the dst rank). Use `gather` if you need "
        "the per-rank values (e.g., to compute a median, or log per-rank "
        "loss separately). `gather` uses N× the memory."
    ),
    "exercise_index": 1,
    "exercise_title": "compare reduce vs gather + manual sum",
    "slug": "compare-reduce-vs-gather-plus-manual-sum",
    "bloom_level": "Analyze",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["reduce", "gather", "dst", "gather_list", "aggregation"],
    "kcs": ["reduce-collapses-to-aggregate", "gather-preserves-per-rank-values"],
    "lo": (
        "Apply both `dist.reduce(SUM)` and `dist.gather` + manual list sum at "
        "rank 0 to confirm they produce the same total but differ in what "
        "intermediate values are accessible."
    ),
    "prompt_body": (
        "Implement `ex1_reduce_vs_gather_worker(rank, world_size, port, "
        "out_queue)`. Each rank starts with `tensor = t.tensor([float(rank "
        "+ 1)])`.\n\n"
        "On EVERY rank, do BOTH collective patterns sequentially:\n\n"
        "**Path A — `reduce`:**\n"
        "```python\n"
        "a = t.tensor([float(rank + 1)])\n"
        "dist.reduce(a, dst=0, op=dist.ReduceOp.SUM)\n"
        "# rank 0 now holds the SUM; other ranks hold their ORIGINAL value\n"
        "```\n"
        "\n"
        "**Path B — `gather` + manual sum:**\n"
        "```python\n"
        "b = t.tensor([float(rank + 1)])\n"
        "if rank == 0:\n"
        "    gather_list = [t.zeros(1) for _ in range(world_size)]\n"
        "else:\n"
        "    gather_list = None\n"
        "dist.gather(b, gather_list=gather_list, dst=0)\n"
        "if rank == 0:\n"
        "    gathered_sum = sum(g.item() for g in gather_list)\n"
        "    per_rank = [g.item() for g in gather_list]\n"
        "else:\n"
        "    gathered_sum = None\n"
        "    per_rank = None\n"
        "```\n"
        "\n"
        "Push `(rank, a.item(), gathered_sum, per_rank)` onto `out_queue`. "
        "On rank>0, `gathered_sum` and `per_rank` will be `None`.\n\n"
        "**Expected (world_size=3, inputs [1,2,3]).** Rank 0: `a=6.0` (the "
        "sum), `gathered_sum=6.0`, `per_rank=[1.0,2.0,3.0]`. On non-dst "
        "ranks (1, 2) the `a` value is implementation-defined garbage — the "
        "test does NOT assert what it is, only that it is NOT silently the "
        "sum (which would mean you accidentally used `all_reduce`)."
    ),
    "stub": (
        "import os, datetime\n"
        "import torch as t\n"
        "import torch.distributed as dist\n"
        "\n"
        "def ex1_reduce_vs_gather_worker(rank, world_size, port, out_queue):\n"
        '    """Init gloo, run BOTH reduce and gather on the same tensor, queue both outcomes."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FORK_HARNESS + "\n\n"
        "manager = _mp.Manager()\n"
        "q = manager.Queue()\n"
        "codes = _dd_run_workers(ex1_reduce_vs_gather_worker, 3, 29616, q)\n"
        "assert codes == [0, 0, 0], f'workers failed: {codes}'\n"
        "\n"
        "results = {}\n"
        "while not q.empty():\n"
        "    rank, a_val, gathered_sum, per_rank = q.get()\n"
        "    results[rank] = (a_val, gathered_sum, per_rank)\n"
        "\n"
        "assert set(results.keys()) == {0, 1, 2}\n"
        "\n"
        "# --- Rank 0: reduce result + gather list + manual sum ---\n"
        "a0, gs0, pr0 = results[0]\n"
        "assert abs(a0 - 6.0) < 1e-6, f'rank 0 reduce: expected 6.0, got {a0}'\n"
        "assert gs0 is not None and abs(gs0 - 6.0) < 1e-6, (\n"
        "    f'rank 0 gather sum: expected 6.0, got {gs0}'\n"
        ")\n"
        "assert pr0 == [1.0, 2.0, 3.0], (\n"
        "    f'rank 0 per_rank list: expected [1.0, 2.0, 3.0], got {pr0} '\n"
        "    f'(gather preserves per-rank values in rank order)'\n"
        ")\n"
        "\n"
        "# --- Rank 1, rank 2: reduce leaves non-dst tensors in an\n"
        "# implementation-defined state. We DON'T assert a specific value;\n"
        "# we ONLY assert the rank still ran (got past reduce) and that\n"
        "# gather metadata is None on non-dst ranks (key contrast vs all_*).\n"
        "for non_dst in (1, 2):\n"
        "    a_v, gs_v, pr_v = results[non_dst]\n"
        "    assert isinstance(a_v, float), (\n"
        "        f'rank {non_dst} reduce: expected a scalar float (non-dst value is\\n'\n"
        "        f'implementation-defined, but the call must still complete), got {a_v!r}'\n"
        "    )\n"
        "    assert gs_v is None, (\n"
        "        f'rank {non_dst} gather sum should be None — only dst rank computes the sum '\n"
        "        f'(if not None, you ran the sum on all ranks → that\\'s all_gather behavior, not gather)'\n"
        "    )\n"
        "    assert pr_v is None, f'rank {non_dst} per_rank should be None, got {pr_v!r}'"
    ),
    "solution_body": (
        "def ex1_reduce_vs_gather_worker(rank, world_size, port, out_queue):\n"
        "    os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        "    os.environ['MASTER_PORT'] = str(port)\n"
        "    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size,\n"
        "                            timeout=datetime.timedelta(seconds=20))\n"
        "    # --- Path A: reduce collapses to aggregate at dst=0 only ---\n"
        "    a = t.tensor([float(rank + 1)])\n"
        "    dist.reduce(a, dst=0, op=dist.ReduceOp.SUM)\n"
        "    # --- Path B: gather preserves per-rank values at dst=0 only ---\n"
        "    b = t.tensor([float(rank + 1)])\n"
        "    if rank == 0:\n"
        "        gather_list = [t.zeros(1) for _ in range(world_size)]\n"
        "    else:\n"
        "        gather_list = None\n"
        "    dist.gather(b, gather_list=gather_list, dst=0)\n"
        "    if rank == 0:\n"
        "        gathered_sum = sum(g.item() for g in gather_list)\n"
        "        per_rank = [g.item() for g in gather_list]\n"
        "    else:\n"
        "        gathered_sum = None\n"
        "        per_rank = None\n"
        "    out_queue.put((rank, a.item(), gathered_sum, per_rank))\n"
        "    dist.destroy_process_group()"
    ),
    "solution_notes": (
        "**Key insight: only dst is guaranteed.** Beginners often assume "
        "every rank gets the sum after `reduce` — that's `all_reduce`, not "
        "`reduce`. With `reduce`, ONLY the dst rank's tensor is guaranteed "
        "to hold the aggregate. Non-dst ranks may hold their original "
        "value, a partial sum from the tree-reduction, or whatever the "
        "backend's implementation produced. **Never read the non-dst "
        "tensor.** This drill enforces that: the test does not assert the "
        "non-dst value, only that the call returned a scalar.\n\n"
        "**Different backends, different garbage.** On `gloo` you'll often "
        "see tree-reduction partials on non-dst ranks (e.g. for "
        "world_size=3 with inputs [1,2,3], rank 1 might end with 5.0 = "
        "1+(2*2) due to pairwise reduce). On `nccl` you may see the "
        "original value untouched. Both are correct per the API contract.\n\n"
        "**When to choose `gather`.** Any aggregation that's not a binary "
        "op: median, percentile, min-with-rank-tag, top-k. The trade-off is "
        "memory: gather_list at dst holds N tensors; reduce holds 1.\n\n"
        "**Symmetric variants.** `all_reduce` = reduce-to-everyone (the sum "
        "IS guaranteed on all ranks). `all_gather` = gather-to-everyone "
        "(each rank ends with the same length-N list). Pick based on "
        "whether downstream code needs the result on every rank or just on "
        "rank 0."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# 5. segment-line-intersect-2d
# ---------------------------------------------------------------------------

RECAP_SEGLINE = (
    "## 2-D segment vs line intersection — quick refresher\n"
    "\n"
    "**Parametric form.** A segment from `P` to `P + d` (direction = endpoint - start) "
    "has points `P + t*d` for `t ∈ [0, 1]`. A line through `Q` with direction `e` has "
    "points `Q + s*e` for `s ∈ ℝ` (no bound).\n"
    "\n"
    "**Intersection.** Solve `P + t*d = Q + s*e`, i.e. `t*d - s*e = Q - P`. In matrix "
    "form, stack `d` and `-e` as columns:\n"
    "```\n"
    "[d_x  -e_x] [t]   [Q_x - P_x]\n"
    "[d_y  -e_y] [s] = [Q_y - P_y]\n"
    "```\n"
    "Solve via Cramer's rule (or `torch.linalg.solve`). If the determinant is zero, "
    "segment and line are parallel — no unique intersection.\n"
    "\n"
    "**Hit test.** The segment hits the line iff a solution exists AND `0 ≤ t ≤ 1`. "
    "`s` is unconstrained (the line is infinite). Returning `(t, s, hit_bool)` is the "
    "canonical signature."
)


SPEC_SEGLINE = {
    "atom_id": "segment-line-intersect-2d",
    "subtopic": "Geometry: Segment-line intersect 2-D",
    "topic_folder": TOPIC_GEOM,
    "atom_recap_md": RECAP_SEGLINE,
    "exercise_index": 1,
    "exercise_title": "intersect a segment with an infinite line in 2-D",
    "slug": "intersect-a-segment-with-an-infinite-line-in-2d",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["2D-geometry", "cramer", "linalg-solve", "parametric"],
    "kcs": ["segment-line-as-linear-system", "t-in-zero-one-hit-test"],
    "lo": (
        "Apply the parametric-form linear system (with `torch.linalg.solve`) "
        "to find segment-vs-line intersection in 2-D and return whether the "
        "segment actually crosses the line."
    ),
    "prompt_body": (
        "Implement `ex1_seg_line_intersect(P, Q1, Q2, R1, R2)`. Inputs are "
        "all 2-D points:\n"
        "- `P`, `Q1, Q2`: the SEGMENT goes from `Q1` to `Q2`; ignore `P` "
        "(legacy name compat).\n"
        "  Actually let's simplify: drop unused names.\n\n"
        "Let's use clean names. Inputs:\n"
        "- `S0`, `S1`: 2-D endpoints of the segment.\n"
        "- `L0`, `L1`: 2-D points defining the infinite line.\n\n"
        "Steps:\n"
        "1. Direction of segment: `d = S1 - S0`. Direction of line: "
        "`e = L1 - L0`.\n"
        "2. Solve `t*d - s*e = L0 - S0` as a 2×2 linear system:\n"
        "   ```python\n"
        "   A = t.stack([d, -e], dim=1)  # shape (2, 2)\n"
        "   b = L0 - S0                  # shape (2,)\n"
        "   ts = t.linalg.solve(A, b)    # ts = [t, s]\n"
        "   ```\n"
        "3. Compute `hit = (ts[0] >= 0) & (ts[0] <= 1)`. The segment "
        "parameter `t` must be in `[0, 1]`; line `s` is unconstrained.\n"
        "4. Return `(ts[0].item(), ts[1].item(), bool(hit.item()))`.\n\n"
        "Assume the matrix is non-singular for this drill (parallel case is "
        "covered separately in `try-except-solve`).\n\n"
        "Inputs are `(2,)` float tensors. Output is `(t_seg, s_line, hit)`."
    ),
    "stub": (
        "def ex1_seg_line_intersect(S0: Tensor, S1: Tensor, L0: Tensor, L1: Tensor):\n"
        '    """Return (t_seg, s_line, hit) for segment S0->S1 vs line through L0,L1."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Case 1: segment (-1, 0) -> (1, 0), line y=0 (along x-axis through (0,0)-(1,0)).\n"
        "# They are colinear → matrix singular. Skip; instead test crossing.\n"
        "# Segment (-1, -1) -> (1, 1) vs line y=0 (through (0,0) and (1,0)).\n"
        "# Crosses at origin: segment t=0.5, line s=0.\n"
        "S0 = t.tensor([-1.0, -1.0]); S1 = t.tensor([1.0, 1.0])\n"
        "L0 = t.tensor([0.0, 0.0]);   L1 = t.tensor([1.0, 0.0])\n"
        "ts, ls, hit = ex1_seg_line_intersect(S0, S1, L0, L1)\n"
        "assert abs(ts - 0.5) < 1e-5, f'expected t=0.5, got {ts}'\n"
        "assert abs(ls - 0.0) < 1e-5, f'expected s=0, got {ls}'\n"
        "assert hit is True, f'expected hit=True, got {hit!r}'\n"
        "\n"
        "# Case 2: segment misses the line (segment lies above x-axis entirely).\n"
        "# Segment (-1, 1) -> (1, 2) vs line y=0. Solve gives t outside [0,1].\n"
        "S0 = t.tensor([-1.0, 1.0]); S1 = t.tensor([1.0, 2.0])\n"
        "L0 = t.tensor([0.0, 0.0]);  L1 = t.tensor([1.0, 0.0])\n"
        "ts, ls, hit = ex1_seg_line_intersect(S0, S1, L0, L1)\n"
        "# Segment doesn't reach y=0 → t would be < 0 (extrapolating backward).\n"
        "assert hit is False, f'segment is above line, expected hit=False, got {hit!r}'\n"
        "assert (ts < 0.0) or (ts > 1.0), f'expected t outside [0,1], got {ts}'\n"
        "\n"
        "# Case 3: segment endpoint exactly on the line (t=1.0 boundary).\n"
        "S0 = t.tensor([0.0, 1.0]); S1 = t.tensor([0.0, 0.0])\n"
        "L0 = t.tensor([-1.0, 0.0]); L1 = t.tensor([1.0, 0.0])\n"
        "ts, ls, hit = ex1_seg_line_intersect(S0, S1, L0, L1)\n"
        "assert abs(ts - 1.0) < 1e-5, f'expected t=1.0, got {ts}'\n"
        "assert hit is True, 'endpoint on the line counts as a hit (closed interval)'\n"
        "\n"
        "# Case 4: line is the y-axis (x=0); segment (-2, 3) -> (2, 3) crosses at midpoint.\n"
        "S0 = t.tensor([-2.0, 3.0]); S1 = t.tensor([2.0, 3.0])\n"
        "L0 = t.tensor([0.0, 0.0]);  L1 = t.tensor([0.0, 1.0])  # y-axis\n"
        "ts, ls, hit = ex1_seg_line_intersect(S0, S1, L0, L1)\n"
        "assert abs(ts - 0.5) < 1e-5, f'expected t=0.5, got {ts}'\n"
        "assert hit is True\n"
        "\n"
        "# --- Visualization: scatter a batch of random segments + the line ---\n"
        "import matplotlib.pyplot as plt\n"
        "rng = t.Generator().manual_seed(7)\n"
        "fig, ax = plt.subplots(figsize=(6, 6))\n"
        "L0 = t.tensor([-3.0, 0.0]); L1 = t.tensor([3.0, 0.0])  # x-axis line\n"
        "ax.axhline(0, color='red', linewidth=1, label='line y=0')\n"
        "for _ in range(20):\n"
        "    a = (t.rand(2, generator=rng) * 6) - 3  # in [-3,3]^2\n"
        "    b = (t.rand(2, generator=rng) * 6) - 3\n"
        "    ts_, _, hit_ = ex1_seg_line_intersect(a, b, L0, L1)\n"
        "    color = 'green' if hit_ else 'gray'\n"
        "    ax.plot([a[0].item(), b[0].item()], [a[1].item(), b[1].item()],\n"
        "            color=color, alpha=0.6)\n"
        "ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)\n"
        "ax.set_aspect('equal'); ax.grid(True, alpha=0.3)\n"
        "ax.set_title('green = crosses y=0, gray = misses')\n"
        "ax.legend()\n"
        "plt.tight_layout(); plt.show()"
    ),
    "solution_body": (
        "def ex1_seg_line_intersect(S0: Tensor, S1: Tensor, L0: Tensor, L1: Tensor):\n"
        "    d = S1 - S0\n"
        "    e = L1 - L0\n"
        "    A = t.stack([d, -e], dim=1)   # columns are d and -e\n"
        "    b = L0 - S0\n"
        "    ts = t.linalg.solve(A, b)\n"
        "    t_seg, s_line = ts[0].item(), ts[1].item()\n"
        "    hit = (t_seg >= 0.0) and (t_seg <= 1.0)\n"
        "    return t_seg, s_line, hit"
    ),
    "solution_notes": (
        "**Why stack `[d, -e]` not `[d, e]`.** The equation is "
        "`P + t*d = Q + s*e`, rearranged to `t*d - s*e = Q - P`. "
        "Multiplying through the system, the second column is `-e`. Get this "
        "wrong and `s` flips sign — easy bug to miss because `t` stays "
        "correct.\n\n"
        "**Why `[0, 1]` not `(0, 1)`.** Closed interval — endpoints on the "
        "line count as hits. Half-open conventions (`[0, 1)`) appear in "
        "raycasting (next-segment continuation) but for the canonical "
        "intersection test, both endpoints are part of the segment.\n\n"
        "**This generalizes to 3-D.** Replace the 2×2 with a 3×2 system "
        "(under-determined — line and segment in 3-D *usually* miss). For "
        "3-D you'd use `torch.linalg.lstsq` and check residual to decide "
        "whether 'close enough' is a hit. Different drill."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


# ---------------------------------------------------------------------------
# 6. rotation-matrix-3d (FULL — Rodrigues, general axis)
# ---------------------------------------------------------------------------

RECAP_ROT3D = (
    "## General 3-D rotation (Rodrigues' formula) — quick refresher\n"
    "\n"
    "**The matrix.** Right-hand rotation by `θ` about a UNIT axis "
    "`k = (kx, ky, kz)`:\n"
    "```\n"
    "R = I + sin(θ) * K + (1 - cos(θ)) * K²\n"
    "```\n"
    "where `K` is the skew-symmetric cross-product matrix of `k`:\n"
    "```\n"
    "K = [[ 0,  -kz,  ky],\n"
    "     [ kz,  0,  -kx],\n"
    "     [-ky,  kx,  0]]\n"
    "```\n"
    "**Special cases.** Axis = X gives `R_x(θ)`, axis = Y gives `R_y(θ)`, "
    "axis = Z gives `R_z(θ)`. The previous drill (`rotation-matrix-3d-y-axis`) "
    "is the Y-axis special case.\n"
    "\n"
    "**Axis must be unit length.** If `||k|| != 1`, Rodrigues' formula returns "
    "a scaled rotation — wrong. Always `k = k / k.norm()` before use.\n"
    "\n"
    "**Verifying orthogonality.** `R @ R.T == I` and `det(R) == +1`. Use these "
    "as numerical sanity checks (tolerance ~1e-6)."
)


SPEC_ROT3D = {
    "atom_id": "rotation-matrix-3d",
    "subtopic": "Geometry: Rotation matrix 3-D (full)",
    "topic_folder": TOPIC_GEOM,
    "atom_recap_md": RECAP_ROT3D,
    "exercise_index": 1,
    "exercise_title": "Rodrigues rotation about an arbitrary 3-D axis",
    "slug": "rodrigues-rotation-about-an-arbitrary-3d-axis",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["rotation", "rodrigues", "skew-symmetric", "axis-angle", "3D"],
    "kcs": ["build-skew-symmetric-matrix", "rodrigues-formula-assemble"],
    "lo": (
        "Apply Rodrigues' formula `R = I + sinθ K + (1-cosθ) K²` to construct "
        "the 3×3 rotation matrix for an arbitrary unit-axis + angle, with a "
        "3-D scatter showing rotated points."
    ),
    "prompt_body": (
        "Implement `ex1_rot3d(axis, theta)`. Inputs:\n"
        "- `axis`: `(3,)` float tensor (NOT assumed unit — normalize inside).\n"
        "- `theta`: scalar float angle in radians.\n\n"
        "Steps:\n"
        "1. Normalize: `k = axis / axis.norm()`.\n"
        "2. Build the skew-symmetric `K`:\n"
        "   ```python\n"
        "   K = t.tensor([[    0, -k[2],  k[1]],\n"
        "                 [ k[2],     0, -k[0]],\n"
        "                 [-k[1],  k[0],    0]])\n"
        "   ```\n"
        "   (Use `k[0].item()` etc. if you build via `t.tensor`, OR use "
        "`t.stack` / `t.zeros` to keep the tensor backend.)\n"
        "3. Apply Rodrigues:\n"
        "   ```python\n"
        "   R = t.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)\n"
        "   ```\n"
        "4. Return `R` as a `(3, 3)` `float32` tensor.\n\n"
        "**Sanity checks** (asserted in the test, but worth running yourself):\n"
        "- `R @ R.T ≈ I`\n"
        "- `det(R) ≈ +1`\n"
        "- Rotating `k` itself gives back `k` (the axis is fixed)."
    ),
    "stub": (
        "import math\n"
        "\n"
        "def ex1_rot3d(axis: Tensor, theta: float) -> Tensor:\n"
        '    """Rodrigues 3-D rotation matrix for axis (any length) by theta radians."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import math\n"
        "# Case 1: axis = +z, theta = pi/2 → R_z(90°) = [[0,-1,0],[1,0,0],[0,0,1]]\n"
        "R = ex1_rot3d(t.tensor([0.0, 0.0, 1.0]), math.pi / 2)\n"
        "assert R.shape == (3, 3), f'shape: {tuple(R.shape)}'\n"
        "expected_z90 = t.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])\n"
        "assert t.allclose(R, expected_z90, atol=1e-5), f'R_z(90) wrong:\\n{R}\\nvs\\n{expected_z90}'\n"
        "\n"
        "# Case 2: axis = +y, theta = pi/2 → R_y(90°) = [[0,0,1],[0,1,0],[-1,0,0]]\n"
        "R = ex1_rot3d(t.tensor([0.0, 1.0, 0.0]), math.pi / 2)\n"
        "expected_y90 = t.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])\n"
        "assert t.allclose(R, expected_y90, atol=1e-5), f'R_y(90) wrong:\\n{R}'\n"
        "\n"
        "# Case 3: orthogonality + det=1 for a non-axis-aligned axis.\n"
        "axis = t.tensor([1.0, 2.0, 3.0])  # NOT unit\n"
        "theta = 0.7\n"
        "R = ex1_rot3d(axis, theta)\n"
        "I = t.eye(3)\n"
        "assert t.allclose(R @ R.T, I, atol=1e-5), f'R not orthogonal:\\n{R @ R.T}'\n"
        "assert abs(t.linalg.det(R).item() - 1.0) < 1e-5, f'det(R) = {t.linalg.det(R).item()}, expected 1.0'\n"
        "# The axis is a fixed direction.\n"
        "k_unit = axis / axis.norm()\n"
        "assert t.allclose(R @ k_unit, k_unit, atol=1e-5), 'axis must be a fixed point of R'\n"
        "\n"
        "# Case 4: theta = 0 → identity.\n"
        "R0 = ex1_rot3d(t.tensor([1.0, 0.0, 0.0]), 0.0)\n"
        "assert t.allclose(R0, t.eye(3), atol=1e-6), f'theta=0 must give I, got\\n{R0}'\n"
        "\n"
        "# Case 5: theta = 2*pi → identity (full revolution).\n"
        "R2pi = ex1_rot3d(t.tensor([0.3, 0.6, 0.7]), 2 * math.pi)\n"
        "assert t.allclose(R2pi, t.eye(3), atol=1e-5), 'theta=2π must give I'\n"
        "\n"
        "# --- 3-D visualization: rotate a cube's corners by a tilted axis ---\n"
        "import matplotlib.pyplot as plt\n"
        "from mpl_toolkits.mplot3d import Axes3D  # noqa: F401\n"
        "axis = t.tensor([1.0, 1.0, 1.0])  # diagonal\n"
        "theta = math.pi / 3\n"
        "R = ex1_rot3d(axis, theta)\n"
        "# 8 cube corners at +/-1.\n"
        "corners = t.tensor([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])\n"
        "rotated = corners @ R.T\n"
        "fig = plt.figure(figsize=(6, 5))\n"
        "ax = fig.add_subplot(111, projection='3d')\n"
        "ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c='blue', s=60, label='original')\n"
        "ax.scatter(rotated[:, 0], rotated[:, 1], rotated[:, 2], c='red', s=60, label='rotated 60° about (1,1,1)')\n"
        "for c0, c1 in zip(corners, rotated):\n"
        "    ax.plot([c0[0], c1[0]], [c0[1], c1[1]], [c0[2], c1[2]], color='gray', alpha=0.3)\n"
        "ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')\n"
        "ax.set_title('ex1 — Rodrigues rotation of a cube')\n"
        "ax.legend()\n"
        "plt.tight_layout(); plt.show()"
    ),
    "solution_body": (
        "def ex1_rot3d(axis: Tensor, theta: float) -> Tensor:\n"
        "    k = axis / axis.norm()\n"
        "    kx, ky, kz = k[0], k[1], k[2]\n"
        "    K = t.stack([\n"
        "        t.stack([t.zeros_like(kx),          -kz,                  ky]),\n"
        "        t.stack([                 kz, t.zeros_like(kx),          -kx]),\n"
        "        t.stack([                -ky,                kx, t.zeros_like(kx)]),\n"
        "    ])\n"
        "    s, c = math.sin(theta), math.cos(theta)\n"
        "    return t.eye(3) + s * K + (1 - c) * (K @ K)"
    ),
    "solution_notes": (
        "**Why normalize the axis.** Rodrigues' formula assumes `||k||=1`. "
        "If you pass `(1, 2, 3)` directly, `K @ K` scales by `||k||^2 = 14`, "
        "and you get a wildly non-orthogonal matrix. Always `k = axis / "
        "axis.norm()` first.\n\n"
        "**Why `K @ K` not `K^2` via element-wise square.** `K**2` (Python) "
        "does element-wise squaring on tensors — that's NOT the matrix square. "
        "Use `K @ K` (matmul) or `t.linalg.matrix_power(K, 2)`.\n\n"
        "**Why the cube viz.** Single-axis rotations (X/Y/Z) keep one axis "
        "fixed, which doesn't visually exercise the off-diagonal Rodrigues "
        "terms. Rotating about the body diagonal `(1, 1, 1)` involves ALL "
        "components of `K` — a more interesting test of the formula.\n\n"
        "**Generalizing to a batch.** For a stack of `(B, 3)` axes and `(B,)` "
        "angles, broadcast `K` along the batch dim. PyTorch's `linalg` ops "
        "handle this natively — useful for robotics IK and rigid-body sims."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt", "import math"],
}


# ---------------------------------------------------------------------------
# 7. try-except-solve  (graceful singular-matrix fallback)
# ---------------------------------------------------------------------------

RECAP_TRYSOLVE = (
    "## `try`/`except` around `torch.linalg.solve` — quick refresher\n"
    "\n"
    "`torch.linalg.solve(A, b)` raises `torch._C._LinAlgError` (a subclass of "
    "`RuntimeError`) when `A` is singular. In a hot batch this means ONE bad "
    "matrix kills the whole solve.\n"
    "\n"
    "**Two strategies for singular A:**\n"
    "1. **`try`/`except` fallback to None.** Wrap the call; on error return "
    "a sentinel (`None`, `nan`, a flag). Caller decides what to do. Simple, "
    "honest, exposes the failure.\n"
    "2. **Identity-substitution trick.** Replace singular A's with I before "
    "solving (different atom: `singular-matrix-mask-trick`). Lets you keep the "
    "tensor shape and mask out failures downstream.\n"
    "\n"
    "**This drill drills strategy 1.** Use the try/except pattern for "
    "non-batched solves where graceful per-call failure is acceptable."
)


SPEC_TRYSOLVE = {
    "atom_id": "try-except-solve",
    "subtopic": "LinAlg: try/except solve",
    "topic_folder": TOPIC_GEOM,
    "atom_recap_md": RECAP_TRYSOLVE,
    "exercise_index": 1,
    "exercise_title": "graceful single-call solve with try/except",
    "slug": "graceful-single-call-solve-with-try-except",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["try-except", "linalg-solve", "singular", "graceful-failure", "fallback"],
    "kcs": ["catch-linalg-runtime-error", "return-none-on-singular"],
    "lo": (
        "Apply `try`/`except RuntimeError` around `torch.linalg.solve` so a "
        "singular matrix returns `None` instead of crashing."
    ),
    "prompt_body": (
        "Implement `ex1_safe_solve(A, b)`:\n\n"
        "1. Try `t.linalg.solve(A, b)`. If it succeeds, return the result.\n"
        "2. If it raises `RuntimeError` (covers `_LinAlgError`), return "
        "`None`.\n\n"
        "Signature: `(A: Tensor, b: Tensor) -> Optional[Tensor]`.\n\n"
        "`A` is `(n, n)`, `b` is `(n,)` for this drill. Returns either the "
        "solution `(n,)` or `None`.\n\n"
        "**Contrast** with `singular-matrix-mask-trick`: that atom would "
        "substitute identity and return a tensor of shape `(n,)` with `nan` "
        "/ zero markers. This atom is for the non-batched case where you "
        "want a clean Python None and the caller branches on it."
    ),
    "stub": (
        "from typing import Optional\n"
        "\n"
        "def ex1_safe_solve(A: Tensor, b: Tensor) -> Optional[Tensor]:\n"
        '    """Solve A @ x = b; return None if A is singular."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Case 1: well-conditioned A → returns x s.t. A @ x ≈ b.\n"
        "A = t.tensor([[3.0, 1.0], [1.0, 2.0]])\n"
        "b = t.tensor([9.0, 8.0])\n"
        "x = ex1_safe_solve(A, b)\n"
        "assert x is not None, 'well-conditioned solve should NOT return None'\n"
        "assert x.shape == (2,), f'shape: {tuple(x.shape)}'\n"
        "assert t.allclose(A @ x, b, atol=1e-5), f'A @ x = {A @ x}, expected {b}'\n"
        "\n"
        "# Case 2: singular A → returns None.\n"
        "A_sing = t.tensor([[1.0, 2.0], [2.0, 4.0]])  # second row = 2 * first → rank 1\n"
        "b = t.tensor([3.0, 6.0])\n"
        "x = ex1_safe_solve(A_sing, b)\n"
        "assert x is None, f'singular A must return None, got {x!r}'\n"
        "\n"
        "# Case 3: zero matrix → singular.\n"
        "A_zero = t.zeros(3, 3)\n"
        "b = t.ones(3)\n"
        "x = ex1_safe_solve(A_zero, b)\n"
        "assert x is None, 'zero matrix is singular → must return None'\n"
        "\n"
        "# Case 4: 3x3 well-conditioned.\n"
        "A3 = t.eye(3) * 2 + t.ones(3, 3) * 0.1\n"
        "b3 = t.tensor([1.0, 2.0, 3.0])\n"
        "x3 = ex1_safe_solve(A3, b3)\n"
        "assert x3 is not None\n"
        "assert t.allclose(A3 @ x3, b3, atol=1e-4)\n"
        "\n"
        "# Case 5: identity → x == b.\n"
        "x_eye = ex1_safe_solve(t.eye(4), t.tensor([5.0, -3.0, 2.0, 1.0]))\n"
        "assert x_eye is not None\n"
        "assert t.allclose(x_eye, t.tensor([5.0, -3.0, 2.0, 1.0]), atol=1e-6)"
    ),
    "solution_body": (
        "def ex1_safe_solve(A: Tensor, b: Tensor) -> Optional[Tensor]:\n"
        "    try:\n"
        "        return t.linalg.solve(A, b)\n"
        "    except RuntimeError:\n"
        "        return None"
    ),
    "solution_notes": (
        "**Why `RuntimeError` not `_LinAlgError`.** "
        "`torch._C._LinAlgError` is a SUBCLASS of `RuntimeError`. Catching "
        "the parent is portable across PyTorch versions (the linalg error "
        "type was renamed in 2.0+) AND across CUDA-vs-CPU codepaths (CUDA "
        "sometimes raises raw `RuntimeError` with a different message).\n\n"
        "**Why None not nan.** A `None` return is a Python sentinel that "
        "forces the caller to branch. A `nan` tensor silently propagates and "
        "can poison downstream computation — usually the OPPOSITE of what "
        "you want for a graceful-failure abstraction.\n\n"
        "**When NOT to use this.** In a hot batched loop (1000s of solves), "
        "the try/except overhead is per-call and Python-side. For that case "
        "use the masked-substitution trick (`singular-matrix-mask-trick`) "
        "which stays vectorized."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# 8. cross-product-normal  (surface normal of a triangle)
# ---------------------------------------------------------------------------

RECAP_NORMAL = (
    "## Surface normal via cross product — quick refresher\n"
    "\n"
    "Given three triangle vertices `P1, P2, P3` (3-D), the surface normal is:\n"
    "```\n"
    "n = (P2 - P1) × (P3 - P1)\n"
    "n_hat = n / ||n||\n"
    "```\n"
    "**Right-hand rule.** Curling the fingers from `(P2 - P1)` toward "
    "`(P3 - P1)` makes the thumb point along `n`. Swap the order of the two "
    "edges → normal flips sign.\n"
    "\n"
    "**Degenerate case.** If the three vertices are colinear, the cross "
    "product is the zero vector and `||n|| = 0` — normalization gives nan/inf. "
    "Real renderers check `||n|| > eps` before normalizing.\n"
    "\n"
    "**Why this matters.** Lighting (`L · n`), backface culling (`view · n < 0`), "
    "and ray-triangle intersection all need a consistent surface normal."
)


SPEC_NORMAL = {
    "atom_id": "cross-product-normal",
    "subtopic": "Geometry: Cross-product surface normal",
    "topic_folder": TOPIC_GEOM,
    "atom_recap_md": RECAP_NORMAL,
    "exercise_index": 1,
    "exercise_title": "unit surface normal of a triangle via cross product",
    "slug": "unit-surface-normal-of-a-triangle-via-cross-product",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["cross-product", "normal", "triangle", "rendering"],
    "kcs": ["edges-from-shared-vertex", "cross-then-normalize"],
    "lo": (
        "Apply `cross(P2-P1, P3-P1)` followed by normalization to compute the "
        "unit surface normal of a triangle in 3-D, returning a `(3,)` tensor."
    ),
    "prompt_body": (
        "Implement `ex1_triangle_normal(P1, P2, P3)`:\n\n"
        "1. Compute edges from the shared vertex `P1`:\n"
        "   ```python\n"
        "   e1 = P2 - P1\n"
        "   e2 = P3 - P1\n"
        "   ```\n"
        "2. Cross product:\n"
        "   ```python\n"
        "   n = t.linalg.cross(e1, e2)\n"
        "   ```\n"
        "3. Normalize:\n"
        "   ```python\n"
        "   return n / n.norm()\n"
        "   ```\n"
        "\n"
        "Inputs: three `(3,)` float tensors. Output: `(3,)` float unit "
        "vector.\n\n"
        "**Order matters.** `cross(e1, e2)` and `cross(e2, e1)` differ in "
        "sign. The drill convention is `cross(P2-P1, P3-P1)` — counter-"
        "clockwise winding when viewed from the front."
    ),
    "stub": (
        "def ex1_triangle_normal(P1: Tensor, P2: Tensor, P3: Tensor) -> Tensor:\n"
        '    """Unit surface normal of triangle (P1, P2, P3) via cross product."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Case 1: triangle in the z=0 plane, CCW → normal points +z.\n"
        "P1 = t.tensor([0.0, 0.0, 0.0])\n"
        "P2 = t.tensor([1.0, 0.0, 0.0])\n"
        "P3 = t.tensor([0.0, 1.0, 0.0])\n"
        "n = ex1_triangle_normal(P1, P2, P3)\n"
        "assert n.shape == (3,), f'shape: {tuple(n.shape)}'\n"
        "expected = t.tensor([0.0, 0.0, 1.0])\n"
        "assert t.allclose(n, expected, atol=1e-5), f'expected +z normal, got {n}'\n"
        "assert abs(n.norm().item() - 1.0) < 1e-5, f'must be unit length, got {n.norm().item()}'\n"
        "\n"
        "# Case 2: swap P2/P3 → CW winding → normal points -z.\n"
        "n_flipped = ex1_triangle_normal(P1, P3, P2)\n"
        "assert t.allclose(n_flipped, t.tensor([0.0, 0.0, -1.0]), atol=1e-5), (\n"
        "    f'CW winding should flip the normal, got {n_flipped}'\n"
        ")\n"
        "\n"
        "# Case 3: triangle in the x=5 plane → normal is +/- x.\n"
        "P1 = t.tensor([5.0, 0.0, 0.0])\n"
        "P2 = t.tensor([5.0, 1.0, 0.0])\n"
        "P3 = t.tensor([5.0, 0.0, 1.0])\n"
        "n = ex1_triangle_normal(P1, P2, P3)\n"
        "# (P2-P1)=(0,1,0); (P3-P1)=(0,0,1); cross = (1,0,0). Normal = +x.\n"
        "assert t.allclose(n, t.tensor([1.0, 0.0, 0.0]), atol=1e-5), f'expected +x, got {n}'\n"
        "\n"
        "# Case 4: tilted triangle — verify orthogonality to both edges.\n"
        "P1 = t.tensor([0.0, 0.0, 0.0])\n"
        "P2 = t.tensor([2.0, 1.0, 0.0])\n"
        "P3 = t.tensor([0.0, 1.0, 3.0])\n"
        "n = ex1_triangle_normal(P1, P2, P3)\n"
        "e1 = P2 - P1\n"
        "e2 = P3 - P1\n"
        "assert abs((n * e1).sum().item()) < 1e-5, f'n must be perp to e1, dot = {(n*e1).sum().item()}'\n"
        "assert abs((n * e2).sum().item()) < 1e-5, f'n must be perp to e2, dot = {(n*e2).sum().item()}'\n"
        "assert abs(n.norm().item() - 1.0) < 1e-5, 'unit length'\n"
        "\n"
        "# --- Visualization: triangle + outward normal in 3-D ---\n"
        "import matplotlib.pyplot as plt\n"
        "from mpl_toolkits.mplot3d import Axes3D  # noqa: F401\n"
        "P1 = t.tensor([0.0, 0.0, 0.0])\n"
        "P2 = t.tensor([2.0, 0.0, 0.0])\n"
        "P3 = t.tensor([0.5, 2.0, 0.0])\n"
        "n = ex1_triangle_normal(P1, P2, P3)\n"
        "centroid = (P1 + P2 + P3) / 3\n"
        "fig = plt.figure(figsize=(6, 5))\n"
        "ax = fig.add_subplot(111, projection='3d')\n"
        "# triangle as a closed loop\n"
        "tri = t.stack([P1, P2, P3, P1])\n"
        "ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], 'b-', linewidth=2)\n"
        "ax.scatter(*[tri[:3, i] for i in range(3)], c='blue', s=50)\n"
        "# normal arrow from centroid\n"
        "ax.quiver(centroid[0], centroid[1], centroid[2],\n"
        "          n[0], n[1], n[2], length=1.0, color='red', label='unit normal')\n"
        "ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')\n"
        "ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 2.5); ax.set_zlim(-0.5, 1.5)\n"
        "ax.set_title('ex1 — triangle + outward unit normal')\n"
        "ax.legend()\n"
        "plt.tight_layout(); plt.show()"
    ),
    "solution_body": (
        "def ex1_triangle_normal(P1: Tensor, P2: Tensor, P3: Tensor) -> Tensor:\n"
        "    e1 = P2 - P1\n"
        "    e2 = P3 - P1\n"
        "    n = t.linalg.cross(e1, e2)\n"
        "    return n / n.norm()"
    ),
    "solution_notes": (
        "**Why `t.linalg.cross` and not `t.cross`.** `t.cross` requires a "
        "`dim` argument in modern PyTorch (it used to default to dim=-1 for "
        "size-3 tensors, but that default emits a deprecation warning since "
        "1.8). `t.linalg.cross` is the explicit replacement.\n\n"
        "**Why edges share `P1`.** You can also compute `cross(P2-P1, P3-P2)` "
        "— the cross product is the same up to sign (vectors are coplanar). "
        "But `P1` as the shared vertex is the convention because it matches "
        "how barycentric coordinates are defined.\n\n"
        "**Robustness.** Real renderers compute `n.norm()` first; if "
        "`||n|| < 1e-8`, treat as a degenerate triangle and skip it. Without "
        "that guard, division by ~0 produces `inf` or `nan` normals which "
        "poison the lighting calculation downstream."
    ),
    "extra_imports": ["import matplotlib.pyplot as plt"],
}


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

SPECS = [
    SPEC_BROADCAST_WEIGHTS,
    SPEC_EVAL_METRICS,
    SPEC_MODEL_SAVE,
    SPEC_REDUCE_GATHER,
    SPEC_SEGLINE,
    SPEC_ROT3D,
    SPEC_TRYSOLVE,
    SPEC_NORMAL,
]

for spec in SPECS:
    path = emit_standalone(spec)
    rel = path.relative_to(path.parents[3])
    print(f"wrote {rel}")
