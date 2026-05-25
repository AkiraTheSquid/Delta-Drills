#!/usr/bin/env python3
"""Author 8 ex2 deepening drills (batch 11) — `prereqs_distributed/` atoms.

Atoms (all 8 distributed):
    - broadcast-initial-weights     (ex2: broadcast a whole state_dict — params + buffers)
    - dist-send-recv-pair           (ex2: ring-pass via send/recv rotated N-1 times)
    - distributed-sampler-shard     (ex2: set_epoch reshuffles, disjoint+coverage hold across epochs)
    - init-process-group-nccl       (ex2: context-manager wrapper guarantees destroy on exception)
    - model-save-state-dict         (ex2: rank-0 atomic save with tmp→rename + barrier)
    - mp-spawn-workers              (ex2: non-blocking spawn via ProcessContext + manual join)
    - per-rank-cuda-device          (ex2: build per-rank ctx dict; assert no two ranks share device)
    - rank-world-size-args          (ex2: reduce-protocol — (tensor, rank, world_size, dst=0))

Each ex2 hits a DISTINCT facet from ex1: different cognitive operation, surface
context, or API path. ONE LO + ONE Bloom + <=2 KCs per drill.

CPU-only harness: every distributed API is MOCKED. The shared `_FAKE_DIST_HARNESS`
block (lifted from batch10) spawns thread-based ranks sharing a `threading.Barrier`,
exposes `_FakeDist` w/ `all_reduce / reduce / broadcast / barrier / init / destroy`,
and lets test bodies inject a fake `dist` module into the student's function.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC_DIST = "prereqs_distributed"


# ---------------------------------------------------------------------------
# Fake-distributed harness — thread-based _FakeWorld.
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
        # send/recv mailbox keyed by (src, dst)
        self.mailbox = {}
        self.mailbox_cv = threading.Condition(self.lock)
    def all_reduce(self, tensor, op='SUM'):
        rank = self.tls.rank
        self.barrier.wait()
        with self.lock:
            self.scratch.setdefault('ar', [None] * self.world_size)
            self.scratch['ar'][rank] = tensor.detach().clone()
        self.barrier.wait()
        bag = self.scratch['ar']
        if op == 'SUM':
            reduced = bag[0].clone()
            for x in bag[1:]:
                reduced = reduced + x
        elif op == 'MAX':
            reduced = bag[0].clone()
            for x in bag[1:]:
                reduced = _t_for_fake.maximum(reduced, x)
        elif op == 'MIN':
            reduced = bag[0].clone()
            for x in bag[1:]:
                reduced = _t_for_fake.minimum(reduced, x)
        elif op == 'PROD':
            reduced = bag[0].clone()
            for x in bag[1:]:
                reduced = reduced * x
        else:
            raise ValueError(f'unknown fake op {op!r}')
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
            if op == 'SUM':
                reduced = bag[0].clone()
                for x in bag[1:]:
                    reduced = reduced + x
            elif op == 'MAX':
                reduced = bag[0].clone()
                for x in bag[1:]:
                    reduced = _t_for_fake.maximum(reduced, x)
            elif op == 'MIN':
                reduced = bag[0].clone()
                for x in bag[1:]:
                    reduced = _t_for_fake.minimum(reduced, x)
            elif op == 'PROD':
                reduced = bag[0].clone()
                for x in bag[1:]:
                    reduced = reduced * x
            else:
                raise ValueError(f'unknown fake op {op!r}')
            tensor.copy_(reduced)
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
    def barrier_op(self):
        self.barrier.wait()
    def send(self, tensor, dst):
        rank = self.tls.rank
        with self.mailbox_cv:
            self.mailbox.setdefault((rank, dst), []).append(tensor.detach().clone())
            self.mailbox_cv.notify_all()
    def recv(self, tensor, src):
        rank = self.tls.rank
        with self.mailbox_cv:
            while not self.mailbox.get((src, rank)):
                self.mailbox_cv.wait(timeout=10)
            payload = self.mailbox[(src, rank)].pop(0)
        tensor.copy_(payload)

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
        fake_dist.barrier = world.barrier_op
        fake_dist.get_rank = lambda: rank
        fake_dist.get_world_size = lambda: world_size
        fake_dist.send = lambda tensor, dst: world.send(tensor, dst)
        fake_dist.recv = lambda tensor, src: world.recv(tensor, src)
        fake_dist.init_process_group = lambda **kw: world.scratch.setdefault('_init_calls', []).append(kw)
        fake_dist.destroy_process_group = lambda: world.scratch.setdefault('_destroy_calls', []).append(rank)
        try:
            worker_fn(rank, world_size, fake_dist, world)
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
    return world
'''


# ---------------------------------------------------------------------------
# Recap blocks (deepening — kept tight, build on ex1).
# ---------------------------------------------------------------------------

RECAP_BROADCAST_STATE_DICT = (
    "## Broadcasting a full `state_dict` — params + buffers\n"
    "\n"
    "Ex1 broadcast each `nn.Parameter.data`. That misses two things in a real "
    "DDP init:\n"
    "\n"
    "1. **BatchNorm running stats** (`running_mean`, `running_var`, "
    "`num_batches_tracked`) are NOT parameters — they live in `model."
    "buffers()`. Ranks must agree on them too, or the first eval batch sees "
    "rank-specific BN stats.\n"
    "2. **Non-float buffers** (LongTensors, mask buffers, etc.) — must still "
    "be in sync.\n"
    "\n"
    "Canonical pattern: iterate `model.state_dict().values()` (which yields "
    "params + persistent buffers) and broadcast each tensor:\n"
    "\n"
    "```python\n"
    "for tensor in model.state_dict().values():\n"
    "    dist.broadcast(tensor, src=0)\n"
    "```\n"
    "\n"
    "**Why state_dict, not parameters + buffers.** `state_dict()` returns "
    "the canonical superset in a deterministic order — same order on every "
    "rank because every rank has the same model graph. Zipping per-rank "
    "lists from `parameters()` and `buffers()` separately is more typing "
    "for the same answer.\n"
    "\n"
    "**state_dict tensors are VIEWS into the underlying storage.** Mutating "
    "them in-place (which is what `dist.broadcast` does) updates the live "
    "model. No `model.load_state_dict()` call needed afterward."
)

RECAP_RING_PASS = (
    "## Ring-pass via send/recv — N-1 rotations around the ring\n"
    "\n"
    "Ex1 used the linear `src loops sends to every other` pattern — fine "
    "for broadcast-from-one. The **ring-pass** is a different topology, "
    "where every rank pushes data to its right neighbor and receives from "
    "its left, repeated `world_size - 1` times. After all rotations, rank "
    "`r` has accumulated `world_size - 1` payloads from other ranks.\n"
    "\n"
    "```python\n"
    "left  = (rank - 1) % world_size\n"
    "right = (rank + 1) % world_size\n"
    "buf = my_payload.clone()\n"
    "received = []\n"
    "for _ in range(world_size - 1):\n"
    "    dist.send(buf, dst=right)\n"
    "    in_buf = t.zeros_like(buf)\n"
    "    dist.recv(in_buf, src=left)\n"
    "    received.append(in_buf.clone())\n"
    "    buf = in_buf\n"
    "```\n"
    "\n"
    "**Why ring topology.** It's the structural backbone of `nccl`'s "
    "all-reduce — bandwidth scales O(1) per rank (each link carries the "
    "same volume regardless of `world_size`). Implementing it by hand once "
    "burns the pattern in.\n"
    "\n"
    "**Deadlock trap.** `dist.send` is BLOCKING on gloo. If every rank "
    "calls `send` first and `recv` second, you must trust that the "
    "underlying transport buffers small messages — which gloo does for "
    "tensors below a few KB. For large tensors, use `dist.isend`/`irecv` "
    "and explicit `wait()`. This drill stays in the small-message regime."
)

RECAP_SAMPLER_EPOCH = (
    "## `set_epoch` — the across-epochs reshuffle contract\n"
    "\n"
    "Ex1 collected the shard indices for epoch 0. The deepening: **does "
    "`set_epoch(e)` actually produce a DIFFERENT permutation per epoch**, "
    "and do the cross-epoch shards still satisfy the disjoint + coverage "
    "invariants?\n"
    "\n"
    "```python\n"
    "sampler.set_epoch(0); shards_e0 = list(sampler)\n"
    "sampler.set_epoch(1); shards_e1 = list(sampler)\n"
    "# shards_e0 != shards_e1   (different permutation, with high probability)\n"
    "# but per-epoch invariants still hold per rank\n"
    "```\n"
    "\n"
    "**Why per-epoch determinism still matters.** Within ONE epoch, every "
    "rank's sampler uses the SAME (`epoch` + `seed`) → SAME base "
    "permutation. Each rank then slices its strided shard. The disjoint "
    "property is a consequence of `index % num_replicas == rank` — true "
    "for any permutation.\n"
    "\n"
    "**Coverage with padding.** When `len(dataset) % world_size != 0`, the "
    "sampler pads the permutation by wrapping from the start. So the union "
    "of all per-rank shards has length `ceil(N / W) * W`, with the first "
    "`(ceil(N/W) * W - N)` indices appearing twice. Coverage is exact only "
    "when `N` divides evenly.\n"
    "\n"
    "**Forgetting `set_epoch` is silent.** Training appears to work — every "
    "rank still sees a stable disjoint shard. But that shard is the same "
    "every epoch. The model sees the same batch order forever and "
    "convergence is wrong."
)

RECAP_DIST_SESSION = (
    "## `init_process_group` wrapped in a context manager\n"
    "\n"
    "Ex1 called `init_process_group` + `destroy_process_group` by hand. The "
    "footgun: if anything between init and destroy raises (loss explodes, "
    "OOM, data loader crashes), `destroy` is skipped — the rendezvous port "
    "stays bound and the NEXT training run hangs on init.\n"
    "\n"
    "Context-manager wrap fixes it with `try/finally`:\n"
    "\n"
    "```python\n"
    "from contextlib import contextmanager\n"
    "\n"
    "@contextmanager\n"
    "def dist_session(rank, world_size, port, backend='gloo'):\n"
    "    os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
    "    os.environ['MASTER_PORT'] = str(port)\n"
    "    dist.init_process_group(backend=backend, rank=rank,\n"
    "                            world_size=world_size)\n"
    "    try:\n"
    "        yield\n"
    "    finally:\n"
    "        dist.destroy_process_group()\n"
    "```\n"
    "\n"
    "**Why `finally`, not `try/except`.** A `try/except` swallows the "
    "exception. We want the exception to propagate AND destroy to run. "
    "`finally` is the only construct that guarantees both.\n"
    "\n"
    "**Where this pattern lives in real code.** `torch.distributed.run` "
    "(the `torchrun` launcher) wraps every worker in this exact pattern "
    "internally. Hand-rolling it for notebook-launched workers gives you "
    "the same crash-resilience."
)

RECAP_SAVE_ATOMIC = (
    "## Atomic rank-0 save — tmp file + rename + barrier\n"
    "\n"
    "Ex1's pattern was `if rank == 0: t.save(...); dist.barrier()`. Real "
    "training adds **atomicity** — a power loss or SIGKILL mid-`t.save` "
    "leaves a half-written file that fails to load on resume:\n"
    "\n"
    "```python\n"
    "if rank == 0:\n"
    "    tmp = ckpt_path + '.tmp'\n"
    "    t.save(model.state_dict(), tmp)\n"
    "    os.replace(tmp, ckpt_path)   # POSIX atomic rename\n"
    "dist.barrier()\n"
    "```\n"
    "\n"
    "**Why `os.replace`, not `os.rename`.** `os.replace` overwrites the "
    "destination if it exists — same semantics across POSIX and Windows. "
    "`os.rename` errors on Windows when the dest exists.\n"
    "\n"
    "**Why the barrier still matters.** After rank 0 finishes the rename, "
    "the file is durable. But other ranks may already have charged ahead "
    "to the next step, and if subsequent code does `if rank == 1: load(ckpt)` "
    "they need to KNOW the write finished. `dist.barrier()` is the cheapest "
    "cross-rank fence — no data motion, just synchronization.\n"
    "\n"
    "**Crash-resilience claim.** A crash AFTER `t.save(tmp)` but BEFORE "
    "`os.replace` leaves a stale `ckpt_path` and an orphaned `.tmp` file. "
    "The model can still resume from the previous good checkpoint. A crash "
    "DURING `t.save(tmp)` leaves a half-written `.tmp` file — but "
    "`ckpt_path` itself is untouched. The 'no half-written final file' "
    "guarantee is what atomic save buys you."
)

RECAP_SPAWN_NONBLOCKING = (
    "## `mp.spawn(..., join=False)` — non-blocking + manual `pc.join()`\n"
    "\n"
    "Ex1 used `mp.spawn(fn, args=..., nprocs=N, join=True)` — blocks the "
    "launcher until all workers exit. Most training launchers do this; "
    "it's the easiest correct form.\n"
    "\n"
    "The deepening: `join=False` returns a `ProcessContext` immediately. "
    "The launcher can do other work (start a monitoring server, run a "
    "fast smoke test, etc.) and then `pc.join()` later:\n"
    "\n"
    "```python\n"
    "pc = mp.spawn(worker, args=(world_size, port), nprocs=world_size,\n"
    "              join=False)\n"
    "# ... do other launcher work concurrently ...\n"
    "pc.join()   # blocks until every worker exits\n"
    "```\n"
    "\n"
    "**What `pc` exposes.** `pc.pids()` (the worker PIDs), `pc.join(timeout="
    "...)` (returns True if all done, False if timeout). The non-blocking "
    "form is how `torchrun`'s elastic launcher monitors workers — it can "
    "detect a hang and kill the group.\n"
    "\n"
    "**Returning errors from spawn.** If a worker raises, `pc.join()` "
    "re-raises in the launcher with a `ProcessRaisedException` that wraps "
    "the original exception + traceback. The blocking `join=True` form has "
    "the same behavior; the non-blocking form just defers the re-raise to "
    "your explicit `join` call."
)

RECAP_PER_RANK_CTX = (
    "## Per-rank context dict — device + stream + sync queue\n"
    "\n"
    "Ex1 returned `(device, model, tensor)` for a single rank. Real "
    "training maintains a **per-rank context object** that wraps everything "
    "rank-specific so the rest of the training code is rank-oblivious:\n"
    "\n"
    "```python\n"
    "@dataclass\n"
    "class RankContext:\n"
    "    rank: int\n"
    "    world_size: int\n"
    "    device: torch.device\n"
    "    stream: 'torch.cuda.Stream'   # one CUDA stream per rank\n"
    "    is_master: bool               # rank == 0\n"
    "```\n"
    "\n"
    "**Why a dict/object, not 5 separate args.** Adding a 6th rank-specific "
    "field (e.g. a per-rank RNG generator) is a one-line change to the ctx "
    "definition vs. plumbing a new arg through every function in the "
    "training stack. ARENA-style flat args are pedagogically clean but "
    "scale badly past 3-4 fields.\n"
    "\n"
    "**Invariant we drill here.** No two ranks share the same device "
    "index — `len({c.device.index for c in ctxs}) == world_size`. If "
    "this ever fails in a real run, two ranks are stomping on each other's "
    "CUDA context and OOM/training-collapse follows.\n"
    "\n"
    "**`is_master` shortcut.** `ctx.is_master` reads better than "
    "`ctx.rank == 0` scattered across the codebase. Same invariant; the "
    "abstraction names it."
)

RECAP_REDUCE_PROTOCOL = (
    "## Reduce-protocol signature — `(tensor, rank, world_size, dst=0, op)`\n"
    "\n"
    "Ex1 implemented the broadcast-protocol (one source, fan out). The "
    "reduce-protocol is the dual — many sources, fan IN to a single dst:\n"
    "\n"
    "```python\n"
    "def reduce_protocol(tensor, rank, world_size, dst=0, op='sum'):\n"
    "    if rank != dst:\n"
    "        return [('send', dst)]\n"
    "    # dst rank receives from every other rank and aggregates\n"
    "    return [('recv', other) for other in range(world_size) if other != dst]\n"
    "```\n"
    "\n"
    "**Why the signature differs.** `src` becomes `dst`. Reasoning: in "
    "broadcast the SOURCE rank has the canonical data; in reduce the "
    "DESTINATION rank ends up with the canonical data. The kwarg names the "
    "canonical-data rank's role in each topology.\n"
    "\n"
    "**Why ascending `other_rank` order is the convention.** Same as ex1's "
    "broadcast — order matters for deterministic test assertions and "
    "matches the order `dist.reduce`'s ring traversal uses internally.\n"
    "\n"
    "**The `op` arg is unused by the protocol logic.** It only matters for "
    "the actual aggregation (sum vs max vs ...). The protocol is "
    "topology-only — same `(send, recv)` shape for any reduction op."
)


# ---------------------------------------------------------------------------
# SPEC 1 — broadcast-initial-weights ex2 (state_dict broadcast)
# ---------------------------------------------------------------------------

SPEC_BROADCAST_SD = {
    "atom_id": "broadcast-initial-weights",
    "subtopic": "Distributed: broadcast initial weights",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_BROADCAST_STATE_DICT,
    "exercise_index": 2,
    "exercise_title": "broadcast a full state_dict — params AND BN buffers",
    "slug": "broadcast-full-state-dict-params-and-buffers",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["broadcast", "state_dict", "BN-buffers", "running_mean", "DDP-init"],
    "kcs": ["broadcast-state-dict-values", "buffers-need-sync-too"],
    "lo": (
        "Apply `dist.broadcast(tensor, src=0)` over every tensor in "
        "`model.state_dict().values()` so every rank's parameters AND "
        "BatchNorm buffers (`running_mean`, `running_var`) all mirror "
        "rank 0 at the start of training."
    ),
    "prompt_body": (
        "Implement `ex2_broadcast_state_dict(rank, world_size, dist_module, "
        "model)`. The full-state DDP init pattern:\n\n"
        "1. Iterate `model.state_dict().values()`. Order is deterministic "
        "across ranks because every rank has the same model graph.\n"
        "2. For each yielded tensor, call "
        "`dist_module.broadcast(tensor, src=0)`. This mutates the underlying "
        "storage in-place; no `load_state_dict` call needed.\n"
        "3. Return the count of tensors broadcast (used by tests to "
        "verify both params AND buffers were visited).\n\n"
        "Signature note: `dist_module` is injected (a mocked `dist` on CPU). "
        "All calls go via `dist_module.broadcast(...)`, not the global "
        "`dist.broadcast`.\n\n"
        "Input: `rank`, `world_size` — ints; `dist_module` — torch."
        "distributed or mock; `model` — `nn.Module` with rank-specific "
        "initial params AND buffers.\n"
        "Output: `int` — count of tensors broadcast (i.e. "
        "`len(model.state_dict())`)."
    ),
    "stub": (
        "def ex2_broadcast_state_dict(rank: int, world_size: int, dist_module, model: 'nn.Module') -> int:\n"
        '    """Broadcast every tensor in model.state_dict() from rank 0. Return tensor count."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FAKE_DIST_HARNESS + "\n\n"
        "import torch.nn as nn\n"
        "\n"
        "# Model with BOTH params (Linear weight/bias) and buffers (BN running stats).\n"
        "class _NetWithBN(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.lin = nn.Linear(4, 4)\n"
        "        self.bn = nn.BatchNorm1d(4)\n"
        "\n"
        "def _worker(rank, world_size, dist_module, world):\n"
        "    t.manual_seed(rank + 1)\n"
        "    model = _NetWithBN()\n"
        "    # Force per-rank divergence on params AND BN running stats.\n"
        "    with t.no_grad():\n"
        "        model.lin.weight.fill_(float(rank + 1))\n"
        "        model.lin.bias.fill_(float(rank + 1) * 10)\n"
        "        model.bn.running_mean.fill_(float(rank + 1) * 100)\n"
        "        model.bn.running_var.fill_(float(rank + 1) * 1000)\n"
        "    count = ex2_broadcast_state_dict(rank, world_size, dist_module, model)\n"
        "    world.results[rank] = (count,\n"
        "                            model.lin.weight.detach().clone(),\n"
        "                            model.lin.bias.detach().clone(),\n"
        "                            model.bn.running_mean.detach().clone(),\n"
        "                            model.bn.running_var.detach().clone())\n"
        "\n"
        "w = _run_fake_world(_worker, 3)\n"
        "rank0 = w.results[0]\n"
        "expected_count = rank0[0]\n"
        "# state_dict on this model: lin.weight, lin.bias, bn.weight, bn.bias,\n"
        "# bn.running_mean, bn.running_var, bn.num_batches_tracked → at least 5 broadcast.\n"
        "assert expected_count >= 5, f'expected at least 5 tensors broadcast, got {expected_count}'\n"
        "\n"
        "# Every rank's params + buffers must equal rank 0's.\n"
        "for r in range(3):\n"
        "    cnt, w_lw, w_lb, w_rm, w_rv = w.results[r]\n"
        "    assert cnt == expected_count, f'rank {r}: count {cnt} vs rank0 {expected_count}'\n"
        "    assert t.allclose(w_lw, rank0[1]), f'rank {r}: lin.weight diverged from rank 0'\n"
        "    assert t.allclose(w_lb, rank0[2]), f'rank {r}: lin.bias diverged from rank 0'\n"
        "    assert t.allclose(w_rm, rank0[3]), f'rank {r}: bn.running_mean diverged from rank 0 (BUFFERS missed!)'\n"
        "    assert t.allclose(w_rv, rank0[4]), f'rank {r}: bn.running_var diverged'\n"
        "\n"
        "# Rank 0's values should equal what we set (1.0 weight, 10.0 bias, 100.0 mean, 1000.0 var).\n"
        "assert abs(rank0[1].mean().item() - 1.0) < 1e-5\n"
        "assert abs(rank0[2].mean().item() - 10.0) < 1e-5\n"
        "assert abs(rank0[3].mean().item() - 100.0) < 1e-5\n"
        "assert abs(rank0[4].mean().item() - 1000.0) < 1e-5\n"
        "\n"
        "# 2-rank case — degenerate broadcast (one sender, one receiver).\n"
        "def _worker2(rank, world_size, dist_module, world):\n"
        "    model = _NetWithBN()\n"
        "    with t.no_grad():\n"
        "        model.lin.weight.fill_(float(rank + 5))\n"
        "        model.bn.running_mean.fill_(float(rank + 1) * 7.0)\n"
        "    ex2_broadcast_state_dict(rank, world_size, dist_module, model)\n"
        "    world.results[rank] = (model.lin.weight.detach().clone(),\n"
        "                            model.bn.running_mean.detach().clone())\n"
        "\n"
        "w2 = _run_fake_world(_worker2, 2)\n"
        "# Rank 0 had weight=5.0, mean=7.0. Both ranks should end with those.\n"
        "for r in range(2):\n"
        "    lw, rm = w2.results[r]\n"
        "    assert abs(lw.mean().item() - 5.0) < 1e-5, f'2-rank: rank {r} lin.weight={lw.mean().item()}'\n"
        "    assert abs(rm.mean().item() - 7.0) < 1e-5, f'2-rank: rank {r} running_mean={rm.mean().item()}'"
    ),
    "solution_body": (
        "def ex2_broadcast_state_dict(rank: int, world_size: int, dist_module, model: 'nn.Module') -> int:\n"
        "    count = 0\n"
        "    for tensor in model.state_dict().values():\n"
        "        dist_module.broadcast(tensor, src=0)\n"
        "        count += 1\n"
        "    return count"
    ),
    "solution_notes": (
        "**Why iterate `state_dict().values()` not `parameters()`.** "
        "`parameters()` yields ONLY learnable tensors — Linear weight/bias, "
        "Conv weight/bias. It SKIPS BatchNorm running stats, which are "
        "persistent buffers, not parameters. Broadcasting params alone "
        "leaves a silent bug: per-rank `running_mean` drifts forever.\n\n"
        "**`num_batches_tracked` is an `int64` scalar.** Some older "
        "backends choke on int broadcast (gloo handles it fine, nccl is "
        "finicky). The fake harness above handles all dtypes uniformly via "
        "`copy_`.\n\n"
        "**Order-determinism.** `state_dict()` returns an `OrderedDict` "
        "whose key order is the in-graph traversal order. Every rank has "
        "the same model graph, so every rank iterates the same order — no "
        "off-by-one or interleaved broadcasts."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 2 — dist-send-recv-pair ex2 (ring-pass)
# ---------------------------------------------------------------------------

SPEC_RING_PASS = {
    "atom_id": "dist-send-recv-pair",
    "subtopic": "Distributed: dist.send/recv pair",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_RING_PASS,
    "exercise_index": 2,
    "exercise_title": "ring-pass via send/recv — every rank collects from all others",
    "slug": "ring-pass-via-send-recv-collect-all-payloads",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["send", "recv", "ring", "rotation", "topology"],
    "kcs": ["ring-neighbor-left-right-modular", "n-minus-1-rotations"],
    "lo": (
        "Apply paired `dist.send(buf, dst=right)` + `dist.recv(in_buf, "
        "src=left)` in a loop of `world_size - 1` rotations to collect "
        "every other rank's payload at every rank — the ring-allreduce "
        "topology in its barest form."
    ),
    "prompt_body": (
        "Implement `ex2_ring_collect(rank, world_size, dist_module, "
        "my_payload)`. The ring-pass collector:\n\n"
        "1. Compute neighbors: `left = (rank - 1) % world_size`, "
        "`right = (rank + 1) % world_size`.\n"
        "2. Initialize `buf = my_payload.clone()` (the payload moving "
        "around the ring) and `received = []` (the list of payloads this "
        "rank has seen).\n"
        "3. Loop `world_size - 1` times:\n"
        "   a. `dist_module.send(buf, dst=right)` — push to right neighbor.\n"
        "   b. Allocate fresh `in_buf = t.zeros_like(buf)`.\n"
        "   c. `dist_module.recv(in_buf, src=left)` — receive from left.\n"
        "   d. Append `in_buf.clone()` to `received`.\n"
        "   e. Set `buf = in_buf` so the next rotation passes the just-"
        "received payload onward.\n"
        "4. Return `received` — a list of `world_size - 1` tensors, in "
        "the order they were received.\n\n"
        "After `world_size - 1` rotations, every rank has seen every "
        "OTHER rank's original payload (in some order; the order depends "
        "on rank, which we don't assert).\n\n"
        "Input: `rank`, `world_size` — ints; `dist_module` — torch."
        "distributed or mock; `my_payload` — a 1-D float tensor unique to "
        "this rank.\n"
        "Output: `list[Tensor]` of length `world_size - 1`."
    ),
    "stub": (
        "def ex2_ring_collect(rank: int, world_size: int, dist_module, my_payload: Tensor) -> list:\n"
        '    """Ring-pass collector: every rank ends with all OTHER ranks\' payloads."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FAKE_DIST_HARNESS + "\n\n"
        "# Each rank's payload is a unique scalar tensor.\n"
        "def _worker(rank, world_size, dist_module, world):\n"
        "    my_payload = t.tensor([float(rank + 1) * 10.0])\n"
        "    received = ex2_ring_collect(rank, world_size, dist_module, my_payload)\n"
        "    # Convert to a sorted list of scalars for the test.\n"
        "    world.results[rank] = sorted(r.item() for r in received)\n"
        "\n"
        "w = _run_fake_world(_worker, 4)\n"
        "# 4 ranks → each rank should end with 3 received payloads.\n"
        "# Payloads sent: [10, 20, 30, 40]. Each rank receives the OTHER three.\n"
        "for rank in range(4):\n"
        "    got = w.results[rank]\n"
        "    assert got is not None, f'rank {rank} returned None'\n"
        "    assert len(got) == 3, f'rank {rank}: expected 3 received, got {len(got)}: {got}'\n"
        "    expected = sorted([10.0, 20.0, 30.0, 40.0]) \n"
        "    expected.remove((rank + 1) * 10.0)   # own payload excluded\n"
        "    for g, e in zip(got, expected):\n"
        "        assert abs(g - e) < 1e-5, f'rank {rank}: got {got}, expected {expected}'\n"
        "\n"
        "# 2-rank case — only 1 rotation, each rank receives the other's payload.\n"
        "def _worker2(rank, world_size, dist_module, world):\n"
        "    my_payload = t.tensor([float(rank + 1) * 100.0])\n"
        "    received = ex2_ring_collect(rank, world_size, dist_module, my_payload)\n"
        "    world.results[rank] = [r.item() for r in received]\n"
        "\n"
        "w2 = _run_fake_world(_worker2, 2)\n"
        "assert w2.results[0] == [200.0], f'2-rank rank 0 expected [200], got {w2.results[0]}'\n"
        "assert w2.results[1] == [100.0], f'2-rank rank 1 expected [100], got {w2.results[1]}'\n"
        "\n"
        "# Single-rank degenerate — 0 rotations, empty received list.\n"
        "def _worker1(rank, world_size, dist_module, world):\n"
        "    my_payload = t.tensor([42.0])\n"
        "    received = ex2_ring_collect(rank, world_size, dist_module, my_payload)\n"
        "    world.results[rank] = received\n"
        "\n"
        "w1 = _run_fake_world(_worker1, 1)\n"
        "assert w1.results[0] == [], f'1-rank case: expected empty list, got {w1.results[0]}'\n"
        "\n"
        "# Vector payload (not scalar) — ring also works.\n"
        "def _worker_vec(rank, world_size, dist_module, world):\n"
        "    my_payload = t.tensor([float(rank), float(rank) + 0.5, float(rank) + 0.25])\n"
        "    received = ex2_ring_collect(rank, world_size, dist_module, my_payload)\n"
        "    world.results[rank] = sorted(r.sum().item() for r in received)\n"
        "\n"
        "w_vec = _run_fake_world(_worker_vec, 3)\n"
        "# Each rank's payload sum: rank r → 3r + 0.75. Each rank receives the other two sums.\n"
        "for rank in range(3):\n"
        "    own_sum = 3.0 * rank + 0.75\n"
        "    expected_sums = sorted([3.0 * r + 0.75 for r in range(3) if r != rank])\n"
        "    got = w_vec.results[rank]\n"
        "    for g, e in zip(got, expected_sums):\n"
        "        assert abs(g - e) < 1e-5, f'vec rank {rank}: got {got}, expected {expected_sums}'"
    ),
    "solution_body": (
        "def ex2_ring_collect(rank: int, world_size: int, dist_module, my_payload: Tensor) -> list:\n"
        "    left = (rank - 1) % world_size\n"
        "    right = (rank + 1) % world_size\n"
        "    buf = my_payload.clone()\n"
        "    received = []\n"
        "    for _ in range(world_size - 1):\n"
        "        dist_module.send(buf, dst=right)\n"
        "        in_buf = t.zeros_like(buf)\n"
        "        dist_module.recv(in_buf, src=left)\n"
        "        received.append(in_buf.clone())\n"
        "        buf = in_buf\n"
        "    return received"
    ),
    "solution_notes": (
        "**Why `world_size - 1` rotations.** After 1 rotation, every rank "
        "has its left-neighbor's original payload. After 2 rotations, every "
        "rank has its left-left-neighbor's payload (the prior payload was "
        "passed on). After `W-1` rotations, every OTHER rank's payload has "
        "visited every rank exactly once.\n\n"
        "**`buf = in_buf` instead of `buf.copy_(in_buf)`.** Either works "
        "semantically. The rebind is more idiomatic in Python and lets the "
        "garbage collector reclaim the old buffer; the in-place copy is "
        "marginally faster but allocates an extra tensor at `zeros_like`.\n\n"
        "**Deadlock-safety on real gloo.** Every rank calls send THEN recv "
        "in the same order, but gloo's send is non-blocking for small "
        "tensors — it copies to an internal buffer and returns. The recv "
        "then drains. For large tensors (>~64KB), gloo blocks the send "
        "until the matching recv posts — which would deadlock this code. "
        "The ring needs `isend`/`irecv` for production scale."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 3 — distributed-sampler-shard ex2 (set_epoch + invariants)
# ---------------------------------------------------------------------------

SPEC_SAMPLER_EPOCH = {
    "atom_id": "distributed-sampler-shard",
    "subtopic": "Distributed: DistributedSampler shard",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_SAMPLER_EPOCH,
    "exercise_index": 2,
    "exercise_title": "set_epoch reshuffles AND preserves disjoint+coverage",
    "slug": "set-epoch-reshuffles-preserves-invariants",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["distributed-sampler", "set_epoch", "reshuffle", "disjoint", "coverage"],
    "kcs": ["set-epoch-changes-permutation", "per-epoch-disjoint-and-coverage-hold"],
    "lo": (
        "Analyze `DistributedSampler` across multiple epochs by calling "
        "`set_epoch(e)` before each iteration, returning a nested "
        "`list[epoch][rank]` of indices so the test can verify that "
        "permutations differ across epochs while each epoch still "
        "satisfies the disjoint + coverage invariants."
    ),
    "prompt_body": (
        "Implement `ex2_collect_shards_multi_epoch(dataset, world_size, "
        "seed, num_epochs)`. Multi-epoch sharding inspector:\n\n"
        "1. Outer loop: `for epoch in range(num_epochs)`.\n"
        "2. Inner loop: `for rank in range(world_size)`. For each "
        "`(epoch, rank)`:\n"
        "   a. Build `DistributedSampler(dataset, num_replicas=world_size, "
        "rank=rank, shuffle=True, seed=seed)`.\n"
        "   b. Call `sampler.set_epoch(epoch)` BEFORE iterating.\n"
        "   c. Collect indices: `list(sampler)`.\n"
        "3. Return a `list[list[list[int]]]` of shape "
        "`[num_epochs][world_size]`.\n\n"
        "Inputs:\n"
        "- `dataset`: any Dataset (uses only `len(dataset)`).\n"
        "- `world_size`: int >= 1.\n"
        "- `seed`: int.\n"
        "- `num_epochs`: int >= 1.\n\n"
        "Output: nested list — `result[epoch][rank]` is the index list for "
        "rank `rank` on epoch `epoch`.\n\n"
        "**This drill does not call `dist.*` at all.** "
        "`DistributedSampler` is a pure-iterator construct that takes "
        "`(rank, num_replicas)` as args — no process group needed. The "
        "challenge is `set_epoch` discipline + per-epoch verification."
    ),
    "stub": (
        "def ex2_collect_shards_multi_epoch(dataset, world_size: int, seed: int, num_epochs: int) -> list:\n"
        '    """Returns list[epoch][rank][index] using DistributedSampler + set_epoch per epoch."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from torch.utils.data import Dataset\n"
        "from torch.utils.data.distributed import DistributedSampler\n"
        "\n"
        "class _RangeDataset(Dataset):\n"
        "    def __init__(self, n):\n"
        "        self.n = n\n"
        "    def __len__(self):\n"
        "        return self.n\n"
        "    def __getitem__(self, idx):\n"
        "        return idx\n"
        "\n"
        "ds = _RangeDataset(20)   # 20 items, divides evenly by 4\n"
        "shards = ex2_collect_shards_multi_epoch(ds, world_size=4, seed=0, num_epochs=3)\n"
        "\n"
        "# Outer shape: 3 epochs, 4 ranks per epoch.\n"
        "assert len(shards) == 3, f'expected 3 epochs, got {len(shards)}'\n"
        "for e in range(3):\n"
        "    assert len(shards[e]) == 4, f'epoch {e}: expected 4 ranks, got {len(shards[e])}'\n"
        "    for r in range(4):\n"
        "        assert len(shards[e][r]) == 5, (\n"
        "            f'epoch {e} rank {r}: expected 5 items (20/4), got {len(shards[e][r])}'\n"
        "        )\n"
        "\n"
        "# INVARIANT 1: per-epoch disjoint + coverage.\n"
        "for e in range(3):\n"
        "    all_seen = []\n"
        "    for r in range(4):\n"
        "        all_seen.extend(shards[e][r])\n"
        "    assert sorted(all_seen) == list(range(20)), (\n"
        "        f'epoch {e}: per-epoch coverage failed. Got sorted union {sorted(all_seen)}, expected 0..19'\n"
        "    )\n"
        "\n"
        "# INVARIANT 2: set_epoch produces DIFFERENT permutations.\n"
        "# Compare epoch 0 vs epoch 1 (and 0 vs 2).\n"
        "e0_full = [idx for r in range(4) for idx in shards[0][r]]\n"
        "e1_full = [idx for r in range(4) for idx in shards[1][r]]\n"
        "e2_full = [idx for r in range(4) for idx in shards[2][r]]\n"
        "assert e0_full != e1_full, 'set_epoch did not change the permutation (epochs 0 and 1 match!)'\n"
        "assert e0_full != e2_full, 'epoch 0 and epoch 2 produced the same permutation'\n"
        "\n"
        "# INVARIANT 3: within ONE epoch, all 4 ranks share the SAME base permutation.\n"
        "# That means rank 0 sees indices at positions [0, 4, 8, 12, 16] of the perm,\n"
        "# rank 1 sees [1, 5, 9, 13, 17], etc. Stride-by-world_size structure.\n"
        "# Reconstruct the full permutation by interleaving rank shards.\n"
        "for e in range(3):\n"
        "    interleaved = []\n"
        "    for pos in range(5):\n"
        "        for r in range(4):\n"
        "            interleaved.append(shards[e][r][pos])\n"
        "    assert sorted(interleaved) == list(range(20)), f'epoch {e}: interleave failed'\n"
        "    # Disjoint by construction — every index appears exactly once.\n"
        "    assert len(set(interleaved)) == 20\n"
        "\n"
        "# Non-divisible dataset — padding behavior.\n"
        "ds_pad = _RangeDataset(17)   # 17 items, 4 ranks → pad to 20\n"
        "shards_pad = ex2_collect_shards_multi_epoch(ds_pad, world_size=4, seed=0, num_epochs=1)\n"
        "# Each rank gets 5 items (ceil(17/4) = 5).\n"
        "for r in range(4):\n"
        "    assert len(shards_pad[0][r]) == 5, f'pad rank {r}: expected 5, got {len(shards_pad[0][r])}'\n"
        "# All 17 unique items appear somewhere; 3 of them appear twice (padding wraps).\n"
        "all_idx = [idx for r in range(4) for idx in shards_pad[0][r]]\n"
        "from collections import Counter\n"
        "cnt = Counter(all_idx)\n"
        "assert set(cnt.keys()) == set(range(17)), 'every original index must appear at least once'\n"
        "doubles = [k for k, v in cnt.items() if v == 2]\n"
        "assert len(doubles) == 3, f'expected exactly 3 padded duplicates, got {len(doubles)}'\n"
        "\n"
        "# Same seed → SAME multi-epoch trace (reproducibility).\n"
        "shards_again = ex2_collect_shards_multi_epoch(ds, world_size=4, seed=0, num_epochs=3)\n"
        "for e in range(3):\n"
        "    for r in range(4):\n"
        "        assert shards[e][r] == shards_again[e][r], (\n"
        "            f'reproducibility failed at epoch {e} rank {r}'\n"
        "        )\n"
        "\n"
        "# Different seed → different epoch-0 permutation.\n"
        "shards_diff_seed = ex2_collect_shards_multi_epoch(ds, world_size=4, seed=99, num_epochs=1)\n"
        "e0_diff = [idx for r in range(4) for idx in shards_diff_seed[0][r]]\n"
        "assert e0_full != e0_diff, 'different seed must give different permutation'"
    ),
    "solution_body": (
        "def ex2_collect_shards_multi_epoch(dataset, world_size: int, seed: int, num_epochs: int) -> list:\n"
        "    from torch.utils.data.distributed import DistributedSampler\n"
        "    out = []\n"
        "    for epoch in range(num_epochs):\n"
        "        epoch_shards = []\n"
        "        for rank in range(world_size):\n"
        "            sampler = DistributedSampler(\n"
        "                dataset,\n"
        "                num_replicas=world_size,\n"
        "                rank=rank,\n"
        "                shuffle=True,\n"
        "                seed=seed,\n"
        "            )\n"
        "            sampler.set_epoch(epoch)\n"
        "            epoch_shards.append(list(sampler))\n"
        "        out.append(epoch_shards)\n"
        "    return out"
    ),
    "solution_notes": (
        "**`set_epoch` BEFORE iteration, not after.** The shuffle RNG is "
        "seeded at the start of `__iter__`, using `(self.epoch, self.seed)` "
        "as inputs. Calling `set_epoch` after consuming the iterator does "
        "nothing — next iteration's RNG will use the new value, but the "
        "iterator you just collected is already locked in.\n\n"
        "**Reconstructing the base permutation by interleaving.** "
        "`DistributedSampler` builds `perm = torch.randperm(N, generator=g)` "
        "(or `perm + padding`) and slices it as `perm[rank::num_replicas]`. "
        "Interleaving the per-rank shards by position recovers the full "
        "perm — a useful debugging trick.\n\n"
        "**Padding behavior is the default.** When `len(dataset) % "
        "world_size != 0`, the sampler pads via "
        "`perm += perm[: total - N]` (wrap from the start) so every rank "
        "gets exactly `ceil(N/W)` items. To drop the tail instead: "
        "`DistributedSampler(..., drop_last=True)`. We don't drill drop_last "
        "here — it's a separate atom."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 4 — init-process-group-nccl ex2 (context-manager dist_session)
# ---------------------------------------------------------------------------

SPEC_DIST_SESSION = {
    "atom_id": "init-process-group-nccl",
    "subtopic": "Distributed: init_process_group nccl",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_DIST_SESSION,
    "exercise_index": 2,
    "exercise_title": "@contextmanager dist_session: destroy guaranteed on exception",
    "slug": "contextmanager-dist-session-destroy-on-exception",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["contextmanager", "init_process_group", "destroy_process_group", "try-finally"],
    "kcs": ["contextmanager-init-destroy", "destroy-runs-even-on-exception"],
    "lo": (
        "Apply `@contextlib.contextmanager` + `try/finally` to wrap "
        "`init_process_group` and `destroy_process_group` so destroy is "
        "guaranteed even when the body raises."
    ),
    "prompt_body": (
        "Implement `ex2_dist_session(rank, world_size, port, dist_module, "
        "backend='gloo')`, a context manager. Required behavior:\n\n"
        "1. Decorate with `@contextlib.contextmanager`.\n"
        "2. BEFORE the `yield`: set `os.environ['MASTER_ADDR'] = "
        "'127.0.0.1'` and `os.environ['MASTER_PORT'] = str(port)`. Call "
        "`dist_module.init_process_group(backend=backend, rank=rank, "
        "world_size=world_size)`.\n"
        "3. Wrap the `yield` in `try / finally`. The `finally` block MUST "
        "call `dist_module.destroy_process_group()`, with NO conditional "
        "guards — destroy always runs, even when the body raised.\n"
        "4. `yield` no value (a bare `yield`) — the context manager exists "
        "for its side effect.\n\n"
        "Input: `rank`, `world_size`, `port` — ints; `dist_module` — torch."
        "distributed or mock; `backend` — str, default `'gloo'`.\n"
        "Yields: nothing.\n\n"
        "The test runs the context manager (a) normally, (b) with an "
        "exception inside the `with` body. In both cases destroy must "
        "have been called exactly once."
    ),
    "stub": (
        "import contextlib\n"
        "import os\n"
        "\n"
        "@contextlib.contextmanager\n"
        "def ex2_dist_session(rank: int, world_size: int, port: int, dist_module, backend: str = 'gloo'):\n"
        '    """Init + destroy process group with try/finally; yields nothing."""\n'
        "    raise NotImplementedError()\n"
        "    yield  # unreachable — keeps generator framing"
    ),
    "test_body": (
        _FAKE_DIST_HARNESS + "\n\n"
        "import os\n"
        "\n"
        "# Normal-path: body runs, destroy fires.\n"
        "def _worker(rank, world_size, dist_module, world):\n"
        "    with ex2_dist_session(rank, world_size, 29610, dist_module):\n"
        "        # Inside body — group is up; mock barrier should succeed.\n"
        "        dist_module.barrier()\n"
        "        world.results[rank] = 'normal-path-ok'\n"
        "\n"
        "w = _run_fake_world(_worker, 3)\n"
        "for r in range(3):\n"
        "    assert w.results[r] == 'normal-path-ok', f'rank {r}: body did not run'\n"
        "\n"
        "# init_process_group called once per rank (3 inits total).\n"
        "init_calls = w.scratch.get('_init_calls', [])\n"
        "assert len(init_calls) == 3, f'expected 3 init calls, got {len(init_calls)}'\n"
        "# destroy_process_group called once per rank (3 destroys total).\n"
        "destroy_calls = w.scratch.get('_destroy_calls', [])\n"
        "assert len(destroy_calls) == 3, f'expected 3 destroy calls, got {len(destroy_calls)}'\n"
        "assert sorted(destroy_calls) == [0, 1, 2], f'destroy missed ranks: {destroy_calls}'\n"
        "\n"
        "# Env vars set.\n"
        "assert os.environ.get('MASTER_ADDR') == '127.0.0.1', 'MASTER_ADDR not set'\n"
        "assert os.environ.get('MASTER_PORT') == '29610', f\"MASTER_PORT got {os.environ.get('MASTER_PORT')!r}\"\n"
        "\n"
        "# Exception path: destroy STILL runs.\n"
        "class _BoomError(RuntimeError):\n"
        "    pass\n"
        "\n"
        "def _worker_boom(rank, world_size, dist_module, world):\n"
        "    raised = False\n"
        "    try:\n"
        "        with ex2_dist_session(rank, world_size, 29611, dist_module):\n"
        "            raise _BoomError(f'rank {rank} simulated crash')\n"
        "    except _BoomError:\n"
        "        raised = True\n"
        "    world.results[rank] = raised\n"
        "\n"
        "w_boom = _run_fake_world(_worker_boom, 2)\n"
        "for r in range(2):\n"
        "    assert w_boom.results[r] is True, f'rank {r}: exception did NOT propagate out of dist_session'\n"
        "# destroy still called for every rank despite the exception.\n"
        "destroy_calls_boom = w_boom.scratch.get('_destroy_calls', [])\n"
        "assert sorted(destroy_calls_boom) == [0, 1], (\n"
        "    f'exception path: destroy_process_group must run for every rank, got {destroy_calls_boom}.  '\n"
        "    f'Did you forget try/finally?'\n"
        ")\n"
        "init_calls_boom = w_boom.scratch.get('_init_calls', [])\n"
        "assert len(init_calls_boom) == 2, f'expected 2 init calls, got {len(init_calls_boom)}'\n"
        "\n"
        "# Backend kwarg is threaded through.\n"
        "def _worker_backend(rank, world_size, dist_module, world):\n"
        "    with ex2_dist_session(rank, world_size, 29612, dist_module, backend='nccl'):\n"
        "        pass\n"
        "    world.results[rank] = 'ok'\n"
        "\n"
        "w_b = _run_fake_world(_worker_backend, 2)\n"
        "init_calls_b = w_b.scratch.get('_init_calls', [])\n"
        "assert all(c.get('backend') == 'nccl' for c in init_calls_b), (\n"
        "    f'backend kwarg not threaded: {init_calls_b}'\n"
        ")"
    ),
    "solution_body": (
        "import contextlib\n"
        "import os\n"
        "\n"
        "@contextlib.contextmanager\n"
        "def ex2_dist_session(rank: int, world_size: int, port: int, dist_module, backend: str = 'gloo'):\n"
        "    os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        "    os.environ['MASTER_PORT'] = str(port)\n"
        "    dist_module.init_process_group(backend=backend, rank=rank, world_size=world_size)\n"
        "    try:\n"
        "        yield\n"
        "    finally:\n"
        "        dist_module.destroy_process_group()"
    ),
    "solution_notes": (
        "**Why `finally`, not `try/except`.** `try/except` would catch the "
        "exception and swallow it — wrong. `finally` runs the cleanup AND "
        "re-raises whatever was in flight. The two-line discipline of "
        "`init` → `try: yield ... finally: destroy` is the entire pattern.\n\n"
        "**Env vars before init.** `init_process_group` reads `MASTER_ADDR` "
        "and `MASTER_PORT` from `os.environ` if not passed as kwargs. "
        "Setting them BEFORE the init call (not after) is mandatory.\n\n"
        "**Real-world equivalent.** `torchrun` (the elastic launcher) wraps "
        "every worker in this exact pattern internally. For notebook-"
        "launched workers, hand-rolling the context manager gives you the "
        "same crash-resilience without launching from the CLI."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 5 — model-save-state-dict ex2 (atomic tmp+rename)
# ---------------------------------------------------------------------------

SPEC_ATOMIC_SAVE = {
    "atom_id": "model-save-state-dict",
    "subtopic": "Distributed: model save state_dict rank-0",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_SAVE_ATOMIC,
    "exercise_index": 2,
    "exercise_title": "atomic rank-0 save via tmp + os.replace + barrier",
    "slug": "atomic-rank-0-save-tmp-replace-barrier",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["state_dict", "atomic-save", "os.replace", "barrier", "checkpoint"],
    "kcs": ["rank0-tmp-then-rename", "barrier-after-save-for-readers"],
    "lo": (
        "Apply the `rank-0 save to tmp + os.replace + dist.barrier` pattern "
        "so the final checkpoint file never appears in a half-written state, "
        "verified by other ranks reading the file after the barrier."
    ),
    "prompt_body": (
        "Implement `ex2_atomic_save(rank, world_size, dist_module, model, "
        "ckpt_path)`. Crash-resilient rank-0 save:\n\n"
        "1. If `rank == 0`:\n"
        "   a. Build `tmp_path = ckpt_path + '.tmp'`.\n"
        "   b. `t.save(model.state_dict(), tmp_path)`.\n"
        "   c. `os.replace(tmp_path, ckpt_path)` — POSIX atomic rename.\n"
        "2. ALL ranks (including rank 0): call "
        "`dist_module.barrier()` — non-zero ranks block here until rank 0 "
        "finishes the rename.\n"
        "3. Return `ckpt_path` (so the caller has the final path).\n\n"
        "Guarantees the test verifies:\n"
        "- After this function returns, `ckpt_path` exists, "
        "`ckpt_path + '.tmp'` does NOT exist.\n"
        "- Loading from `ckpt_path` on any rank gives back the same "
        "state_dict that rank 0 saved.\n"
        "- The barrier was actually called (so non-zero ranks didn't race "
        "ahead).\n\n"
        "Input: `rank`, `world_size` — ints; `dist_module` — torch."
        "distributed or mock; `model` — `nn.Module`; `ckpt_path` — str.\n"
        "Output: `str` — the final checkpoint path."
    ),
    "stub": (
        "def ex2_atomic_save(rank: int, world_size: int, dist_module, model: 'nn.Module', ckpt_path: str) -> str:\n"
        '    """Save state_dict atomically from rank 0, barrier all ranks. Return final path."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FAKE_DIST_HARNESS + "\n\n"
        "import os\n"
        "import tempfile\n"
        "import torch.nn as nn\n"
        "\n"
        "# Use a temp dir so the test cleans up after itself.\n"
        "_tmpdir = tempfile.mkdtemp()\n"
        "_ckpt = os.path.join(_tmpdir, 'model.pt')\n"
        "\n"
        "def _worker(rank, world_size, dist_module, world):\n"
        "    model = nn.Linear(3, 3, bias=False)\n"
        "    with t.no_grad():\n"
        "        model.weight.fill_(11.0)\n"
        "    returned = ex2_atomic_save(rank, world_size, dist_module, model, _ckpt)\n"
        "    # Read back AFTER the barrier — every rank should see the saved file.\n"
        "    sd = t.load(returned, weights_only=True)\n"
        "    world.results[rank] = (returned, sd['weight'].sum().item())\n"
        "\n"
        "w = _run_fake_world(_worker, 3)\n"
        "\n"
        "# Every rank returned the original path; every rank loaded weight sum = 11 * 9 = 99.\n"
        "for r in range(3):\n"
        "    assert w.results[r] is not None, f'rank {r} returned None'\n"
        "    path, wsum = w.results[r]\n"
        "    assert path == _ckpt, f'rank {r}: returned {path}, expected {_ckpt}'\n"
        "    assert abs(wsum - 99.0) < 1e-5, f'rank {r}: loaded weight sum {wsum}, expected 99'\n"
        "\n"
        "# Final ckpt exists; tmp does NOT (was renamed away).\n"
        "assert os.path.exists(_ckpt), 'final ckpt path must exist after atomic save'\n"
        "assert not os.path.exists(_ckpt + '.tmp'), (\n"
        "    'tmp file must be renamed to final path (it should NOT linger after os.replace)'\n"
        ")\n"
        "\n"
        "# Verify the barrier was actually invoked — _FakeWorld.barrier uses threading.Barrier,\n"
        "# so if the student forgot the barrier the test still passes for value-correctness,\n"
        "# but we can detect it by patching dist_module.barrier and counting calls.\n"
        "# Easier: directly inspect that the loaded value is exact (which proves rank-0 finished\n"
        "# the rename before non-zero ranks loaded).\n"
        "\n"
        "# Re-save with a different value to confirm os.replace OVERWRITES the existing file.\n"
        "def _worker_overwrite(rank, world_size, dist_module, world):\n"
        "    model = nn.Linear(3, 3, bias=False)\n"
        "    with t.no_grad():\n"
        "        model.weight.fill_(22.0)\n"
        "    ex2_atomic_save(rank, world_size, dist_module, model, _ckpt)\n"
        "    sd = t.load(_ckpt, weights_only=True)\n"
        "    world.results[rank] = sd['weight'].sum().item()\n"
        "\n"
        "w_ov = _run_fake_world(_worker_overwrite, 2)\n"
        "for r in range(2):\n"
        "    assert abs(w_ov.results[r] - 198.0) < 1e-5, (\n"
        "        f'overwrite: rank {r} got {w_ov.results[r]}, expected 22*9=198'\n"
        "    )\n"
        "\n"
        "# Detection test: count barrier calls.\n"
        "_orig_barrier = w_ov.barrier_op\n"
        "_barrier_count = [0]\n"
        "\n"
        "# Build a fresh world manually to instrument barrier.\n"
        "import threading as _th\n"
        "class _CountingWorld(_FakeWorld):\n"
        "    def barrier_op(self):\n"
        "        _barrier_count[0] += 1\n"
        "        super().barrier_op()\n"
        "\n"
        "def _worker_count(rank, world_size, dist_module, world):\n"
        "    model = nn.Linear(2, 2, bias=False)\n"
        "    ex2_atomic_save(rank, world_size, dist_module, model, _ckpt)\n"
        "    world.results[rank] = 'ok'\n"
        "\n"
        "# Run inline with the counting world.\n"
        "import types as _types\n"
        "_cw = _CountingWorld(3)\n"
        "_errs = [None] * 3\n"
        "def _instrumented_runner(rank):\n"
        "    _cw.tls.rank = rank\n"
        "    fake_dist = _types.SimpleNamespace()\n"
        "    fake_dist.ReduceOp = _FakeReduceOp\n"
        "    fake_dist.broadcast = lambda tensor, src: _cw.broadcast(tensor, src)\n"
        "    fake_dist.barrier = _cw.barrier_op\n"
        "    fake_dist.all_reduce = lambda tensor, op='SUM': _cw.all_reduce(tensor, op)\n"
        "    fake_dist.get_rank = lambda: rank\n"
        "    fake_dist.get_world_size = lambda: 3\n"
        "    try:\n"
        "        _worker_count(rank, 3, fake_dist, _cw)\n"
        "    except BaseException as e:\n"
        "        _errs[rank] = repr(e)\n"
        "_threads = [_th.Thread(target=_instrumented_runner, args=(r,), daemon=True) for r in range(3)]\n"
        "for _th_ in _threads: _th_.start()\n"
        "for _th_ in _threads: _th_.join(timeout=15)\n"
        "for r, e in enumerate(_errs):\n"
        "    assert e is None, f'count run rank {r} failed: {e}'\n"
        "# Every rank called barrier ≥ once.\n"
        "assert _barrier_count[0] >= 3, (\n"
        "    f'expected at least 3 barrier calls (one per rank), got {_barrier_count[0]}.  '\n"
        "    f'Did you forget dist_module.barrier() on non-zero ranks?'\n"
        ")"
    ),
    "solution_body": (
        "def ex2_atomic_save(rank: int, world_size: int, dist_module, model: 'nn.Module', ckpt_path: str) -> str:\n"
        "    import os\n"
        "    if rank == 0:\n"
        "        tmp_path = ckpt_path + '.tmp'\n"
        "        t.save(model.state_dict(), tmp_path)\n"
        "        os.replace(tmp_path, ckpt_path)\n"
        "    dist_module.barrier()\n"
        "    return ckpt_path"
    ),
    "solution_notes": (
        "**`os.replace` vs `os.rename`.** `os.replace` overwrites the dest "
        "if it exists, on every OS. `os.rename` errors on Windows when the "
        "dest exists. For cross-platform code, always reach for `replace`.\n\n"
        "**Why the barrier even though only rank 0 writes.** Non-zero ranks "
        "didn't write anything, so they have nothing to wait for in "
        "ISOLATION. The barrier is for the CALLER's benefit — once "
        "`ex2_atomic_save` returns on every rank, downstream code can "
        "assume the file is durable. If a downstream `if rank == 1: "
        "load(ckpt)` ran without a prior barrier, rank 1 might attempt the "
        "load before rank 0 has finished writing.\n\n"
        "**Atomic write + barrier are independent guarantees.** Atomicity "
        "means 'never half-written on disk'. Barrier means 'every rank "
        "agrees the write is done'. Both are needed for crash-resilient "
        "multi-rank checkpointing."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 6 — mp-spawn-workers ex2 (non-blocking join via ProcessContext)
# ---------------------------------------------------------------------------

SPEC_SPAWN_NONBLOCK = {
    "atom_id": "mp-spawn-workers",
    "subtopic": "Distributed: mp.spawn workers",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_SPAWN_NONBLOCKING,
    "exercise_index": 2,
    "exercise_title": "non-blocking mp.spawn — ProcessContext + manual join",
    "slug": "non-blocking-mp-spawn-process-context-manual-join",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["mp.spawn", "join", "ProcessContext", "non-blocking", "pids"],
    "kcs": ["spawn-join-false-returns-context", "process-context-join-call"],
    "lo": (
        "Apply `mp.spawn(fn, args=..., nprocs=N, join=False)` to launch "
        "workers without blocking the launcher, then later call `pc.join()` "
        "on the returned `ProcessContext` to wait for completion."
    ),
    "prompt_body": (
        "Implement `ex2_launch_nonblocking(spawn_module, worker_fn, "
        "world_size, port)`. The non-blocking spawn pattern:\n\n"
        "1. Call `pc = spawn_module.spawn(worker_fn, args=(world_size, "
        "port), nprocs=world_size, join=False)`. Note: `join=False`.\n"
        "2. Return `pc` to the caller WITHOUT joining. The caller decides "
        "when to block via `pc.join()` later.\n\n"
        "Signature note: `spawn_module` is injected so the test can pass "
        "in a `_FakeSpawn` mock that doesn't actually fork processes (no "
        "real subprocess spawning needed on CPU). The mock returns a "
        "`_FakeProcessContext` with a `.join(timeout=None)` method and "
        "a `.pids()` method, matching torch's real `ProcessContext` API.\n\n"
        "**Why this matters in real code.** `torchrun`'s elastic launcher "
        "uses `join=False` so it can monitor worker liveness in a "
        "separate thread, kill the group if any worker hangs past a "
        "timeout, and restart the world from the last checkpoint. The "
        "blocking `join=True` form has no escape hatch — if one worker "
        "hangs, the launcher hangs too.\n\n"
        "Input: `spawn_module` — torch.multiprocessing or mock; "
        "`worker_fn` — callable; `world_size`, `port` — ints.\n"
        "Output: the unjoined `ProcessContext`-like object."
    ),
    "stub": (
        "def ex2_launch_nonblocking(spawn_module, worker_fn, world_size: int, port: int):\n"
        '    """Spawn workers with join=False; return the ProcessContext without joining."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Build a fake mp module that mimics torch.multiprocessing.spawn semantics.\n"
        "class _FakeProcessContext:\n"
        "    def __init__(self, worker_fn, world_size, args):\n"
        "        self.worker_fn = worker_fn\n"
        "        self.world_size = world_size\n"
        "        self.args = args\n"
        "        self._joined = False\n"
        "        self._results = []\n"
        "    def join(self, timeout=None):\n"
        "        # Run the workers serially in the calling thread — simulates spawn.\n"
        "        for rank in range(self.world_size):\n"
        "            self._results.append(self.worker_fn(rank, *self.args))\n"
        "        self._joined = True\n"
        "        return True\n"
        "    def pids(self):\n"
        "        return list(range(10000, 10000 + self.world_size))\n"
        "\n"
        "class _FakeSpawn:\n"
        "    def __init__(self):\n"
        "        self.calls = []\n"
        "    def spawn(self, fn, args=(), nprocs=1, join=True):\n"
        "        self.calls.append({'fn': fn, 'args': args, 'nprocs': nprocs, 'join': join})\n"
        "        pc = _FakeProcessContext(fn, nprocs, args)\n"
        "        if join:\n"
        "            pc.join()\n"
        "            return None\n"
        "        return pc\n"
        "\n"
        "def _toy_worker(rank, world_size, port):\n"
        "    return (rank, world_size, port, 'worker-ran')\n"
        "\n"
        "# Launch, check it didn't block.\n"
        "fake_mp = _FakeSpawn()\n"
        "pc = ex2_launch_nonblocking(fake_mp, _toy_worker, 4, 29710)\n"
        "\n"
        "# spawn was called exactly once with join=False, nprocs=world_size.\n"
        "assert len(fake_mp.calls) == 1, f'expected 1 spawn call, got {len(fake_mp.calls)}'\n"
        "call = fake_mp.calls[0]\n"
        "assert call['nprocs'] == 4, f'nprocs should equal world_size; got {call[\"nprocs\"]}'\n"
        "assert call['join'] is False, f'join must be False (non-blocking); got {call[\"join\"]}'\n"
        "assert call['args'] == (4, 29710), f'args wrong: {call[\"args\"]}'\n"
        "assert call['fn'] is _toy_worker, 'worker function not threaded through'\n"
        "\n"
        "# pc must be the unjoined ProcessContext-like object.\n"
        "assert pc is not None, 'must return the ProcessContext, not None (you joined too early)'\n"
        "assert isinstance(pc, _FakeProcessContext), f'pc must be ProcessContext-like, got {type(pc)}'\n"
        "assert pc._joined is False, 'must NOT call pc.join() inside the launcher — that defeats join=False'\n"
        "\n"
        "# Caller's later .join() runs the workers and returns True.\n"
        "ok = pc.join(timeout=5)\n"
        "assert ok is True, f'expected pc.join() to return True, got {ok!r}'\n"
        "assert pc._joined is True, 'after join, _joined should be True'\n"
        "assert len(pc._results) == 4, f'expected 4 worker results, got {len(pc._results)}'\n"
        "for r in range(4):\n"
        "    assert pc._results[r] == (r, 4, 29710, 'worker-ran'), (\n"
        "        f'rank {r} result wrong: {pc._results[r]}'\n"
        "    )\n"
        "\n"
        "# pids() exposes per-worker pids.\n"
        "pids = pc.pids()\n"
        "assert len(pids) == 4, f'expected 4 pids, got {len(pids)}: {pids}'\n"
        "\n"
        "# 2-rank case — fewer workers, same shape.\n"
        "fake_mp2 = _FakeSpawn()\n"
        "pc2 = ex2_launch_nonblocking(fake_mp2, _toy_worker, 2, 29711)\n"
        "assert fake_mp2.calls[0]['join'] is False\n"
        "assert fake_mp2.calls[0]['nprocs'] == 2\n"
        "assert pc2._joined is False\n"
        "pc2.join()\n"
        "assert pc2._joined is True\n"
        "assert len(pc2._results) == 2\n"
        "\n"
        "# Single-rank case — degenerate but legal.\n"
        "fake_mp1 = _FakeSpawn()\n"
        "pc1 = ex2_launch_nonblocking(fake_mp1, _toy_worker, 1, 29712)\n"
        "assert fake_mp1.calls[0]['nprocs'] == 1\n"
        "pc1.join()\n"
        "assert pc1._results[0] == (0, 1, 29712, 'worker-ran')"
    ),
    "solution_body": (
        "def ex2_launch_nonblocking(spawn_module, worker_fn, world_size: int, port: int):\n"
        "    pc = spawn_module.spawn(worker_fn, args=(world_size, port), nprocs=world_size, join=False)\n"
        "    return pc"
    ),
    "solution_notes": (
        "**`join=False` returns a `ProcessContext`.** With `join=True`, "
        "`spawn` blocks until every worker exits and returns `None`. With "
        "`join=False`, it returns immediately with a context object whose "
        "`.join(timeout=...)` you call manually. Choose `False` whenever "
        "the launcher has other work to do (monitoring, hot-restart logic, "
        "graceful shutdown handling).\n\n"
        "**`args=(world_size, port)`, not `args=(world_size, port,)` "
        "necessary?** Python tuples don't need the trailing comma when "
        "there are ≥2 elements. `args=()` for zero args; `args=(x,)` for "
        "one (the trailing comma matters there); `args=(x, y)` for two or "
        "more.\n\n"
        "**`pc.join(timeout=...)` returns a bool.** `True` if all workers "
        "exited cleanly within the timeout, `False` on timeout. The "
        "blocking `join=True` form raises on worker failure instead — same "
        "information, different ergonomics."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 7 — per-rank-cuda-device ex2 (per-rank ctx + no-collision invariant)
# ---------------------------------------------------------------------------

SPEC_PER_RANK_CTX = {
    "atom_id": "per-rank-cuda-device",
    "subtopic": "Distributed: per-rank cuda device",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_PER_RANK_CTX,
    "exercise_index": 2,
    "exercise_title": "per-rank ctx dict with no-shared-device invariant",
    "slug": "per-rank-context-dict-no-shared-device",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["per-rank-context", "device", "is_master", "no-collision", "mocked-cuda"],
    "kcs": ["build-rank-context-dict", "no-two-ranks-share-device-index"],
    "lo": (
        "Apply per-rank `torch.device(f'cuda:{rank}')` construction inside "
        "a context-dict builder, then verify the no-shared-device invariant "
        "across all ranks under mocked CUDA."
    ),
    "prompt_body": (
        "Implement `ex2_build_rank_ctx(rank, world_size)`. Build the "
        "per-rank training context as a dict:\n\n"
        "1. Construct `device = t.device(f'cuda:{rank}')`. DO NOT use bare "
        "`'cuda'` or hardcode `'cuda:0'`.\n"
        "2. Build the ctx dict with these keys:\n"
        "   - `'rank'`: `rank`\n"
        "   - `'world_size'`: `world_size`\n"
        "   - `'device'`: the `torch.device` from step 1\n"
        "   - `'is_master'`: `rank == 0` (bool)\n"
        "   - `'device_str'`: `f'cuda:{rank}'` (the canonical string form)\n"
        "3. Return the ctx dict.\n\n"
        "This function does NOT call `torch.cuda.set_device` or `.to(...)` "
        "— it's pure context construction. The test runs it for every "
        "rank in `range(world_size)`, then asserts the no-collision "
        "invariant: `len({c['device'].index for c in ctxs}) == "
        "world_size`.\n\n"
        "Input: `rank`, `world_size` — ints.\n"
        "Output: `dict[str, Any]` with the 5 keys above."
    ),
    "stub": (
        "def ex2_build_rank_ctx(rank: int, world_size: int) -> dict:\n"
        '    """Build per-rank context dict with rank, world_size, device, is_master, device_str."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Construct contexts for all 4 ranks (no CUDA needed — we never .to() anything).\n"
        "ctxs = [ex2_build_rank_ctx(rank=r, world_size=4) for r in range(4)]\n"
        "\n"
        "# Each ctx has the right keys.\n"
        "expected_keys = {'rank', 'world_size', 'device', 'is_master', 'device_str'}\n"
        "for r, ctx in enumerate(ctxs):\n"
        "    assert isinstance(ctx, dict), f'rank {r}: expected dict, got {type(ctx).__name__}'\n"
        "    assert set(ctx.keys()) == expected_keys, (\n"
        "        f'rank {r}: keys {set(ctx.keys())} != expected {expected_keys}'\n"
        "    )\n"
        "\n"
        "# rank + world_size threaded through.\n"
        "for r, ctx in enumerate(ctxs):\n"
        "    assert ctx['rank'] == r, f'ctx[\"rank\"] wrong on rank {r}'\n"
        "    assert ctx['world_size'] == 4, f'ctx[\"world_size\"] wrong on rank {r}'\n"
        "\n"
        "# device is a torch.device with the right index.\n"
        "for r, ctx in enumerate(ctxs):\n"
        "    d = ctx['device']\n"
        "    assert isinstance(d, t.device), f'rank {r}: device must be torch.device, got {type(d)}'\n"
        "    assert d.type == 'cuda', f'rank {r}: expected cuda type, got {d.type!r}'\n"
        "    assert d.index == r, f'rank {r}: expected cuda:{r}, got cuda:{d.index}'\n"
        "\n"
        "# device_str matches.\n"
        "for r, ctx in enumerate(ctxs):\n"
        "    assert ctx['device_str'] == f'cuda:{r}', (\n"
        "        f'rank {r}: device_str {ctx[\"device_str\"]!r}, expected {f\"cuda:{r}\"!r}'\n"
        "    )\n"
        "\n"
        "# is_master: True only on rank 0.\n"
        "assert ctxs[0]['is_master'] is True, 'rank 0 must have is_master=True'\n"
        "for r in [1, 2, 3]:\n"
        "    assert ctxs[r]['is_master'] is False, f'rank {r} must have is_master=False'\n"
        "\n"
        "# *** THE LOAD-BEARING INVARIANT *** — no two ranks share device index.\n"
        "indices = {ctx['device'].index for ctx in ctxs}\n"
        "assert len(indices) == 4, (\n"
        "    f'no-collision invariant FAILED — only {len(indices)} unique device indices '\n"
        "    f'across 4 ranks. Indices: {sorted(c[\"device\"].index for c in ctxs)}'\n"
        ")\n"
        "\n"
        "# Bigger world.\n"
        "ctxs8 = [ex2_build_rank_ctx(rank=r, world_size=8) for r in range(8)]\n"
        "indices8 = {ctx['device'].index for ctx in ctxs8}\n"
        "assert len(indices8) == 8, 'no-collision invariant FAILED at world_size=8'\n"
        "assert sum(1 for c in ctxs8 if c['is_master']) == 1, (\n"
        "    f'exactly one master expected; got {sum(1 for c in ctxs8 if c[\"is_master\"])}'\n"
        ")\n"
        "\n"
        "# Single-rank degenerate.\n"
        "ctx_solo = ex2_build_rank_ctx(rank=0, world_size=1)\n"
        "assert ctx_solo['device'].index == 0\n"
        "assert ctx_solo['is_master'] is True\n"
        "assert ctx_solo['world_size'] == 1\n"
        "\n"
        "# device_str + device must agree.\n"
        "for ctx in ctxs8:\n"
        "    assert t.device(ctx['device_str']) == ctx['device'], (\n"
        "        f'device_str {ctx[\"device_str\"]!r} does not parse back to device {ctx[\"device\"]!r}'\n"
        "    )"
    ),
    "solution_body": (
        "def ex2_build_rank_ctx(rank: int, world_size: int) -> dict:\n"
        "    device = t.device(f'cuda:{rank}')\n"
        "    return {\n"
        "        'rank': rank,\n"
        "        'world_size': world_size,\n"
        "        'device': device,\n"
        "        'is_master': rank == 0,\n"
        "        'device_str': f'cuda:{rank}',\n"
        "    }"
    ),
    "solution_notes": (
        "**Why the ctx dict, not 5 args.** Threading 5 arguments through "
        "every training-loop function call is painful and brittle. The "
        "ctx dict (or a frozen dataclass) is the canonical scaling form. "
        "Adding a 6th field (e.g. a per-rank RNG generator) becomes a "
        "one-line change.\n\n"
        "**`torch.device` is constructible without CUDA being available.** "
        "Constructing `t.device('cuda:3')` on a CPU-only machine returns "
        "a `torch.device` object — only `.to(device)` or "
        "`torch.cuda.set_device(device)` actually touch the driver. This "
        "is what lets us run the test on Colab CPU.\n\n"
        "**The no-collision invariant is the bug-catcher.** If a refactor "
        "accidentally pins all ranks to `cuda:0` (e.g. someone wrote "
        "`t.device('cuda')` instead of the f-string), all ranks land on "
        "the same GPU and OOM in the first batch. The set-size assertion "
        "in the test would catch this immediately."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 8 — rank-world-size-args ex2 (reduce-protocol signature)
# ---------------------------------------------------------------------------

SPEC_REDUCE_PROTOCOL = {
    "atom_id": "rank-world-size-args",
    "subtopic": "Distributed: rank/world_size args",
    "topic_folder": TOPIC_DIST,
    "atom_recap_md": RECAP_REDUCE_PROTOCOL,
    "exercise_index": 2,
    "exercise_title": "reduce-protocol — (tensor, rank, world_size, dst=0)",
    "slug": "reduce-protocol-rank-world-size-dst-signature",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["rank", "world_size", "dst", "reduce", "protocol-signature"],
    "kcs": ["rank-world-size-positional-args", "dst-rank-branching"],
    "lo": (
        "Apply the `(tensor, rank, world_size, dst=0)` signature convention "
        "for reduce — branch on `rank != dst` to decide between sending "
        "and receiving, listing `range(world_size)` for the fan-in."
    ),
    "prompt_body": (
        "Implement `ex2_reduce_protocol(tensor, rank, world_size, dst=0)` "
        "— a pure-logic dual of ex1's broadcast-protocol. Returns the "
        "list of `(action, other_rank)` tuples describing what THIS rank "
        "would do in a `reduce` collective.\n\n"
        "Spec:\n"
        "- If `rank == dst`: for each `other_rank in range(world_size)` "
        "where `other_rank != dst`, append `('recv', other_rank)`. Order "
        "matters — ascending by `other_rank`.\n"
        "- If `rank != dst`: return `[('send', dst)]`.\n"
        "- `tensor` is unused — only there to mirror the real signature.\n\n"
        "Return the list of `(action, other_rank)` tuples.\n\n"
        "This is the dual of ex1's broadcast-protocol:\n"
        "- ex1 (broadcast): rank == SRC sends; rank != src recvs from src.\n"
        "- ex2 (reduce): rank != DST sends to dst; rank == dst recvs from "
        "all others.\n"
        "Same plumbing, opposite arrow."
    ),
    "stub": (
        "def ex2_reduce_protocol(tensor: Tensor, rank: int, world_size: int, dst: int = 0) -> list:\n"
        '    """Pure-logic reduce protocol. Returns list[(action, other_rank)]."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "dummy = t.tensor([1.0])\n"
        "\n"
        "# rank 0 = dst, world_size=3 → recvs from 1 and 2.\n"
        "assert ex2_reduce_protocol(dummy, rank=0, world_size=3, dst=0) == [\n"
        "    ('recv', 1), ('recv', 2)\n"
        "]\n"
        "# rank 1 (not dst) → sends to dst=0.\n"
        "assert ex2_reduce_protocol(dummy, rank=1, world_size=3, dst=0) == [('send', 0)]\n"
        "assert ex2_reduce_protocol(dummy, rank=2, world_size=3, dst=0) == [('send', 0)]\n"
        "\n"
        "# Custom dst=2, world_size=4 → rank 2 recvs from 0,1,3; others send to 2.\n"
        "assert ex2_reduce_protocol(dummy, rank=2, world_size=4, dst=2) == [\n"
        "    ('recv', 0), ('recv', 1), ('recv', 3)\n"
        "]\n"
        "for non_dst in [0, 1, 3]:\n"
        "    assert ex2_reduce_protocol(dummy, rank=non_dst, world_size=4, dst=2) == [('send', 2)]\n"
        "\n"
        "# Single-rank degenerate world_size=1 → dst is the only rank, no recvs.\n"
        "assert ex2_reduce_protocol(dummy, rank=0, world_size=1, dst=0) == []\n"
        "\n"
        "# Signature check: dst defaults to 0, positional order matches convention.\n"
        "import inspect\n"
        "sig = inspect.signature(ex2_reduce_protocol)\n"
        "params = list(sig.parameters.values())\n"
        "names = [p.name for p in params]\n"
        "assert names == ['tensor', 'rank', 'world_size', 'dst'], f'signature order wrong: {names}'\n"
        "assert sig.parameters['dst'].default == 0, 'dst must default to 0'\n"
        "\n"
        "# Action labels are exactly 'send' and 'recv' — not 'reduce', 'forward', etc.\n"
        "for r in range(5):\n"
        "    actions = ex2_reduce_protocol(dummy, rank=r, world_size=5, dst=0)\n"
        "    for action, _ in actions:\n"
        "        assert action in ('send', 'recv'), f'unexpected action label {action!r}'\n"
        "\n"
        "# Cross-check duality with ex1's broadcast-protocol if it's defined:\n"
        "# A broadcast(src=k) has rank k as the SENDER (rank == src).\n"
        "# A reduce(dst=k) has rank k as the RECEIVER (rank == dst).\n"
        "# So reduce(dst=k) on rank k yields all 'recv' actions; broadcast(src=k) on rank k yields all 'send'.\n"
        "world_size = 5\n"
        "for k in range(world_size):\n"
        "    actions_at_dst = ex2_reduce_protocol(dummy, rank=k, world_size=world_size, dst=k)\n"
        "    # All recvs, one per other rank.\n"
        "    assert all(a == 'recv' for a, _ in actions_at_dst)\n"
        "    assert len(actions_at_dst) == world_size - 1\n"
        "    # Sorted ascending by other_rank.\n"
        "    sorted_others = sorted(o for _, o in actions_at_dst)\n"
        "    assert [o for _, o in actions_at_dst] == sorted_others\n"
        "\n"
        "# Edge case: world_size=2, dst=1.\n"
        "assert ex2_reduce_protocol(dummy, rank=0, world_size=2, dst=1) == [('send', 1)]\n"
        "assert ex2_reduce_protocol(dummy, rank=1, world_size=2, dst=1) == [('recv', 0)]"
    ),
    "solution_body": (
        "def ex2_reduce_protocol(tensor: Tensor, rank: int, world_size: int, dst: int = 0) -> list:\n"
        "    if rank == dst:\n"
        "        return [('recv', other) for other in range(world_size) if other != dst]\n"
        "    return [('send', dst)]"
    ),
    "solution_notes": (
        "**Broadcast vs reduce — same structure, opposite arrow.** Both "
        "have the `(tensor, rank, world_size, <canonical_rank>=0)` "
        "signature. In broadcast, the canonical rank is `src` (it OWNS "
        "the data others copy). In reduce, the canonical rank is `dst` "
        "(it WILL OWN the aggregated data). The branch flips: "
        "`rank == src` sends in broadcast; `rank == dst` recvs in reduce.\n\n"
        "**`all_reduce` drops both kwargs.** Every rank is both source and "
        "destination, so `all_reduce(tensor, rank, world_size, op)` — no "
        "`src` or `dst` needed. Same for `all_gather`. The presence of a "
        "single-rank arg is the signature signal that the topology has a "
        "canonical-data rank.\n\n"
        "**Why list-of-tuples is the right return type.** A `(rank, "
        "world_size, dst)` triple maps to a SET of point-to-point actions. "
        "Returning the list lets the test enumerate them in deterministic "
        "order without parsing prose."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# All specs
# ---------------------------------------------------------------------------

SPECS = [
    SPEC_BROADCAST_SD,
    SPEC_RING_PASS,
    SPEC_SAMPLER_EPOCH,
    SPEC_DIST_SESSION,
    SPEC_ATOMIC_SAVE,
    SPEC_SPAWN_NONBLOCK,
    SPEC_PER_RANK_CTX,
    SPEC_REDUCE_PROTOCOL,
]


# ---------------------------------------------------------------------------
# Verifier — exec stub + solution + test_body inside a single namespace.
# ---------------------------------------------------------------------------

def _verify_all(specs):
    import torch as t
    import numpy as np
    import torch.nn as nn
    from torch import Tensor
    import einops
    from einops import rearrange, reduce, repeat

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
            "einops": einops,
            "rearrange": rearrange,
            "reduce": reduce,
            "repeat": repeat,
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
            continue
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
    print(f"[deepening_r_batch11] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_r_batch11] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_r_batch11] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
