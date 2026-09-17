#!/usr/bin/env python3
"""Author 8 standalone Colab drills for PyTorch distributed-training atoms.

These atoms underpin ARENA chap-0 part-3 exercises 0_3_9 (broadcast),
0_3_10 (all_reduce), 0_3_11 (DistResNetTrainer). The composite ARENA
exercises bundle 13-32 atoms each — this batch breaks them into the
8 single-skill drills the procedural-drill atoms enumerate.

Colab/CPU notes:
- All drills use the `gloo` backend (works on CPU; `nccl` requires GPUs).
  The atom name `init-process-group-nccl` preserves ARENA's GPU-flavored
  framing in the recap text — students learn the pattern parametrically.
- Multi-process tests use either:
  * `mp.get_context('fork').Process` — works for worker fns defined in a
    notebook cell (Colab is Linux, fork is safe pre-fork-of-cuda).
  * `mp.spawn` — used only in the `mp-spawn-workers` drill, where the
    worker is written to a tempfile + imported (the canonical Colab
    pattern, since spawn cannot pickle functions defined in __main__).
- The `per-rank-cuda-device` drill is the only one that mocks: real CUDA
  is unavailable on Colab CPU runtimes. The student writes the canonical
  `torch.device(f'cuda:{rank}')` + `.to(device)` pattern; the test patches
  `torch.cuda.set_device` and `Tensor.to` to assert correct args.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_distributed"

# ---------------------------------------------------------------------------
# Shared boilerplate fragments
# ---------------------------------------------------------------------------

# Block used inside test bodies that need to spawn worker procs and capture
# results via a multiprocessing.Manager queue. Defined here so each test
# stays a focused assertion block.
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


RECAP_CORE = (
    "## torch.distributed quick refresher\n"
    "\n"
    "PyTorch's collective-communication library (`torch.distributed`, aliased "
    "`dist`) lets multiple processes coordinate over tensors. The standard "
    "workflow:\n"
    "\n"
    "1. **Each rank** runs the same function, parameterized by `rank` and "
    "`world_size`. Rank 0 is conventionally the 'driver'.\n"
    "2. **`dist.init_process_group(backend=...)`** establishes the rendezvous. "
    "Backends:\n"
    "   - `'nccl'` — NVIDIA's GPU-to-GPU primitive. Used in ARENA's multi-GPU "
    "setup. Requires CUDA + one process per GPU.\n"
    "   - `'gloo'` — CPU-friendly. What you'll use in these drills (Colab "
    "CPU runtimes have no real GPUs).\n"
    "3. **Pin a device** per rank: `torch.device(f'cuda:{rank}')` so each "
    "process owns exactly one GPU.\n"
    "4. **Collective ops** (`all_reduce`, `broadcast`, `send`, `recv`) operate "
    "in-place on tensors of identical shape across all ranks.\n"
    "5. **`dist.destroy_process_group()`** tears down at the end.\n"
    "\n"
    "**Two ways to launch multiple ranks:**\n"
    "- `torch.multiprocessing.spawn(fn, args=(...), nprocs=world_size)` — "
    "what ARENA uses. Spawn requires the worker fn be importable (not "
    "defined in `__main__`/a notebook cell).\n"
    "- `mp.get_context('fork').Process(target=fn, args=...)` — Linux-only "
    "but works with cell-defined fns. The drills use this in tests so the "
    "worker can stay in the cell.\n"
    "\n"
    "**Two-rank trick.** Colab gives ~2 CPU cores, so `world_size=2` is the "
    "right scale: enough to exercise the protocol, cheap enough to finish "
    "in seconds."
)


# ---------------------------------------------------------------------------
# 1. init-process-group-nccl  (use gloo in the actual test)
# ---------------------------------------------------------------------------

SPEC_INIT_PG = {
    "atom_id": "init-process-group-nccl",
    "subtopic": "Distributed: init_process_group nccl",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_CORE + (
        "\n\n### This drill's atom: `init_process_group`\n"
        "ARENA's solution calls `dist.init_process_group(backend='nccl', "
        "rank=rank, world_size=world_size)` and pairs it with "
        "`dist.destroy_process_group()`. The drill below uses **`'gloo'`** "
        "(CPU) so you can run it on Colab, but the call pattern is the same."
    ),
    "exercise_index": 1,
    "exercise_title": "init + destroy a process group with gloo",
    "slug": "init-and-destroy-a-process-group-with-gloo",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["init_process_group", "gloo", "nccl", "destroy_process_group", "lifecycle"],
    "kcs": ["init-process-group-call", "destroy-process-group-pair"],
    "lo": (
        "Apply `dist.init_process_group` + `dist.destroy_process_group` to "
        "open and cleanly tear down a 2-rank `gloo` process group from a "
        "worker function."
    ),
    "prompt_body": (
        "Implement `ex1_worker(rank, world_size, port)`. The minimum-viable "
        "distributed worker:\n\n"
        "1. Set the rendezvous env vars: `os.environ['MASTER_ADDR'] = "
        "'127.0.0.1'`, `os.environ['MASTER_PORT'] = str(port)`.\n"
        "2. Call `dist.init_process_group(backend='gloo', rank=rank, "
        "world_size=world_size, timeout=datetime.timedelta(seconds=20))`.\n"
        "3. After init, call `dist.get_rank()` and `dist.get_world_size()` "
        "and **print** them prefixed with `f'[rank {rank}] '` so the test "
        "harness can capture them.\n"
        "4. Call `dist.destroy_process_group()` before returning.\n\n"
        "The function takes a 3rd `port` arg so multiple tests can pick "
        "different ports and avoid collisions. The test spawns 2 forked "
        "procs and asserts both exit cleanly (exitcode 0).\n\n"
        "**Why `gloo`, not `nccl`?** `nccl` requires real GPUs. Colab CPU "
        "runtimes have none. ARENA uses `'nccl'` because they're on multi-"
        "GPU boxes. The argument is literally the only thing that changes "
        "between the two backends."
    ),
    "stub": (
        "import os\n"
        "import datetime\n"
        "import torch.distributed as dist\n"
        "\n"
        "def ex1_worker(rank: int, world_size: int, port: int) -> None:\n"
        '    """Init a gloo process group, print rank, destroy."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FORK_HARNESS + "\n\n"
        "# 2-rank fork test — both ranks must init, print, destroy, exit 0.\n"
        "codes = _dd_run_workers(ex1_worker, 2, 29510)\n"
        "assert codes == [0, 0], f'expected both ranks exit 0, got {codes}'\n"
        "\n"
        "# A 4-rank stress test on a different port.\n"
        "codes4 = _dd_run_workers(ex1_worker, 4, 29511)\n"
        "assert codes4 == [0, 0, 0, 0], f'expected 4 clean exits, got {codes4}'\n"
        "\n"
        "# Re-running on the SAME port must also work because each previous\n"
        "# run called destroy_process_group(). If the student skipped destroy,\n"
        "# this re-init will hang or error.\n"
        "codes_again = _dd_run_workers(ex1_worker, 2, 29510)\n"
        "assert codes_again == [0, 0], (\n"
        "    f'second init on same port failed ({codes_again}) — did you forget destroy_process_group?'\n"
        ")"
    ),
    "solution_body": (
        "import os\n"
        "import datetime\n"
        "import torch.distributed as dist\n"
        "\n"
        "def ex1_worker(rank: int, world_size: int, port: int) -> None:\n"
        "    os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        "    os.environ['MASTER_PORT'] = str(port)\n"
        "    dist.init_process_group(\n"
        "        backend='gloo',\n"
        "        rank=rank,\n"
        "        world_size=world_size,\n"
        "        timeout=datetime.timedelta(seconds=20),\n"
        "    )\n"
        "    print(f'[rank {rank}] dist.get_rank()={dist.get_rank()} dist.get_world_size()={dist.get_world_size()}')\n"
        "    dist.destroy_process_group()"
    ),
    "solution_notes": (
        "**The four env vars `init_process_group` needs.** `MASTER_ADDR` + "
        "`MASTER_PORT` (rendezvous endpoint) are mandatory. `RANK` + "
        "`WORLD_SIZE` are also looked up from env if not passed as kwargs — "
        "the explicit-kwargs form (used here) is clearer.\n\n"
        "**Why `timeout=20s` is generous.** The default is 30 minutes (yes, "
        "really). For a CPU test we want fast failure if rendezvous fails. "
        "Production code on slow networks uses the default.\n\n"
        "**`destroy_process_group()` matters.** Without it, the port stays "
        "bound and the next init on the same port hangs. The third test "
        "case above (re-using port 29510) is the regression check."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# 2. mp-spawn-workers — write worker to tempfile + import + spawn
# ---------------------------------------------------------------------------

SPEC_MP_SPAWN = {
    "atom_id": "mp-spawn-workers",
    "subtopic": "Distributed: mp.spawn workers",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_CORE + (
        "\n\n### This drill's atom: `mp.spawn`\n"
        "`torch.multiprocessing.spawn(fn, args=(...), nprocs=N, join=True)` "
        "is the canonical ARENA pattern. **Gotcha:** `spawn` pickles `fn` "
        "and sends it to the child — but functions defined in `__main__` "
        "(or a notebook cell) cannot be pickled. The drill teaches the "
        "two-line workaround: write the worker to a real `.py` file, "
        "`importlib.import_module` it, then spawn that imported attribute."
    ),
    "exercise_index": 1,
    "exercise_title": "launch a 2-rank distributed job with mp.spawn",
    "slug": "launch-a-2-rank-distributed-job-with-mp-spawn",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["mp.spawn", "nprocs", "join", "tempfile-worker", "pickling"],
    "kcs": ["mp-spawn-with-nprocs", "spawn-worker-must-be-importable"],
    "lo": (
        "Apply `mp.spawn` with `nprocs=2` to launch two worker procs that "
        "each call `init_process_group(gloo)`, by writing the worker to a "
        "temp `.py` file and importing it (Colab/notebook-correct pattern)."
    ),
    "prompt_body": (
        "Implement `ex1_launch_with_spawn(port)`. The full Colab-safe "
        "`mp.spawn` recipe:\n\n"
        "1. Use the pre-defined `WORKER_SRC` string (below in the stub) as "
        "the source of the worker function `worker(rank, world_size, port)`. "
        "It already does init/destroy.\n"
        "2. Write `WORKER_SRC` to `/tmp/dd_spawn_worker.py` (text mode).\n"
        "3. Add `/tmp` to `sys.path` if it's not there, then "
        "`importlib.import_module('dd_spawn_worker')` (use "
        "`importlib.reload` if it's already cached from a prior cell run).\n"
        "4. Call `mp.spawn(mod.worker, args=(2, port), nprocs=2, "
        "join=True)`.\n"
        "5. Return `True` on success.\n\n"
        "**Why the tempfile dance?** `mp.spawn` pickles the function ref and "
        "ships it to the child interpreter, which then needs to *import* "
        "the module to unpickle. Cell-defined functions live in "
        "`__main__` and aren't reachable from a fresh child. Writing to a "
        "real module file is the workaround.\n\n"
        "**ARENA cheat.** In ARENA's `.py` runner files this isn't needed — "
        "the worker is already at module scope. The dance is only required "
        "when launching from a Jupyter/Colab cell."
    ),
    "stub": (
        "import os, sys, importlib, tempfile\n"
        "import torch.multiprocessing as mp\n"
        "\n"
        "WORKER_SRC = '''\n"
        "import os, datetime\n"
        "import torch as t\n"
        "import torch.distributed as dist\n"
        "\n"
        "def worker(rank, world_size, port):\n"
        "    os.environ[\"MASTER_ADDR\"] = \"127.0.0.1\"\n"
        "    os.environ[\"MASTER_PORT\"] = str(port)\n"
        "    dist.init_process_group(backend=\"gloo\", rank=rank,\n"
        "                            world_size=world_size,\n"
        "                            timeout=datetime.timedelta(seconds=20))\n"
        "    x = t.tensor([float(rank + 1)])\n"
        "    dist.all_reduce(x, op=dist.ReduceOp.SUM)\n"
        "    assert x.item() == float(sum(range(1, world_size + 1)))\n"
        "    dist.destroy_process_group()\n"
        "'''\n"
        "\n"
        "def ex1_launch_with_spawn(port: int) -> bool:\n"
        '    """Write worker to /tmp, import, spawn 2 ranks, return True."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "ok = ex1_launch_with_spawn(29520)\n"
        "assert ok is True, f'expected True, got {ok!r}'\n"
        "\n"
        "# Re-run on a different port: file should already exist, reload path works.\n"
        "ok2 = ex1_launch_with_spawn(29521)\n"
        "assert ok2 is True\n"
        "\n"
        "# Verify the worker file actually ended up on disk.\n"
        "import os as _os\n"
        "assert _os.path.exists('/tmp/dd_spawn_worker.py'), 'worker file must be written to /tmp'\n"
        "with open('/tmp/dd_spawn_worker.py') as _f:\n"
        "    contents = _f.read()\n"
        "assert 'def worker' in contents, 'worker file must contain def worker'\n"
        "assert 'init_process_group' in contents, 'worker file must call init_process_group'"
    ),
    "solution_body": (
        "def ex1_launch_with_spawn(port: int) -> bool:\n"
        "    path = '/tmp/dd_spawn_worker.py'\n"
        "    with open(path, 'w') as f:\n"
        "        f.write(WORKER_SRC)\n"
        "    if '/tmp' not in sys.path:\n"
        "        sys.path.insert(0, '/tmp')\n"
        "    if 'dd_spawn_worker' in sys.modules:\n"
        "        mod = importlib.reload(sys.modules['dd_spawn_worker'])\n"
        "    else:\n"
        "        mod = importlib.import_module('dd_spawn_worker')\n"
        "    mp.spawn(mod.worker, args=(2, port), nprocs=2, join=True)\n"
        "    return True"
    ),
    "solution_notes": (
        "**`nprocs` vs `world_size`.** `nprocs` says how many child procs "
        "`spawn` should create. The worker fn receives `rank` as its FIRST "
        "positional arg automatically (spawn injects it); `world_size` you "
        "pass yourself via `args=(...)`. ARENA's convention: "
        "`args=(world_size,)` and `nprocs=world_size`.\n\n"
        "**`join=True` blocks the launcher.** Without it, `mp.spawn` returns "
        "immediately and you race against the children. Always `join=True` "
        "unless you have a specific reason to detach.\n\n"
        "**Why we accept the tempfile.** The whole point of teaching the "
        "drill on Colab is the gotcha — once you've felt the pickling "
        "error once, you remember to put worker code in `.py` files. In a "
        "real training repo, your trainer module is already importable, so "
        "the dance disappears."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# 3. per-rank-cuda-device  (MOCK: real cuda unavailable on Colab CPU)
# ---------------------------------------------------------------------------

SPEC_PER_RANK_CUDA = {
    "atom_id": "per-rank-cuda-device",
    "subtopic": "Distributed: per-rank cuda device",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_CORE + (
        "\n\n### This drill's atom: per-rank GPU pinning\n"
        "In multi-GPU DDP, **rank `r` owns GPU `r`**. The canonical pattern:\n"
        "```python\n"
        "device = torch.device(f'cuda:{rank}')\n"
        "torch.cuda.set_device(device)   # optional but recommended\n"
        "model = SimpleModel().to(device)\n"
        "x = torch.tensor([rank], dtype=torch.float32, device=device)\n"
        "```\n"
        "**Why the `f-string` over `cuda`?** Bare `'cuda'` means 'whatever "
        "the current device is' — fine for single-GPU, catastrophic for "
        "DDP (all ranks land on cuda:0 and OOM). The explicit `cuda:{rank}` "
        "guarantees correct sharding.\n\n"
        "**Why this drill mocks.** Colab CPU runtimes have no real GPUs, so "
        "we patch `torch.cuda.set_device` and `Tensor.to` to record the "
        "args, then assert on those records. The pattern you write is "
        "identical to what runs on a real multi-GPU box."
    ),
    "exercise_index": 1,
    "exercise_title": "pin per-rank cuda device with a mocked GPU",
    "slug": "pin-per-rank-cuda-device-with-a-mocked-gpu",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["cuda", "device", "rank-pinning", "mock", "DDP"],
    "kcs": ["device-fstring-per-rank", "model-to-device"],
    "lo": (
        "Apply the `torch.device(f'cuda:{rank}')` + `.to(device)` pattern to "
        "pin a fresh model and a fresh tensor to the rank's GPU, verified by "
        "mocking `torch.cuda.set_device` and `Tensor.to`."
    ),
    "prompt_body": (
        "Implement `ex1_pin_to_rank_device(rank, model, scalar_value)`. The "
        "canonical per-rank pinning recipe:\n\n"
        "1. Build a `torch.device` instance for `cuda:{rank}` using an "
        "f-string. DO NOT hardcode `'cuda:0'` or use bare `'cuda'`.\n"
        "2. Call `torch.cuda.set_device(device)` to make it the current "
        "device (defensive — keeps any subsequent op from accidentally "
        "landing on the wrong GPU).\n"
        "3. Move `model` to the device with `.to(device)` and re-assign.\n"
        "4. Build a tensor `t.tensor([scalar_value], dtype=t.float32, "
        "device=device)`.\n"
        "5. Return the tuple `(device, model, tensor)`.\n\n"
        "**The test mocks CUDA.** Colab has no real GPUs, so the test uses "
        "`unittest.mock.patch` to replace `torch.cuda.set_device` and "
        "`Tensor.to` with stubs that record what the student passed. Your "
        "code never actually moves anything to GPU memory — the test just "
        "asserts you called the right APIs with the right args."
    ),
    "stub": (
        "import torch as t\n"
        "from torch import Tensor\n"
        "\n"
        "def ex1_pin_to_rank_device(rank: int, model: t.nn.Module, scalar_value: float):\n"
        '    """Pin model + a scalar tensor to cuda:{rank}. Return (device, model, tensor)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from unittest.mock import patch, MagicMock\n"
        "\n"
        "class _ToyModel(t.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.p = t.nn.Parameter(t.tensor([1.0]))\n"
        "\n"
        "# Mock cuda.set_device + Tensor.to so the call works on CPU-only.\n"
        "set_device_calls = []\n"
        "\n"
        "def _fake_set_device(dev):\n"
        "    set_device_calls.append(dev)\n"
        "\n"
        "to_calls = []\n"
        "_real_to = t.Tensor.to\n"
        "_real_module_to = t.nn.Module.to\n"
        "\n"
        "def _fake_tensor_to(self, *a, **k):\n"
        "    to_calls.append(('Tensor', a, k))\n"
        "    return self  # stay on CPU — we just record the call\n"
        "\n"
        "def _fake_module_to(self, *a, **k):\n"
        "    to_calls.append(('Module', a, k))\n"
        "    return self\n"
        "\n"
        "with patch('torch.cuda.set_device', _fake_set_device), \\\n"
        "     patch.object(t.Tensor, 'to', _fake_tensor_to), \\\n"
        "     patch.object(t.nn.Module, 'to', _fake_module_to):\n"
        "    model = _ToyModel()\n"
        "    device, returned_model, tensor = ex1_pin_to_rank_device(3, model, 5.0)\n"
        "\n"
        "# Device must be cuda:3 exactly.\n"
        "assert isinstance(device, t.device), f'expected torch.device, got {type(device)}'\n"
        "assert device.type == 'cuda', f'expected cuda type, got {device.type!r}'\n"
        "assert device.index == 3, f'expected cuda:3, got cuda:{device.index}'\n"
        "\n"
        "# set_device must have been called with the same device.\n"
        "assert len(set_device_calls) == 1, f'expected exactly one set_device call, got {len(set_device_calls)}'\n"
        "assert set_device_calls[0].index == 3, f'set_device got wrong index: {set_device_calls[0]}'\n"
        "\n"
        "# Module.to and Tensor.to must each have been called once with the device.\n"
        "module_to_calls = [c for c in to_calls if c[0] == 'Module']\n"
        "tensor_to_calls = [c for c in to_calls if c[0] == 'Tensor']\n"
        "assert len(module_to_calls) == 1, f'model.to was called {len(module_to_calls)} times, expected 1'\n"
        "# The Tensor.to call comes from t.tensor(..., device=device) — torch internally\n"
        "# may or may not route through Tensor.to; the SOLE requirement is that the\n"
        "# returned tensor's device is cuda:3 (well, would be — we kept it on CPU\n"
        "# in the mock so we just check the device arg we asked for).\n"
        "# The MODEL.to call's first positional arg must be the device.\n"
        "_args, _kwargs = module_to_calls[0][1], module_to_calls[0][2]\n"
        "_passed = _args[0] if _args else _kwargs.get('device')\n"
        "assert _passed == device, f'model.to was called with {_passed!r}, expected {device!r}'\n"
        "\n"
        "# Returned model must be the same instance (.to returned self in mock).\n"
        "assert returned_model is model, 'must reassign model = model.to(device) and return it'\n"
        "\n"
        "# 4 different ranks → 4 different device indices.\n"
        "for r in [0, 1, 2, 7]:\n"
        "    with patch('torch.cuda.set_device', lambda d: None), \\\n"
        "         patch.object(t.Tensor, 'to', _fake_tensor_to), \\\n"
        "         patch.object(t.nn.Module, 'to', _fake_module_to):\n"
        "        d, _, _ = ex1_pin_to_rank_device(r, _ToyModel(), float(r))\n"
        "    assert d.index == r, f'rank {r} produced cuda:{d.index}'"
    ),
    "solution_body": (
        "def ex1_pin_to_rank_device(rank: int, model: t.nn.Module, scalar_value: float):\n"
        "    device = t.device(f'cuda:{rank}')\n"
        "    t.cuda.set_device(device)\n"
        "    model = model.to(device)\n"
        "    tensor = t.tensor([scalar_value], dtype=t.float32, device=device)\n"
        "    return device, model, tensor"
    ),
    "solution_notes": (
        "**Why `t.device(f'cuda:{rank}')` not `f'cuda:{rank}'`.** Both work as "
        "args to `.to(...)`, but constructing the `t.device` object once and "
        "reusing it (a) catches typos at construction time, (b) lets you "
        "pass the same canonical object to set_device / to() / tensor(...) — "
        "no chance of mismatched strings drifting apart.\n\n"
        "**`set_device` vs `to(device)`.** `set_device` makes a CUDA context "
        "the *default* for the current thread; `.to(device)` moves a "
        "specific tensor/module. ARENA solutions tend to skip `set_device` "
        "and rely purely on explicit `.to(device)` — both are valid, but "
        "calling `set_device` first prevents accidental cuda:0 allocations "
        "from third-party libs that don't take a device arg.\n\n"
        "**Reassignment matters for modules but not tensors.** "
        "`module.to(device)` mutates in place AND returns self; "
        "`tensor.to(device)` returns a NEW tensor (not in-place). Either "
        "way, write `x = x.to(device)` — it's the only form that's safe "
        "for both."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# 4. rank-world-size-args
# ---------------------------------------------------------------------------

SPEC_RANK_WORLD = {
    "atom_id": "rank-world-size-args",
    "subtopic": "Distributed: rank/world_size args",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_CORE + (
        "\n\n### This drill's atom: explicit `(rank, world_size)` plumbing\n"
        "Every distributed-aware function in ARENA's chap-0 part-3 has "
        "`rank` and `world_size` in its signature — `broadcast(tensor, "
        "rank, world_size, src=0)`, `reduce(tensor, rank, world_size, "
        "dst, op)`, etc. **Why repeat them everywhere?**\n"
        "- The launcher (`mp.spawn`) passes `rank` to the worker as the "
        "first positional arg. You then thread it through every call so "
        "each function can decide what THIS rank does.\n"
        "- `world_size` is constant across all ranks but needed for loops "
        "like `for other_rank in range(world_size):`.\n"
        "- Functions could read them from `dist.get_rank()` / "
        "`dist.get_world_size()`, but explicit args are cheaper, "
        "test-friendlier (you can call them without an active process "
        "group), and make the data-flow obvious."
    ),
    "exercise_index": 1,
    "exercise_title": "thread rank and world_size through a broadcast signature",
    "slug": "thread-rank-and-world-size-through-a-broadcast-signature",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["rank", "world_size", "signature", "src", "broadcast-protocol"],
    "kcs": ["rank-world-size-positional-args", "src-rank-branching"],
    "lo": (
        "Apply the `(tensor, rank, world_size, src=0)` signature convention "
        "from ARENA's broadcast — branch on `rank == src` to decide between "
        "sending and receiving, looping `range(world_size)` for the fanout."
    ),
    "prompt_body": (
        "Implement `ex1_broadcast_protocol(tensor, rank, world_size, src=0)` "
        "— a **pure-logic** stand-in for `dist.broadcast` that does NOT "
        "actually transmit. Instead, it returns a list of `(action, "
        "other_rank)` tuples describing what THIS rank would do, so we can "
        "test the protocol without a real process group.\n\n"
        "Spec:\n"
        "- If `rank == src`: for each `other_rank in range(world_size)` "
        "where `other_rank != src`, append `('send', other_rank)`. Order "
        "matters — must be ascending by `other_rank`.\n"
        "- If `rank != src`: return `[('recv', src)]`.\n"
        "- `tensor` is unused by this dry-run protocol — it's only there "
        "to mirror the real signature.\n\n"
        "Return the list of `(action, other_rank)` tuples.\n\n"
        "This drill is the **signature** atom — once you've internalized "
        "`(tensor, rank, world_size, src=0)`, the matching `reduce(tensor, "
        "rank, world_size, dst=0, op)` and `all_reduce(tensor, rank, "
        "world_size, op)` slot into muscle memory."
    ),
    "stub": (
        "import torch as t\n"
        "from torch import Tensor\n"
        "\n"
        "def ex1_broadcast_protocol(tensor: Tensor, rank: int, world_size: int, src: int = 0):\n"
        '    """Pure-logic broadcast protocol. Returns list[(action, other_rank)]."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "dummy = t.tensor([1.0])\n"
        "\n"
        "# rank 0 = src, world_size=3 → sends to 1 and 2 in order.\n"
        "assert ex1_broadcast_protocol(dummy, rank=0, world_size=3, src=0) == [\n"
        "    ('send', 1), ('send', 2)\n"
        "]\n"
        "# rank 1 (not src) → recv from src=0.\n"
        "assert ex1_broadcast_protocol(dummy, rank=1, world_size=3, src=0) == [('recv', 0)]\n"
        "assert ex1_broadcast_protocol(dummy, rank=2, world_size=3, src=0) == [('recv', 0)]\n"
        "\n"
        "# Custom src=2 → rank 2 sends to 0, 1, 3; others recv from 2.\n"
        "assert ex1_broadcast_protocol(dummy, rank=2, world_size=4, src=2) == [\n"
        "    ('send', 0), ('send', 1), ('send', 3)\n"
        "]\n"
        "for non_src in [0, 1, 3]:\n"
        "    assert ex1_broadcast_protocol(dummy, rank=non_src, world_size=4, src=2) == [('recv', 2)]\n"
        "\n"
        "# Single-rank degenerate world_size=1 → src is the only rank, no sends.\n"
        "assert ex1_broadcast_protocol(dummy, rank=0, world_size=1, src=0) == []\n"
        "\n"
        "# Signature check: src defaults to 0.\n"
        "import inspect\n"
        "sig = inspect.signature(ex1_broadcast_protocol)\n"
        "params = list(sig.parameters.values())\n"
        "names = [p.name for p in params]\n"
        "assert names == ['tensor', 'rank', 'world_size', 'src'], f'signature order wrong: {names}'\n"
        "assert sig.parameters['src'].default == 0, 'src must default to 0'"
    ),
    "solution_body": (
        "def ex1_broadcast_protocol(tensor: Tensor, rank: int, world_size: int, src: int = 0):\n"
        "    if rank == src:\n"
        "        return [('send', other) for other in range(world_size) if other != src]\n"
        "    return [('recv', src)]"
    ),
    "solution_notes": (
        "**The src/dst convention.** `src` for ops where one rank pushes to "
        "many (`broadcast`, `scatter`). `dst` for ops where many ranks "
        "push to one (`reduce`, `gather`). `all_*` variants drop both "
        "because every rank is both source and destination.\n\n"
        "**Why dry-run-able protocols are a debugging superpower.** When "
        "your real `broadcast` hangs in production, you cannot inspect what "
        "each rank was trying to do (the process is stuck inside C++). "
        "Writing a parallel pure-logic version like this one lets you call "
        "it for every `(rank, world_size, src)` combo and confirm the "
        "intended message pattern. ARENA does exactly this when "
        "demonstrating the broadcast diagram in the markdown."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# 5. dist-send-recv-pair
# ---------------------------------------------------------------------------

SPEC_SEND_RECV = {
    "atom_id": "dist-send-recv-pair",
    "subtopic": "Distributed: dist.send/recv pair",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_CORE + (
        "\n\n### This drill's atom: matched `send` / `recv`\n"
        "`dist.send(tensor, dst=R)` on the sending rank pairs with "
        "`dist.recv(buffer, src=S)` on the receiving rank. Both calls "
        "BLOCK until the match completes. **Critical invariant:** the "
        "`tensor` on the sender and the `buffer` on the receiver must "
        "have the **same shape and dtype**, otherwise the receiver hangs "
        "(or in newer torch, errors with a confusing message).\n\n"
        "The standard ARENA pattern (used in `broadcast`):\n"
        "```python\n"
        "if rank == src:\n"
        "    for other in range(world_size):\n"
        "        if other != src:\n"
        "            dist.send(tensor, dst=other)\n"
        "else:\n"
        "    buf = t.zeros_like(tensor)\n"
        "    dist.recv(buf, src=src)\n"
        "    tensor.copy_(buf)\n"
        "```\n"
        "(The `tensor.copy_(buf)` step mutates the caller's tensor so the "
        "result is visible without rebinding.)"
    ),
    "exercise_index": 1,
    "exercise_title": "implement broadcast via paired dist.send and dist.recv",
    "slug": "implement-broadcast-via-paired-dist-send-and-dist-recv",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["dist.send", "dist.recv", "broadcast", "tensor.copy_", "gloo"],
    "kcs": ["send-recv-shape-match", "recv-into-zeros_like-buffer"],
    "lo": (
        "Apply matched `dist.send` / `dist.recv` (with a `torch.zeros_like` "
        "buffer + `tensor.copy_` writeback) to build a 1-source-to-N-rank "
        "broadcast on the `gloo` backend."
    ),
    "prompt_body": (
        "Implement `ex1_broadcast_worker(rank, world_size, port, src, "
        "payload, out_queue)`. Each worker:\n\n"
        "1. Init `gloo` process group with `(rank, world_size, port)`.\n"
        "2. Build the tensor: if `rank == src`, use `t.tensor(payload, "
        "dtype=t.float32)`; else, use `t.zeros(len(payload), "
        "dtype=t.float32)` (the receive buffer).\n"
        "3. Run the matched send/recv broadcast:\n"
        "   - If `rank == src`: loop `for other in range(world_size)` and "
        "`dist.send(tensor, dst=other)` for every other rank.\n"
        "   - Else: `dist.recv(tensor, src=src)`. (No `copy_` needed here "
        "since we're receiving directly into the buffer we just built.)\n"
        "4. Push the final tensor's values onto `out_queue` as a list, "
        "tagged with the rank: `out_queue.put((rank, tensor.tolist()))`.\n"
        "5. `dist.destroy_process_group()`.\n\n"
        "**Why every non-src rank ends up with the same data.** Send/recv "
        "is a point-to-point primitive; broadcasting from rank `src` means "
        "running `world_size - 1` matched pairs. The test confirms that "
        "all ranks see the original `payload` after the protocol runs."
    ),
    "stub": (
        "import os, datetime\n"
        "import torch as t\n"
        "import torch.distributed as dist\n"
        "\n"
        "def ex1_broadcast_worker(rank, world_size, port, src, payload, out_queue):\n"
        '    """Init gloo, broadcast `payload` from `src` via send/recv, queue result."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FORK_HARNESS + "\n\n"
        "manager = _mp.Manager()\n"
        "q = manager.Queue()\n"
        "payload = [3.0, 1.0, 4.0, 1.0, 5.0]\n"
        "codes = _dd_run_workers(ex1_broadcast_worker, 3, 29530, 0, payload, q)\n"
        "assert codes == [0, 0, 0], f'workers failed: {codes}'\n"
        "\n"
        "# Drain queue.\n"
        "results = {}\n"
        "while not q.empty():\n"
        "    rank, vals = q.get()\n"
        "    results[rank] = vals\n"
        "assert set(results.keys()) == {0, 1, 2}, f'expected ranks 0,1,2, got {sorted(results.keys())}'\n"
        "for rank, vals in results.items():\n"
        "    assert vals == payload, f'rank {rank} got {vals}, expected {payload}'\n"
        "\n"
        "# Test with src=1 (not rank 0).\n"
        "q2 = manager.Queue()\n"
        "payload2 = [7.0, 7.0, 7.0]\n"
        "codes2 = _dd_run_workers(ex1_broadcast_worker, 3, 29531, 1, payload2, q2)\n"
        "assert codes2 == [0, 0, 0], f'src=1 workers failed: {codes2}'\n"
        "results2 = {}\n"
        "while not q2.empty():\n"
        "    rank, vals = q2.get()\n"
        "    results2[rank] = vals\n"
        "for rank in [0, 1, 2]:\n"
        "    assert results2[rank] == payload2, f'src=1: rank {rank} got {results2[rank]}'"
    ),
    "solution_body": (
        "def ex1_broadcast_worker(rank, world_size, port, src, payload, out_queue):\n"
        "    os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        "    os.environ['MASTER_PORT'] = str(port)\n"
        "    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size,\n"
        "                            timeout=datetime.timedelta(seconds=20))\n"
        "    if rank == src:\n"
        "        tensor = t.tensor(payload, dtype=t.float32)\n"
        "        for other in range(world_size):\n"
        "            if other != src:\n"
        "                dist.send(tensor, dst=other)\n"
        "    else:\n"
        "        tensor = t.zeros(len(payload), dtype=t.float32)\n"
        "        dist.recv(tensor, src=src)\n"
        "    out_queue.put((rank, tensor.tolist()))\n"
        "    dist.destroy_process_group()"
    ),
    "solution_notes": (
        "**`send`/`recv` blocks**. Both are synchronous. If you accidentally "
        "have rank A wait to recv from rank B while rank B is waiting to "
        "recv from rank A, you get a deadlock and the process hangs forever. "
        "The 30s timeout in the test harness will eventually kill it, but "
        "in production you'd want `dist.isend` / `dist.irecv` (non-blocking) "
        "or — better — a higher-level collective like `dist.broadcast`.\n\n"
        "**Why we don't use the real `dist.broadcast`.** `dist.broadcast` "
        "does exactly what this drill builds — in fewer lines and with NCCL "
        "tree-reduction acceleration. ARENA reimplements it by hand because "
        "the point is to internalize the protocol. In real code, always "
        "call `dist.broadcast(tensor, src=0)` and let the backend optimize.\n\n"
        "**`zeros_like` vs `zeros`.** ARENA's solution uses `t.zeros_like(t)` "
        "where `t` is a freshly-constructed sender-side template. Here we "
        "use `t.zeros(len(payload), dtype=t.float32)` because the non-src "
        "rank doesn't have access to the payload — only its length and dtype, "
        "which must be agreed in advance. Same idea, slightly different "
        "ergonomics."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# 6. all-reduce-compose  (reduce-then-broadcast)
# ---------------------------------------------------------------------------

SPEC_ALL_REDUCE_COMPOSE = {
    "atom_id": "all-reduce-compose",
    "subtopic": "Distributed: all_reduce composition",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_CORE + (
        "\n\n### This drill's atom: `all_reduce = reduce ∘ broadcast`\n"
        "ARENA's pedagogy: build `all_reduce` on top of the two primitives "
        "you already implemented:\n"
        "```python\n"
        "def all_reduce(tensor, rank, world_size, op='sum'):\n"
        "    reduce(tensor, rank, world_size, dst=0, op=op)   # → rank 0 holds the result\n"
        "    broadcast(tensor, rank, world_size, src=0)       # → every rank gets it\n"
        "```\n"
        "After `reduce`, only rank 0's tensor is correct. After the "
        "`broadcast`, every rank holds the same final value. "
        "**Real `dist.all_reduce`** does both in a single tree-reduction "
        "pass (faster, less memory) — but the composed form is the "
        "easy-to-reason-about correct baseline."
    ),
    "exercise_index": 1,
    "exercise_title": "compose all_reduce from reduce plus broadcast",
    "slug": "compose-all-reduce-from-reduce-plus-broadcast",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["all_reduce", "reduce", "broadcast", "composition", "tree-pattern"],
    "kcs": ["all-reduce-equals-reduce-then-broadcast", "rank0-as-aggregation-point"],
    "lo": (
        "Apply the `reduce`-then-`broadcast` composition to build a custom "
        "`all_reduce` that sums tensors across ranks, using real "
        "`dist.reduce` and `dist.broadcast` on the `gloo` backend."
    ),
    "prompt_body": (
        "Implement `ex1_all_reduce_worker(rank, world_size, port, initial, "
        "out_queue)`. Each rank starts with `initial * (rank + 1)` and the "
        "composed `all_reduce` should leave EVERY rank holding the sum "
        "`initial * (1 + 2 + ... + world_size)`.\n\n"
        "Steps inside the worker:\n"
        "1. Init `gloo` process group.\n"
        "2. Build `tensor = t.tensor([initial * (rank + 1)], "
        "dtype=t.float32)`.\n"
        "3. **Compose `all_reduce` from `reduce` + `broadcast`:**\n"
        "   ```python\n"
        "   dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)\n"
        "   dist.broadcast(tensor, src=0)\n"
        "   ```\n"
        "   (We're using the REAL `dist.reduce` and `dist.broadcast` — "
        "this drill is about the composition pattern, not reimplementing "
        "send/recv. ARENA's hand-rolled versions live in the previous "
        "drills.)\n"
        "4. Push `(rank, tensor.item())` onto `out_queue`.\n"
        "5. Destroy process group.\n\n"
        "**Check.** With `initial=1.0, world_size=3`, every rank should "
        "end with `6.0` (= 1+2+3). The test asserts this for ranks 0, 1, 2."
    ),
    "stub": (
        "import os, datetime\n"
        "import torch as t\n"
        "import torch.distributed as dist\n"
        "\n"
        "def ex1_all_reduce_worker(rank, world_size, port, initial, out_queue):\n"
        '    """Init gloo, compose all_reduce from reduce+broadcast, queue result."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FORK_HARNESS + "\n\n"
        "manager = _mp.Manager()\n"
        "q = manager.Queue()\n"
        "codes = _dd_run_workers(ex1_all_reduce_worker, 3, 29540, 1.0, q)\n"
        "assert codes == [0, 0, 0], f'workers failed: {codes}'\n"
        "\n"
        "results = {}\n"
        "while not q.empty():\n"
        "    rank, val = q.get()\n"
        "    results[rank] = val\n"
        "assert set(results.keys()) == {0, 1, 2}\n"
        "expected = 1.0 + 2.0 + 3.0\n"
        "for rank, val in results.items():\n"
        "    assert abs(val - expected) < 1e-5, f'rank {rank}: got {val}, expected {expected}'\n"
        "\n"
        "# 2-rank case with different initial.\n"
        "q2 = manager.Queue()\n"
        "codes2 = _dd_run_workers(ex1_all_reduce_worker, 2, 29541, 2.5, q2)\n"
        "assert codes2 == [0, 0]\n"
        "results2 = {}\n"
        "while not q2.empty():\n"
        "    rank, val = q2.get()\n"
        "    results2[rank] = val\n"
        "expected2 = 2.5 * 1 + 2.5 * 2  # = 7.5\n"
        "for rank, val in results2.items():\n"
        "    assert abs(val - expected2) < 1e-5, f'2-rank case rank {rank}: got {val}, expected {expected2}'"
    ),
    "solution_body": (
        "def ex1_all_reduce_worker(rank, world_size, port, initial, out_queue):\n"
        "    os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        "    os.environ['MASTER_PORT'] = str(port)\n"
        "    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size,\n"
        "                            timeout=datetime.timedelta(seconds=20))\n"
        "    tensor = t.tensor([initial * (rank + 1)], dtype=t.float32)\n"
        "    # Compose: reduce → rank 0, then broadcast back to all ranks.\n"
        "    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)\n"
        "    dist.broadcast(tensor, src=0)\n"
        "    out_queue.put((rank, tensor.item()))\n"
        "    dist.destroy_process_group()"
    ),
    "solution_notes": (
        "**Why the composition matters pedagogically.** Real DDP libs use "
        "`dist.all_reduce` directly — but understanding it as "
        "`reduce ∘ broadcast` explains:\n"
        "- Why the result is identical on every rank (the broadcast step).\n"
        "- Why summing across ranks needs O(world_size) bandwidth (each "
        "rank's tensor must visit rank 0 once, then return).\n"
        "- Why mean is just `sum / world_size` after the all_reduce (the "
        "next drill).\n\n"
        "**Tree reduction beats this.** `NCCL`'s `all_reduce` uses a "
        "ring-allreduce pattern: each chunk of the tensor flows around "
        "the ring of GPUs, getting summed pairwise. Total time is "
        "`2 * (N-1) * chunk_size / bandwidth`, vs `2 * N * tensor_size` "
        "for the naive compose. For large tensors on many GPUs, the "
        "speedup is enormous.\n\n"
        "**The `dist.ReduceOp` enum.** Other ops: `SUM`, `PRODUCT`, `MAX`, "
        "`MIN`, `BAND`, `BOR`, `BXOR`, `PREMUL_SUM`. There's NO `MEAN` op "
        "— you always sum then divide (next drill)."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# 7. all-reduce-grad-sync
# ---------------------------------------------------------------------------

SPEC_GRAD_SYNC = {
    "atom_id": "all-reduce-grad-sync",
    "subtopic": "Distributed: all_reduce grad sync",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_CORE + (
        "\n\n### This drill's atom: gradient synchronization via `all_reduce`\n"
        "The heart of data-parallel training:\n"
        "```python\n"
        "loss.backward()                  # each rank fills its OWN .grad tensors\n"
        "for p in model.parameters():\n"
        "    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)\n"
        "    p.grad /= world_size         # convert sum → mean\n"
        "optimizer.step()                 # now every rank takes the SAME step\n"
        "```\n"
        "Without grad sync, each rank's optimizer would drift independently "
        "— after a few steps the models diverge and training collapses. "
        "After grad sync, ranks stay bit-identical (modulo cross-GPU "
        "non-determinism in NCCL).\n\n"
        "**Why iterate parameters.** Each parameter has its own `.grad` "
        "tensor of independent shape; `all_reduce` operates on a single "
        "tensor at a time. Real DDP fuses these into 'buckets' for "
        "bandwidth efficiency — we're doing the unfused version."
    ),
    "exercise_index": 1,
    "exercise_title": "synchronize gradients across ranks with all_reduce",
    "slug": "synchronize-gradients-across-ranks-with-all-reduce",
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "difficulty_dots": "🔴🔴🔴🔴⚪",
    "keywords": ["DDP", "grad-sync", "all_reduce", "parameters", "mean"],
    "kcs": ["loop-parameters-all-reduce-grad", "divide-by-world-size-for-mean"],
    "lo": (
        "Apply `dist.all_reduce(p.grad, SUM)` + divide-by-world_size to "
        "average gradients across ranks for every parameter in a small model, "
        "after each rank has its own `.grad` populated."
    ),
    "prompt_body": (
        "Implement `ex1_grad_sync_worker(rank, world_size, port, "
        "out_queue)`. Each rank:\n\n"
        "1. Inits the `gloo` process group.\n"
        "2. Builds a fresh model: `model = SimpleModel()` (defined inline "
        "in the test — a `nn.Module` with one `nn.Parameter` `p` "
        "initialized to `[2.0, 4.0]`).\n"
        "3. Computes a rank-dependent loss so each rank gets a DIFFERENT "
        "gradient: `loss = (model.p * (rank + 1)).sum()`. Then "
        "`loss.backward()`. Each rank now has `p.grad = [rank+1, rank+1]`.\n"
        "4. **Synchronizes grads across ranks:**\n"
        "   ```python\n"
        "   for param in model.parameters():\n"
        "       dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)\n"
        "       param.grad /= world_size\n"
        "   ```\n"
        "5. Pushes `(rank, model.p.grad.tolist())` onto `out_queue`.\n"
        "6. Destroys process group.\n\n"
        "Expected: with `world_size=2`, rank 0's pre-sync grad is `[1, 1]` "
        "and rank 1's is `[2, 2]`. After sync (mean), BOTH ranks hold "
        "`[1.5, 1.5]`."
    ),
    "stub": (
        "import os, datetime\n"
        "import torch as t\n"
        "import torch.distributed as dist\n"
        "\n"
        "class SimpleModel(t.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.p = t.nn.Parameter(t.tensor([2.0, 4.0]))\n"
        "\n"
        "def ex1_grad_sync_worker(rank, world_size, port, out_queue):\n"
        '    """Init gloo, compute rank-specific grad, all_reduce mean across ranks."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FORK_HARNESS + "\n\n"
        "manager = _mp.Manager()\n"
        "q = manager.Queue()\n"
        "codes = _dd_run_workers(ex1_grad_sync_worker, 2, 29550, q)\n"
        "assert codes == [0, 0], f'workers failed: {codes}'\n"
        "\n"
        "results = {}\n"
        "while not q.empty():\n"
        "    rank, grad_list = q.get()\n"
        "    results[rank] = grad_list\n"
        "# Pre-sync grads: rank 0 has [1, 1], rank 1 has [2, 2]. Mean is [1.5, 1.5].\n"
        "expected = [1.5, 1.5]\n"
        "for rank in [0, 1]:\n"
        "    got = results[rank]\n"
        "    assert len(got) == 2, f'rank {rank}: grad has wrong length {len(got)}'\n"
        "    for i, (a, b) in enumerate(zip(got, expected)):\n"
        "        assert abs(a - b) < 1e-5, f'rank {rank} idx {i}: got {a}, expected {b}'\n"
        "\n"
        "# 3-rank case: pre-grads [1,1], [2,2], [3,3] → mean [2, 2].\n"
        "q2 = manager.Queue()\n"
        "codes2 = _dd_run_workers(ex1_grad_sync_worker, 3, 29551, q2)\n"
        "assert codes2 == [0, 0, 0]\n"
        "results2 = {}\n"
        "while not q2.empty():\n"
        "    rank, gl = q2.get()\n"
        "    results2[rank] = gl\n"
        "expected2 = [2.0, 2.0]\n"
        "for rank in [0, 1, 2]:\n"
        "    got = results2[rank]\n"
        "    for i, (a, b) in enumerate(zip(got, expected2)):\n"
        "        assert abs(a - b) < 1e-5, f'3-rank: rank {rank} idx {i}: got {a}, expected {b}'"
    ),
    "solution_body": (
        "def ex1_grad_sync_worker(rank, world_size, port, out_queue):\n"
        "    os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        "    os.environ['MASTER_PORT'] = str(port)\n"
        "    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size,\n"
        "                            timeout=datetime.timedelta(seconds=20))\n"
        "    model = SimpleModel()\n"
        "    loss = (model.p * (rank + 1)).sum()\n"
        "    loss.backward()\n"
        "    for param in model.parameters():\n"
        "        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)\n"
        "        param.grad /= world_size\n"
        "    out_queue.put((rank, model.p.grad.tolist()))\n"
        "    dist.destroy_process_group()"
    ),
    "solution_notes": (
        "**Order matters: `backward()` → `all_reduce` → `step()`.** If you "
        "all_reduce BEFORE backward, there's no grad to reduce yet (it's "
        "`None`). If you all_reduce AFTER step, you've already moved the "
        "parameters using a rank-local gradient — divergence.\n\n"
        "**Why `param.grad /= world_size` not `dist.all_reduce(..., "
        "op=ReduceOp.AVG)`.** `AVG` exists in newer torch (>= 1.10) for "
        "nccl, but `gloo` doesn't support it. Sum-then-divide is the "
        "portable form. ARENA's `reduce_op_mean_divide` atom is exactly "
        "this division step — the next drill drills it in isolation.\n\n"
        "**Real DDP does this for you.** `torch.nn.parallel."
        "DistributedDataParallel` wraps your model and hooks "
        "`backward` so grads are all_reduced as soon as each parameter's "
        "grad is computed (during backward, not after). The naïve loop "
        "in this drill is the conceptual model; DDP is the optimized "
        "production form."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# 8. reduce-op-mean-divide
# ---------------------------------------------------------------------------

SPEC_MEAN_DIVIDE = {
    "atom_id": "reduce-op-mean-divide",
    "subtopic": "Distributed: reduce-op mean divide",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_CORE + (
        "\n\n### This drill's atom: 'mean' = sum then in-place divide\n"
        "`dist.ReduceOp` enum has `SUM`, `PRODUCT`, `MAX`, `MIN`, but **no "
        "`MEAN`**. To average tensors across ranks portably:\n"
        "```python\n"
        "dist.all_reduce(tensor, op=dist.ReduceOp.SUM)\n"
        "tensor /= world_size   # in-place division\n"
        "```\n"
        "The `/=` matters — it mutates the tensor in place so anyone "
        "holding a reference (e.g. `param.grad`) sees the new value. "
        "`tensor = tensor / world_size` rebinds the local name but leaves "
        "the upstream `param.grad` reference pointing at the unscaled sum.\n\n"
        "ARENA's implementation:\n"
        "```python\n"
        "if op == 'mean':\n"
        "    tensor /= world_size\n"
        "```\n"
        "This drill turns the divide into a standalone function so you "
        "can verify the in-place semantics."
    ),
    "exercise_index": 1,
    "exercise_title": "implement mean reduction as sum-then-in-place-divide",
    "slug": "implement-mean-reduction-as-sum-then-in-place-divide",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["ReduceOp.SUM", "mean", "in-place-divide", "aliasing", "gloo"],
    "kcs": ["sum-then-divide-for-mean", "inplace-divide-preserves-references"],
    "lo": (
        "Apply `dist.all_reduce(SUM)` followed by `tensor /= world_size` "
        "(in-place) to mean-reduce across ranks, and verify the in-place "
        "step preserves an upstream `.grad` reference that points at the "
        "same storage."
    ),
    "prompt_body": (
        "Implement `ex1_mean_reduce_worker(rank, world_size, port, "
        "out_queue)`. Each rank:\n\n"
        "1. Inits `gloo` process group.\n"
        "2. Builds `param = t.nn.Parameter(t.zeros(3))` so `param.grad` "
        "can be assigned. Then sets `param.grad = t.tensor([float(rank+1), "
        "float(rank+1), float(rank+1)])` — rank-dependent gradient.\n"
        "3. Calls `ex1_mean_reduce_inplace(param.grad, world_size)` "
        "(your function — defined below).\n"
        "4. Pushes `(rank, param.grad.tolist())` AND a boolean "
        "`param.grad is grad_ref_at_start` (where you captured "
        "`grad_ref_at_start = param.grad` BEFORE step 3) onto `out_queue` "
        "as `(rank, grad_vals, ref_preserved_bool)`.\n"
        "5. Destroys process group.\n\n"
        "Implement `ex1_mean_reduce_inplace(tensor, world_size)`:\n"
        "1. `dist.all_reduce(tensor, op=dist.ReduceOp.SUM)`.\n"
        "2. **In-place** divide by world_size: `tensor /= world_size` "
        "(or equivalently `tensor.div_(world_size)`). DO NOT use "
        "`tensor = tensor / world_size` — that rebinds the local name and "
        "breaks the reference invariant.\n"
        "3. Return `None` (function mutates in-place).\n\n"
        "The test asserts:\n"
        "- All ranks end with the same mean grad.\n"
        "- The `param.grad is grad_ref_at_start` check is `True` on every "
        "rank (proves the in-place semantics)."
    ),
    "stub": (
        "import os, datetime\n"
        "import torch as t\n"
        "import torch.distributed as dist\n"
        "\n"
        "def ex1_mean_reduce_inplace(tensor, world_size):\n"
        '    """All-reduce with sum then divide IN PLACE by world_size."""\n'
        "    raise NotImplementedError()\n"
        "\n"
        "def ex1_mean_reduce_worker(rank, world_size, port, out_queue):\n"
        '    """Build param.grad, mean-reduce, queue (grad_vals, ref_preserved)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        _FORK_HARNESS + "\n\n"
        "manager = _mp.Manager()\n"
        "q = manager.Queue()\n"
        "codes = _dd_run_workers(ex1_mean_reduce_worker, 3, 29560, q)\n"
        "assert codes == [0, 0, 0], f'workers failed: {codes}'\n"
        "\n"
        "results = {}\n"
        "while not q.empty():\n"
        "    rank, grad_vals, ref_preserved = q.get()\n"
        "    results[rank] = (grad_vals, ref_preserved)\n"
        "\n"
        "# Pre-grads: rank 0 = [1,1,1], rank 1 = [2,2,2], rank 2 = [3,3,3]. Mean = [2,2,2].\n"
        "expected_grad = [2.0, 2.0, 2.0]\n"
        "for rank in [0, 1, 2]:\n"
        "    grad_vals, ref_preserved = results[rank]\n"
        "    for i, (a, b) in enumerate(zip(grad_vals, expected_grad)):\n"
        "        assert abs(a - b) < 1e-5, f'rank {rank} idx {i}: got {a}, expected {b}'\n"
        "    assert ref_preserved is True, (\n"
        "        f'rank {rank}: param.grad reference was rebound — '\n"
        "        f'did you write `tensor = tensor / world_size` instead of `tensor /= world_size`?'\n"
        "    )\n"
        "\n"
        "# 2-rank case: [1,1,1] and [2,2,2] → [1.5, 1.5, 1.5].\n"
        "q2 = manager.Queue()\n"
        "codes2 = _dd_run_workers(ex1_mean_reduce_worker, 2, 29561, q2)\n"
        "assert codes2 == [0, 0]\n"
        "while not q2.empty():\n"
        "    rank, gv, rp = q2.get()\n"
        "    for v in gv:\n"
        "        assert abs(v - 1.5) < 1e-5, f'2-rank: rank {rank} got {gv}'\n"
        "    assert rp is True"
    ),
    "solution_body": (
        "def ex1_mean_reduce_inplace(tensor, world_size):\n"
        "    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)\n"
        "    tensor /= world_size  # in-place — preserves caller's reference\n"
        "\n"
        "def ex1_mean_reduce_worker(rank, world_size, port, out_queue):\n"
        "    os.environ['MASTER_ADDR'] = '127.0.0.1'\n"
        "    os.environ['MASTER_PORT'] = str(port)\n"
        "    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size,\n"
        "                            timeout=datetime.timedelta(seconds=20))\n"
        "    param = t.nn.Parameter(t.zeros(3))\n"
        "    param.grad = t.tensor([float(rank + 1)] * 3)\n"
        "    grad_ref_at_start = param.grad\n"
        "    ex1_mean_reduce_inplace(param.grad, world_size)\n"
        "    ref_preserved = param.grad is grad_ref_at_start\n"
        "    out_queue.put((rank, param.grad.tolist(), ref_preserved))\n"
        "    dist.destroy_process_group()"
    ),
    "solution_notes": (
        "**`/=` vs `/`.** In Python, `x /= y` calls `x.__itruediv__(y)` "
        "which (for `Tensor`) mutates in place. `x = x / y` calls "
        "`x.__truediv__(y)`, allocates a new tensor, and rebinds `x` to it.\n\n"
        "For a free-standing local variable, both produce the right *value* "
        "but different *storage*. For a parameter's `.grad`, only the "
        "in-place form keeps the optimizer's grad reference in sync.\n\n"
        "**Why no `ReduceOp.AVG` on gloo.** NCCL added `AVG` in PyTorch "
        "1.10. Gloo never did. The portable, framework-agnostic recipe is "
        "always 'sum then divide'. If you only ever ship NCCL, "
        "`dist.all_reduce(t, op=dist.ReduceOp.AVG)` is cleaner — but the "
        "drill is gloo, so we use the portable form.\n\n"
        "**Float-precision footgun.** Summing then dividing accumulates "
        "rounding error proportional to `world_size`. For most ML use "
        "cases this is negligible (`world_size < 1024` in practice). If "
        "you ever do see drift between ranks, suspect non-determinism in "
        "the reduction order, not the divide."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------

SPECS = [
    SPEC_INIT_PG,
    SPEC_MP_SPAWN,
    SPEC_PER_RANK_CUDA,
    SPEC_RANK_WORLD,
    SPEC_SEND_RECV,
    SPEC_ALL_REDUCE_COMPOSE,
    SPEC_GRAD_SYNC,
    SPEC_MEAN_DIVIDE,
]


if __name__ == "__main__":
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"wrote {rel}")
