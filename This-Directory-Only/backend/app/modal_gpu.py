"""
GPU kernels for the ARENA sections whose model does not fit a CPU sandbox.

Spec: `This-Directory-Only/SPEC_MODAL_COMPUTE_BUDGET.md`, pass 2.

Every section opens on the CPU kernel. A section in `SECTIONS` with a
`gpu_from` cell moves to an L4 once the learner runs that cell or a later one:
the client sends the notebook's context with `GPU_SUFFIX`, and `modal_kernel`
treats the new context the way it treats a notebook switch. The CPU sandbox
closes, a GPU sandbox starts, and the client restores setup on it. Nothing
hops back per cell: the model and every variable live in that kernel's memory.

What bounds the spend, in the order it bites:
  * `GPU_IDLE_SECONDS` (5 min): a GPU sandbox closes sooner than a CPU one.
  * `GPU_HOURS_PER_LEARNER` a month, counted from the usage log.
  * `GPU_MAX` sandboxes at once across everyone.
  * The workspace ceiling in `modal_spend`, which counts GPU time at its rate.

Gated weights (gemma, Llama) are prefetched into the `WEIGHTS_VOLUME` with
Seth's HF token (`scripts/prefetch_gpu_weights.py`) and mounted read-only. A
sandbox never holds a token.
"""

from __future__ import annotations

import os
import threading

from app import modal_spend

GPU_SUFFIX = "#gpu"
GPU_TYPE = os.environ.get("DD_KERNEL_GPU", "L4")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


GPU_IDLE_SECONDS = int(_env_float("DD_KERNEL_GPU_IDLE_SECONDS", 300))
GPU_MAX = int(_env_float("DD_KERNEL_GPU_MAX", 2))
GPU_HOURS_PER_LEARNER = _env_float("DD_KERNEL_GPU_HOURS", 5)
# Requests beside the GPU. Weights load from the Volume through
# device_map="auto" straight onto the card, so host memory is a staging
# area, not the model's home; the sandbox bursts above it, billed as used.
GPU_CPU = _env_float("DD_KERNEL_GPU_CPU", 0.5)
GPU_MEMORY_MB = int(_env_float("DD_KERNEL_GPU_MEMORY_MB", 4096))
# A GPU sandbox boots the multi-GB CUDA image, so it gets longer than the
# CPU kernel's SPAWN_SECONDS before the request gives up on it.
SPAWN_SECONDS = int(_env_float("DD_KERNEL_GPU_SPAWN_SECONDS", 300))
# Modal list prices, $/h (modal.com/pricing, 2026-10-01). Sandbox GPUs bill
# at the standard rate; only sandbox CPU and memory carry the 3x.
GPU_USD_PER_HOUR = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "L40S": 1.95,
                    "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95}

WEIGHTS_VOLUME = "dd-kernel-weights"
WEIGHTS_MOUNT = "/hf-ro"
HF_CACHE = "/root/.cache/huggingface/hub"
# The Volume is read-only, and huggingface_hub writes lock files beside what
# it reads. So the sandbox's own (writable) cache gets a tree of symlinks into
# the Volume: reads go to the prefetched files, writes land locally.
# 1.3.1 asks for Llama 3.1 under its old name. The Hub redirects that name,
# but a sandbox with no token never reaches the Hub for a gated repo, and the
# cache is keyed by the name asked for, so the old name points at the new.
ALIASES = {"models--meta-llama--Meta-Llama-3.1-8B-Instruct": "models--meta-llama--Llama-3.1-8B-Instruct"}
LINK_WEIGHTS = (f"mkdir -p {HF_CACHE} && cp -rs {WEIGHTS_MOUNT}/hub/. {HF_CACHE}/ 2>/dev/null; cd {HF_CACHE} && "
                + "".join(f"[ -d {new} ] && [ ! -e {old} ] && ln -s {new} {old}; "
                          for old, new in ALIASES.items())
                + "true")
# 1.3.3 / 1.4.2 open their Gemma cells with `assert HF_TOKEN`. A single space
# passes that assert, and huggingface_hub strips it to "no token": nothing
# secret is in the sandbox, public repos still download, and the gated
# weights are already in the cache.
SANDBOX_ENV = {"HF_TOKEN": " "}

# Section (the notebook id, `arena:<id>`) → what it needs. Unlisted = cpu.
# `gpu_from` is the compiled-notebook cell that first loads a model a CPU
# sandbox cannot hold (scanned 2026-10-01 at ARENA 527f937; re-check when
# ARENA_SHA moves). 1.3.1's falls inside its setup cells, so it goes to the
# GPU as soon as setup runs.
SECTIONS = {
    "1-3-1": {"class": "gpu", "gpu_from": "1-3-1-c012"},   # Llama-2-13b 8-bit, Llama-3.1-8B
    "1-3-2": {"class": "gpu", "gpu_from": "1-3-2-c012"},   # gpt-j-6b, gpt2-xl
    "1-3-3": {"class": "gpu", "gpu_from": "1-3-3-c115"},   # gemma-2-2b; gpt2-small before it
    "1-3-4": {"class": "gpu", "gpu_from": "1-3-4-c012"},   # Qwen3-8B, Llama-3.1-8B
    "1-4-2": {"class": "gpu", "gpu_from": "1-4-2-c078"},   # gemma-3-1b-it + transcoders
    "4-3": {"class": "gpu", "gpu_from": "4-3-c016"},       # R1-Distill 1.5B / 8B
    "4-1": {"class": "gpu-big"},                           # Qwen2.5-14B + LoRA, ~30 GB
    "4-4": {"class": "gpu-big", "gpu_from": "4-4-c118"},   # 27B/32B; the Qwen2.5-7B route fits
    "3-1": {"class": "api"}, "3-2": {"class": "api"}, "3-3": {"class": "api"},
    "3-4": {"class": "api"}, "3-5": {"class": "api"}, "4-2": {"class": "api"},
    "4-5": {"class": "api"},
}

GPU_BUSY_MESSAGE = ("Both notebook GPUs are in use right now — try this cell again in a few "
                    "minutes. Cells before the GPU part still run.")


def _cap_message(used: float) -> str:
    return (f"You have used your {GPU_HOURS_PER_LEARNER:g} GPU hours for this month "
            f"({used:.1f} h). The GPU part of this section comes back on the 1st; cells "
            "before it still run.")


def section_of(context: str) -> str:
    """`arena:1-3-3#gpu` → `1-3-3`. Anything not an ARENA notebook → ''."""
    if not context.startswith("arena:"):
        return ""
    return context[len("arena:"):].split("#", 1)[0]


def wants_gpu(context: str) -> bool:
    """A GPU context for a section that has a GPU part. The section check is
    the spend guard: a client cannot put 0.0 on a GPU by adding the suffix."""
    return (context or "").endswith(GPU_SUFFIX) and bool(SECTIONS.get(section_of(context), {}).get("gpu_from"))


def idle_seconds(context: str, cpu_idle: int) -> int:
    return GPU_IDLE_SECONDS if wants_gpu(context) else cpu_idle


def usd_per_hour() -> float:
    """What one GPU sandbox bills at its requests."""
    return GPU_USD_PER_HOUR.get(GPU_TYPE, 1.0) + modal_spend.request_rate(GPU_CPU, GPU_MEMORY_MB)


def hours_used(session_id: str, live_starts: list[float]) -> float:
    return modal_spend.gpu_hours(session_id, live_starts)


def admit(session_id: str, live_gpu: int, live_starts: list[float]) -> None:
    """Raise with the learner-facing reason when a GPU sandbox may not start.
    `live_gpu` counts every open GPU sandbox; `live_starts` are this learner's."""
    if live_gpu >= GPU_MAX:
        raise RuntimeError(GPU_BUSY_MESSAGE)
    used = hours_used(session_id, live_starts)
    if used >= GPU_HOURS_PER_LEARNER:
        raise RuntimeError(_cap_message(used))


def sandbox_options(modal) -> dict:
    """`Sandbox.create` keywords for a GPU sandbox (its image is `image()`)."""
    volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
    return {"gpu": GPU_TYPE, "cpu": GPU_CPU, "memory": GPU_MEMORY_MB,
            "volumes": {WEIGHTS_MOUNT: volume.read_only()}}


def link_weights(sandbox) -> None:
    proc = sandbox.exec("sh", "-c", LINK_WEIGHTS)
    proc.wait()


_image = None
_image_lock = threading.Lock()


def image(modal):
    global _image
    with _image_lock:
        if _image is None:
            from app import modal_image
            _image = modal_image.build(modal, gpu=True)
        return _image


def learner_view(session_id: str | None, live_starts: list[float], enabled: bool) -> dict:
    """What the notebook needs to place its banner and its GPU switch."""
    used = hours_used(session_id, live_starts) if session_id else 0.0
    return {
        "gpu_enabled": enabled,
        "gpu_type": GPU_TYPE,
        "gpu_idle_seconds": GPU_IDLE_SECONDS,
        "gpu_hours_cap": GPU_HOURS_PER_LEARNER,
        "gpu_hours_used": round(used, 2),
        "gpu_hours_left": round(max(0.0, GPU_HOURS_PER_LEARNER - used), 2),
        "sections": SECTIONS,
    }
