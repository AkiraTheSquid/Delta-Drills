"""
The kernel images: what a learner's Python finds already installed.

Two images, one recipe. The CPU image is the one every section opens on. The
GPU image is the same recipe on CUDA torch, plus what the gpu sections load
their big models with (`modal_gpu`). Modal hashes an image definition, so an
unchanged recipe costs one lookup, and ANY change here is a rebuild: prebuild
it (`image.build(app)` under `modal.enable_output()`) before deploying, or the
first spawn after the deploy blows `SPAWN_SECONDS` while the image builds.

Moved out of `modal_kernel.py` (2026-10-01) unchanged, so the CPU image hash
did not move.
"""

from __future__ import annotations

from pathlib import Path

# The ARENA edition the notebooks under lessons/notebooks/ were compiled from.
# Bump both together (scripts/compile_arena_notebooks.py reads the checkout at
# content/ARENA_5.0-main, so `git -C … rev-parse HEAD` is the value to paste).
ARENA_REPO = "https://github.com/callummcdougall/ARENA_3.0.git"
ARENA_SHA = "527f9376b40ad9a12ecd80490884b0009b54dd55"
ARENA_CHAPTERS = (
    "chapter0_fundamentals",
    "chapter1_transformer_interp",
    "chapter2_rl",
    "chapter3_llm_evals",
    "chapter4_alignment_science",
)

_CH1_PACKAGES = (
    "transformer_lens==2.17.0", "sae-lens>=4.0.0,<5.0.0", "openai==1.56.1",
    "tabulate", "umap-learn", "hdbscan", "frozendict",  # frozendict: sae_vis needs it, does not declare it
    "git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python",
    "git+https://github.com/callummcdougall/sae_vis.git@callum/v3",
    "git+https://github.com/neelnanda-io/neel-plotly",
)
# Chapter 2's setup line, adapted to Python 3.12. The `atari` extra pins
# ale-py 0.8, which has no 3.12 wheel; ale-py 0.9.1 bundles the ROMs and
# registers `ALE/*` with gymnasium 0.29. `mujoco-py` (2.3 bonus only) needs the
# old MuJoCo binaries; the modern `mujoco` wheel covers 2.3's own env. numpy
# stays on chapter 1's pin: the `other` extra's opencv/moviepy pull numpy 2,
# which transformer_lens 2.17 and circuitsvis refuse.
_CH2_PACKAGES = (
    "gymnasium[other]==0.29.0", "ale-py==0.9.1", "pygame", "wandb==0.18.7", "mujoco",
    "numpy==1.26.4", "opencv-python<4.11",
)
# Weights the gpt2-small sections load (1.1, 1.2, 1.3.3, 1.4.x, 2.4), into the
# image's HF cache so a fresh sandbox reads them from disk instead of the Hub.
_CH1_WEIGHTS = (
    "from transformer_lens import HookedTransformer; "
    "HookedTransformer.from_pretrained('gpt2-small'); "
    "from huggingface_hub import hf_hub_download; "
    "hf_hub_download('callummcdougall/attn_only_2L_half', 'attn_only_2L_half.pth')"
)

SHIM_PATH = Path(__file__).with_name("modal_kernel_shim.py")
SHIM_REMOTE = "/opt/delta/kernel_shim.py"


CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
# On top of the recipe, GPU only: every gpu section loads with
# device_map="auto" (accelerate), 1.3.1 loads Llama-2-13b in 8-bit
# (bitsandbytes), and 1.3.4 / 4.x load LoRA adapters (peft).
GPU_PACKAGES = ("accelerate", "bitsandbytes", "peft")


def build(modal, gpu: bool = False):
    """The kernel image. CPU: torch from the CPU wheel index (the CUDA wheel is
    2 GB of libraries nothing there can use), and every later pip layer keeps
    that index so a re-resolve cannot swap it in. GPU: PyPI's CUDA torch."""
    index = {} if gpu else {"extra_index_url": CPU_TORCH_INDEX}
    sparse = " ".join(f"{c}/exercises" for c in ARENA_CHAPTERS)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("wget", "unzip", "sudo", "git")
        .pip_install("torch", "torchvision", **index)
        .pip_install(
            "ipykernel", "jupyter_client", "ipython", "jupyter",
            "einops", "jaxtyping", "numpy", "plotly", "pandas", "tqdm", "rich",
            "torchinfo", "datasets", "pillow", "matplotlib", "wandb", "eindex-callum",
        )
        # Colab-shaped: /root/<chapter>/exercises exists, so ARENA's own
        # setup cell skips its download branch. Sparse + blobless keeps
        # the 100 MB repo down to the ~15 MB the exercises need.
        .run_commands(
            "cd /tmp && git clone --filter=blob:none --no-checkout --depth 1 "
            f"{ARENA_REPO} arena || git clone --filter=blob:none --no-checkout {ARENA_REPO} arena",
            f"cd /tmp/arena && git fetch --depth 1 origin {ARENA_SHA} && "
            f"git sparse-checkout init --cone && git sparse-checkout set {sparse} && "
            f"git checkout {ARENA_SHA}",
            *(f"mv /tmp/arena/{c} /root/{c}" for c in ARENA_CHAPTERS),
            "rm -rf /tmp/arena",
        )
        # Chapter 1 preloaded. Its setup cells guard the WHOLE %pip line on
        # `import transformer_lens`, so baking transformer_lens means baking
        # everything those lines install too — or 1.3.x / 1.4.2 / 1.5.x / 4.x
        # skip the install and lose sae-lens and openai. The CPU index keeps
        # any torch re-resolve off the 2 GB CUDA wheel.
        .pip_install(*_CH1_PACKAGES,
                     **index)
        .run_commands(f"python -c \"{_CH1_WEIGHTS}\"")
        # Chapter 2's setup cells guard their %pip line on `import
        # jaxtyping`, which is above — so without this layer gymnasium was
        # never installed. libgl/glib are for the opencv the `other` extra
        # pulls in.
        .apt_install("libgl1", "libglib2.0-0")
        .pip_install(*_CH2_PACKAGES,
                     **index)
        # ale-py 0.9 registers `ALE/*` only when imported (0.8 did it via
        # shimmy's plugin), and ARENA never imports it — so gymnasium
        # imports it, at the END of its own __init__ (ale_py calls
        # gymnasium.register, so any earlier is a circular import). The
        # make() fails the build if that ever stops working.
        .run_commands(
            "printf '\\ntry:\\n    import ale_py  # delta-drills: registers ALE/*\\n"
            "except ImportError:\\n    pass\\n' >> "
            "$(python -c 'import gymnasium as g; print(g.__file__)')",
            "python -c \"import gymnasium as g; g.make('ALE/Breakout-v5')\"",
        )
    )
    if gpu:
        image = image.pip_install(*GPU_PACKAGES)
    return image.add_local_file(SHIM_PATH, remote_path=SHIM_REMOTE, copy=True)
