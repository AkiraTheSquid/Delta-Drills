#!/usr/bin/env python3
"""
Fill the `dd-kernel-weights` Modal Volume with the model weights the ARENA GPU
sections load (spec: SPEC_MODAL_COMPUTE_BUDGET.md, pass 2, decision P4).

Why: gemma and Llama are gated on Hugging Face. A learner's GPU sandbox holds
no token (`modal_gpu.SANDBOX_ENV`), so it can only load a gated model that is
already in its cache, and this Volume is that cache, mounted read-only and
linked in at spawn. The big ungated models are here too, so a GPU sandbox
does not spend its first billed minutes downloading 16 GB.

The download runs on Modal, next to the Volume, not on this machine. Seth's
token comes from the Modal secret `dd-hf-token` (create it in the Modal
dashboard from the Hugging Face template, and accept the gemma and Llama
licences on huggingface.co with the same account). Without the secret the
gated repos are skipped and reported; rerun once it exists. A repo already
complete in the Volume costs one listing, so rerunning is cheap.

    backend/.venv/bin/python scripts/prefetch_gpu_weights.py           # everything
    backend/.venv/bin/python scripts/prefetch_gpu_weights.py --list    # what and how big
    backend/.venv/bin/python scripts/prefetch_gpu_weights.py --only Qwen/Qwen3-8B

Storage: Modal Volumes are free up to 1 TiB a month (2026-10-01); this is
about 130 GB.
"""

from __future__ import annotations

import argparse
import sys
import time

import modal

VOLUME = "dd-kernel-weights"      # modal_gpu.WEIGHTS_VOLUME
SECRET = "dd-hf-token"
MOUNT = "/vol"
CACHE = f"{MOUNT}/hub"            # modal_gpu links <mount>/hub into the HF cache

# repo -> (sections, gated). Only what a section loads at or after its
# `gpu_from` cell and fits an L4; 4.1's 14B and 4.4's 27B/32B are not here.
# Small ungated models (SAEs, gpt2-xl, tiny demos) download at run time.
MODELS = {
    "meta-llama/Llama-2-13b-hf": ("1-3-1", True),
    "meta-llama/Llama-3.1-8B-Instruct": ("1-3-1 1-3-4", True),
    "EleutherAI/gpt-j-6b": ("1-3-2", False),
    "google/gemma-2-2b": ("1-3-3 1-4-2", True),
    "google/gemma-2b-it": ("1-3-3", True),
    "Qwen/Qwen3-8B": ("1-3-4", False),
    "google/gemma-3-1b-it": ("1-4-2", True),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": ("4-3", False),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": ("4-3", False),
    "Qwen/Qwen2.5-7B-Instruct": ("4-4", False),
}
# Weights in safetensors when the repo has them, else the PyTorch .bin
# (gpt-j-6b has only that). Never the TF/Flax copies or Meta's `original/`.
META = ["*.json", "tokenizer*", "*.model", "*.txt", "*.tiktoken"]
IGNORE = ["original/*", "*.h5", "*.msgpack", "*.onnx", "*.gguf"]

app = modal.App("dd-prefetch-weights")
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_xet]==0.36.2")
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)


def _secret():
    try:
        secret = modal.Secret.from_name(SECRET)
        secret.hydrate()
        return [secret]
    except Exception:
        return []


SECRETS = _secret()


@app.function(image=image, volumes={MOUNT: volume}, secrets=SECRETS, timeout=3 * 3600,
              cpu=2, memory=4096)
def fetch(repos: list[str]) -> list[dict]:
    import os
    from huggingface_hub import HfApi, snapshot_download

    token = (os.environ.get("HF_TOKEN") or "").strip() or None
    api = HfApi(token=token)
    report = []
    for repo in repos:
        started = time.time()
        try:
            files = [f.rfilename for f in api.model_info(repo, files_metadata=False).siblings]
            weights = ["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["*.bin"]
            path = snapshot_download(repo, cache_dir=CACHE, token=token,
                                     allow_patterns=META + weights, ignore_patterns=IGNORE)
            size = sum(os.path.getsize(os.path.join(root, name))
                       for root, _, names in os.walk(path) for name in names)
            volume.commit()
            report.append({"repo": repo, "ok": True, "gb": round(size / 1e9, 1),
                           "seconds": round(time.time() - started)})
        except Exception as exc:
            report.append({"repo": repo, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", action="store_true", help="print the plan, download nothing")
    parser.add_argument("--only", nargs="*", help="these repos only")
    args = parser.parse_args()

    repos = args.only or list(MODELS)
    have_token = bool(SECRETS)
    plan = [r for r in repos if have_token or not MODELS.get(r, ("", False))[1]]
    skipped = [r for r in repos if r not in plan]
    for repo in repos:
        sections, gated = MODELS.get(repo, ("?", False))
        mark = "SKIP (gated, no dd-hf-token secret)" if repo in skipped else "fetch"
        print(f"  {mark:38s} {repo:45s} {'gated' if gated else '':6s} {sections}")
    if args.list or not plan:
        return 0 if args.list else 1
    with modal.enable_output(), app.run():
        rows = fetch.remote(plan)
    for row in rows:
        print(("  ok    " if row["ok"] else "  FAIL  ") + row["repo"],
              f"{row.get('gb')} GB in {row.get('seconds')} s" if row["ok"] else row["error"])
    if skipped:
        print(f"\n{len(skipped)} gated repo(s) skipped. Create the Modal secret '{SECRET}' "
              "(HF_TOKEN, an account that accepted the gemma + Llama licences) and rerun.")
    # Nonzero unless every requested repo is in the Volume: a partial cache is
    # not a ready one.
    return 0 if all(row["ok"] for row in rows) and not skipped else 1


if __name__ == "__main__":
    sys.exit(main())
