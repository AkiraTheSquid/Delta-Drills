#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


SHARED_DIRNAME = "Local_Deployed_Shared"
THIS_DIRNAME = "This-Directory-Only"
ALLOWED_ROOT_NAMES = {".git", ".gitignore", ".dockerignore", ".vercel", SHARED_DIRNAME, THIS_DIRNAME}
ALLOWED_SPLIT_METADATA_NAMES = {".gitignore", ".vercelignore", ".vercel"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh root compatibility symlinks for split Delta Drills trees.")
    parser.add_argument("--root", type=Path, default=None, help="Repo root to refresh. Defaults to the current Delta Drills repo root.")
    args = parser.parse_args()

    if args.root is None:
        root = Path(__file__).resolve().parents[2]
    else:
        root = args.root.resolve()

    shared_dir = root / SHARED_DIRNAME
    env_dir = root / THIS_DIRNAME
    if not env_dir.is_dir():
        raise RuntimeError(f"Expected {THIS_DIRNAME}/ under {root}")
    if not shared_dir.is_dir():
        raise RuntimeError(f"Expected {SHARED_DIRNAME}/ under {root}")

    for directory in (shared_dir, env_dir):
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if child.name == ".git":
                raise RuntimeError(f"Hidden repo metadata must stay at root, not under split dirs: {child}")
            if child.name.startswith(".") and child.name not in ALLOWED_SPLIT_METADATA_NAMES:
                raise RuntimeError(f"Unexpected hidden metadata inside split dir: {child}")

    for child in root.iterdir():
        if child.name in ALLOWED_ROOT_NAMES:
            continue
        if child.is_symlink():
            child.unlink()
            continue
        raise RuntimeError(f"Unexpected root-level path outside split layout: {child}")


if __name__ == "__main__":
    main()
