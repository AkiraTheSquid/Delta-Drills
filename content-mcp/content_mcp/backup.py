"""The one-day-old safety net.

Exactly ONE snapshot of the content tree is kept. It is refreshed lazily,
before the first write of any day: if the snapshot on disk is younger than
`MIN_AGE_SECONDS` it is left alone, so a snapshot always predates today's
editing session and a bad afternoon cannot overwrite the good copy of the
morning. Restoring puts every content file back exactly as the snapshot found
it.

Deliberately NOT a version history. Seth asked for one backup that is about a
day old, and a single file is a thing an author can reason about — `restore`
has one meaning. `keep_extra_on_restore` is the one hedge: restoring first
parks the CURRENT tree beside the snapshot, so an accidental restore is itself
undoable.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import time
from pathlib import Path

from . import paths

MIN_AGE_SECONDS = 24 * 60 * 60
ARCHIVE_NAME = "content-backup.tar.gz"
META_NAME = "content-backup.json"
PRE_RESTORE_NAME = "pre-restore.tar.gz"


def archive_path() -> Path:
    return paths.state_dir() / ARCHIVE_NAME


def meta_path() -> Path:
    return paths.state_dir() / META_NAME


def _members() -> list[tuple[Path, str]]:
    """Every file the snapshot carries, as (absolute path, arcname)."""
    out: list[tuple[Path, str]] = []
    for rel, globs in paths.CONTENT_PATHS:
        source = paths.REPO / rel
        if not source.exists():
            continue
        if source.is_file():
            out.append((source, rel))
            continue
        candidates = []
        if globs is None:
            candidates = [p for p in source.rglob("*") if p.is_file()]
        else:
            for pattern in globs:
                candidates.extend(p for p in source.glob(pattern) if p.is_file())
        for path in sorted(candidates):
            if any(part in paths.BACKUP_EXCLUDE for part in path.relative_to(source).parts):
                continue
            if path.name in paths.BACKUP_EXCLUDE:
                continue
            out.append((path, str(path.relative_to(paths.REPO))))
    return out


def _write_archive(target: Path) -> dict:
    members = _members()
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        for path, arcname in members:
            tar.add(path, arcname=arcname)
    tmp.replace(target)
    return {
        "created_epoch": time.time(),
        "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "file_count": len(members),
        "bytes": target.stat().st_size,
        "repo": str(paths.REPO),
    }


def status() -> dict:
    archive = archive_path()
    if not archive.exists():
        return {"exists": False, "path": str(archive), "age_hours": None, "stale": True}
    meta = {}
    if meta_path().exists():
        try:
            meta = json.loads(meta_path().read_text())
        except (json.JSONDecodeError, OSError):
            meta = {}
    age = time.time() - archive.stat().st_mtime
    return {
        "exists": True,
        "path": str(archive),
        "age_hours": round(age / 3600, 2),
        "stale": age >= MIN_AGE_SECONDS,
        "created_iso": meta.get("created_iso"),
        "file_count": meta.get("file_count"),
        "bytes": archive.stat().st_size,
    }


def snapshot(force: bool = False) -> dict:
    """Refresh the snapshot if it is missing or at least a day old.

    Returns what it did, so callers can report it: `rotated` False means the
    existing snapshot was young enough to keep, which is the normal answer on
    the second and later edits of a day.
    """
    current = status()
    if current["exists"] and not current["stale"] and not force:
        return {"rotated": False, "reason": "snapshot is younger than 24h", **current}
    meta = _write_archive(archive_path())
    meta_path().write_text(json.dumps(meta, indent=2) + "\n")
    return {"rotated": True, "reason": "forced" if force else "snapshot missing or stale", **status()}


class SnapshotError(RuntimeError):
    """Raised when the safety copy cannot be written."""


def ensure() -> dict:
    """Called before every mutating op. Fails CLOSED.

    A full disk or a permission problem must stop the write, not proceed
    without the recovery copy the caller was promised. Swallowing this is how
    you end up with an unrecoverable mess and a green result line.
    """
    try:
        return snapshot(force=False)
    except OSError as err:
        raise SnapshotError(
            f"Refusing to write: the content snapshot could not be created ({err}). "
            f"Free space or fix permissions on {paths.state_dir()}, then retry."
        ) from err


def contents() -> list[str]:
    archive = archive_path()
    if not archive.exists():
        return []
    with tarfile.open(archive, "r:gz") as tar:
        return sorted(m.name for m in tar.getmembers() if m.isfile())


def restore(paths_filter: list[str] | None = None, keep_current: bool = True) -> dict:
    """Put the snapshot back. Optionally only the given repo-relative paths."""
    archive = archive_path()
    if not archive.exists():
        raise FileNotFoundError(f"No snapshot to restore from at {archive}")

    parked = None
    if keep_current:
        parked = paths.state_dir() / PRE_RESTORE_NAME
        _write_archive(parked)

    restored: list[str] = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if paths_filter and not any(
                member.name == p or member.name.startswith(p.rstrip("/") + "/")
                for p in paths_filter
            ):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            target = paths.REPO / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as handle:
                shutil.copyfileobj(extracted, handle)
            restored.append(member.name)

    # Restore overwrites; it does not DELETE. A file created after the snapshot
    # is not in the archive, so it survives — which is the safe default in a
    # repo where other sessions are working, but it means the tree is not
    # byte-identical to the snapshot. Say so, and name them, rather than
    # letting the caller assume otherwise.
    in_snapshot = set(contents())
    added_since = []
    if not paths_filter:
        for path, arcname in _members():
            if arcname not in in_snapshot:
                added_since.append(arcname)

    return {
        "restored_count": len(restored),
        "restored": restored[:50],
        "truncated": len(restored) > 50,
        "current_tree_parked_at": str(parked) if parked else None,
        "snapshot_created": status().get("created_iso"),
        "added_since_snapshot_NOT_removed": added_since,
        "note": (
            "Files created after the snapshot were left in place — restore "
            "overwrites, it never deletes. Remove any listed above by hand if "
            "you want the tree exactly as the snapshot found it."
        ) if added_since else None,
    }
