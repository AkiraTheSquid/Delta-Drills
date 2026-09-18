"""Historical KC ids → current ids, applied at every read of learner data.

The first tensor lessons (np-1..np-3) were authored as NumPy, converted to the
PyTorch dialect in 2026-08, and kept `numpy.*` KC ids until 2026-09-18 — long
enough that a learner who had only ever drilled `torch.arange` was reading
reports about "NumPy problems". The ids are now `torch.*`, and every learner
file on Fly still carries the old ones: `kc_ladder`, `kc_posteriors`, `kc_prefs`
and `kc_exposure` keys (the last also as `kc#segment`), placement probes,
diagnostic priors, and the `kc` field of every attempt row.

The map lives beside the registry (`lessons/kc_renames.json`) so content and
code rename together. It is append-only: an entry may only be dropped once no
user file anywhere can still hold the old id, which in practice means never.

`migrate` walks any JSON-shaped value and rewrites old ids wherever they sit —
as a dict key, as a string value, or as the prefix of a `kc#segment` key. It is
idempotent and leaves everything else byte-for-byte alone, so calling it on a
save that predates the rename and on one that follows it costs the same nothing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RENAMES_PATH = _REPO_ROOT / "Local_Deployed_Shared" / "lessons" / "kc_renames.json"

_cache: Dict[str, str] | None = None


def renames() -> Dict[str, str]:
    """The old→new map, loaded once. A missing file is an empty map, not an error."""
    global _cache
    if _cache is None:
        try:
            raw = json.loads(_RENAMES_PATH.read_text(encoding="utf-8"))
            _cache = {str(k): str(v) for k, v in (raw.get("renames") or {}).items()}
        except (OSError, ValueError, AttributeError) as e:
            logger.warning("kc_renames.json unreadable (%s) — no KC renames applied", e)
            _cache = {}
    return _cache


def canon(kc: Any) -> Any:
    """Current id for `kc`. Non-strings and unknown ids pass through untouched.

    Handles the exposure-key form `kc#segment` as well as a bare id.
    """
    if not isinstance(kc, str):
        return kc
    m = renames()
    if not m:
        return kc
    head, sep, tail = kc.partition("#")
    new = m.get(head)
    return (new + sep + tail) if new else kc


def migrate(value: Any) -> Any:
    """Return `value` with every old KC id rewritten, recursively.

    Dict keys and string values are both candidates; anything else is copied
    through. When two keys collapse onto one (a save that somehow holds both
    the old and the new id), the NEW id's entry wins — it is the more recent.
    """
    m = renames()
    if not m:
        return value
    return _walk(value)


def _walk(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[Any, Any] = {}
        for k, v in value.items():
            nk = canon(k)
            if nk in out and nk != k:
                continue  # the current-id entry is already there; keep it
            out[nk] = _walk(v)
        return out
    if isinstance(value, list):
        return [_walk(v) for v in value]
    return canon(value)
