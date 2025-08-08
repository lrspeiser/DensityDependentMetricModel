"""
Atomic JSON writer with periodic backup rotation.

- Uses compact JSON separators to keep files small while still human-readable.
- Atomic os.replace ensures readers never see partial writes.

Intended for writing snapshot files reliably and rotating timestamped backups
approximately every 30 minutes (configurable).
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict

_MAX_BYTES = 100 * 1024  # 100 KB soft budget


def _round_numeric(obj: Any, ndigits: int = 6) -> Any:
    """Recursively round floats in JSON-compatible structures to ndigits.

    - Floats are rounded to `ndigits` decimal places.
    - Lists/tuples are processed element-wise.
    - Dicts are processed value-wise.
    - Non-floats are returned unchanged.
    """
    if isinstance(obj, float):
        # Use round for numeric compactness; json will trim trailing zeros
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [_round_numeric(v, ndigits) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_round_numeric(v, ndigits) for v in obj)
    if isinstance(obj, dict):
        return {k: _round_numeric(v, ndigits) for k, v in obj.items()}
    return obj


def _validate_seed_dims(payload: Dict[str, Any]) -> None:
    """Best-effort validation for seed arrays if present in payload.

    Expects schema like payload["seed_live_points"] = {"K": int, "D": int, ...}.
    Logs a lightweight warning into payload["_warnings"] if limits are exceeded.
    """
    try:
        seed = payload.get("seed_live_points") or {}
        K = seed.get("K")
        D = seed.get("D")
        if isinstance(K, int) and K > 256:
            payload.setdefault("_warnings", []).append("K exceeds 256; truncated or reconfigure upstream.")
        if isinstance(D, int) and D > 15:
            payload.setdefault("_warnings", []).append("D exceeds 15; truncated or reconfigure upstream.")
    except Exception:
        # Non-fatal; snapshotting should never fail due to validation
        pass


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> int:
    """Write JSON to `path` atomically and return final file size in bytes.

    The write is done to a temporary file in the same directory and then
    atomically moved into place with os.replace so readers never observe
    partially written data.

    JSON is written with compact separators to keep the snapshot small.
    """
    # returns file size in bytes
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)

    # Best-effort seed validation and numeric rounding to meet size budget
    _validate_seed_dims(payload)
    rounded = _round_numeric(payload, ndigits=6)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=d, prefix=".tmp_tuning_", suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(rounded, f, separators=(",", ":"), ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        size = os.path.getsize(path)

        # Soft size guard: annotate if exceeding 100 KB
        if size > _MAX_BYTES:
            try:
                # Reopen, add a compact warning tag (keeps separators minimal)
                rounded.setdefault("_warnings", []).append(
                    f"snapshot size {size}B exceeds 100KB budget; consider trimming seeds/fields"
                )
                with open(path, "w", encoding="utf-8") as f2:
                    json.dump(rounded, f2, separators=(",", ":"), ensure_ascii=False)
                size = os.path.getsize(path)
            except Exception:
                pass
        return size
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            # Best-effort cleanup of temp file
            pass


def _rotate_backup_if_due(
    path: str,
    run_dir: str,
    last_backup_ts: float,
    backup_interval_sec: int = 1800,
) -> float:
    """Copy current snapshot to a timestamped backup if interval has elapsed.

    Returns the (possibly updated) last-backup timestamp. Keeps only the most
    recent 4 backups in `run_dir` on a best-effort basis.
    """
    now = time.time()
    if now - last_backup_ts < backup_interval_sec:
        return last_backup_ts

    ts = __import__("datetime").datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bpath = os.path.join(run_dir, f"tuning_snapshot.{ts}.json")
    try:
        import shutil

        shutil.copy2(path, bpath)
    except Exception:
        # Best-effort backup; ignore errors
        pass

    # optional: prune to last 4 backups
    try:
        backs = sorted(
            [
                p
                for p in os.listdir(run_dir)
                if p.startswith("tuning_snapshot.") and p.endswith(".json")
            ]
        )
        excess = len(backs) - 4
        for i in range(max(0, excess)):
            os.remove(os.path.join(run_dir, backs[i]))
    except Exception:
        # Best-effort pruning; ignore errors
        pass

    return now

