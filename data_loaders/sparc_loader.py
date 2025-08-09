"""
SPARC/Rotmod loader utilities.

Supports files like external_data/Rotmod_LTG/NGC3198_rotmod.dat with whitespace-separated
columns. We try to parse common SPARC/rotmod formats where columns include:

R  Vobs  eVobs  Vgas  Vdisk  Vbul [optional extra columns]

We ignore extra columns if present. Units are typically:
- R in kpc
- velocities in km/s

If uncertainties (eVobs) are missing, we assign a small default (5 km/s).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

EXPECTED_MIN_COLS = 5  # R, Vobs, eVobs, Vgas, Vdisk (bulge optional)


def load_rotmod(path: str | Path) -> Dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SPARC/Rotmod file not found: {p}")
    # Read, skipping comment lines starting with # or ;
    rows = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith(";"):
                continue
            parts = s.split()
            # Require at least R, Vobs, eVobs, Vgas, Vdisk
            if len(parts) < EXPECTED_MIN_COLS:
                continue
            try:
                vals = [float(x) for x in parts]
            except Exception:
                continue
            rows.append(vals)
    if not rows:
        raise ValueError(f"No data rows parsed from {p}")
    arr = np.array(rows, dtype=float)
    # Map columns
    R = arr[:, 0]
    Vobs = arr[:, 1]
    # If eVobs not present, synthesize
    if arr.shape[1] >= 3:
        eVobs = np.clip(arr[:, 2], 1.0, None)
    else:
        eVobs = np.full_like(Vobs, 5.0)
    Vgas = arr[:, 3] if arr.shape[1] >= 4 else np.zeros_like(Vobs)
    Vdisk = arr[:, 4] if arr.shape[1] >= 5 else np.zeros_like(Vobs)
    Vbul = arr[:, 5] if arr.shape[1] >= 6 else np.zeros_like(Vobs)
    return {
        "R_kpc": R.astype(float),
        "Vobs_kms": Vobs.astype(float),
        "eVobs_kms": eVobs.astype(float),
        "Vgas_kms": Vgas.astype(float),
        "Vdisk_kms": Vdisk.astype(float),
        "Vbul_kms": Vbul.astype(float),
    }
