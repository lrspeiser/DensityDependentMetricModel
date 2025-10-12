# data_io/loaders.py
# Tiny, forgiving loaders for CMB ℓ_A, SNe, BAO, and RAR/SPARC-like CSV files.

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Literal


# -----------------------------
# CMB acoustic scale ℓ_A constraint
# -----------------------------
def load_cmb_ellA_constraint(value: float = 301.0, sigma: float = 0.1) -> Tuple[float, float]:
    """
    Returns (ℓ_A_obs, σ_ℓA). Defaults put you in Planck ballpark; override with your exact numbers.
    """
    return float(value), float(sigma)


# -----------------------------
# Supernovae (Pantheon+ style)
# -----------------------------
def load_sne_table(csv_or_ascii: str | Path) -> pd.DataFrame:
    """
    Load a simple SN table.
    Accepts:
      - CSV with columns [z, mu, sigma_mu] OR
      - Pantheon+SH0ES.dat ASCII (official release) where we pick zHD (col 3), MU_SH0ES (col 11), MU_SH0ES_ERR_DIAG (col 12).
    """
    p = Path(csv_or_ascii)
    path = str(p)
    if path.endswith('.csv'):
        df = pd.read_csv(path)
        if "mu_err" in df.columns and "sigma_mu" not in df.columns:
            df = df.rename(columns={"mu_err": "sigma_mu"})
        return df
    # Fallback: try to parse Pantheon+SH0ES.dat ASCII
    z_list, mu_list, muerr_list = [], [], []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('CID'):
                continue
            cols = s.split()
            try:
                z_val = float(cols[2])       # zHD
                mu_val = float(cols[10])     # MU_SH0ES
                mu_err = float(cols[11])     # MU_SH0ES_ERR_DIAG
            except (IndexError, ValueError):
                continue
            z_list.append(z_val)
            mu_list.append(mu_val)
            muerr_list.append(mu_err)
    if not z_list:
        raise ValueError(f"Unrecognized SN format: {path}")
    return pd.DataFrame({"z": z_list, "mu": mu_list, "sigma_mu": muerr_list})


def maybe_load_covariance(table_path: str | Path) -> np.ndarray | None:
    """Try to locate a sidecar covariance next to a table.
    If `table_path` ends with .dat (Pantheon+), look for Pantheon+SH0ES_STAT+SYS.cov in the same base dir.
    Otherwise try *_cov.npy next to the CSV/Parquet.
    """
    p = Path(table_path)
    # Pantheon+ .dat → look for STAT+SYS.cov; if found, attempt to parse ASCII matrix
    if p.suffix == '.dat':
        base_dir = p.parent
        cand = base_dir / 'Pantheon+SH0ES_STAT+SYS.cov'
        if cand.exists():
            # ASCII matrix; parse numbers and build array
            rows = []
            with open(cand, 'r') as f:
                for ln in f:
                    s = ln.strip()
                    if not s or s.startswith('#'):
                        continue
                    parts = [pp for pp in s.replace(',', ' ').split() if pp]
                    try:
                        row = [float(x) for x in parts]
                    except ValueError:
                        continue
                    rows.append(row)
            if rows:
                # If first row is a single integer N, flatten read; else treat as square
                if len(rows[0]) == 1 and int(rows[0][0]) > 1:
                    N = int(rows[0][0])
                    flat = []
                    for r in rows[1:]:
                        flat.extend(r)
                    arr = np.asarray(flat, float)
                    if arr.size >= N*N:
                        return arr[:N*N].reshape(N, N)
                # else try square
                arr = np.array(rows, float)
                if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                    return arr
        return None
    # CSV/Parquet sidecar
    cov_path = Path(str(p).replace('.csv','_cov.npy').replace('.parquet','_cov.npy'))
    if cov_path.exists():
        try:
            return np.load(cov_path)
        except Exception:
            try:
                return np.load(cov_path, allow_pickle=True)
            except Exception:
                return None
    return None


# -----------------------------
# BAO table
# -----------------------------
def load_bao_table(csv: str | Path) -> pd.DataFrame:
    """
    Expects (long-form) columns:
      - z
      - observable: one of {"DV_over_rd", "DM_over_rd", "DH_over_rd"}
      - value
      - sigma
    """
    return pd.read_csv(csv)


# -----------------------------
# RAR/SPARC-like points
# -----------------------------
def load_rar_table(csv: str | Path) -> pd.DataFrame:
    """
    Expects columns:
      - gbar_m_s2
      - gobs_m_s2
      - sigma_gobs_m_s2 (optional)
      - (optional grouping columns like galaxy_name, R_kpc — not required for the pure RAR fit)
    """
    return pd.read_csv(csv)