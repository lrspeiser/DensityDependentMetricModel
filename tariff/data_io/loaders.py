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
def load_sne_table(csv_or_parquet: str | Path) -> pd.DataFrame:
    """
    Expects columns:
      - z: redshift
      - mu: (optional) distance modulus; if absent, we use mB - M as provided and treat M via anchoring
      - mu_err or sigma_mu: (optional, used if no covariance provided)
      - mB, dmb: (optional) light-curve stats (ignored here; we fit μ directly)

    If you have a full covariance matrix C (N×N), store it as a .npy alongside with suffix *_cov.npy.
    """
    df = pd.read_csv(csv_or_parquet) if str(csv_or_parquet).endswith(".csv") else pd.read_parquet(csv_or_parquet)
    if "mu_err" in df.columns and "sigma_mu" not in df.columns:
        df = df.rename(columns={"mu_err": "sigma_mu"})
    return df


def maybe_load_covariance(csv_path: str | Path) -> np.ndarray | None:
    cov_path = Path(str(csv_path).replace(".csv", "_cov.npy").replace(".parquet", "_cov.npy"))
    if cov_path.exists():
        return np.load(cov_path)
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