#!/usr/bin/env python3
"""
pantheon_plus_tools.py — Full-covariance Pantheon+ helpers with analytic anchor and GLS regression.

Functions:
- load_pantheon_plus(base_dir, basename='Pantheon+SH0ES') -> SimpleNamespace(z, mu, cov)
- analytic_anchor_fullcov(mu_model, mu_data, C) -> dict(anchor, chi2, resid)
- gls_linefit_fullcov(S, resid, C) -> dict(slope, stderr, t_stat, r2_weighted, dof, intercept)

Notes:
- Ensures SPD covariance via small jitter if needed; symmetrizes.
- Anchor a is an additive magnitude offset applied to the model (equivalently MB/H0).
"""
from __future__ import annotations

import os
import numpy as np
from types import SimpleNamespace


def _ensure_spd(C: np.ndarray, max_trials: int = 5, jitter_frac: float = 1e-10) -> np.ndarray:
    C = 0.5 * (C + C.T)
    diag = np.diag(C)
    scale = float(np.median(diag)) if np.all(np.isfinite(diag)) and np.any(diag > 0) else 1.0
    for i in range(max_trials + 1):
        try:
            # Cholesky will throw if not SPD
            np.linalg.cholesky(C)
            return C
        except np.linalg.LinAlgError:
            C = C + np.eye(C.shape[0]) * (jitter_frac * scale)
    # Last resort: pseudo-inverse-based SPD projection (nearest SPD approx. simplified)
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, 1e-15, None)
    return (evecs * evals) @ evecs.T


def load_pantheon_plus(base_dir: str, basename: str = 'Pantheon+SH0ES') -> SimpleNamespace:
    """Load Pantheon+ table and covariance (if available) from a directory.
    Expects a data file matching basename (e.g., Pantheon+SH0ES.dat) and a covariance file with
    names like '*cov.npy', '*_STAT+SYS_cov.txt', or the release ASCII '*.cov'. Returns z, mu, cov (or None).
    """
    # Data file
    candidates = [
        os.path.join(base_dir, f'{basename}.dat'),
        os.path.join(base_dir, f'{basename}.txt'),
    ]
    data_path = next((p for p in candidates if os.path.exists(p)), None)
    if data_path is None:
        raise FileNotFoundError(f'Pantheon+ data not found in {base_dir}')
    # Load z, mu, mu_err (we need mu only; covariance supersedes diagonal)
    z_list, mu_list = [], []
    with open(data_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('CID'):
                continue
            cols = s.split()
            try:
                z_val = float(cols[2])
                mu_val = float(cols[10])
            except (IndexError, ValueError):
                continue
            z_list.append(z_val)
            mu_list.append(mu_val)
    z = np.asarray(z_list, float)
    mu = np.asarray(mu_list, float)

    # Covariance candidates (include official ASCII .cov)
    cov_candidates = [
        os.path.join(base_dir, f'{basename}_cov.npy'),
        os.path.join(base_dir, f'{basename}_STAT+SYS_cov.npy'),
        os.path.join(base_dir, f'{basename}_cov.csv'),
        os.path.join(base_dir, f'{basename}_STAT+SYS_cov.csv'),
        os.path.join(base_dir, f'{basename}_cov.txt'),
        os.path.join(base_dir, f'{basename}_STAT+SYS_cov.txt'),
        os.path.join(base_dir, f'{basename}_STAT+SYS.cov'),
    ]
    C = None
    for cp in cov_candidates:
        if not os.path.exists(cp):
            continue
        try:
            if cp.endswith('.npy'):
                C = np.load(cp)
            elif cp.endswith('.csv') or cp.endswith('.txt') or cp.endswith('.cov'):
                arr = []
                with open(cp, 'r') as f:
                    for line in f:
                        s = line.strip()
                        if not s or s.startswith('#'):
                            continue
                        arr.append([float(x) for x in s.replace(',', ' ').split()])
                C = np.array(arr, dtype=float)
            if C is not None:
                break
        except Exception:
            C = None
            continue
    if C is not None:
        if C.shape[0] != len(z):
            # try to select a matching submatrix if extra rows/cols present
            n = min(C.shape[0], len(z))
            C = C[:n, :n]
            z = z[:n]
            mu = mu[:n]
        C = _ensure_spd(C)
    return SimpleNamespace(z=z, mu=mu, cov=C)


def analytic_anchor_fullcov(mu_model: np.ndarray, mu_data: np.ndarray, C: np.ndarray) -> dict:
    """Analytic minimization over additive anchor a for full covariance.
    Uses Cholesky-based whitening (no explicit C^{-1}). Returns dict(anchor, chi2, resid).
    resid are anchored residuals y = mu_data - (mu_model + a).
    """
    y = np.asarray(mu_data, float)
    m = np.asarray(mu_model, float)
    if C is None:
        raise ValueError('Covariance matrix C is required for full-covariance analytic anchor')
    # Ensure SPD and build Cholesky with tiny jitter
    C = 0.5 * (C + C.T)
    jitter = 0.0
    for _ in range(5):
        try:
            L = np.linalg.cholesky(C + jitter * np.eye(C.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter = max(1e-12, 10.0 * (jitter if jitter else 1e-12))
    else:
        # last resort pseudo-SPD via eigen cleanup
        evals, evecs = np.linalg.eigh(C)
        evals = np.clip(evals, 1e-15, None)
        C = (evecs * evals) @ evecs.T
        L = np.linalg.cholesky(C)
    def solve_L(v):
        return np.linalg.solve(L, v)
    one = np.ones_like(y)
    y_t = solve_L(y)
    m_t = solve_L(m)
    one_t = solve_L(one)
    dy_t = m_t - y_t
    num = float(one_t @ dy_t)
    den = float(one_t @ one_t)
    a = -num / max(den, 1e-300)
    resid = y - (m + a)
    r_t = solve_L(resid)
    chi2 = float(r_t @ r_t)
    return {'anchor': a, 'chi2': chi2, 'resid': resid}


def gls_linefit_fullcov(S: np.ndarray, resid: np.ndarray, C: np.ndarray) -> dict:
    """Generalized least squares fit of resid = a + b*S using full covariance C.
    Uses Cholesky whitening; returns slope b, stderr, t_stat, weighted R^2, dof, intercept.
    """
    y = np.asarray(resid, float)
    x = np.asarray(S, float)
    n = len(y)
    if C is None:
        raise ValueError('Covariance matrix C is required for GLS')
    C = 0.5 * (C + C.T)
    jitter = 0.0
    for _ in range(5):
        try:
            L = np.linalg.cholesky(C + jitter * np.eye(C.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter = max(1e-12, 10.0 * (jitter if jitter else 1e-12))
    else:
        evals, evecs = np.linalg.eigh(C)
        evals = np.clip(evals, 1e-15, None)
        C = (evecs * evals) @ evecs.T
        L = np.linalg.cholesky(C)
    def whiten(v):
        return np.linalg.solve(L, v)
    X = np.vstack([np.ones_like(x), x]).T
    y_t = whiten(y)
    X_t = np.column_stack([whiten(X[:, j]) for j in range(X.shape[1])])
    XtX = X_t.T @ X_t
    XtY = X_t.T @ y_t
    try:
        beta = np.linalg.solve(XtX, XtY)
        cov_beta = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtX) @ XtY
        cov_beta = np.linalg.pinv(XtX)
    y_hat_t = X_t @ beta
    r_t = y_t - y_hat_t
    chi2 = float(r_t @ r_t)
    dof = max(n - 2, 1)
    sigma2 = chi2 / dof
    cov_beta = cov_beta * sigma2
    b = float(beta[1])
    b_se = float(np.sqrt(max(cov_beta[1, 1], 0.0)))
    t_stat = float(b / b_se) if b_se > 0 else float('inf')
    # Weighted R^2 in whitened space
    ybar_t = float(np.mean(y_t))
    ss_tot = float(np.sum((y_t - ybar_t) ** 2))
    ss_res = float(np.sum(r_t ** 2))
    r2_w = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return {
        'slope': b,
        'stderr': b_se,
        't_stat': t_stat,
        'r2_weighted': r2_w,
        'dof': dof,
        'intercept': float(beta[0]),
    }
