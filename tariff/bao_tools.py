#!/usr/bin/env python3
"""
bao_tools.py — BAO compilation loader and shape-only rd fit against model DM(z), DH(z).

Functions:
- load_bao_compilation(path) -> dict with z and either {DM_over_rd, DH_over_rd, (optional) rho, D_M_err, D_H_err}
  or DV_over_rd (+ optional DV_err)
- rd_shape_only_fit(df, z_mod, DM_mod, DH_mod) -> dict(rd_best_Mpc, chi2, red_chi2, dof)

Notes:
- Uses 1D golden-section search over rd to minimize chi^2.
- Supports anisotropic (DM/DH) or isotropic (DV) inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


def load_bao_compilation(path: str) -> Dict[str, np.ndarray]:
    df = pd.read_csv(path)
    out = {'z': df['z'].to_numpy(float)}
    cols = df.columns
    if 'D_M_over_rd' in cols and 'D_H_over_rd' in cols:
        out['D_M_over_rd'] = df['D_M_over_rd'].to_numpy(float)
        out['D_H_over_rd'] = df['D_H_over_rd'].to_numpy(float)
        if 'D_M_err' in cols:
            out['D_M_err'] = df['D_M_err'].to_numpy(float)
        if 'D_H_err' in cols:
            out['D_H_err'] = df['D_H_err'].to_numpy(float)
        if 'rho' in cols:
            out['rho'] = df['rho'].to_numpy(float)
    elif 'DV_over_rd' in cols:
        out['DV_over_rd'] = df['DV_over_rd'].to_numpy(float)
        if 'DV_err' in cols:
            out['DV_err'] = df['DV_err'].to_numpy(float)
    else:
        raise ValueError('BAO CSV must contain either D_M_over_rd & D_H_over_rd or DV_over_rd')
    return out


def _chi2_aniso(rd: float, z_b: np.ndarray, DM_b: np.ndarray, DH_b: np.ndarray, eDM: np.ndarray, eDH: np.ndarray, rho: np.ndarray,
                z_mod: np.ndarray, DM_mod: np.ndarray, DH_mod: np.ndarray) -> float:
    DM_m = np.interp(z_b, z_mod, DM_mod)
    DH_m = np.interp(z_b, z_mod, DH_mod)
    # Build per-point covariance and accumulate chi^2
    chi2 = 0.0
    for i in range(len(z_b)):
        mu = np.array([DM_m[i] / rd, DH_m[i] / rd])
        y = np.array([DM_b[i], DH_b[i]])
        C = np.array([[eDM[i]**2, rho[i]*eDM[i]*eDH[i]], [rho[i]*eDM[i]*eDH[i], eDH[i]**2]])
        try:
            Ci = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            Ci = np.linalg.pinv(C, rcond=1e-10)
        d = y - mu
        chi2 += float(d.T @ Ci @ d)
    return chi2


def _chi2_iso(rd: float, z_b: np.ndarray, DV_b: np.ndarray, eDV: np.ndarray, z_mod: np.ndarray, DM_mod: np.ndarray, DH_mod: np.ndarray) -> float:
    # DV = [z^2 DM^2 DH]^(1/3)
    DM_m = np.interp(z_b, z_mod, DM_mod)
    DH_m = np.interp(z_b, z_mod, DH_mod)
    DV_m = ( (z_b**2) * (DM_m**2) * DH_m ) ** (1.0/3.0)
    mu = DV_m / rd
    d = DV_b - mu
    return float(np.sum((d / eDV)**2))


def _golden_section(f, a: float, b: float, tol: float = 1e-6, max_iter: int = 200) -> float:
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    it = 0
    while abs(b - a) > tol and it < max_iter:
        if fc < fd:
            b, fd = d, fc
            d = a + gr * (b - a)
            fc = f(c)
        else:
            a, fc = c, fd
            c = b - gr * (b - a)
            fd = f(d)
        it += 1
    return (a + b) / 2.0


def rd_shape_only_fit(df: Dict[str, np.ndarray], z_mod: np.ndarray, DM_mod: np.ndarray, DH_mod: np.ndarray) -> Dict[str, float]:
    z_b = df['z']
    if 'D_M_over_rd' in df and 'D_H_over_rd' in df:
        DM_b = df['D_M_over_rd']; DH_b = df['D_H_over_rd']
        eDM = df.get('D_M_err', np.full_like(DM_b, 0.05))
        eDH = df.get('D_H_err', np.full_like(DH_b, 0.05))
        rho = df.get('rho', np.zeros_like(DM_b))
        f = lambda rd: _chi2_aniso(rd, z_b, DM_b, DH_b, eDM, eDH, rho, z_mod, DM_mod, DH_mod)
    elif 'DV_over_rd' in df:
        DV_b = df['DV_over_rd']
        eDV = df.get('DV_err', np.full_like(DV_b, 0.05))
        f = lambda rd: _chi2_iso(rd, z_b, DV_b, eDV, z_mod, DM_mod, DH_mod)
    else:
        raise ValueError('BAO dict missing required keys')
    # Find a decent bracket around rd ~ [80, 200] Mpc
    rd_best = _golden_section(f, a=60.0, b=200.0, tol=1e-6, max_iter=300)
    chi2 = float(f(rd_best))
    dof = int(len(z_b) * (2 if ('D_M_over_rd' in df and 'D_H_over_rd' in df) else 1) - 1)
    return {'rd_best_Mpc': float(rd_best), 'bao_chi2': chi2, 'bao_red_chi2': float(chi2/dof), 'bao_dof': dof}


def rd_shape_only_fit_analytic(df: Dict[str, np.ndarray], z_mod: np.ndarray, DM_mod: np.ndarray, DH_mod: np.ndarray) -> Dict[str, float]:
    """Analytic rd fit for anisotropic BAO using block-diagonal per-bin 2x2 covariances.
    Requires columns: z, D_M_over_rd, D_H_over_rd and, ideally, D_M_err, D_H_err, rho.
    Returns rd_best, rd_err, chi2, dof, red_chi2.
    """
    if not ('D_M_over_rd' in df and 'D_H_over_rd' in df):
        # Fallback to numeric if only DV present
        return rd_shape_only_fit(df, z_mod, DM_mod, DH_mod)
    z_b = df['z']
    DM_b = df['D_M_over_rd']; DH_b = df['D_H_over_rd']
    eDM = df.get('D_M_err', np.full_like(DM_b, 0.05))
    eDH = df.get('D_H_err', np.full_like(DH_b, 0.05))
    rho = df.get('rho', np.zeros_like(DM_b))
    # Build y (6,) and C6 (6x6) block-diagonal
    y = np.empty(2*len(z_b), float)
    C6 = np.zeros((2*len(z_b), 2*len(z_b)), float)
    for i in range(len(z_b)):
        y[2*i] = DM_b[i]; y[2*i+1] = DH_b[i]
        C2 = np.array([[eDM[i]**2, rho[i]*eDM[i]*eDH[i]], [rho[i]*eDM[i]*eDH[i], eDH[i]**2]], float)
        C6[2*i:2*i+2, 2*i:2*i+2] = C2
    # Model x (6,) from DM_mod, DH_mod
    DM_m = np.interp(z_b, z_mod, DM_mod)
    DH_m = np.interp(z_b, z_mod, DH_mod)
    x = np.empty_like(y)
    x[0::2] = DM_m; x[1::2] = DH_m
    # Analytic s=1/rd
    try:
        Ci = np.linalg.inv(C6)
    except np.linalg.LinAlgError:
        Ci = np.linalg.pinv(C6, rcond=1e-10)
    xtC = x @ Ci
    den = xtC @ x
    num = xtC @ y
    if den <= 0:
        # Fallback numeric
        return rd_shape_only_fit(df, z_mod, DM_mod, DH_mod)
    s_best = num/den
    rd = 1.0/s_best
    sigma_s = den**(-0.5)
    rd_err = sigma_s/(s_best**2)
    r = y - s_best*x
    chi2 = float(r @ Ci @ r)
    dof = 2*len(z_b) - 1
    return {'rd_best_Mpc': float(rd), 'rd_err_Mpc': float(rd_err), 'bao_chi2': float(chi2), 'bao_red_chi2': float(chi2/dof), 'bao_dof': int(dof)}
