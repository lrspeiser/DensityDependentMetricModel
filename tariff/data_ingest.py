#!/usr/bin/env python3
"""
data_ingest.py — tariff-only data loaders for baseline and comparison analyses.

Functions:
- load_pantheon(path) -> (z, mu, mu_err)
- load_cmb_spectrum_csv(path) -> (nu_Hz, I_Wsr_m2_Hz)
- load_bao_csv(path) -> dict with keys 'z' and one of {'DM_over_rd','DH_over_rd','DV_over_rd'} (+ optional error columns)
- load_tolman_csv(path) -> (z, SB, SB_err)
- load_sntd_csv(path) -> (z, t, terr) using 'timescale' or 'stretch' columns

All remain confined to tariff/.
"""
from __future__ import annotations

import numpy as np
import csv

def load_pantheon(path: str):
    z_list, mu_list, muerr_list = [], [], []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('CID'):
                continue
            cols = s.split()
            try:
                z_val = float(cols[2])
                mu_val = float(cols[10])
                mu_err = float(cols[11])
            except (IndexError, ValueError):
                continue
            z_list.append(z_val)
            mu_list.append(mu_val)
            muerr_list.append(mu_err)
    return np.asarray(z_list, float), np.asarray(mu_list, float), np.asarray(muerr_list, float)

def load_cmb_spectrum_csv(path: str):
    nu, I = [], []
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            nu.append(float(row['frequency_GHz'])*1e9)
            I.append(float(row['intensity_Wsr_m2_Hz']))
    return np.asarray(nu, float), np.asarray(I, float)

def load_bao_csv(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    out = {'z': df['z'].to_numpy(float)}
    for key in ['D_M_over_rd','D_H_over_rd','DV_over_rd']:
        if key in df.columns:
            out[key] = df[key].to_numpy(float)
    for key in ['D_M_err','D_H_err','DV_err']:
        if key in df.columns:
            out[key] = df[key].to_numpy(float)
    return out

def load_tolman_csv(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    z = df['z'].to_numpy(float)
    SB = df['SB'].to_numpy(float)
    SB_err = df['SB_err'].to_numpy(float) if 'SB_err' in df.columns else np.full_like(SB, 0.1*np.median(SB))
    return z, SB, SB_err

def load_sntd_csv(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    z = df['z'].to_numpy(float)
    if 'timescale' in df.columns:
        t = df['timescale'].to_numpy(float)
        terr = df['timescale_err'].to_numpy(float) if 'timescale_err' in df.columns else np.full_like(t, 0.1*np.median(t))
    elif 'stretch' in df.columns:
        t = df['stretch'].to_numpy(float)
        terr = df['stretch_err'].to_numpy(float) if 'stretch_err' in df.columns else np.full_like(t, 0.1*np.median(t))
    else:
        raise ValueError("CSV must contain timescale or stretch")
    return z, t, terr
