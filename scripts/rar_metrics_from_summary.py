#!/usr/bin/env python3
import json, numpy as np
from pathlib import Path
import cupy as cp
import pandas as pd
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from core.data_io import process_gaia_data
from core.density_metric_cupy import v_total_kms_cupy, DEFAULT_DTYPE

raw = REPO/'external_data'/'gaia_sky_slices'/'all_sky_gaia.csv'
df_raw = pd.read_csv(raw)
df = process_gaia_data(df_raw)
mask = np.isfinite(df['R_kpc']) & np.isfinite(df['v_obs']) & np.isfinite(df['sigma_v'])
df = df.loc[mask, ['R_kpc','v_obs','sigma_v']].copy()
R = df['R_kpc'].to_numpy(float)
V = df['v_obs'].to_numpy(float)
E = df['sigma_v'].to_numpy(float)

# RAR params from prior academic run
sum_path = REPO/'runs'/'rar_gate_20250818_164443'/'run_summary_enhanced.json'
js = json.loads(sum_path.read_text(encoding='utf-8'))
pb = js['parameter_estimates']['best_fit']
# Build param dict
baryon_keys = ['M_thin_disk_solar','R_thin_disk_kpc','hz_thin_disk_kpc','M_thick_disk_solar','R_thick_disk_kpc','hz_thick_disk_kpc','M_bulge_solar','R_bulge_kpc','M_gas_solar','R_gas_kpc','hz_gas_kpc']
P_b = {k: float(pb[k]) for k in baryon_keys if k in pb}
P_b.update({'include_disk_thin': True,'include_disk_thick': True,'include_bulge': True,'include_gas': True})
P = dict(P_b)
for k in ['a0_m_s2','gamma_exp','lambda_max','T0','sigma_lnT','wmin']:
    if k in pb:
        P[k] = float(pb[k])
P['allow_experimental'] = True

Rg = cp.asarray(R, dtype=DEFAULT_DTYPE)
V_model = v_total_kms_cupy(Rg, P, xi_type='rar_gate')
V_model = cp.asnumpy(V_model)

out = {}

def metrics(rmin, rmax):
    m = (R >= rmin) & (R <= rmax)
    if not np.any(m):
        return None
    resid = V[m] - V_model[m]
    rmse = float(np.sqrt(np.mean(resid**2)))
    chi2 = float(np.sum(((V[m] - V_model[m]) / np.maximum(E[m], 1e-6))**2))
    n = int(np.sum(m))
    return {"rmse_kms": rmse, "chi2": chi2, "N": n, "sqrt_chi2_over_N": float(np.sqrt(chi2/max(n,1)))}

out['window_6_14'] = metrics(6.0, 14.0)
out['window_8_14'] = metrics(8.0, 14.0)
out['window_12_16'] = metrics(12.0, 16.0)

# V_inf
m_outer = (R >= 12.0) & (R <= 16.0)
if np.any(m_outer):
    v_inf = float(np.median(V_model[m_outer]))
else:
    idx = np.argsort(R)
    sel = idx[int(0.9*len(R)):] if len(R)>0 else []
    v_inf = float(np.median(V_model[sel])) if len(sel)>0 else float(np.median(V_model))

out['v_infty_kms'] = v_inf

# BTFR masses
G_SI = 6.67430e-11
MSUN = 1.98847e30
v_si = v_inf * 1000.0
for tag, a0 in [("a0_fitted", float(P.get('a0_m_s2', 1.2e-10))), ("a0_canonical", 1.2e-10)]:
    M_b_kg = (v_si**4) / (G_SI * a0)
    out[f'M_b_BTFR_Msun_{tag}'] = float(M_b_kg / MSUN)

M_b_model = 0.0
for k in ['M_thin_disk_solar','M_thick_disk_solar','M_bulge_solar','M_gas_solar']:
    if k in P_b:
        M_b_model += float(P_b[k])
out['M_b_model_Msun'] = M_b_model

print(json.dumps(out))

