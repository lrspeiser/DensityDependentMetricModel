import gzip
import pickle
import numpy as np

input_file = "chains_dynesty/mw_grav_color_DTf_DKf_Bf_Gf_20250716/dynesty_mw_grav_color_Bf_DTf_DKf_Gf_results.pkl.gz"
output_file = input_file.replace(".pkl.gz", "_samples_FIXED.npz")

with gzip.open(input_file, "rb") as f:
    res = pickle.load(f)

samples = res.samples
weights = np.exp(res.logwt - res.logz[-1]) if hasattr(res, 'logwt') else np.ones(len(samples)) / len(samples)

param_names = [
    'rho_c_solar_kpc3',
    'M_disk_thin_solar',
    'gamma_exp',
    'h_z_thin_kpc',
    'M_disk_thick_solar',
    'R_d_thin_kpc',
    'lambda_g',
    'M_gas_solar',
    'R_d_thick_kpc',
    'M_bulge_solar',
    'a_bulge_kpc',
    'h_z_thick_kpc'
]

np.savez(output_file,
    samples=samples,
    weights=weights,
    param_names=np.array(param_names),
    logl=res.logl,
    logz=res.logz,
    logzerr=res.logzerr,
    blob=res.blob if hasattr(res, 'blob') else None
)

print(f"✅ Saved fixed results to {output_file}")
