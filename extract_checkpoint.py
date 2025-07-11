import os
import numpy as np
import dill as pickle
from pathlib import Path
from dynesty import utils as dyfunc

# === Configuration ===
CHECKPOINT_PATH = Path("chains_truly_data_driven/dynesty_checkpoint.pkl")
RESULTS_PATH = Path("chains_truly_data_driven/dynesty_mw_power_Bf_DTf_DKf_Gf_samples.npz")
FALLBACK_OUTPUT = Path("chains_truly_data_driven/resumed_results.npz")

# === Try loading existing .npz results ===
if RESULTS_PATH.exists():
    print(f"📦 Found final results: {RESULTS_PATH}")
    data = np.load(RESULTS_PATH)
    samples = data["samples"]
    weights = data["weights"]
    logz = data["logz"]
    logzerr = data["logzerr"]
    logl = data["logl"]
    print("✅ Loaded final .npz results.")
else:
    print(f"🔄 No .npz results found. Attempting to resume from checkpoint: {CHECKPOINT_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError("❌ No checkpoint or final results found.")

    with open(CHECKPOINT_PATH, "rb") as f:
        sampler = pickle.load(f)

    # Resume sampling to complete it
    print("▶️ Resuming dynesty sampling...")
    sampler.run_nested(dlogz_init=0.01, maxcall=2_000_000, print_progress=True)

    res = sampler.results
    samples = res.samples
    logz = res.logz
    logzerr = res.logzerr
    logl = res.logl
    logwt = res.logwt
    weights = np.exp(logwt - logz[-1])

    # Save to .npz
    np.savez(FALLBACK_OUTPUT,
             samples=samples,
             weights=weights,
             logl=logl,
             logz=logz,
             logzerr=logzerr,
             logwt=logwt,
             ncall=res.ncall if hasattr(res, 'ncall') else None)

    print(f"✅ Resumed and saved final results to: {FALLBACK_OUTPUT}")

# === Compute and print summary stats ===
print(f"\n🔍 Analysis Summary:")
print(f"  Total samples: {len(samples)}")
print(f"  log(Z): {logz[-1]:.3f} ± {logzerr[-1]:.3f}")
if len(logz) > 1:
    print(f"  dlogz: {logz[-1] - logz[-2]:.4f}")
eff_samples = 1.0 / np.sum(weights**2)
print(f"  Effective samples: {eff_samples:.1f}")

# Weighted mean and std
weighted_mean = np.average(samples, weights=weights, axis=0)
weighted_std = np.sqrt(np.average((samples - weighted_mean)**2, weights=weights, axis=0))

# Parameter names (can be adjusted)
param_names = [
    'rho_c', 'n', 'M_thin', 'R_thin', 'h_thin', 
    'M_thick', 'R_thick', 'h_thick', 
    'M_bulge', 'a_bulge', 'M_gas', 'R_gas', 'h_gas'
]

print("\n📊 Final Parameter Estimates:")
for i in range(len(weighted_mean)):
    name = param_names[i] if i < len(param_names) else f"param_{i}"
    print(f"  {name:10s}: {weighted_mean[i]:.3e} ± {weighted_std[i]:.3e}")
