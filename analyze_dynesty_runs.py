import numpy as np
from pathlib import Path
from collections import defaultdict

# --- Physical plausibility checks ---
PHYSICAL_BOUNDS = {
    'M_total': (5e10, 2e11),
    'thick_thin_ratio': 0.7,
    'xi_Rsun': (0.7, 1.0),
    'v_model_Rsun': (150, 250),
}

G_ASTRO = 4.30091e-6  # (kpc * (km/s)^2) / M_sun
R_SUN_KPC = 8.122

param_names = [
    'rho_c', 'n', 'M_thin', 'R_thin', 'h_thin',
    'M_thick', 'R_thick', 'h_thick',
    'M_bulge', 'a_bulge',
    'M_gas', 'R_gas', 'h_gas'
]

def estimate_xi(rho_c, n, rho_typical=1e8):
    return 1.0 / (1.0 + (rho_typical / rho_c)**n)

def estimate_v_model(m_total, xi):
    return np.sqrt(G_ASTRO * m_total * 0.6 / R_SUN_KPC) * np.sqrt(xi)

def check_physical_constraints(params):
    notes = []
    plausible = True

    m_total = params['M_thin'] + params['M_thick'] + params['M_bulge'] + params['M_gas']
    if not PHYSICAL_BOUNDS['M_total'][0] <= m_total <= PHYSICAL_BOUNDS['M_total'][1]:
        notes.append(f"total mass out of bounds ({m_total:.2e})")
        plausible = False

    if params['M_thin'] > 0:
        ratio = params['M_thick'] / params['M_thin']
        if ratio > PHYSICAL_BOUNDS['thick_thin_ratio']:
            notes.append(f"thick/thin mass ratio = {ratio:.2f}")
            plausible = False

    xi_solar = estimate_xi(params['rho_c'], params['n'])
    if not PHYSICAL_BOUNDS['xi_Rsun'][0] <= xi_solar <= PHYSICAL_BOUNDS['xi_Rsun'][1]:
        notes.append(f"xi(R☉) = {xi_solar:.3f}")
        plausible = False

    v_model = estimate_v_model(m_total, xi_solar)
    if not PHYSICAL_BOUNDS['v_model_Rsun'][0] <= v_model <= PHYSICAL_BOUNDS['v_model_Rsun'][1]:
        notes.append(f"v(R☉) = {v_model:.1f} km/s")
        plausible = False

    return plausible, notes, m_total, xi_solar, v_model

def analyze_file(path):
    try:
        d = np.load(path)
        samples = d['samples']
        weights = d['weights']
        logz = d['logz'][-1]
        dlogz = d['logz'][-1] - d['logz'][-2] if len(d['logz']) > 1 else np.nan
        ess = 1.0 / np.sum(weights**2)

        mean = np.average(samples, weights=weights, axis=0)
        param_dict = dict(zip(param_names, mean))
        plausible, notes, m_total, xi_solar, v_model = check_physical_constraints(param_dict)

        return {
            'file': str(path),
            'logz': logz,
            'dlogz': dlogz,
            'ess': ess,
            'plausible': plausible,
            'notes': notes,
            'M_total': m_total,
            'xi_solar': xi_solar,
            'v_model': v_model,
            'params': param_dict
        }
    except Exception as e:
        return {'file': str(path), 'error': str(e), 'plausible': False}

def print_runs(label, runs, limit=5):
    print(f"\n=== {label} ({len(runs)} runs) ===")
    for r in runs[:limit]:
        print(f"• {r['file']}")
        print(f"   logZ={r['logz']:.2f}  dlogz={r['dlogz']:.4f}  ESS={r['ess']:.1f}")
        print(f"   M_total={r['M_total']:.2e}  xi(R☉)={r['xi_solar']:.3f}  v(R☉)={r['v_model']:.1f} km/s")
        if not r['plausible']:
            print(f"   ❌ {', '.join(r['notes'])}")
        print()

def main():
    files = list(Path('.').rglob('*.npz'))
    print(f"🔍 Scanning {len(files)} result files...\n")

    results = [analyze_file(f) for f in files if 'checkpoint' not in str(f)]
    plausible = [r for r in results if r['plausible'] and 'error' not in r]
    failed = [r for r in results if not r['plausible'] and 'error' not in r]

    # Different slices
    best_logz = sorted(plausible, key=lambda r: r['logz'])[:5]
    best_dlogz = sorted(plausible, key=lambda r: abs(r['dlogz']))[:5]
    best_ess = sorted(plausible, key=lambda r: -r['ess'])[:5]

    borderline = [r for r in failed if len(r['notes']) <= 1]  # nearly plausible
    borderline = sorted(borderline, key=lambda r: r['logz'])[:5]

    print_runs("Physically Plausible (All Checks Passed)", plausible)
    print_runs("Lowest dlogz (Best Converged)", best_dlogz)
    print_runs("Highest Effective Samples", best_ess)
    print_runs("Best logZ Scores", best_logz)
    print_runs("Borderline Plausible (Only 1 Failing Test)", borderline)

if __name__ == "__main__":
    main()
