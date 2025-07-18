#!/usr/bin/env python3
"""
Create multiple test parameter sets for Gaussian xi function
to explore different configurations
"""
import json
import os


# Base MW parameters (same for all tests)
base_params = {
    "xi_function": "gaussian",
    "fit_xi_params": True,
    
    # Standard MW parameters
    "M_disk_thin_solar": 5e10,
    "R_d_thin_kpc": 2.6,
    "h_z_thin_kpc": 0.3,
    
    "M_disk_thick_solar": 0.15e10,
    "R_d_thick_kpc": 3.6,
    "h_z_thick_kpc": 0.9,
    
    "M_bulge_solar": 1.5e10,
    "a_bulge_kpc": 0.5,
    
    "M_gas_solar": 0.5e10,
    "R_d_gas_kpc": 7.0,
    "h_z_gas_kpc": 0.15,
    
    # Include components
    "include_disk_thin": True,
    "include_disk_thick": True,
    "include_bulge": True,
    "include_gas": True,
    "include_bulge_density": True,
    
    # Fitting flags
    "fit_disk_thin": True,
    "fit_disk_thick": True,
    "fit_bulge": True,
    "fit_gas": False
}

# Different xi parameter configurations to test
test_configs = [
    {
        "name": "conservative",
        "desc": "Modest enhancement, narrow peak",
        "rho_c": 0.5,    # Peak at galaxy density
        "n_exp": 0.8,    # Narrow width
        "A": 1.5         # 2.5x max enhancement
    },
    {
        "name": "standard",
        "desc": "Standard enhancement for galaxies", 
        "rho_c": 0.5,
        "n_exp": 1.0,
        "A": 2.0         # 3x max enhancement
    },
    {
        "name": "broad",
        "desc": "Broader enhancement profile",
        "rho_c": 0.3,    # Peak at lower density
        "n_exp": 1.5,    # Wider profile
        "A": 2.0
    },
    {
        "name": "strong",
        "desc": "Stronger enhancement",
        "rho_c": 0.5,
        "n_exp": 1.0,
        "A": 3.0         # 4x max enhancement
    },
    {
        "name": "shifted",
        "desc": "Peak shifted to lower densities",
        "rho_c": 0.1,    # Peak at halo density
        "n_exp": 1.2,
        "A": 2.5
    }
]

# Create test directory
os.makedirs("gaussian_test_params", exist_ok=True)

# Generate parameter files
for config in test_configs:
    params = base_params.copy()
    params.update({
        "rho_c": config["rho_c"],
        "n_exp": config["n_exp"], 
        "A": config["A"]
    })
    
    filename = f"gaussian_test_params/params_{config['name']}.json"
    with open(filename, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"\nCreated {filename}")
    print(f"  {config['desc']}")
    print(f"  rho_peak={config['rho_c']}, sigma={config['n_exp']}, lambda={config['A']}")

# Also create a quick test script
test_script = """#!/usr/bin/env python3
import numpy as np
from density_metric2 import xi_gaussian_enhancement

# Test each configuration
configs = [
    ("conservative", 0.5, 0.8, 1.5),
    ("standard", 0.5, 1.0, 2.0),
    ("broad", 0.3, 1.5, 2.0),
    ("strong", 0.5, 1.0, 3.0),
    ("shifted", 0.1, 1.2, 2.5)
]

# Key density checkpoints
test_points = [
    (0.01, "Halo edge"),
    (0.5, "Galaxy disk"),
    (10, "Galaxy center"),
    (100, "Solar System"),
    (1e6, "Stellar")
]

print("\\nGaussian Xi Function Tests")
print("="*80)

for name, rho_c, n_exp, A in configs:
    print(f"\\n{name.upper()}: rho_c={rho_c}, sigma={n_exp}, lambda={A}")
    print("-"*60)
    print("Density      | Location      | xi    | v_factor | Status")
    print("-"*60)
    
    for rho, loc in test_points:
        xi = xi_gaussian_enhancement(rho, rho_c, n_exp, A)[0]
        v_factor = np.sqrt(xi)
        
        # Check constraints
        if loc == "Solar System":
            status = "✓" if abs(xi - 1.0) < 0.1 else "✗ FAIL"
        elif loc in ["Galaxy disk", "Halo edge"]:
            status = "✓" if xi > 1.5 else "weak"
        else:
            status = "ok"
            
        print(f"{rho:12.2e} | {loc:13s} | {xi:5.3f} | {v_factor:8.3f} | {status}")

print("\\n" + "="*80)
print("Best configs should have:")
print("- Solar System: xi ≈ 1.0 (✓)")
print("- Galaxy disk: xi > 1.5 (✓)")
print("- Smooth transitions between regimes")
"""

with open("test_gaussian_configs.py", "w") as f:
    f.write(test_script)
os.chmod("test_gaussian_configs.py", 0o755)

print("\n\nTo test all configurations:")
print("  python3 test_gaussian_configs.py")
print("\nTo validate a specific config:")
print("  python3 validate_ddmm.py gaussian_test_params/params_standard.json --output_dir validation_gaussian_standard")
print("\nTo run dynesty with best config:")
print("  python3 run_dynesty.py --params_file gaussian_test_params/params_standard.json --nlive_init 500")