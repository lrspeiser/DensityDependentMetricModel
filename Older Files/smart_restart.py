#!/usr/bin/env python3
"""Smart restart using insights from 2-day run console output."""

import numpy as np

# Based on your console output from the 2-day run
console_insights = {
    'rho_c_solar_kpc3': 9.78e8,  # from console
    'n_exp': 1.946,               # near upper bound
    'M_disk_thin_solar': 9.24e10,
    'R_d_thin_kpc': 4.974,        # was at upper bound
    'h_z_thin_kpc': 0.502,
    'M_disk_thick_solar': 2.83e10,
    'R_d_thick_kpc': 2.529,       # was at lower bound - problematic!
    'h_z_thick_kpc': 0.695,
    'M_bulge_solar': 7.15e9,
    'a_bulge_kpc': 1.260,
    'M_gas_solar': 2.86e10,       # near upper bound
    'R_d_gas_kpc': 9.715,
    'h_z_gas_kpc': 0.229
}

# Correct the physical violations
corrected_params = console_insights.copy()
corrected_params['R_d_thin_kpc'] = 5.5      # Allow it to be larger
corrected_params['R_d_thick_kpc'] = 6.5     # Must be > R_d_thin!
corrected_params['n_exp'] = 2.2             # Allow higher values

# Generate restart command
cmd_parts = [
    "python run_dynesty.py \\",
    "    --output_dir chains_corrected_physics \\",
]

# Set initial values from corrected params
for param, value in corrected_params.items():
    fixed_name = param.replace('_solar', '_fixed').replace('_kpc3', '_fixed').replace('_kpc', '_fixed')
    cmd_parts.append(f"    --{fixed_name} {value:.3e} \\")

# Sampler settings
cmd_parts.extend([
    "    --nlive_init 1500 \\",  # More live points for complex space
    "    --sample_method rslice \\",
    "    --enlarge_factor 1.2 \\",  # Tighter bounds
    "    --bound_method ellipsoid \\",  # Better for concentrated sampling
    "    --checkpoint_every 180 \\",
    "    --monitor_interval_s 300 \\",
    "    --fit_xi_params --fit_disk_thin --fit_disk_thick --fit_bulge --fit_gas \\",
    "    --include_disk_thin --include_disk_thick --include_bulge --include_gas"
])

print("🚀 RESTART COMMAND (with corrected physics):")
print("-" * 80)
print("\n".join(cmd_parts))

# Save as executable script
with open("restart_corrected.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    f.write("\n".join(cmd_parts))
    
print("\n✅ Saved as restart_corrected.sh")
print("\nTo run: bash restart_corrected.sh")