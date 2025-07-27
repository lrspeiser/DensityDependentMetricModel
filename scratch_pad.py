# Test the enhanced model at galaxy scale
import numpy as np

def xi_enhanced(rho, rho_c, n, A):
    return 1 + A / (1 + (rho/rho_c)**n)

# Galaxy scale test
rho_galaxy = 1e9  # Typical galaxy density
rho_c = 1e13
n = 1.5
A = 1.0

xi_gal = xi_enhanced(rho_galaxy, rho_c, n, A)
print(f"Galaxy: rho={rho_galaxy:.1e}, xi={xi_gal:.3f}")

# Saturn test
rho_saturn = 2.3e21
xi_sat = xi_enhanced(rho_saturn, rho_c, n, A)
print(f"Saturn: rho={rho_saturn:.1e}, xi={xi_sat:.10f}")
print(f"Cassini test: |xi-1| = {abs(xi_sat-1):.2e} < 2.3e-5? {abs(xi_sat-1) < 2.3e-5}")