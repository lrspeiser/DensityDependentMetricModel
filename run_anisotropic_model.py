#!/usr/bin/env python3
"""
Wrapper to run dynesty with anisotropic xi model.
This requires modifying how xi is called in the likelihood.
"""
import sys
import os

# Modify the xi_type argument before importing run_dynesty
if '--xi' not in sys.argv:
    sys.argv.extend(['--xi', 'anisotropic_radial'])

# Now import and run
from run_dynesty import main_dynesty

if __name__ == "__main__":
    # Note: The likelihood function needs to be modified to handle
    # different xi for radial vs vertical calculations
    print("WARNING: This requires modifying the likelihood to use")
    print("xi_anisotropic(rho, 'radial') for rotation curve")
    print("xi_anisotropic(rho, 'vertical') for K_z calculations")
    
    main_dynesty()
