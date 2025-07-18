#!/usr/bin/env python3
"""
test_sparc_loader.py - Test that SPARCDataLoader works with your data
"""

from sparc_data_loader import SPARCDataLoader
import matplotlib.pyplot as plt

# The loader NEEDS to know where your SPARC files are!
sparc_directory = "Rotmod_LTG"

# Create loader instance
print(f"Creating loader for directory: {sparc_directory}")
loader = SPARCDataLoader(sparc_directory)

# Load all galaxies
galaxies = loader.load_all_galaxies()

if not galaxies:
    print("ERROR: No galaxies loaded! Check your directory path.")
    exit(1)

# Show what we loaded
print(f"\nLoaded {len(galaxies)} galaxies:")
for i, name in enumerate(list(galaxies.keys())[:10]):
    galaxy = galaxies[name]
    print(f"  {name}: {len(galaxy['r_kpc'])} data points, "
          f"max v_obs = {max(galaxy['v_obs']):.1f} km/s")

# Plot first galaxy as a test
first_name = list(galaxies.keys())[0]
first_galaxy = galaxies[first_name]

plt.figure(figsize=(8, 6))
plt.errorbar(first_galaxy['r_kpc'], first_galaxy['v_obs'], 
             yerr=first_galaxy['v_err'], fmt='o', label='Observed')
plt.plot(first_galaxy['r_kpc'], first_galaxy['v_baryon'], 
         'g--', label='Baryonic')
plt.xlabel('R (kpc)')
plt.ylabel('V (km/s)')
plt.title(f'{first_name} Rotation Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('test_loader_plot.png')
print(f"\nTest plot saved to test_loader_plot.png")