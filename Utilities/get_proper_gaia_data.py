#!/usr/bin/env python3
"""
get_proper_gaia_data.py - Quick script to download Gaia data with proper radial coverage
"""
import sys
import shutil
from pathlib import Path

print("="*60)
print("GETTING PROPER GAIA DATA")
print("="*60)

# Option 1: Use the comprehensive loader
use_comprehensive = "--full" in sys.argv

if use_comprehensive:
    print("\nUsing comprehensive multi-region query...")
    from gaia_proper_loader import load_gaia_with_validation
    
    # Get 1000 stars with good radial distribution
    df = load_gaia_with_validation(n_stars=1000, force_new=True)
    
else:
    # Option 2: Quick fix - add the enhanced functions to existing data_io
    print("\nPatching existing data_io.py with enhanced query...")
    
    # Backup original
    if Path("data_io.py").exists():
        shutil.copy("data_io.py", "data_io.py.backup_original")
        print("✓ Backed up original data_io.py")
    
    # Import and run
    from data_io_enhanced_final import load_gaia_validated, perform_gaia_adql_query_multiregion, validate_data_distribution
    
    # Force new query with multi-region sampling
    print("\nQuerying Gaia with multi-region sampling for better coverage...")
    data = load_gaia_validated(
        sample_max=1000,
        force_new_query_gaia=True,
        force_reprocess_raw=True,
        require_validation=True
    )
    
    if data is None:
        print("\n❌ Failed to get valid Gaia data!")
        print("Falling back to synthetic data...")
        import subprocess
        subprocess.run([sys.executable, "create_test_gaia_data.py"])
    else:
        print(f"\n✅ Success! Got {len(data['R_kpc'])} stars with proper distribution")

print("\n" + "="*60)
print("NEXT STEPS:")
print("1. Your data is ready in: gaia_query_cache_DR3_processed_for_fit.parquet")
print("2. Run dynesty with better parameters:")
print()
print("python3 run_dynesty.py \\")
print("    --output_dir chains_1param_proper \\")
print("    --max_sample_gaia 1000 \\")
print("    --nlive_init 400 \\")  
print("    --maxcall 20000 \\")
print("    --include_disk_thin --fit_disk_thin \\")
print("    --rho_c_fixed 1e9 --n_exp_fixed 1.0 \\")
print("    --M_disk_thin_fixed 6e10 --R_d_thin_fixed 3.0")
print("="*60)