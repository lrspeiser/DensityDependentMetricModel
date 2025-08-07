@echo off
echo Starting balanced screening model with FIXED gravitational constant
echo.
python runners/run_dynesty_cupy.py --xi_type balanced_screening --n_samples 144000 --maxcall 10000000 --output_dir runs
echo.
echo Run complete!
pause