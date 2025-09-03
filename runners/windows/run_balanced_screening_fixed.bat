@echo off
echo Starting balanced screening model with FIXED gravitational constant
echo.
python runners/run_dynesty_cupy.py --xi balanced_screening --max_sample_gaia 144000 --maxcall 10000000 --output_dir runs
echo.
echo Run complete!
pause