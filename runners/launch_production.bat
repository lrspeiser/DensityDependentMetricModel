@echo off
REM Launch full production run with all available stars and maximum iterations

echo ================================================================================
echo LAUNCHING FULL PRODUCTION RUN
echo ================================================================================
echo.
echo This will run stellar fitting with:
echo   - ALL available stars (144,000+)
echo   - 40 million likelihood evaluations per model
echo   - 2000-3000 live points for thorough exploration
echo.
echo Estimated runtime: 5-10 hours per model (25-50 hours total for 5 priority models)
echo.
echo Press Ctrl+C at any time to interrupt
echo ================================================================================
echo.

REM Check if we should use auto configuration
echo Checking GPU memory and selecting optimal configuration...
python run_production_fits.py --auto --priority

REM Alternative: Run with explicit high-precision settings
REM python run_production_fits.py --config high_precision --priority

REM Alternative: Run specific models with custom settings
REM python run_all_stellar_fits.py --models power grav_color tidal_band --sample_max 144000 --maxcall 40000000 --nlive 3000

pause
