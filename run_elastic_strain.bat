@echo off
REM Run elastic_strain model with enhanced summaries

echo ========================================
echo STARTING ELASTIC STRAIN MODEL RUN
echo ========================================
echo.
echo This run will test the elastic_strain xi function
echo with enhanced summary output every 30 seconds
echo.

REM Run with enhanced summaries and reasonable parameters
python runners/run_dynesty_cupy.py ^
    --xi elastic_strain ^
    --nlive 500 ^
    --dlogz_target 0.01 ^
    --maxcall 2000000 ^
    --max_sample_gaia 10000 ^
    --summary_interval 30 ^
    --periodic_analysis ^
    --analysis_interval_min 10 ^
    --num_threads 4 ^
    --sample_method rslice ^
    --bound_method multi ^
    --checkpoint_every 300

echo.
echo ========================================
echo RUN COMPLETED
echo ========================================
pause