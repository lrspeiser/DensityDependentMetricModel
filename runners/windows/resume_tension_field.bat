@echo off
REM Resume the tension_field run with enhanced summaries

echo ========================================
echo RESUMING TENSION FIELD RUN
echo ========================================
echo.

REM Copy the checkpoint to the expected location for resume
echo Preparing checkpoint for resume...
copy "runs\tension_field_20250806_174323\dynesty_checkpoint_tension_field_latest.npz" "runs\tension_field_20250806_174323\dynesty_checkpoint.pkl" >nul 2>&1

REM Run with resume flag and enhanced summaries
python runners/run_dynesty_cupy.py ^
    --xi tension_field ^
    --nlive 400 ^
    --dlogz_target 0.01 ^
    --output_dir runs/tension_field_20250806_174323 ^
    --maxcall 5000000 ^
    --max_sample_gaia 10000 ^
    --resume ^
    --summary_interval 30 ^
    --periodic_analysis ^
    --analysis_interval_min 5

echo.
echo ========================================
echo RUN COMPLETED
echo ========================================
pause