@echo off
echo Directory Reorganization Script for DensityDependentMetricModel
echo ============================================================
echo.
echo This will reorganize your project files into a cleaner structure:
echo.
echo   /core/         - Core physics and data handling modules  
echo   /runners/      - Execution scripts
echo   /data_loaders/ - Data loading modules
echo   /tests/        - Test files
echo   /analysis/     - Analysis scripts
echo   /validation/   - Validation tools
echo   /results/      - Output files
echo   /external_data/- External datasets
echo   /utils/        - Utility scripts
echo   /docs/         - Documentation
echo   /logs/         - Log files
echo.
echo WARNING: This will move many files. A backup is recommended\!
echo.
echo To run the reorganization:
echo   1. First run: python reorganize_directory.py (preview mode)
echo   2. Then run:  python reorganize_directory.py --execute
echo.
pause
