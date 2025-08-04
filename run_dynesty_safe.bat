@echo off
echo Setting up UTF-8 encoding for Windows...
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

echo Testing Unicode fixes...
py test_fixes.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Unicode test failed!
    pause
    exit /b 1
)

echo Testing resource monitoring...
py test_resource_monitor.py
if %ERRORLEVEL% neq 0 (
    echo WARNING: Resource monitoring test failed - continuing anyway
)

echo Starting dynesty run with safe parameters...
echo Output will be redirected to files to avoid terminal issues.

timeout /t 5 /nobreak > nul

echo Running main script...
py run_dynesty.py --xi grav_color --fit_target milkyway --fit_xi_params --maxcall 1000 --nlive 100 > output.log 2> error.log

if %ERRORLEVEL% neq 0 (
    echo ERROR: Script failed with exit code %ERRORLEVEL%
    echo Check error.log for details
    type error.log
) else (
    echo SUCCESS: Script completed successfully
    echo Check output.log for results
)

pause 