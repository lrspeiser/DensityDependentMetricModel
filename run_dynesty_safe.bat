@echo off
echo Starting dynesty script with safety measures...
echo Timestamp: %date% %time%

REM Set timeout (30 seconds)
timeout /t 30 /nobreak > nul

REM Run the script with output redirection
echo Running: py run_dynesty.py --xi grav_color --fit_xi_params --nlive_init 50 --maxcall 500 --disable_cassini_penalty
py run_dynesty.py --xi grav_color --fit_xi_params --nlive_init 50 --maxcall 500 --disable_cassini_penalty > output.log 2>&1

REM Check if the script completed
if %errorlevel% equ 0 (
    echo Script completed successfully
) else (
    echo Script failed with error code: %errorlevel%
)

echo Final timestamp: %date% %time%
echo Check output.log for details
pause 