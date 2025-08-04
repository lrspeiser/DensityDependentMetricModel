# PowerShell script to run dynesty with UTF-8 encoding and safety measures
Write-Host "Setting up UTF-8 encoding for Windows..." -ForegroundColor Green

# Set UTF-8 encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Testing Unicode fixes..." -ForegroundColor Yellow
try {
    $testResult = & py test_fixes.py 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Unicode test failed!" -ForegroundColor Red
        Write-Host $testResult
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Unicode test passed!" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to run test script" -ForegroundColor Red
    Write-Host $_.Exception.Message
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Testing resource monitoring..." -ForegroundColor Yellow
try {
    $resourceResult = & py test_resource_monitor.py 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Resource monitoring test failed - continuing anyway" -ForegroundColor Yellow
        Write-Host $resourceResult
    } else {
        Write-Host "Resource monitoring test passed!" -ForegroundColor Green
    }
} catch {
    Write-Host "WARNING: Failed to run resource monitoring test - continuing anyway" -ForegroundColor Yellow
}

Write-Host "Starting dynesty run with safe parameters..." -ForegroundColor Green
Write-Host "Output will be redirected to files to avoid terminal issues." -ForegroundColor Yellow

Start-Sleep -Seconds 5

Write-Host "Running main script..." -ForegroundColor Green
try {
    $output = & py run_dynesty.py --xi grav_color --fit_target milkyway --fit_xi_params --maxcall 1000 --nlive 100 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: Script completed successfully" -ForegroundColor Green
        $output | Out-File -FilePath "output.log" -Encoding UTF8
        Write-Host "Check output.log for results" -ForegroundColor Yellow
    } else {
        Write-Host "ERROR: Script failed with exit code $LASTEXITCODE" -ForegroundColor Red
        $output | Out-File -FilePath "error.log" -Encoding UTF8
        Write-Host "Check error.log for details" -ForegroundColor Yellow
        Write-Host $output
    }
} catch {
    Write-Host "ERROR: Failed to run main script" -ForegroundColor Red
    Write-Host $_.Exception.Message
    $_.Exception.Message | Out-File -FilePath "error.log" -Encoding UTF8
}

Read-Host "Press Enter to exit" 