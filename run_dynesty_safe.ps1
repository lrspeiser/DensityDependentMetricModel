Write-Host "Starting dynesty script with safety measures..." -ForegroundColor Green
Write-Host "Timestamp: $(Get-Date)" -ForegroundColor Yellow

# Set timeout (30 seconds)
$timeout = 30

try {
    # Run the script with timeout
    Write-Host "Running: py run_dynesty.py --xi grav_color --fit_xi_params --nlive_init 50 --maxcall 500 --disable_cassini_penalty" -ForegroundColor Cyan
    
    $job = Start-Job -ScriptBlock {
        param($scriptPath, $args)
        & py $scriptPath $args
    } -ArgumentList "run_dynesty.py", "--xi", "grav_color", "--fit_xi_params", "--nlive_init", "50", "--maxcall", "500", "--disable_cassini_penalty"
    
    # Wait for job with timeout
    if (Wait-Job $job -Timeout $timeout) {
        $result = Receive-Job $job
        Remove-Job $job
        Write-Host "Script completed successfully" -ForegroundColor Green
        $result | Out-File -FilePath "output.log" -Append
    } else {
        Write-Host "Script timed out after $timeout seconds" -ForegroundColor Red
        Stop-Job $job
        Remove-Job $job
    }
} catch {
    Write-Host "Error running script: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "Final timestamp: $(Get-Date)" -ForegroundColor Yellow
Write-Host "Check output.log for details" -ForegroundColor Cyan
Read-Host "Press Enter to continue" 