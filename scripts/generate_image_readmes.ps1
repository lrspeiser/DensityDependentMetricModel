# Generates README.md files for directories containing images.
#
# Notes:
# - Prefers Rust CLI tools (fd/rg) for discovery per Windows Rust-first workflow.
# - Falls back to PowerShell enumeration only if fd/rg are unavailable.
# - No web services or API keys are used here.

Param(
    [string]$Root = ".",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Test-Cli {
    param([string]$name)
    try { $null = Get-Command $name -ErrorAction Stop; return $true } catch { return $false }
}

function Get-RelPath {
    param([string]$Base, [string]$Target)
    $basePath = (Resolve-Path $Base).Path
    if (Test-Path $Target) {
        $baseUri = New-Object System.Uri($basePath)
        $targetUri = New-Object System.Uri((Resolve-Path $Target))
        $rel = $baseUri.MakeRelativeUri($targetUri).ToString()
        return ($rel -replace '/', [System.IO.Path]::DirectorySeparatorChar)
    } else {
        # Target may not exist yet (e.g., README.md during dry-run). Build a relative path safely.
        $normalizedTarget = $Target -replace '/', '\\'
        if (-not [System.IO.Path]::IsPathRooted($normalizedTarget)) {
            $normalizedTarget = Join-Path $basePath $normalizedTarget
        }
        $bp = $basePath.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
        $tp = $normalizedTarget
        if ($tp.ToLower().StartsWith(($bp.ToLower() + '\\'))) {
            return $tp.Substring($bp.Length + 1)
        } else {
            return $Target
        }
    }
}

function Get-ImageDirectories {
    param([string]$root)
    $exts = @('png','jpg','jpeg','gif','bmp','tiff','webp','svg')
    if (Test-Cli -name 'fd') {
        $args = @('-H','-t','f')
        foreach ($e in $exts) { $args += @('-e', $e) }
        $args += @('-E','.git','-E','node_modules','.', $root)
        $files = & fd @args
    } elseif (Test-Cli -name 'rg') {
        $pattern = '\\.(png|jpg|jpeg|gif|bmp|tiff|webp|svg)$'
        $files = & rg -n --hidden -g '!node_modules' -g '!.git' $pattern $root | ForEach-Object { ($_ -split ':',2)[0] }
    } else {
        # Fallback to PowerShell enumeration if fd/rg are unavailable.
        $files = Get-ChildItem -LiteralPath $root -Recurse -File -Force -Include *.png,*.jpg,*.jpeg,*.gif,*.bmp,*.tiff,*.webp,*.svg | Select-Object -ExpandProperty FullName
    }
    $dirs = $files | ForEach-Object { Split-Path -Parent $_ } | Sort-Object -Unique
    return $dirs
}

function Parse-ParamsFromPath {
    param([string]$path)
    $p = [ordered]@{}
    if ($path -match 'gamma(?<gamma>[0-9p\.]+)') { $g = $Matches['gamma'] -replace 'p','.'; $p['gamma'] = $g }
    if ($path -match 'lambda(?<lambda>[0-9p\.]+)') { $l = $Matches['lambda'] -replace 'p','.'; $p['lambda'] = $l }
    if ($path -match 'alpha[_]?(?<alpha>[0-9p\.]+)') { $a = $Matches['alpha'] -replace 'p','.'; $p['alpha'] = $a }
    if ($path -match 'zeta[_\-]?(?<zeta>m?[0-9p\.]+)') { $z = $Matches['zeta'] -replace 'p','.'; if ($z -like 'm*') { $z = '-' + $z.Substring(1) }; $p['zeta'] = $z }
    if ($path -match '(hernquist|jaffe|nfw|gaussian|logistic|exponential|mond)') { $p['profile'] = $Matches[0] }
    if ($path -match '(rar_gate|gr_only|gr_nfw|rar)') { $p['mode'] = $Matches[0] }
    if ($path -match '(sparc|mw|lensing|btfr)') { $p['dataset'] = $Matches[0] }
    return $p
}

function Get-ImageFilesInDir {
    param([string]$dir)
    $exts = @('png','jpg','jpeg','gif','bmp','tiff','webp','svg')
    $files = @()
    if (Test-Cli -name 'fd') {
        $args = @('-H','-t','f')
        foreach ($e in $exts) { $args += @('-e',$e) }
        $args += @('.', $dir)
        $files = & fd @args
        $files = $files | ForEach-Object { if ($_ -like '.\\*' -or $_ -like './*') { Join-Path $dir ($_ -replace '^\.\\|^\./','') } else { $_ } }
    } else {
        $files = Get-ChildItem -LiteralPath $dir -File -Force -Include *.png,*.jpg,*.jpeg,*.gif,*.bmp,*.tiff,*.webp,*.svg | Select-Object -ExpandProperty FullName
    }
    return @($files | Sort-Object)
}

function Find-RunMetadataForDir {
    param([string]$root, [string]$dir)
    # Map images/... to results/... and try common metadata files
    $rel = Get-RelPath -Base $root -Target $dir
    if ($rel -match '^images\\(.+)$') {
        $sub = $Matches[1]
        $baseResults = Join-Path (Join-Path $root 'results') $sub
        $candidates = @(
            (Join-Path $baseResults 'run_metadata.json'),
            (Join-Path $baseResults 'btfr_fit_summary.json'),
            (Join-Path $baseResults 'summary_min.json'),
            (Join-Path $baseResults 'summary.json')
        )
        $found = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($found) {
            try {
                $raw = Get-Content -LiteralPath $found -Raw -ErrorAction Stop | ConvertFrom-Json
                $meta = [ordered]@{}
                if ($raw.PSObject.Properties.Name -contains 'raw_command') { $meta['raw_command'] = $raw.raw_command }
                if ($raw.PSObject.Properties.Name -contains 'timestamp') { $meta['timestamp'] = $raw.timestamp }
                if ($raw.PSObject.Properties.Name -contains 'run_dir') { $meta['run_dir'] = $raw.run_dir }
                $meta['source_file'] = Get-RelPath -Base $root -Target $found
                return $meta
            } catch {
                return @{ source_file = Get-RelPath -Base $root -Target $found; parse_error = $_.Exception.Message }
            }
        }
    }
    return $null
}

function Get-ImageDateRange {
    param([string[]]$images)
    if (-not $images -or $images.Count -eq 0) { return $null }
    $times = @()
    foreach ($img in $images) {
        try { $times += (Get-Item -LiteralPath $img).LastWriteTime } catch {}
    }
    if ($times.Count -eq 0) { return $null }
    return @{ first = ($times | Sort-Object | Select-Object -First 1); last = ($times | Sort-Object | Select-Object -Last 1) }
}

function New-ReadmeContent {
    param(
        [string]$root,
        [string]$dir,
        [hashtable]$params,
        [string[]]$images,
        [hashtable]$runMeta
    )
    $relDir = Get-RelPath -Base $root -Target $dir
    $nl = "`r`n"
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append("# $relDir$nl$nl")
    [void]$sb.Append("This folder contains generated images. Fill in the sections below to document the generation parameters and what each image depicts.$nl$nl")
    [void]$sb.Append("## Parameters$nl")
    if ($params.Keys.Count -gt 0) {
        foreach ($k in $params.Keys) {
            [void]$sb.Append("- $($k): $($params[$k])$nl")
        }
    } else {
        [void]$sb.Append("- TODO: parameters not auto-detected from folder name. Add details here.$nl")
    }
    [void]$sb.Append("$nl## What these images are$nl- TODO: Describe the subject and purpose of these images.$nl")
    [void]$sb.Append("$nl## Image list$nl")
    foreach ($img in $images) {
        $name = Split-Path -Leaf $img
        [void]$sb.Append("- $name - TODO: description$nl")
    }

    # Reproduction details from run metadata and date ranges
    $dateRange = Get-ImageDateRange -images $images
    [void]$sb.Append("$nl## Reproduction$nl")
    if ($runMeta) {
        if ($runMeta['raw_command']) { [void]$sb.Append("- Raw command: `$($runMeta['raw_command'])`$nl") } else { [void]$sb.Append("- Raw command: TODO$nl") }
        if ($runMeta['timestamp']) { [void]$sb.Append("- Run timestamp: $($runMeta['timestamp'])$nl") }
        if ($runMeta['run_dir']) { [void]$sb.Append("- Run dir: $($runMeta['run_dir'])$nl") }
        if ($runMeta['source_file']) { [void]$sb.Append("- Metadata source: $($runMeta['source_file'])$nl") }
        if ($runMeta['parse_error']) { [void]$sb.Append("- Metadata parse error: $($runMeta['parse_error'])$nl") }
    } else {
        [void]$sb.Append("- Run metadata: not found - TODO: add the exact command and config used.$nl")
    }
    if ($dateRange) {
        [void]$sb.Append("- Image file modified (local time): first=$($dateRange.first), last=$($dateRange.last)$nl")
    }

    [void]$sb.Append("$nl## Notes$nl- TODO: Any caveats or additional details.$nl")
    return $sb.ToString()
}

$Root = Resolve-Path $Root
$dirs = Get-ImageDirectories -root $Root

foreach ($dir in $dirs) {
    $readmePath = Join-Path $dir 'README.md'
    if (Test-Path $readmePath) {
        Write-Host "Exists: $(Get-RelPath -Base $Root -Target $readmePath)"
        continue
    }
    $params = Parse-ParamsFromPath -path $dir
    $images = Get-ImageFilesInDir -dir $dir
    $runMeta = Find-RunMetadataForDir -root $Root -dir $dir
    $content = New-ReadmeContent -root $Root -dir $dir -params $params -images $images -runMeta $runMeta
    if ($DryRun) {
        Write-Host "Would create: $(Get-RelPath -Base $Root -Target $readmePath) (images: $(@($images).Count))"
        if ($params.Keys.Count -gt 0) {
            $pv = ($params.Keys | ForEach-Object { "$_=$($params[$_])" }) -join ', '
            Write-Host "  Detected params: $pv"
        }
        if ($runMeta) {
            $keys = ($runMeta.Keys -join ', ')
            Write-Host "  Found run metadata keys: $keys"
        }
        continue
    }
    $content | Out-File -FilePath $readmePath -Encoding UTF8 -Force
    Write-Host "Created: $(Get-RelPath -Base $Root -Target $readmePath)"
}

