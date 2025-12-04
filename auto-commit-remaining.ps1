#!/usr/bin/env pwsh
<#
.SYNOPSIS
Improved auto-commit script for remaining changes
.DESCRIPTION
Commits each remaining file change separately
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Auto-Commit for Remaining Changes" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$changes = @(git status --short)
$totalChanges = $changes.Count
Write-Host "Remaining changes found: $totalChanges" -ForegroundColor Yellow

if ($totalChanges -eq 0) {
    Write-Host "No changes to commit!" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
$commitCount = 0
$successCount = 0

foreach ($line in $changes) {
    if ($line.Trim() -eq "") { continue }
    
    $commitCount++
    $status = $line.Substring(0, 2).Trim()
    $file = $line.Substring(3).Trim()
    
    # Determine message
    if ($status -eq "D") {
        $msg = "remove: delete $file"
    } elseif ($status -eq "M") {
        $msg = "update: modify $file"
    } else {
        $msg = "chore: update $file"
    }
    
    # Stage and commit
    git add "$file" 2>$null
    $output = git commit -m "$msg" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "$commitCount/$totalChanges [OK] $msg" -ForegroundColor Green
        $successCount++
    } else {
        Write-Host "$commitCount/$totalChanges [FAIL] $file" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Committed: $successCount / $totalChanges changes" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($successCount -gt 0) {
    Write-Host "Pushing to remote..." -ForegroundColor Yellow
    git push origin main
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Push completed successfully!" -ForegroundColor Green
    }
}

Write-Host ""
