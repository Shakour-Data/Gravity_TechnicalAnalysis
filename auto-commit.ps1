#!/usr/bin/env pwsh
<#
.SYNOPSIS
Automated commit script for individual file changes
.DESCRIPTION
Commits each file change separately and then pushes to remote
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

# Colors for output
function Write-Status { Write-Host $args[0] -ForegroundColor Green }
function Write-Info { Write-Host $args[0] -ForegroundColor Cyan }
function Write-Error { Write-Host $args[0] -ForegroundColor Red }
function Write-Success { Write-Host $args[0] -ForegroundColor Yellow }

# Get current branch
$branch = git rev-parse --abbrev-ref HEAD
Write-Info "Current branch: $branch"
Write-Info "Starting automatic commits for individual changes..."
Write-Info ""

# Get all changes
$changes = @()
$statusOutput = git status --short

foreach ($line in $statusOutput) {
    if ($line.Trim()) {
        $changes += $line.Trim()
    }
}

$totalChanges = $changes.Count
Write-Info "Total changes found: $totalChanges"
Write-Info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Info ""

$commitCount = 0
$successCount = 0
$errorCount = 0

# Process each change
foreach ($change in $changes) {
    $commitCount++
    $status = $change.Substring(0, 2).Trim()
    $file = $change.Substring(3).Trim()
    
    # Determine commit message based on status
    $commitMsg = ""
    switch ($status) {
        "D" { 
            $commitMsg = "remove: delete $file"
            $actionIcon = "[DEL]"
        }
        "M" { 
            $commitMsg = "update: modify $file"
            $actionIcon = "[MOD]"
        }
        "A" { 
            $commitMsg = "feat: add $file"
            $actionIcon = "[ADD]"
        }
        "??" { 
            $commitMsg = "feat: add new file $file"
            $actionIcon = "[NEW]"
        }
        default { 
            $commitMsg = "chore: update $file"
            $actionIcon = "[UPD]"
        }
    }
    
    # Stage the file
    try {
        if ($status -eq "D") {
            # For deleted files, we need to stage the deletion
            git add "$file" 2>$null
        } else {
            # For other changes
            git add "$file" 2>$null
        }
        
        # Commit
        $output = git commit -m $commitMsg 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "$commitCount/$totalChanges [$actionIcon] $commitMsg"
            $successCount++
        } else {
            Write-Error "$commitCount/$totalChanges [ERR] Failed: $file"
            $errorCount++
        }
    }
    catch {
        Write-Error "$commitCount/$totalChanges [ERR] Error with $file"
        $errorCount++
    }
}

Write-Info ""
Write-Info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Status "Commit Summary:"
Write-Status "  Total Changes: $totalChanges"
Write-Status "  Successful: $successCount"
Write-Error "  Failed: $errorCount"
Write-Info ""

# Push all commits
if ($successCount -gt 0) {
    Write-Info "Pushing commits to remote..."
    try {
        git push origin $branch
        if ($LASTEXITCODE -eq 0) {
            Write-Success "✅ All commits pushed successfully!"
        } else {
            Write-Error "❌ Failed to push commits"
            exit 1
        }
    }
    catch {
        Write-Error "❌ Error during push: $_"
        exit 1
    }
} else {
    Write-Error "❌ No commits were successful, skipping push"
    exit 1
}

Write-Info ""
Write-Success "✅ ALL OPERATIONS COMPLETED SUCCESSFULLY"
Write-Info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
