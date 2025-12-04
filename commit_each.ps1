$changes = git status --porcelain

foreach ($line in $changes -split "`n") {
    if ($line.Trim() -ne "") {
        $status = $line.Substring(0,2).Trim()
        $file = $line.Substring(3)
        Write-Host "Processing: $file"
        git add "$file"
        $message = "Commit for $file"
        git commit -m "$message"
    }
}

git push
