Get-ChildItem -Path "apps\analysis_api\src\gravity_tech" -Recurse -Filter "*.py" | ForEach-Object { 
  $basePath = (Get-Item "apps\analysis_api\src\gravity_tech").FullName
  $relativePath = $_.FullName.Substring($basePath.Length + 1)
  $filename = $relativePath -replace '\\', '_' -replace '\.py$', '.txt'
  Get-Content $_.FullName | Out-File "main_python_files\$filename" -Encoding utf8 
}
