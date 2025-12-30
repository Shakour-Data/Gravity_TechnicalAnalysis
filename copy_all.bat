@echo off
setlocal enabledelayedexpansion
for /r "apps\analysis_api\src\gravity_tech" %%f in (*.py) do (
  set "fullpath=%%f"
  set "relpath=!fullpath:apps\analysis_api\src\gravity_tech\=!"
  set "filename=!relpath:\=_!"
  set "filename=!filename:.py=.txt!"
  type "%%f" > "main_python_files\!filename!"
)
