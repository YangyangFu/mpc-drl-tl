echo on
rem direct to some folders
for %%a in ("%CD%") do set "p_dir=%%~dpa"
for %%a in (%p_dir:~0,-1%) do set "p2_dir=%%~dpa"
for %%a in (%p2_dir:~0,-1%) do set "p3_dir=%%~dpa"

python run_all_in_once.py