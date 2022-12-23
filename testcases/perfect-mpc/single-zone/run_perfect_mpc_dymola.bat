rem add library dependencies
rem add Buildings library to Modelicapath
echo on
rem direct to some folders
for %%a in ("%CD%") do set "p_dir=%%~dpa"
for %%a in (%p_dir:~0,-1%) do set "p2_dir=%%~dpa"
for %%a in (%p2_dir:~0,-1%) do set "p3_dir=%%~dpa"
for %%a in (%p3_dir:~0,-1%) do set "p4_dir=%%~dpa"
for %%a in (%p4_dir:~0,-1%) do set "p5_dir=%%~dpa"

rem add ThreatInjection library to Modelicapath
set THREATINJECTION_PATH=%p5_dir%
set MODELICAPATH=%THREATINJECTION_PATH%;%MODELICAPATH%
echo %THREATINJECTION_PATH%

python run_perfect_mpc.py