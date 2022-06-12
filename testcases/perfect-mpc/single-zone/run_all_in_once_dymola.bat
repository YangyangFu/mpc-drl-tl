rem add library dependencies
rem add Buildings library to Modelicapath
echo on
rem direct to some folders
for %%a in ("%CD%") do set "p_dir=%%~dpa"
for %%a in (%p_dir:~0,-1%) do set "p2_dir=%%~dpa"
for %%a in (%p2_dir:~0,-1%) do set "p3_dir=%%~dpa"

rem add ThreatInjection library to Modelicapath
rem set THREATINJECTION_PATH=%p3_dir%
rem set MODELICAPATH=%THREATINJECTION_PATH%;%MODELICAPATH%
rem echo %THREATINJECTION_PATH%

python run_all_in_once.py