@echo off
:: Installs onnx_light locally and builds the standalone custom-pattern example.

setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_DIR=%%~fI"
for %%I in ("%SCRIPT_DIR%\..\..") do set "REPO_ROOT=%%~fI"

if "%~1"=="" (set "INSTALL_PREFIX=%REPO_ROOT%\build\install-register-custom-pattern") else (set "INSTALL_PREFIX=%~f1")
if "%~2"=="" (set "LIB_BUILD_DIR=%REPO_ROOT%\build\register-custom-pattern-lib") else (set "LIB_BUILD_DIR=%~f2")
if "%~3"=="" (set "EXAMPLE_BUILD_DIR=%REPO_ROOT%\build\register-custom-pattern-example") else (set "EXAMPLE_BUILD_DIR=%~f3")
if "%CMAKE_BUILD_TYPE%"=="" (set "BUILD_TYPE=Release") else (set "BUILD_TYPE=%CMAKE_BUILD_TYPE%")
if "%CMAKE_BUILD_PARALLEL_LEVEL%"=="" (set "PARALLEL_JOBS=%NUMBER_OF_PROCESSORS%") else (set "PARALLEL_JOBS=%CMAKE_BUILD_PARALLEL_LEVEL%")

cmake -S "%REPO_ROOT%" -B "%LIB_BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DONNX_LIGHT_BUILD_PYTHON=OFF ^
    -DONNX_LIGHT_BUILD_TESTS=OFF ^
    -DCMAKE_INSTALL_PREFIX="%INSTALL_PREFIX%"
if errorlevel 1 exit /b 1

cmake --build "%LIB_BUILD_DIR%" --config %BUILD_TYPE% --parallel %PARALLEL_JOBS%
if errorlevel 1 exit /b 1
cmake --install "%LIB_BUILD_DIR%" --config %BUILD_TYPE%
if errorlevel 1 exit /b 1

cmake -S "%SCRIPT_DIR%" -B "%EXAMPLE_BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_PREFIX_PATH="%INSTALL_PREFIX%"
if errorlevel 1 exit /b 1
cmake --build "%EXAMPLE_BUILD_DIR%" --config %BUILD_TYPE% --parallel %PARALLEL_JOBS%
if errorlevel 1 exit /b 1

echo Example binary: %EXAMPLE_BUILD_DIR%\%BUILD_TYPE%\register_custom_pattern.exe

endlocal
