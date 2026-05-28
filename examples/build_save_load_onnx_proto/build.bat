@echo off
:: build.bat -- installs onnx_light locally and builds the standalone
:: build_save_load_onnx_proto example against that install.
::
:: Usage (run from the repository root or from this directory):
::   examples\build_save_load_onnx_proto\build.bat [install-prefix] [lib-build-dir] [example-build-dir]

setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_DIR=%%~fI"
for %%I in ("%SCRIPT_DIR%\..\..") do set "REPO_ROOT=%%~fI"

if "%~1"=="" (
    set "INSTALL_PREFIX=%REPO_ROOT%\build\install-build-save-load-onnx-proto"
) else (
    set "INSTALL_PREFIX=%~f1"
)

if "%~2"=="" (
    set "LIB_BUILD_DIR=%REPO_ROOT%\build\build-save-load-onnx-proto-lib"
) else (
    set "LIB_BUILD_DIR=%~f2"
)

if "%~3"=="" (
    set "EXAMPLE_BUILD_DIR=%REPO_ROOT%\build\build-save-load-onnx-proto-example"
) else (
    set "EXAMPLE_BUILD_DIR=%~f3"
)

if "%CMAKE_BUILD_TYPE%"=="" (
    set "BUILD_TYPE=Release"
) else (
    set "BUILD_TYPE=%CMAKE_BUILD_TYPE%"
)

if "%CMAKE_BUILD_PARALLEL_LEVEL%"=="" (
    set "PARALLEL_JOBS=%NUMBER_OF_PROCESSORS%"
) else (
    set "PARALLEL_JOBS=%CMAKE_BUILD_PARALLEL_LEVEL%"
)

echo === Step 1: configure and build onnx_light (%BUILD_TYPE%) ===
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

echo === Step 2: configure and build build_save_load_onnx_proto (%BUILD_TYPE%) ===
cmake -S "%SCRIPT_DIR%" -B "%EXAMPLE_BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_PREFIX_PATH="%INSTALL_PREFIX%"
if errorlevel 1 exit /b 1

cmake --build "%EXAMPLE_BUILD_DIR%" --config %BUILD_TYPE% --parallel %PARALLEL_JOBS%
if errorlevel 1 exit /b 1

echo.
echo Example binary:
echo   %EXAMPLE_BUILD_DIR%\%BUILD_TYPE%\build_save_load_onnx_proto.exe

endlocal
