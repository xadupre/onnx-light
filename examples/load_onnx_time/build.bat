@echo off
:: build.bat -- builds the standalone load_onnx_time example against the
:: standard onnx C++ library.
::
:: Prerequisites: the standard onnx C++ library must be installed and
:: findable by CMake (e.g. via CMAKE_PREFIX_PATH).  On Windows you can
:: install onnx using vcpkg:
::   vcpkg install onnx
::
:: Usage (run from the repository root or from this directory):
::   examples\load_onnx_time\build.bat [example-build-dir] [cmake-prefix-path]

setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_DIR=%%~fI"
for %%I in ("%SCRIPT_DIR%\..\..") do set "REPO_ROOT=%%~fI"

if "%~1"=="" (
    set "EXAMPLE_BUILD_DIR=%REPO_ROOT%\build\load-onnx-time-example"
) else (
    set "EXAMPLE_BUILD_DIR=%~f1"
)

if "%CMAKE_BUILD_TYPE%"=="" (
    set "BUILD_TYPE=Release"
) else (
    set "BUILD_TYPE=%CMAKE_BUILD_TYPE%"
)

set "CMAKE_EXTRA_ARGS="
if not "%~2"=="" (
    set "CMAKE_EXTRA_ARGS=-DCMAKE_PREFIX_PATH=%~f2"
)

echo === Configure and build load_onnx_time (%BUILD_TYPE%) ===
cmake -S "%SCRIPT_DIR%" -B "%EXAMPLE_BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    %CMAKE_EXTRA_ARGS%
if errorlevel 1 exit /b 1

cmake --build "%EXAMPLE_BUILD_DIR%" --config %BUILD_TYPE% --parallel
if errorlevel 1 exit /b 1

echo.
echo Example binary:
echo   %EXAMPLE_BUILD_DIR%\%BUILD_TYPE%\load_onnx_time.exe

endlocal
