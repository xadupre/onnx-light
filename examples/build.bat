@echo off
:: build.bat -- Installs onnx_light locally and builds all standalone examples
:: in the examples directory against that install.
::
:: Usage (run from the repository root or from examples\):
::   examples\build.bat [install-prefix] [lib-build-dir] [examples-build-root]

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_DIR=%%~fI"
for %%I in ("%SCRIPT_DIR%\..") do set "REPO_ROOT=%%~fI"

if "%~1"=="" (
    set "INSTALL_PREFIX=%REPO_ROOT%\build\install-examples"
) else (
    set "INSTALL_PREFIX=%~f1"
)

if "%~2"=="" (
    set "LIB_BUILD_DIR=%REPO_ROOT%\build\examples-lib"
) else (
    set "LIB_BUILD_DIR=%~f2"
)

if "%~3"=="" (
    set "EXAMPLES_BUILD_ROOT=%REPO_ROOT%\build\examples"
) else (
    set "EXAMPLES_BUILD_ROOT=%~f3"
)

if "%CMAKE_BUILD_TYPE%"=="" (
    set "BUILD_TYPE=Release"
) else (
    set "BUILD_TYPE=%CMAKE_BUILD_TYPE%"
)

echo === Step 1: Configure and Build onnx_light (%BUILD_TYPE%) ===
cmake -S "%REPO_ROOT%" -B "%LIB_BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DONNX_LIGHT_BUILD_PYTHON=OFF ^
    -DONNX_LIGHT_BUILD_TESTS=OFF ^
    -DCMAKE_INSTALL_PREFIX="%INSTALL_PREFIX%"
if errorlevel 1 exit /b 1

cmake --build "%LIB_BUILD_DIR%" --config %BUILD_TYPE% --parallel
if errorlevel 1 exit /b 1

cmake --install "%LIB_BUILD_DIR%" --config %BUILD_TYPE%
if errorlevel 1 exit /b 1

if not exist "%EXAMPLES_BUILD_ROOT%" mkdir "%EXAMPLES_BUILD_ROOT%"

echo === Step 2: Configure and Build examples (%BUILD_TYPE%) ===
for /d %%D in ("%SCRIPT_DIR%\*") do (
    if exist "%%D\CMakeLists.txt" (
        set "example_name=%%~nxD"
        set "example_build_dir=%EXAMPLES_BUILD_ROOT%\!example_name!"

        echo --- Building !example_name! ---
        cmake -S "%%D" -B "!example_build_dir!" ^
            -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
            -DCMAKE_PREFIX_PATH="%INSTALL_PREFIX%"
        if errorlevel 1 exit /b 1

        cmake --build "!example_build_dir!" --config %BUILD_TYPE% --parallel
        if errorlevel 1 exit /b 1
    )
)

echo Built examples in: %EXAMPLES_BUILD_ROOT%

endlocal
