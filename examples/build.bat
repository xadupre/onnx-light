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

if not exist "%EXAMPLES_BUILD_ROOT%" mkdir "%EXAMPLES_BUILD_ROOT%"

echo === Building examples (%BUILD_TYPE%) ===
for /d %%D in ("%SCRIPT_DIR%\*") do (
    if exist "%%D\build.bat" (
        set "example_name=%%~nxD"
        set "example_build_dir=%EXAMPLES_BUILD_ROOT%\!example_name!"

        echo --- Building !example_name! ---
        setlocal
        set "CMAKE_BUILD_TYPE=%BUILD_TYPE%"
        call "%%D\build.bat" "%INSTALL_PREFIX%" "%LIB_BUILD_DIR%" "!example_build_dir!"
        if errorlevel 1 (
            endlocal
            exit /b 1
        )
        endlocal
    )
)

echo Built examples in: %EXAMPLES_BUILD_ROOT%

endlocal
