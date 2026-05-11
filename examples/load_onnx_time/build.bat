@echo off
:: build.bat -- builds the standalone load_onnx_time example against the
:: standard onnx C++ library.
::
:: The script has two operating modes:
::
:: 1. System install (default when onnx is found as a CMake package):
::      examples\load_onnx_time\build.bat
::    On Windows you can install onnx via vcpkg: vcpkg install onnx
::
:: 2. From-source build (automatic when system onnx is absent, or explicit):
::      set ONNX_GIT_TAG=v1.17.0 && examples\load_onnx_time\build.bat
::    The script clones onnx from git and passes FETCHCONTENT_SOURCE_DIR_ONNX
::    to cmake so that onnx (and all its transitive dependencies: protobuf,
::    abseil, utf8_range, …) is built inline inside the example's cmake build.
::    No separate protobuf install is needed.
::
:: Usage (run from the repository root or from this directory):
::   examples\load_onnx_time\build.bat [install-prefix] [lib-build-dir] [example-build-dir]
::
:: Arguments:
::   install-prefix    unused (kept for backward compatibility with examples\build.bat)
::   lib-build-dir     directory for library source trees
::                     (default: build\load-onnx-time-lib)
::   example-build-dir load_onnx_time build directory
::                     (default: build\load-onnx-time-example)
::
:: Environment variables:
::   ONNX_GIT_TAG         git tag/branch for onnx (e.g. v1.17.0).  When unset
::                        the script checks whether Python onnx is importable;
::                        if absent it switches to a from-source build using
::                        ONNX_DEFAULT_GIT_TAG.
::   ONNX_GIT_URL         onnx git URL
::   ONNX_DEFAULT_GIT_TAG fallback onnx tag (default: v1.17.0)
::   CMAKE_BUILD_TYPE     build type (default: Release)

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_DIR=%%~fI"
for %%I in ("%SCRIPT_DIR%\..\..") do set "REPO_ROOT=%%~fI"

:: install-prefix is accepted for backward compat but not used in from-source mode
if "%~2"=="" (
    set "LIB_BUILD_DIR=%REPO_ROOT%\build\load-onnx-time-lib"
) else (
    set "LIB_BUILD_DIR=%~f2"
)

if "%~3"=="" (
    set "EXAMPLE_BUILD_DIR=%REPO_ROOT%\build\load-onnx-time-example"
) else (
    set "EXAMPLE_BUILD_DIR=%~f3"
)

if "%CMAKE_BUILD_TYPE%"=="" (
    set "BUILD_TYPE=Release"
) else (
    set "BUILD_TYPE=%CMAKE_BUILD_TYPE%"
)

if "%ONNX_GIT_URL%"=="" set "ONNX_GIT_URL=https://github.com/onnx/onnx.git"
if "%ONNX_DEFAULT_GIT_TAG%"=="" set "ONNX_DEFAULT_GIT_TAG=v1.17.0"

:: ---- auto-detect: switch to from-source if onnx is not in site-packages ---
if "%ONNX_GIT_TAG%"=="" (
    :: Try python3 first, then python.
    set "_onnx_found=0"
    python3 -c "import onnx" > nul 2>&1
    if not errorlevel 1 set "_onnx_found=1"
    if "!_onnx_found!"=="0" (
        python -c "import onnx" > nul 2>&1
        if not errorlevel 1 set "_onnx_found=1"
    )
    if "!_onnx_found!"=="0" (
        :: onnx not importable from Python site-packages; switch to from-source.
        set "ONNX_GIT_TAG=%ONNX_DEFAULT_GIT_TAG%"
        echo onnx not found in Python site-packages; will build onnx from source (!ONNX_GIT_TAG!).
    )
)

set "FETCHCONTENT_ARG="
set "STEP=1"

:: ---- optionally pre-clone onnx for FetchContent ----------------------------
if not "%ONNX_GIT_TAG%"=="" (
    set "ONNX_SRC_DIR=%LIB_BUILD_DIR%\onnx-src"

    echo === Step !STEP!: clone onnx %ONNX_GIT_TAG% ===
    set /a STEP+=1
    if not exist "!ONNX_SRC_DIR!\.git" (
        git clone --depth 1 --branch "%ONNX_GIT_TAG%" "%ONNX_GIT_URL%" "!ONNX_SRC_DIR!"
        if errorlevel 1 exit /b 1
        git -C "!ONNX_SRC_DIR!" submodule update --init --recursive
        if errorlevel 1 exit /b 1
    ) else (
        echo Source directory !ONNX_SRC_DIR! already exists, skipping clone.
    )

    :: Tell cmake FetchContent to use the local source instead of downloading.
    set "FETCHCONTENT_ARG=-DFETCHCONTENT_SOURCE_DIR_ONNX=!ONNX_SRC_DIR! -DONNX_GIT_TAG=%ONNX_GIT_TAG% -DONNX_GIT_URL=%ONNX_GIT_URL%"
)

:: ---- build the example -----------------------------------------------------
:: cmake builds onnx (and all its transitive deps: protobuf, abseil, utf8_range)
:: inline via FetchContent – no manual install step or separate cmake project.
echo === Step !STEP!: configure and build load_onnx_time (%BUILD_TYPE%) ===
cmake -S "%SCRIPT_DIR%" -B "%EXAMPLE_BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    %FETCHCONTENT_ARG%
if errorlevel 1 exit /b 1

cmake --build "%EXAMPLE_BUILD_DIR%" --config %BUILD_TYPE% --parallel
if errorlevel 1 exit /b 1

echo.
echo Example binary:
echo   %EXAMPLE_BUILD_DIR%\%BUILD_TYPE%\load_onnx_time.exe

endlocal
