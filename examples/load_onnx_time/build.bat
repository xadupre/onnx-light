@echo off
:: build.bat -- builds the standalone load_onnx_time example against the
:: standard onnx C++ library.
::
:: By default the script expects the system onnx C++ library to be findable
:: by CMake.  On Windows you can install onnx using vcpkg:
::   vcpkg install onnx
:: and then pass the vcpkg toolchain file via CMAKE_TOOLCHAIN_FILE or set
:: CMAKE_PREFIX_PATH to the vcpkg install directory.
::
:: Alternatively, set the ONNX_GIT_TAG environment variable to a git tag or
:: branch (e.g. v1.17.0) to clone and build onnx from source.
:: Protobuf must already be installed and findable by CMake.
::
:: Usage (run from the repository root or from this directory):
::   examples\load_onnx_time\build.bat [install-prefix] [lib-build-dir] [example-build-dir]
::
:: Arguments:
::   install-prefix    onnx install prefix when ONNX_GIT_TAG is set
::                     (default: build\install-load-onnx-time)
::   lib-build-dir     onnx build directory when ONNX_GIT_TAG is set
::                     (default: build\load-onnx-time-lib)
::   example-build-dir load_onnx_time build directory
::                     (default: build\load-onnx-time-example)

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_DIR=%%~fI"
for %%I in ("%SCRIPT_DIR%\..\..") do set "REPO_ROOT=%%~fI"

if "%~1"=="" (
    set "INSTALL_PREFIX=%REPO_ROOT%\build\install-load-onnx-time"
) else (
    set "INSTALL_PREFIX=%~f1"
)

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

set "CMAKE_PREFIX_PATH_ARG="
if not "%ONNX_GIT_TAG%"=="" (
    set "ONNX_GIT_URL_VAL=%ONNX_GIT_URL%"
    if "!ONNX_GIT_URL_VAL!"=="" set "ONNX_GIT_URL_VAL=https://github.com/onnx/onnx.git"

    set "ONNX_SRC_DIR=%LIB_BUILD_DIR%\onnx-src"
    set "ONNX_BUILD_DIR_LOC=%LIB_BUILD_DIR%\onnx-build"

    echo === Step 1: clone onnx %ONNX_GIT_TAG% ===
    if not exist "!ONNX_SRC_DIR!\.git" (
        git clone --depth 1 --branch "%ONNX_GIT_TAG%" "!ONNX_GIT_URL_VAL!" "!ONNX_SRC_DIR!"
        if errorlevel 1 exit /b 1
        git -C "!ONNX_SRC_DIR!" submodule update --init --recursive
        if errorlevel 1 exit /b 1
    ) else (
        echo Source directory !ONNX_SRC_DIR! already exists, skipping clone.
    )

    echo === Step 2: build and install onnx ===
    cmake -S "!ONNX_SRC_DIR!" -B "!ONNX_BUILD_DIR_LOC!" ^
        -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
        -DONNX_ML=ON ^
        -DONNX_BUILD_TESTS=OFF ^
        -DONNX_BUILD_BENCHMARKS=OFF ^
        -DCMAKE_INSTALL_PREFIX="%INSTALL_PREFIX%"
    if errorlevel 1 exit /b 1
    cmake --build "!ONNX_BUILD_DIR_LOC!" --config %BUILD_TYPE% --parallel
    if errorlevel 1 exit /b 1
    cmake --install "!ONNX_BUILD_DIR_LOC!" --config %BUILD_TYPE%
    if errorlevel 1 exit /b 1

    set "CMAKE_PREFIX_PATH_ARG=-DCMAKE_PREFIX_PATH=%INSTALL_PREFIX%"
)

echo === Configure and build load_onnx_time (%BUILD_TYPE%) ===
cmake -S "%SCRIPT_DIR%" -B "%EXAMPLE_BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    %CMAKE_PREFIX_PATH_ARG%
if errorlevel 1 exit /b 1

cmake --build "%EXAMPLE_BUILD_DIR%" --config %BUILD_TYPE% --parallel
if errorlevel 1 exit /b 1

echo.
echo Example binary:
echo   %EXAMPLE_BUILD_DIR%\%BUILD_TYPE%\load_onnx_time.exe

endlocal
