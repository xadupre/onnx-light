@echo off
:: build.bat -- builds the standalone load_onnx_time example against the
:: standard onnx C++ library.
::
:: The script has three operating modes:
::
:: 1. System install (default when onnx is found as a CMake package):
::      examples\load_onnx_time\build.bat
::    On Windows you can install onnx via vcpkg: vcpkg install onnx
::
:: 2. Explicit from-source build (set ONNX_GIT_TAG):
::      set ONNX_GIT_TAG=v1.17.0 && examples\load_onnx_time\build.bat
::    Set PROTOBUF_GIT_TAG as well if protobuf is not installed:
::      set PROTOBUF_GIT_TAG=v3.21.12 && examples\load_onnx_time\build.bat
::
:: 3. Automatic from-source build (no env vars needed):
::    When ONNX_GIT_TAG is not set, the script checks whether onnx is
::    importable from Python (site-packages).  If not found, it switches to a
::    from-source build using ONNX_DEFAULT_GIT_TAG and also builds protobuf
::    from source (PROTOBUF_DEFAULT_GIT_TAG).
::
:: Usage (run from the repository root or from this directory):
::   examples\load_onnx_time\build.bat [install-prefix] [lib-build-dir] [example-build-dir]
::
:: Arguments:
::   install-prefix    install prefix for onnx (and protobuf) when building
::                     from source (default: build\install-load-onnx-time)
::   lib-build-dir     directory for library source and build trees
::                     (default: build\load-onnx-time-lib)
::   example-build-dir load_onnx_time build directory
::                     (default: build\load-onnx-time-example)
::
:: Environment variables:
::   ONNX_GIT_TAG             git tag/branch for onnx (e.g. v1.17.0)
::   ONNX_GIT_URL             onnx git URL
::   ONNX_DEFAULT_GIT_TAG     fallback onnx tag (default: v1.17.0)
::   PROTOBUF_GIT_TAG         git tag/branch for protobuf (e.g. v3.21.12)
::   PROTOBUF_GIT_URL         protobuf git URL
::   PROTOBUF_DEFAULT_GIT_TAG fallback protobuf tag (default: v3.21.12)
::   CMAKE_BUILD_TYPE         build type (default: Release)

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

if "%ONNX_GIT_URL%"=="" set "ONNX_GIT_URL=https://github.com/onnx/onnx.git"
if "%ONNX_DEFAULT_GIT_TAG%"=="" set "ONNX_DEFAULT_GIT_TAG=v1.17.0"
if "%PROTOBUF_GIT_URL%"=="" set "PROTOBUF_GIT_URL=https://github.com/protocolbuffers/protobuf.git"
if "%PROTOBUF_DEFAULT_GIT_TAG%"=="" set "PROTOBUF_DEFAULT_GIT_TAG=v3.21.12"

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
        :: Assume protobuf is also absent; build it from source too.
        if "%PROTOBUF_GIT_TAG%"=="" (
            set "PROTOBUF_GIT_TAG=%PROTOBUF_DEFAULT_GIT_TAG%"
            echo Protobuf assumed absent; will build Protobuf from source (!PROTOBUF_GIT_TAG!).
        )
    )
)

set "CMAKE_PREFIX_PATH_ARG="
set "STEP=1"

:: ---- optionally build protobuf from source ---------------------------------
if not "%PROTOBUF_GIT_TAG%"=="" (
    set "PROTO_SRC_DIR=%LIB_BUILD_DIR%\protobuf-src"
    set "PROTO_BUILD_DIR=%LIB_BUILD_DIR%\protobuf-build"

    echo === Step !STEP!: clone protobuf %PROTOBUF_GIT_TAG% ===
    set /a STEP+=1
    if not exist "!PROTO_SRC_DIR!\.git" (
        git clone --depth 1 --branch "%PROTOBUF_GIT_TAG%" "%PROTOBUF_GIT_URL%" "!PROTO_SRC_DIR!"
        if errorlevel 1 exit /b 1
        git -C "!PROTO_SRC_DIR!" submodule update --init --recursive
        if errorlevel 1 exit /b 1
    ) else (
        echo Source directory !PROTO_SRC_DIR! already exists, skipping clone.
    )

    echo === Step !STEP!: build and install protobuf ===
    set /a STEP+=1
    cmake -S "!PROTO_SRC_DIR!" -B "!PROTO_BUILD_DIR!" ^
        -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
        -Dprotobuf_BUILD_TESTS=OFF ^
        -Dprotobuf_BUILD_SHARED_LIBS=OFF ^
        -DCMAKE_INSTALL_PREFIX="%INSTALL_PREFIX%"
    if errorlevel 1 exit /b 1
    cmake --build "!PROTO_BUILD_DIR!" --config %BUILD_TYPE% --parallel
    if errorlevel 1 exit /b 1
    cmake --install "!PROTO_BUILD_DIR!" --config %BUILD_TYPE%
    if errorlevel 1 exit /b 1

    set "CMAKE_PREFIX_PATH_ARG=-DCMAKE_PREFIX_PATH=%INSTALL_PREFIX%"
)

:: ---- optionally build onnx from source -------------------------------------
if not "%ONNX_GIT_TAG%"=="" (
    set "ONNX_SRC_DIR=%LIB_BUILD_DIR%\onnx-src"
    set "ONNX_BUILD_DIR_LOC=%LIB_BUILD_DIR%\onnx-build"

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

    echo === Step !STEP!: build and install onnx ===
    set /a STEP+=1
    cmake -S "!ONNX_SRC_DIR!" -B "!ONNX_BUILD_DIR_LOC!" ^
        -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
        -DONNX_ML=ON ^
        -DONNX_BUILD_TESTS=OFF ^
        -DONNX_BUILD_BENCHMARKS=OFF ^
        %CMAKE_PREFIX_PATH_ARG% ^
        -DCMAKE_INSTALL_PREFIX="%INSTALL_PREFIX%"
    if errorlevel 1 exit /b 1
    cmake --build "!ONNX_BUILD_DIR_LOC!" --config %BUILD_TYPE% --parallel
    if errorlevel 1 exit /b 1
    cmake --install "!ONNX_BUILD_DIR_LOC!" --config %BUILD_TYPE%
    if errorlevel 1 exit /b 1

    :: cmake --install may not copy the protobuf-generated headers (e.g.
    :: onnx-ml.pb.h) that live only in the build directory.  Copy them
    :: explicitly so that the example can compile.
    for /r "!ONNX_BUILD_DIR_LOC!" %%F in (*.pb.h) do (
        copy /y "%%F" "%INSTALL_PREFIX%\include\onnx\" > nul
    )

    set "CMAKE_PREFIX_PATH_ARG=-DCMAKE_PREFIX_PATH=%INSTALL_PREFIX%"
)

:: ---- build the example -----------------------------------------------------
echo === Step !STEP!: configure and build load_onnx_time (%BUILD_TYPE%) ===
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
