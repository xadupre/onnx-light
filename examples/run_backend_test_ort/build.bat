@echo off
:: build.bat -- downloads a pre-built onnxruntime CPU release, installs
:: onnx_light locally, and builds the standalone run_backend_test_ort
:: example against both (Windows / x64).
::
:: Usage (run from the repository root or from this directory):
::   examples\run_backend_test_ort\build.bat ^
::     [install-prefix] [lib-build-dir] [example-build-dir] [ort-root]
::
:: Environment overrides:
::   ONNXRUNTIME_VERSION   onnxruntime release tag without the leading 'v'
::                         (default: 1.19.2)
::   ONNXRUNTIME_ROOT_DIR  Skip the download; use this existing extracted
::                         onnxruntime release directory instead.
::   CMAKE_BUILD_TYPE      Build type (default: Release).
::
:: Note: onnxruntime itself cannot reasonably be built from source as part of
:: this example -- the official pre-built CPU release archive is downloaded.

setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_DIR=%%~fI"
for %%I in ("%SCRIPT_DIR%\..\..") do set "REPO_ROOT=%%~fI"

if "%~1"=="" (
    set "INSTALL_PREFIX=%REPO_ROOT%\build\install-run-backend-test-ort"
) else (
    set "INSTALL_PREFIX=%~f1"
)

if "%~2"=="" (
    set "LIB_BUILD_DIR=%REPO_ROOT%\build\run-backend-test-ort-lib"
) else (
    set "LIB_BUILD_DIR=%~f2"
)

if "%~3"=="" (
    set "EXAMPLE_BUILD_DIR=%REPO_ROOT%\build\run-backend-test-ort-example"
) else (
    set "EXAMPLE_BUILD_DIR=%~f3"
)

if "%~4"=="" (
    set "ORT_ROOT_OVERRIDE=%ONNXRUNTIME_ROOT_DIR%"
) else (
    set "ORT_ROOT_OVERRIDE=%~f4"
)

if "%ONNXRUNTIME_VERSION%"=="" (
    set "ORT_VERSION=1.19.2"
) else (
    set "ORT_VERSION=%ONNXRUNTIME_VERSION%"
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

set "ORT_PLATFORM=win-x64"

if not "%ORT_ROOT_OVERRIDE%"=="" (
    set "ORT_ROOT=%ORT_ROOT_OVERRIDE%"
    echo === Step 0: using existing onnxruntime release: %ORT_ROOT% ===
) else (
    set "ORT_DOWNLOAD_DIR=%REPO_ROOT%\build\onnxruntime-downloads"
    set "ORT_ARCHIVE_NAME=onnxruntime-%ORT_PLATFORM%-%ORT_VERSION%.zip"
    set "ORT_ARCHIVE_PATH=%ORT_DOWNLOAD_DIR%\%ORT_ARCHIVE_NAME%"
    set "ORT_ROOT=%ORT_DOWNLOAD_DIR%\onnxruntime-%ORT_PLATFORM%-%ORT_VERSION%"
    set "ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v%ORT_VERSION%/%ORT_ARCHIVE_NAME%"

    if not exist "%ORT_DOWNLOAD_DIR%" mkdir "%ORT_DOWNLOAD_DIR%"

    if not exist "%ORT_ROOT%" (
        echo === Step 0: downloading onnxruntime %ORT_VERSION% for %ORT_PLATFORM% ===
        echo     URL: %ORT_URL%
        powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%ORT_URL%' -OutFile '%ORT_ARCHIVE_PATH%'"
        if errorlevel 1 exit /b 1
        powershell -NoProfile -Command "Expand-Archive -Force -LiteralPath '%ORT_ARCHIVE_PATH%' -DestinationPath '%ORT_DOWNLOAD_DIR%'"
        if errorlevel 1 exit /b 1
    ) else (
        echo === Step 0: reusing cached onnxruntime release: %ORT_ROOT% ===
    )
)

if not exist "%ORT_ROOT%\include\onnxruntime_cxx_api.h" (
    echo ERROR: "%ORT_ROOT%\include\onnxruntime_cxx_api.h" not found. 1>&2
    echo        Pass the correct release directory as the 4th positional argument or via ONNXRUNTIME_ROOT_DIR. 1>&2
    exit /b 1
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

echo === Step 2: configure and build run_backend_test_ort (%BUILD_TYPE%) ===
cmake -S "%SCRIPT_DIR%" -B "%EXAMPLE_BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_PREFIX_PATH="%INSTALL_PREFIX%" ^
    -DONNXRUNTIME_ROOT_DIR="%ORT_ROOT%"
if errorlevel 1 exit /b 1

cmake --build "%EXAMPLE_BUILD_DIR%" --config %BUILD_TYPE% --parallel %PARALLEL_JOBS%
if errorlevel 1 exit /b 1

echo.
echo Example binary:
echo   %EXAMPLE_BUILD_DIR%\%BUILD_TYPE%\run_backend_test_ort.exe
echo.
echo Make sure onnxruntime.dll is on PATH or alongside the .exe:
echo   copy "%ORT_ROOT%\lib\onnxruntime.dll" "%EXAMPLE_BUILD_DIR%\%BUILD_TYPE%\"

endlocal
