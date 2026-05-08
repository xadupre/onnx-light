@echo off
:: profile.bat -- build bench_parse_serialize with RelWithDebInfo and run a
:: gprof or plain-run profile on Windows.
::
:: Usage (run from the repository root):
::   benchmarks\profile.bat [gprof|run]  [extra bench flags]
::
:: Examples:
::   benchmarks\profile.bat gprof  -n 20 -t 1
::   benchmarks\profile.bat run    -n 20 -t 1
::
:: The default tool is gprof (requires MinGW/MSYS2 g++ with -pg support).
:: Use "run" to build and run the benchmark without any profiling tool.
::
:: Note: perf and valgrind are Linux-only and are not supported on Windows.
::
:: Requirements:
::   cmake (on PATH), MinGW g++ for gprof, or any CMake-compatible compiler
::   for the "run" option.

setlocal EnableDelayedExpansion

:: ---------------------------------------------------------------------------
:: Parse arguments
:: ---------------------------------------------------------------------------
set "TOOL=%~1"
if "%TOOL%"=="" set "TOOL=gprof"

:: Shift extra bench flags into BENCH_ARGS
set "BENCH_ARGS="
:parse_loop
shift
if "%~1"=="" goto :parse_done
set "BENCH_ARGS=%BENCH_ARGS% %~1"
goto :parse_loop
:parse_done

:: ---------------------------------------------------------------------------
:: Locate repository root (two levels up from this script)
:: ---------------------------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
:: SCRIPT_DIR ends with a backslash, trim it
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_DIR=%%~fI"
for %%I in ("%SCRIPT_DIR%\..") do set "REPO_ROOT=%%~fI"

set "BUILD_BASE=%REPO_ROOT%\build"

:: ---------------------------------------------------------------------------
:: Select build directory and gprof flag
:: ---------------------------------------------------------------------------
if /I "%TOOL%"=="gprof" (
    set "BUILD_DIR=%BUILD_BASE%\bench_gprof"
    set "GPROF_FLAG=-DONNX_LIGHT_BENCH_GPROF=ON"
) else if /I "%TOOL%"=="run" (
    set "BUILD_DIR=%BUILD_BASE%\bench_rdi"
    set "GPROF_FLAG="
) else (
    echo Unknown tool "%TOOL%". Choose: gprof ^| run>&2
    echo Note: perf and valgrind are Linux-only and are not available on Windows.>&2
    exit /b 1
)

set "BENCH=%BUILD_DIR%\bench_parse_serialize.exe"

:: ---------------------------------------------------------------------------
:: Step 1 -- build
:: ---------------------------------------------------------------------------
echo === Step 1: cmake configure (%TOOL%) ===
cmake -S "%REPO_ROOT%" -B "%BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
    -DONNX_LIGHT_BUILD_BENCHMARKS=ON ^
    -DONNX_LIGHT_BUILD_PYTHON=OFF ^
    %GPROF_FLAG%
if errorlevel 1 exit /b 1

echo === Step 1: cmake build ===
cmake --build "%BUILD_DIR%" --target bench_parse_serialize --config RelWithDebInfo
if errorlevel 1 exit /b 1

echo.
echo Binary: %BENCH%
echo.

:: ---------------------------------------------------------------------------
:: Step 2 -- profile / run
:: ---------------------------------------------------------------------------
if /I "%TOOL%"=="gprof" (
    echo === Step 2: run with gprof instrumentation ===
    pushd "%BUILD_DIR%"
    "%BENCH%" %BENCH_ARGS%
    if errorlevel 1 ( popd & exit /b 1 )
    popd

    echo.
    echo === Step 3: gprof report (top 40 lines) ===
    gprof -b "%BENCH%" "%BUILD_DIR%\gmon.out" > "%BUILD_DIR%\gprof_report.txt"
    if errorlevel 1 (
        echo gprof failed. Make sure you are using a MinGW g++ build with -pg support.>&2
        exit /b 1
    )
    :: Print the first 40 lines of the report
    set "LINE=0"
    for /f "usebackq delims=" %%L in ("%BUILD_DIR%\gprof_report.txt") do (
        set /a LINE+=1
        if !LINE! leq 40 echo %%L
    )
    echo.
    echo Full report saved to: %BUILD_DIR%\gprof_report.txt
) else (
    echo === Step 2: run (no profiling tool) ===
    "%BENCH%" %BENCH_ARGS%
    if errorlevel 1 exit /b 1
)

endlocal
