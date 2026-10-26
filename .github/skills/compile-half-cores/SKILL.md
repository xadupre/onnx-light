---
name: compile-half-cores
description: Builds native code with half the available processors. Use when compiling C++ code with CMake, setup.py, or another parallel build tool.
---

# Build with half the available processors

Use half the processors available to the current process for local builds,
rounded down with a minimum of one worker. An explicit worker count requested
by the user takes precedence.

On Linux, calculate the worker count with `nproc`, not `nproc --all`:

```bash
available=$(nproc)
jobs=$((available / 2))
if [ "$jobs" -lt 1 ]; then
  jobs=1
fi
```

Pass that count explicitly to the existing build command:

```bash
cmake --build build/temp --parallel "$jobs"
python setup.py build_ext --inplace --parallel "$jobs"
```

These are alternatives; run only the command needed for the task. Reuse the
existing build tree and preserve its configuration. For other build tools, use
their equivalent worker-count option or set `CMAKE_BUILD_PARALLEL_LEVEL="$jobs"`
when appropriate.

Do not use an unbounded `--parallel`, all processors, or a fixed default such as
six workers. For example, 96 available processors means 48 build workers.
This local-build policy does not request changes to CI worker counts.
