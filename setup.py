import os
import shlex
import subprocess
import sys
from pathlib import Path


def _spawn(command, dry_run):
    """Prints and executes a command unless dry-run mode is enabled."""
    print(" ".join(shlex.quote(cmd_part) for cmd_part in command))
    if not dry_run:
        subprocess.run(command, check=True)


def _cmake_args_from_env():
    """Retrieves additional CMake configure arguments from ``CMAKE_ARGS``.

    Returns:
        A list of parsed CMake arguments, or an empty list if the variable is not set.
    """
    cmake_args = os.environ.get("CMAKE_ARGS")
    if not cmake_args:
        return []
    return shlex.split(cmake_args)


def _run_build_ext_without_packaging(args):
    """Executes build_ext or build_benchmarks without setuptools or distutils support."""
    if not args or args[0] not in {"build_ext", "build_benchmarks"}:
        return False

    command = args[0]
    inplace = False
    dry_run = False
    build_temp = "build/temp"
    build_lib = "build/lib"
    gprof = False

    i = 1
    while i < len(args):
        arg = args[i]
        if arg in {"--inplace", "-i"}:
            inplace = True
        elif arg in {"--dry-run", "-n"}:
            dry_run = True
        elif arg.startswith("--build-temp="):
            build_temp = arg.split("=", 1)[1]
        elif arg.startswith("--build-lib="):
            build_lib = arg.split("=", 1)[1]
        elif arg == "--build-temp" and i + 1 < len(args):
            build_temp = args[i + 1]
            i += 1
        elif arg == "--build-lib" and i + 1 < len(args):
            build_lib = args[i + 1]
            i += 1
        elif arg == "--gprof":
            gprof = True
        else:
            raise ValueError(
                f"Unsupported argument for {command}: {arg!r}. "
                "Supported arguments include: --inplace, --dry-run, "
                "--build-temp, --build-lib, --gprof."
            )
        i += 1

    root = Path(__file__).resolve().parent
    build_temp_path = Path(build_temp).resolve()
    build_temp_path.mkdir(parents=True, exist_ok=True)

    if command == "build_benchmarks":
        print("running build_benchmarks")
        cmake_args = [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(build_temp_path),
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DONNX_LIGHT_BUILD_BENCHMARKS=ON",
            "-DONNX_LIGHT_BUILD_PYTHON=OFF",
        ]
        if gprof:
            cmake_args.append("-DONNX_LIGHT_BENCH_GPROF=ON")
        _spawn(cmake_args, dry_run)
        _spawn(
            [
                "cmake",
                "--build",
                str(build_temp_path),
                "--target",
                "bench_parse_serialize",
                "-j4",
            ],
            dry_run,
        )
        bench_bin = build_temp_path / "bench_parse_serialize"
        if not dry_run and bench_bin.exists():
            print(f"\nBenchmark binary: {bench_bin}")
            print(f"Usage: {bench_bin} -n 20 -t 1\n")
    else:
        print("running build_ext")
        install_prefix = root if inplace else Path(build_lib).resolve()
        _spawn(
            [
                "cmake",
                "-S",
                str(root),
                "-B",
                str(build_temp_path),
                f"-DPython_EXECUTABLE={sys.executable}",
                *_cmake_args_from_env(),
            ],
            dry_run,
        )
        _spawn(["cmake", "--build", str(build_temp_path), "--config", "Release"], dry_run)
        _spawn(
            ["cmake", "--install", str(build_temp_path), "--prefix", str(install_prefix)], dry_run
        )
    return True


try:
    from setuptools import Command, Distribution, setup
except ModuleNotFoundError:
    try:
        from distutils.cmd import Command
        from distutils.core import Distribution, setup
    except ModuleNotFoundError:
        if _run_build_ext_without_packaging(sys.argv[1:]):
            raise SystemExit(0) from None
        raise


class NoConfigDistribution(Distribution):
    """Skips setup.cfg and pyproject.toml parsing for setup.py commands."""

    def parse_config_files(self, _filenames=None):
        """Skips setuptools configuration file parsing."""
        return None


class BuildExt(Command):
    """Builds the extension with CMake."""

    description = "builds C++ extension with CMake"
    user_options = [
        ("inplace", "i", "build extension in the source tree"),
        ("build-temp=", "t", "temporary build directory"),
        ("build-lib=", "b", "build directory for platform-specific files"),
    ]
    boolean_options = ["inplace"]

    def initialize_options(self):
        """Initializes default values for command options."""
        self.inplace = False
        self.build_temp = None
        self.build_lib = None

    def finalize_options(self):
        """Finalizes build directory paths for unspecified options."""
        build_base = "build"
        if self.build_temp is None:
            self.build_temp = os.path.join(build_base, "temp")
        if self.build_lib is None:
            self.build_lib = os.path.join(build_base, "lib")

    def run(self):
        """Runs CMake configure, build, and install commands."""
        root = Path(__file__).resolve().parent
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        install_prefix = root if self.inplace else Path(self.build_lib).resolve()

        self.spawn(
            [
                "cmake",
                "-S",
                str(root),
                "-B",
                str(build_temp),
                f"-DPython_EXECUTABLE={sys.executable}",
                *_cmake_args_from_env(),
            ]
        )
        self.spawn(["cmake", "--build", str(build_temp), "--config", "Release"])
        self.spawn(["cmake", "--install", str(build_temp), "--prefix", str(install_prefix)])


class BuildBenchmarks(Command):
    """Builds the C++ benchmark executables with RelWithDebInfo and debug symbols."""

    description = "builds C++ benchmark executables (RelWithDebInfo + DWARF debug info)"
    user_options = [
        ("build-temp=", "t", "temporary build directory"),
        ("gprof", None, "add -pg instrumentation for gprof profiling"),
    ]
    boolean_options = ["gprof"]

    def initialize_options(self):
        """Initializes default values for command options."""
        self.build_temp = None
        self.gprof = False

    def finalize_options(self):
        """Sets the build directory to 'build/benchmarks' when not explicitly specified."""
        if self.build_temp is None:
            self.build_temp = os.path.join("build", "benchmarks")

    def run(self):
        """Configures CMake with RelWithDebInfo and ONNX_LIGHT_BUILD_BENCHMARKS=ON.

        Then builds the bench_parse_serialize target.
        """
        root = Path(__file__).resolve().parent
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(build_temp),
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DONNX_LIGHT_BUILD_BENCHMARKS=ON",
            "-DONNX_LIGHT_BUILD_PYTHON=OFF",
        ]
        if self.gprof:
            cmake_args.append("-DONNX_LIGHT_BENCH_GPROF=ON")

        self.spawn(cmake_args)
        self.spawn(
            ["cmake", "--build", str(build_temp), "--target", "bench_parse_serialize", "-j4"]
        )

        bench_bin = build_temp / "bench_parse_serialize"
        if bench_bin.exists():
            print(f"\nBenchmark binary: {bench_bin}")
            print(f"Usage: {bench_bin} -n 20 -t 1\n")


setup(
    name="onnx-light",
    version="0.1.0",
    packages=["onnx_light"],
    distclass=NoConfigDistribution,
    cmdclass={"build_ext": BuildExt, "build_benchmarks": BuildBenchmarks},
)
