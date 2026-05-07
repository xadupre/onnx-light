import os
import shlex
import subprocess
import sys
from pathlib import Path


def _spawn(command, dry_run):
    """Prints and executes a command unless dry-run mode is enabled."""
    print(" ".join(shlex.quote(c) for c in command))
    if not dry_run:
        subprocess.run(command, check=True)


def _run_build_ext_without_packaging(args):
    """Executes build_ext without setuptools or distutils support."""
    if not args or args[0] != "build_ext":
        return False

    inplace = False
    dry_run = False
    build_temp = "build/temp"
    build_lib = "build/lib"

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
        else:
            raise ValueError(
                f"Unsupported argument for build_ext: {arg!r}. "
                "Supported arguments are: --inplace, --dry-run, --build-temp, --build-lib."
            )
        i += 1

    print("running build_ext")
    root = Path(__file__).resolve().parent
    build_temp_path = Path(build_temp).resolve()
    build_temp_path.mkdir(parents=True, exist_ok=True)
    install_prefix = root if inplace else Path(build_lib).resolve()

    _spawn(
        [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(build_temp_path),
            f"-DPython_EXECUTABLE={sys.executable}",
        ],
        dry_run,
    )
    _spawn(["cmake", "--build", str(build_temp_path), "--config", "Release"], dry_run)
    _spawn(["cmake", "--install", str(build_temp_path), "--prefix", str(install_prefix)], dry_run)
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
            ]
        )
        self.spawn(["cmake", "--build", str(build_temp), "--config", "Release"])
        self.spawn(["cmake", "--install", str(build_temp), "--prefix", str(install_prefix)])


setup(
    name="onnx-light",
    version="0.1.0",
    packages=["onnx_light"],
    distclass=NoConfigDistribution,
    cmdclass={"build_ext": BuildExt},
)
