import os
import sys
from pathlib import Path

from setuptools import Command, Distribution, setup


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
