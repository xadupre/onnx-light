"""Runs the Python unit tests with the standard library test runner."""

import argparse
import sys
import unittest
from pathlib import Path


def main(args=None):
    """Runs recursively discovered unit tests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=[Path(__file__).parent])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--durations", type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    parsed = parser.parse_args(args)

    root = Path(__file__).resolve().parent.parent
    paths = [(root / path).resolve() for path in parsed.paths]
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        parser.error(f"test directories do not exist: {', '.join(missing)}")
    excluded = [(root / path).resolve() for path in parsed.exclude]
    test_files = {
        test_file.resolve()
        for path in paths
        for test_file in path.rglob("test_*.py")
        if not any(test_file.resolve().is_relative_to(directory) for directory in excluded)
    }
    test_names = [
        ".".join(test_file.relative_to(root).with_suffix("").parts)
        for test_file in sorted(test_files)
    ]
    if not test_names:
        parser.error("no tests found")
    suite = unittest.defaultTestLoader.loadTestsFromNames(test_names)
    runner = unittest.TextTestRunner(
        verbosity=2 if parsed.verbose else 1, durations=parsed.durations
    )
    return 0 if runner.run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
