#!/usr/bin/env python3
"""Reports the installed size of the lib_onnx_proto shared library."""

import argparse
import pathlib
import re
import shutil
import subprocess
import tempfile
import zipfile

PROTO_LIBRARY_NAMES = {"lib_onnx_proto.dll", "liblib_onnx_proto.dylib", "liblib_onnx_proto.so"}


def _format_bytes(value: int | None) -> str:
    """Formats a byte count for the report."""
    if value is None:
        return "n/a"
    return f"{value:,}"


def _section_sizes(path: pathlib.Path) -> dict[str, int]:
    """Returns section sizes reported by the platform size utility."""
    size_tool = shutil.which("size")
    if size_tool is None:
        return {}
    result = subprocess.run(
        [size_tool, "-A", "-d", str(path)], check=True, capture_output=True, text=True
    )
    sections = {}
    for line in result.stdout.splitlines():
        columns = line.split()
        if len(columns) >= 2 and columns[0] != "Total" and columns[1].isdigit():
            sections[columns[0]] = int(columns[1])
    return sections


def _elf_metadata(path: pathlib.Path) -> tuple[int | None, str]:
    """Returns the defined dynamic-symbol count and ELF dependencies."""
    if path.read_bytes()[:4] != b"\x7fELF":
        return None, "n/a"
    readelf = shutil.which("readelf")
    if readelf is None:
        return None, "readelf unavailable"

    symbols = subprocess.run(
        [readelf, "--dyn-syms", "-W", str(path)], check=True, capture_output=True, text=True
    )
    defined_symbols = 0
    for line in symbols.stdout.splitlines():
        columns = line.split()
        if len(columns) >= 7 and re.fullmatch(r"\d+:", columns[0]) and columns[6] != "UND":
            defined_symbols += 1

    dynamic = subprocess.run(
        [readelf, "-d", "-W", str(path)], check=True, capture_output=True, text=True
    )
    dependencies = re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic.stdout)
    return defined_symbols, ", ".join(dependencies) or "none"


def _measure(path: pathlib.Path, source: str, compressed_size: int | None) -> dict[str, object]:
    """Measures one installed proto library."""
    sections = _section_sizes(path)
    dynamic_symbols, dependencies = _elf_metadata(path)
    allocated_size = sum(sections.values()) if sections else None
    return {
        "source": source,
        "installed_size": path.stat().st_size,
        "compressed_size": compressed_size,
        "allocated_size": allocated_size,
        "text_size": sections.get(".text"),
        "dynamic_symbols": dynamic_symbols,
        "dependencies": dependencies,
    }


def _measure_wheel(path: pathlib.Path) -> list[dict[str, object]]:
    """Measures proto libraries contained in one wheel."""
    measurements = []
    with zipfile.ZipFile(path) as wheel, tempfile.TemporaryDirectory() as temporary_directory:
        for member in wheel.infolist():
            if pathlib.PurePosixPath(member.filename).name not in PROTO_LIBRARY_NAMES:
                continue
            extracted = pathlib.Path(wheel.extract(member, temporary_directory))
            measurements.append(
                _measure(extracted, f"{path.name}:{member.filename}", member.compress_size)
            )
    return measurements


def _measure_target(path: pathlib.Path) -> list[dict[str, object]]:
    """Measures a library, wheel, or directory containing proto libraries."""
    if path.is_dir():
        return [
            _measure(candidate, str(candidate), None)
            for candidate in sorted(path.rglob("*"))
            if candidate.name in PROTO_LIBRARY_NAMES
        ]
    if path.suffix == ".whl":
        return _measure_wheel(path)
    if path.name in PROTO_LIBRARY_NAMES:
        return [_measure(path, str(path), None)]
    raise ValueError(f"unsupported input: {path}")


def _render(measurements: list[dict[str, object]]) -> str:
    """Renders measurements as a Markdown CI report."""
    lines = [
        "## `lib_onnx_proto` binary size",
        "",
        (
            "| Artifact | Installed bytes | Wheel-compressed bytes | Section bytes "
            "| `.text` bytes | Defined dynamic symbols |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for measurement in measurements:
        lines.append(
            (
                "| `{source}` | {installed} | {compressed} | {allocated} "
                "| {text} | {symbols} |"
            ).format(
                source=measurement["source"],
                installed=_format_bytes(measurement["installed_size"]),
                compressed=_format_bytes(measurement["compressed_size"]),
                allocated=_format_bytes(measurement["allocated_size"]),
                text=_format_bytes(measurement["text_size"]),
                symbols=_format_bytes(measurement["dynamic_symbols"]),
            )
        )
    lines.extend(["", "**Required shared libraries**"])
    for measurement in measurements:
        lines.append(f"- `{measurement['source']}`: {measurement['dependencies']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Runs the binary-size reporter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="+", type=pathlib.Path)
    parser.add_argument("--summary", type=pathlib.Path)
    arguments = parser.parse_args()

    measurements = [
        measurement for target in arguments.targets for measurement in _measure_target(target)
    ]
    if not measurements:
        raise RuntimeError("no lib_onnx_proto shared library found")

    report = _render(measurements)
    print(report, end="")
    if arguments.summary is not None:
        with arguments.summary.open("a", encoding="utf-8") as summary:
            summary.write(report)


if __name__ == "__main__":
    main()
