"""Tests for explicit Sphinx-Gallery thumbnails."""

import pathlib
import re
import struct

_ROOT = pathlib.Path(__file__).parents[2]
_DOCS = _ROOT / "docs"
_THUMBNAIL_RE = re.compile(r'^# sphinx_gallery_thumbnail_path = "([^"]+)"$', re.MULTILINE)
_THUMBNAIL_BY_EXAMPLE = {
    "compute/plot_compute_information.py": "compute_information.png",
    "compute/plot_evaluate_shapes.py": "evaluate_shapes.png",
    "compute/plot_shape_inference.py": "shape_inference.png",
    "compute/plot_shape_inference_custom_op.py": "shape_inference_custom_op.png",
    "patterns/plot_pattern_optimization.py": "pattern_optimization.png",
    "patterns/plot_pattern_replay.py": "pattern_replay.png",
    "patterns/plot_pattern_replay_cleanup.py": "pattern_replay_cleanup.png",
    "proto/plot_load_save_external.py": "load_save_external.png",
    "proto/plot_node_callback.py": "node_callback.png",
    "proto/plot_pretty_onnx.py": "pretty_onnx.png",
    "proto/plot_raw_data_callback.py": "raw_data_callback.png",
    "proto/plot_translate.py": "translate.png",
    "runtime/plot_custom_kernel.py": "custom_kernel.png",
    "runtime/plot_register_custom_kernel.py": "register_custom_kernel.png",
    "tuning/plot_kernel_tuning.py": "kernel_tuning.png",
    "tuning/plot_parallel_for_profiling.py": "parallel_for_profiling.png",
}


def test_explicit_gallery_thumbnail_selection():
    """Checks that non-plotting examples select a unique thumbnail."""
    expected = {
        example: f"_static/gallery_thumbnails/{thumbnail}"
        for example, thumbnail in _THUMBNAIL_BY_EXAMPLE.items()
    }
    selected = {}
    for example_path in (_DOCS / "examples").glob("*/plot_*.py"):
        matches = _THUMBNAIL_RE.findall(example_path.read_text(encoding="utf-8"))
        assert len(matches) <= 1, example_path
        if matches:
            selected[example_path.relative_to(_DOCS / "examples").as_posix()] = matches[0]

    assert selected == expected
    assert len(set(selected.values())) == len(selected)


def test_gallery_thumbnail_images():
    """Checks that every selected thumbnail is a 640 by 480 PNG."""
    for thumbnail in _THUMBNAIL_BY_EXAMPLE.values():
        path = _DOCS / "_static" / "gallery_thumbnails" / thumbnail
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n", path
        width, height = struct.unpack(">II", data[16:24])
        assert (width, height) == (640, 480), path
