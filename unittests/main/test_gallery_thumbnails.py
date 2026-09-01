"""Tests for explicit Sphinx-Gallery thumbnails."""

import pathlib
import re
import struct

_ROOT = pathlib.Path(__file__).parents[2]
_DOCS = _ROOT / "docs"
_THUMBNAIL_RE = re.compile(r'^# sphinx_gallery_thumbnail_path = "([^"]+)"$', re.MULTILINE)
_EXAMPLES_BY_THUMBNAIL = {
    "compute.png": {
        "compute/plot_compute_information.py",
        "compute/plot_evaluate_shapes.py",
        "compute/plot_shape_inference.py",
        "compute/plot_shape_inference_custom_op.py",
    },
    "patterns.png": {
        "patterns/plot_pattern_optimization.py",
        "patterns/plot_pattern_replay.py",
        "patterns/plot_pattern_replay_cleanup.py",
    },
    "proto.png": {
        "proto/plot_load_save_external.py",
        "proto/plot_node_callback.py",
        "proto/plot_pretty_onnx.py",
        "proto/plot_raw_data_callback.py",
        "proto/plot_translate.py",
    },
    "runtime.png": {"runtime/plot_custom_kernel.py", "runtime/plot_register_custom_kernel.py"},
    "tuning.png": {"tuning/plot_kernel_tuning.py", "tuning/plot_parallel_for_profiling.py"},
}


def test_explicit_gallery_thumbnail_selection():
    """Checks that non-plotting examples select their category thumbnail."""
    expected = {
        example: f"_static/gallery_thumbnails/{thumbnail}"
        for thumbnail, examples in _EXAMPLES_BY_THUMBNAIL.items()
        for example in examples
    }
    selected = {}
    for example_path in (_DOCS / "examples").glob("*/plot_*.py"):
        matches = _THUMBNAIL_RE.findall(example_path.read_text(encoding="utf-8"))
        assert len(matches) <= 1, example_path
        if matches:
            selected[str(example_path.relative_to(_DOCS / "examples"))] = matches[0]

    assert selected == expected


def test_gallery_thumbnail_images():
    """Checks that every selected thumbnail is a 640 by 480 PNG."""
    for thumbnail in _EXAMPLES_BY_THUMBNAIL:
        path = _DOCS / "_static" / "gallery_thumbnails" / thumbnail
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n", path
        width, height = struct.unpack(">II", data[16:24])
        assert (width, height) == (640, 480), path
