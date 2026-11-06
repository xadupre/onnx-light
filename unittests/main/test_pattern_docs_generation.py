"""Tests for the pattern documentation catalogue."""

import pathlib

from onnx_light.doc import render_rst_pattern_catalog, render_rst_peak_memory_catalog
from onnx_light.onnx_core.optimization import TreeEnsemblePattern, standard_patterns
from onnx_light.onnx_core.shape_inference import peak_memory_dispatch_table_keys


def test_render_rst_pattern_catalog_lists_registered_patterns():
    """Checks that the catalogue links both classes for every registered pattern."""
    patterns_root = (
        pathlib.Path(__file__).parents[2] / "onnx_light" / "onnx_extensions" / "patterns"
    )
    catalogue = render_rst_pattern_catalog(patterns_root)
    patterns = standard_patterns()

    assert catalogue.count("    * - ") == len(patterns) + 1
    positions = [
        catalogue.index(f":cpp:class:`{name} ") for name in sorted(p.name for p in patterns)
    ]
    assert positions == sorted(positions)
    for pattern in patterns:
        class_name = type(pattern).__name__
        assert (
            f":cpp:class:`{pattern.name} <onnx_light::onnx_patterns::{class_name}>`" in catalogue
        )
        python_link = f":class:`Python <onnx_light.onnx_core.optimization.{class_name}>` — "
        assert python_link in catalogue
    assert "    * - #" in catalogue
    assert ":class: sphinx-datatable pattern-catalog" in catalogue
    assert ":class:`Python" in catalogue
    assert ":cpp:class:`C++" not in catalogue
    assert "Before:" not in catalogue
    assert "After:" not in catalogue


def test_tree_ensemble_python_default_priority():
    """Checks that the Python constructor preserves the C++ default priority."""
    assert TreeEnsemblePattern().priority == 1


def test_render_rst_peak_memory_catalog_lists_registered_functions():
    """Checks that the catalogue contains every registered peak-memory function."""
    keys = peak_memory_dispatch_table_keys()
    catalogue = render_rst_peak_memory_catalog()

    assert catalogue.count("    * - ") == len(keys) + 1
    for key in keys:
        domain, op_type, *_ = key.split(":")
        assert f"**{op_type}**" in catalogue
        assert f"``{domain}``" in catalogue


def test_render_rst_peak_memory_catalog_formats_explicit_functions():
    """Checks catalogue formatting independently of the installed kernels."""
    catalogue = render_rst_peak_memory_catalog(["local.test:Scratch:2", "ai.onnx:Attention"])

    assert "**Attention**" in catalogue
    assert "**Scratch**" in catalogue
    assert "``ai.onnx``" in catalogue
    assert "``local.test``" in catalogue
    assert "CPU / default" in catalogue
    assert "GPU2" in catalogue
    assert "onnx_light::core::shapes::ComputePeakMemory" in catalogue
    assert "onnx_light.onnx_core.shape_inference.compute_peak_memory" in catalogue


def test_render_rst_peak_memory_catalog_accepts_empty_registry():
    """Checks that a build without estimators still produces a table header."""
    catalogue = render_rst_peak_memory_catalog([])

    assert catalogue.count("    * - ") == 1
    assert "    * - Operator" in catalogue
    assert ":class: sphinx-datatable memory-peak-catalog" in catalogue
    assert "**Attention**" not in catalogue
