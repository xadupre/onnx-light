# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Feeds every backend test case (excluding the benchmark-sized and ``_big_``
cases) into a :class:`onnx_light.onnx_core.graph_builder.GraphBuilder`, rebuilds
the ONNX model and checks that a further import / rebuild round-trip is stable.

The comparison is order-insensitive for ``opset_import`` and ``metadata_props``:
both are populated from hash maps whose iteration order is not guaranteed, so
the same logical model may serialise them in a different order. The C++
counterpart lives in
``unittests/cc/onnx_extensions/backend_test/test_backend_graph_builder_roundtrip.cc``.
"""

from __future__ import annotations

import unittest

import onnx_light.onnx as onnxl
from onnx_light.ext_test_case import ExtTestCase, import_or_skip

# The backend test registries and the builder extension are only available in
# the full build; skip this module on a reduced build
# (ONNX_LIGHT_BUILD_KERNELS=OFF).
collect_test_cases = import_or_skip("onnx_light.onnx.backend", "collect_test_cases")
GraphBuilder = import_or_skip("onnx_light.onnx_core.graph_builder", "GraphBuilder")

# Operators the GraphBuilder cannot yet re-import faithfully: their real
# signature carries optional / variadic outputs (or control-flow / function
# bodies) that the built-in schema output-count validation or the incremental
# shape inference does not model, so importing such a model raises. Cases
# exercising these operators are skipped; the vast majority of cases (~1900)
# still exercise the round-trip.
UNSUPPORTED_OPS = frozenset(
    {
        "Attention",
        "SoftmaxCrossEntropyLoss",
        "MaxPool",
        "Scan",
        "Dropout",
        "LSTM",
        "Adam",
        "Momentum",
        "Loop",
        "BatchNormalization",
        "SequenceMap",
        "Adagrad",
        "If",
        "Optional",
        "OptionalGetElement",
    }
)


def _iter_op_types(graph):
    """Yields the op_type of every node in ``graph`` and its nested subgraphs."""
    for node in graph.node:
        yield node.op_type
        for attr in node.attribute:
            if len(attr.g.node):
                yield from _iter_op_types(attr.g)
            for subgraph in attr.graphs:
                yield from _iter_op_types(subgraph)


def _should_skip(model):
    """Returns whether ``model`` uses an unsupported construct (see UNSUPPORTED_OPS)."""
    if len(model.functions):
        return True
    return any(op in UNSUPPORTED_OPS for op in _iter_op_types(model.graph))


def _sort_metadata(props):
    """Canonicalises the ``metadata_props`` ordering of a proto in place."""
    items = sorted((entry.key, entry.value) for entry in props)
    del props[:]
    for key, value in items:
        entry = props.add()
        entry.key = key
        entry.value = value


def _normalize_graph(graph):
    """Recursively canonicalises the ``metadata_props`` ordering of a graph."""
    _sort_metadata(graph.metadata_props)
    for node in graph.node:
        _sort_metadata(node.metadata_props)
        for attr in node.attribute:
            if len(attr.g.node):
                _normalize_graph(attr.g)
            for subgraph in attr.graphs:
                _normalize_graph(subgraph)
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        _sort_metadata(value.metadata_props)
    for initializer in graph.initializer:
        _sort_metadata(initializer.metadata_props)


def _normalize(model):
    """Returns a copy of ``model`` with a canonical opset/metadata ordering."""
    copy = onnxl.ModelProto()
    copy.CopyFrom(model)
    opsets = sorted((opset.domain, opset.version) for opset in copy.opset_import)
    del copy.opset_import[:]
    for domain, version in opsets:
        entry = copy.opset_import.add()
        entry.domain = domain
        entry.version = version
    _sort_metadata(copy.metadata_props)
    _normalize_graph(copy.graph)
    return copy


class TestGraphBuilderBackendRoundTrip(ExtTestCase):
    def test_all_collected_cases_are_stable(self):
        # ``collect_test_cases()`` returns the standard correctness cases
        # (``TestMode.TEST``, so no benchmark-sized cases) and excludes the
        # ``_big_`` cases by default: exactly "all backend tests except
        # benchmark and big".
        cases = collect_test_cases()
        self.assertNotEmpty(cases)

        tested = 0
        for tc in cases:
            model = tc.model
            if _should_skip(model):
                continue
            with self.subTest(name=tc.name):
                rebuilt = GraphBuilder(model).to_onnx("model")
                rebuilt_again = GraphBuilder(rebuilt).to_onnx("model")
                self.assertEqual(_normalize(rebuilt), _normalize(rebuilt_again))
            tested += 1
        # Guards against the collector silently returning nothing testable.
        self.assertGreater(tested, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
