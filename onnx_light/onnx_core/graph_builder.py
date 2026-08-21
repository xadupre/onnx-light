"""Incremental ONNX graph builder backed by a C++ library.

This module exposes :class:`GraphBuilder`, an incremental builder for ONNX
graphs, models and functions. A builder starts empty, holds a compute context
and records every value name it hands out so a name can never be reused. Nodes
are added with :meth:`GraphBuilder.make_node`, which resolves the operator
opset, validates the node against the built-in ONNX operator schemas, assigns
output names when the caller leaves them empty and runs incremental shape
inference. :meth:`GraphBuilder.to_onnx` finalises the accumulated graph into a
model (default), a graph or a function, writing the inferred shapes, the
in-place / release-after metadata, the value tags and the peak-memory
estimates.

Typical usage::

    from onnx_light.onnx_core.graph_builder import GraphBuilder
    from onnx_light.onnx_proto import TensorProto

    builder = GraphBuilder("g")
    builder.make_input("x", TensorProto.FLOAT, [2, 3])
    builder.make_input("y", TensorProto.FLOAT, [2, 3])
    (z,) = builder.make_node("Add", ["x", "y"])
    builder.make_output(z)
    model = builder.to_onnx("model")

The module is exposed as ``onnx_light.onnx_core.graph_builder``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy

from ..onnx import helper, numpy_helper
from ..onnx_op import GetAllOnnxOpSchemasWithHistory, LightOpSchema
from ..onnx_py._onnxpycore import builder as _C  # type: ignore[attr-defined]

ConstantFoldingOptions: TypeAlias = _C.ConstantFoldingOptions
PatternOptimization: TypeAlias = _C.PatternOptimization

SchemaLookup: TypeAlias = Callable[[str], "list[LightOpSchema]"]


def _default_schema_lookup(op_type: str) -> list[LightOpSchema]:
    """Returns the built-in ONNX schema history for ``op_type``.

    The schemas live in the ``onnx_op`` extension, which the ``_onnxpycore``
    extension does not link against; this callable bridges the two so the
    builder can resolve opsets and validate nodes without that link.
    """
    return GetAllOnnxOpSchemasWithHistory(op_type, False)


class _OperatorProxy:
    """Exposes compact operator calls for one builder."""

    def __init__(self, builder: GraphBuilder) -> None:
        self._builder = builder

    def __getattr__(self, op_type: str) -> Callable[..., str | tuple[str, ...]]:
        if op_type.startswith("_"):
            raise AttributeError(op_type)

        def make_operator(*inputs: str | numpy.ndarray | None, **kwargs: object):
            outputs = kwargs.pop("outputs", None)
            domain = kwargs.pop("domain", "")
            name = kwargs.pop("name", "")
            if not isinstance(domain, str):
                raise TypeError("The node domain must be a string.")
            if not isinstance(name, str):
                raise TypeError("The node name must be a string.")

            normalized_domain = "ai.onnx" if domain in ("", "ai.onnx") else domain
            schemas = [
                schema
                for schema in _default_schema_lookup(op_type)
                if ("ai.onnx" if schema.domain in ("", "ai.onnx") else schema.domain)
                == normalized_domain
            ]
            if schemas:
                version = self._builder.opset_version(domain)
                matching = (
                    [schema for schema in schemas if schema.since_version <= version]
                    if normalized_domain in self._builder._explicit_opsets
                    else schemas
                )
                if not matching:
                    raise ValueError(
                        f"Standard ONNX operator {op_type!r} is unavailable at opset {version}."
                    )
                schema = max(matching, key=lambda candidate: candidate.since_version)
                attribute_names = {attribute.name for attribute in schema.attributes}
                unknown = sorted(set(kwargs) - attribute_names)
                if unknown:
                    raise ValueError(
                        f"Unknown attribute(s) for standard ONNX operator {op_type!r}: "
                        f"{', '.join(unknown)}."
                    )
            elif normalized_domain == "ai.onnx":
                raise ValueError(f"Unknown standard ONNX operator {op_type!r}.")
            elif self._builder.opset_version(domain) <= 0:
                raise ValueError(
                    f"Custom domain {domain!r} has no imported opset; "
                    "call set_opset_version() first."
                )

            resolved_inputs = []
            for input_value in inputs:
                if input_value is None:
                    resolved_inputs.append("")
                elif isinstance(input_value, numpy.ndarray):
                    resolved_inputs.append(self._builder.init(input_value))
                elif isinstance(input_value, str):
                    resolved_inputs.append(input_value)
                else:
                    raise TypeError(
                        "Operator inputs must be value names, NumPy arrays, or None, "
                        f"not {type(input_value).__name__}."
                    )

            if isinstance(outputs, bool):
                raise TypeError(
                    "outputs must be a name, a sequence of names, or a positive count."
                )
            if isinstance(outputs, int):
                if outputs <= 0:
                    raise ValueError("An output count must be positive.")
                outputs = [""] * outputs

            result = self._builder.make_node(
                op_type,
                resolved_inputs,
                outputs=outputs,
                domain=domain,
                name=name,
                attributes=kwargs,
            )
            return result[0] if len(result) == 1 else tuple(result)

        return make_operator


class GraphBuilder(_C.GraphBuilder):
    """Incrementally builds an ONNX graph, model or function.

    See :mod:`onnx_light.onnx_core.graph_builder` for details. By default the
    builder validates nodes and resolves opsets using the built-in ONNX
    operator schemas; pass ``schema_lookup=None`` to disable this, or a custom
    ``op_type -> list[LightOpSchema]`` callable to use different schemas.
    """

    def __init__(
        self, name: str = "graph", schema_lookup: SchemaLookup | None = _default_schema_lookup
    ) -> None:
        super().__init__(name, schema_lookup)
        self._registered_patterns: dict[str, PatternOptimization] = {}
        self._op = _OperatorProxy(self)
        self._initializer_counter = 0
        self._explicit_opsets: set[str] = (
            set(self.opset_versions()) if not isinstance(name, str) else set()
        )

    def set_opset_version(self, domain: str, version: int) -> None:
        """Records an explicit opset version."""
        super().set_opset_version(domain, version)
        self._explicit_opsets.add("ai.onnx" if domain in ("", "ai.onnx") else domain)

    @property
    def op(self) -> _OperatorProxy:
        """Returns the cached compact operator proxy."""
        return self._op

    def inp(self, name: str, elem_type: int, shape: list[str | int | None]) -> str:
        """Declares and returns a compact graph input."""
        return self.make_input(helper.make_tensor_value_info(name, elem_type, shape))

    def out(
        self, name: str, elem_type: int | None = None, shape: list[str | int | None] | None = None
    ) -> str:
        """Declares and returns a compact graph output."""
        if elem_type is None:
            if shape is not None:
                raise ValueError("An output shape requires an element type.")
            self.make_output(name)
        else:
            self.make_output(helper.make_tensor_value_info(name, elem_type, shape))
        return name

    def init(self, value: numpy.ndarray, name: str | None = None) -> str:
        """Adds a NumPy initializer and returns its final name."""
        if not isinstance(value, numpy.ndarray):
            raise TypeError(f"An initializer must be a NumPy array, not {type(value).__name__}.")
        if name is None:
            while True:
                candidate = (
                    "init"
                    if self._initializer_counter == 0
                    else f"init_{self._initializer_counter}"
                )
                self._initializer_counter += 1
                if not self.has_name(candidate):
                    name = candidate
                    break
        return self.make_initializer(numpy_helper.from_array(value, name=name))

    def register_pattern(self, pattern: PatternOptimization) -> None:
        """Registers or replaces a pattern for this builder."""
        name = str(pattern.name)
        if not name:
            raise ValueError("A registered pattern must have a non-empty name.")
        self._registered_patterns[name] = pattern

    def unregister_pattern(self, name: str) -> bool:
        """Removes a builder-local pattern and returns whether it existed."""
        return self._registered_patterns.pop(name, None) is not None

    def clear_registered_patterns(self) -> None:
        """Removes every builder-local pattern."""
        self._registered_patterns.clear()

    def registered_patterns(self) -> tuple[PatternOptimization, ...]:
        """Returns builder-local patterns in registration order."""
        return tuple(self._registered_patterns.values())

    def registered_pattern_names(self) -> tuple[str, ...]:
        """Returns builder-local pattern names in registration order."""
        return tuple(self._registered_patterns)


__all__ = ["ConstantFoldingOptions", "GraphBuilder", "PatternOptimization", "SchemaLookup"]
