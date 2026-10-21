"""Incremental ONNX graph builder backed by a C++ library.

This module exposes :class:`GraphBuilder`, an incremental builder for ONNX
graphs, models and functions. A builder starts empty, holds a compute context
and records every value name it hands out so a name can never be reused. Nodes
are added with :meth:`GraphBuilder.make_node`, which requires a selected
opset, validates the node against the built-in ONNX operator schemas, assigns
output names when the caller leaves them empty and runs incremental shape
inference. :meth:`GraphBuilder.to_onnx` finalises the accumulated graph into a
``ModelProto`` by default, or a ``GraphProto`` or ``FunctionProto`` selected with
:class:`ProtoKind`, writing the inferred shapes, the
in-place / release-after metadata, the value tags and the peak-memory
estimates. ``to_onnx(optimize=OptimizationMode.kOptimize)`` optimizes a copy and
exports the result; ``to_onnx(optimize=OptimizationMode.kRewritings)`` exports
the canonical model with the captured plan named ``"optimized"`` instead.

Typical usage::

    from onnx_light.onnx_core.graph_builder import GraphBuilder
    from onnx_light.onnx_proto import TensorProto

    builder = GraphBuilder("g", opsets=18)
    builder.make_input("x", TensorProto.FLOAT, [2, 3])
    builder.make_input("y", TensorProto.FLOAT, [2, 3])
    (z,) = builder.make_node("Add", ["x", "y"])
    builder.make_output(z)
    model = builder.to_onnx()

The module is exposed as ``onnx_light.onnx_core.graph_builder``.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from enum import Enum
from typing import TypeAlias

import numpy

import onnx_light.onnx.helper as oh
from ..onnx import FunctionProto, GraphProto, ModelProto, TensorProto
from ..onnx.defs import onnx_opset_version
import onnx_light.onnx.numpy_helper as onh
from ..onnx_op import GetAllOnnxOpSchemasWithHistory, LightOpSchema
from ..onnx_py._onnxpycore import builder as _C  # type: ignore[attr-defined]

# Reduced builds retain graph authoring without kernels; full builds register
# constant-folding kernels without depending on an evaluator import.
if importlib.util.find_spec("onnx_light.onnx_py._onnxpykernels") is not None:
    importlib.import_module("onnx_light.onnx_py._onnxpykernels")

ConstantFoldingOptions: TypeAlias = _C.ConstantFoldingOptions
ProtoKind: TypeAlias = _C.ProtoKind
PatternOptimization: TypeAlias = _C.PatternOptimization

SchemaLookup: TypeAlias = Callable[[str], "list[LightOpSchema]"]


class OptimizationMode(Enum):
    """Selects the optimization workflow used by ``GraphBuilder.to_onnx``."""

    kNone = 0
    kOptimize = 1
    kRewritings = 2


def _default_schema_lookup(op_type: str) -> list[LightOpSchema]:
    """Returns the built-in ONNX schema history for ``op_type``.

    The schemas live in the ``onnx_op`` extension, which the ``_onnxpycore``
    extension does not link against; this callable bridges the two so the
    builder can validate nodes at the selected opset without that link.
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
            version = self._builder.opset_version(domain)
            if version <= 0:
                raise ValueError(
                    f"Domain {domain!r} has no explicit opset; "
                    "supply opsets to the constructor or call set_opset_version() first."
                )
            schemas = [
                schema
                for schema in _default_schema_lookup(op_type)
                if ("ai.onnx" if schema.domain in ("", "ai.onnx") else schema.domain)
                == normalized_domain
            ]
            if schemas:
                matching = [schema for schema in schemas if schema.since_version <= version]
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
    builder validates nodes at selected opsets using the built-in ONNX
    operator schemas; pass ``schema_lookup=None`` to disable this, or a custom
    ``op_type -> list[LightOpSchema]`` callable to use different schemas.
    For an empty builder, ``opsets`` selects the default ONNX version (an integer)
    or explicit domain-to-version imports (a dictionary). The default ``None``
    selects ``{"": onnx_opset_version()}``; an empty dictionary selects no imports.
    When importing a model, ``None`` preserves its declared imports.
    """

    def __init__(
        self,
        name: str = "graph",
        schema_lookup: SchemaLookup | None = _default_schema_lookup,
        *,
        opsets: int | dict[str, int] | None = None,
    ) -> None:
        if opsets is None and isinstance(name, str):
            opsets = {"": onnx_opset_version()}
        if opsets is None:
            super().__init__(name, schema_lookup)
        else:
            super().__init__(
                name, schema_lookup, {"": opsets} if isinstance(opsets, int) else opsets
            )
        self._registered_patterns: dict[str, PatternOptimization] = {}
        self._schema_lookup = schema_lookup
        self._op = _OperatorProxy(self)
        self._initializer_counter = 0

    @property
    def op(self) -> _OperatorProxy:
        """Returns the cached compact operator proxy."""
        return self._op

    def inp(self, name: str, elem_type: int, shape: list[str | int | None]) -> str:
        """Declares and returns a compact graph input."""
        return self.make_input(oh.make_tensor_value_info(name, elem_type, shape))

    def out(
        self, name: str, elem_type: int | None = None, shape: list[str | int | None] | None = None
    ) -> str:
        """Declares and returns a compact graph output."""
        if elem_type is None:
            if shape is not None:
                raise ValueError("An output shape requires an element type.")
            self.make_output(name)
        else:
            self.make_output(oh.make_tensor_value_info(name, elem_type, shape))
        return name

    def init(self, value: numpy.ndarray, name: str | None = None, *, copy: bool = True) -> str:
        """Adds a NumPy initializer and returns its final name.

        By default, copies the payload. With ``copy=False``, borrows C-contiguous,
        dtype-aligned, little-endian storage without conversion. Supports bool, 8/16/32/64-bit
        integers, float16/32/64 and complex64/128; rejects other dtypes and layouts.
        Retains the array until the last borrowed payload owner releases it,
        including models exported from this builder.

        Writable arrays remain writable and mutations are visible to all owners.
        The caller must finish mutations before optimization or creating an
        execution session, which may cache derived values, and must not resize or
        reallocate storage while borrowed. Use ``copy=True`` for an independent
        snapshot; setting an array read-only does not freeze its aliases.
        """
        if not isinstance(value, numpy.ndarray):
            raise TypeError(f"An initializer must be a NumPy array, not {type(value).__name__}.")
        if not copy:
            if not value.flags.c_contiguous:
                raise ValueError("copy=False requires a C-contiguous NumPy array.")
            if not value.flags.aligned:
                raise ValueError("copy=False requires a dtype-aligned NumPy array.")
            if value.dtype.byteorder == ">" or (
                value.dtype.byteorder == "=" and sys.byteorder != "little"
            ):
                raise ValueError("copy=False requires little-endian NumPy storage.")
            if value.dtype.kind not in "biufc" or value.dtype.name not in {
                "bool",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "float16",
                "float32",
                "float64",
                "complex64",
                "complex128",
            }:
                raise TypeError(f"copy=False does not support NumPy dtype {value.dtype}.")
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
        if copy:
            return self.make_initializer(onh.from_array(value, name=name))
        tensor = TensorProto(
            name=name, dims=value.shape, data_type=oh.np_dtype_to_tensor_dtype(value.dtype)
        )
        tensor._set_raw_data_from_buffer(value)
        return self.make_initializer_move(tensor)

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

    def to_onnx(
        self,
        kind: ProtoKind = ProtoKind.kModel,
        ir_version: int = 0,
        domain: str = "",
        *,
        optimize: OptimizationMode = OptimizationMode.kNone,
    ) -> ModelProto | GraphProto | FunctionProto:
        """Exports the current graph, an optimized copy, or captured rewritings.

        With ``OptimizationMode.kNone`` (the default), preserves the existing
        export behavior. With ``OptimizationMode.kOptimize``, optimizes an
        independent builder using the global and builder-local pattern
        registrations, then exports the result without attached plans.
        Model export requires standard ONNX constructs.

        With ``OptimizationMode.kRewritings``, returns the finalized canonical
        ModelProto and attaches the captured plan as ``rewritings["optimized"]``.
        Requires ``kind=ProtoKind.kModel`` and rejects an existing plan with that
        name. Other named plans are retained. Optimization never edits the source
        builder; ordinary export finalization still runs before taking the copy.
        """
        if not isinstance(optimize, OptimizationMode):
            raise TypeError("optimize must be an OptimizationMode.")
        if optimize is OptimizationMode.kNone:
            return super().to_onnx(kind, ir_version, domain)
        if not isinstance(kind, _C.ProtoKind):
            raise TypeError("kind must be a ProtoKind.")
        if optimize is OptimizationMode.kRewritings and kind != ProtoKind.kModel:
            raise ValueError("Exporting rewritings requires kind=ProtoKind.kModel.")

        from .optimization import GraphGraph

        canonical = super().to_model(ir_version)
        if optimize is OptimizationMode.kRewritings and "optimized" in canonical.rewritings:
            raise ValueError("A rewriting plan named 'optimized' already exists.")
        working = GraphBuilder(canonical, self._schema_lookup)
        working.copy_device_targets_from(self)
        for pattern in self.registered_patterns():
            working.register_pattern(pattern)
        plan = GraphGraph(working).optimize()
        if optimize is OptimizationMode.kRewritings:
            restored = working.undo_rewritings(plan).to_model(ir_version)
            if restored.graph.SerializeToString() == canonical.graph.SerializeToString() and [
                function.SerializeToString() for function in restored.functions
            ] == [function.SerializeToString() for function in canonical.functions]:
                for name, existing in canonical.rewritings.items():
                    restored.add_rewriting(name, existing)
            restored.add_rewriting("optimized", plan)
            return restored
        if kind == ProtoKind.kModel:
            return working.to_standard_model(ir_version)
        return working.to_onnx(kind, ir_version, domain)


__all__ = [
    "ConstantFoldingOptions",
    "GraphBuilder",
    "OptimizationMode",
    "PatternOptimization",
    "ProtoKind",
    "SchemaLookup",
]
