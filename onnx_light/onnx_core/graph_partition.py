"""Proposes operator groups with small function interfaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import prod

from ..onnx import GraphProto, ModelProto, TensorProto, ValueInfoProto, collect_node_inputs

__all__ = ["OperatorGroup", "propose_operator_groups"]


@dataclass(frozen=True)
class OperatorGroup:
    """Describes a proposed function body without modifying its source graph.

    ``node_indices`` indexes the original graph. Inputs include constants and
    initializers, in first-use order, although these contribute no transfer cost.
    Outputs retain producer/port order and include values used outside the body,
    including graph outputs and captures by nested graphs.
    """

    node_indices: tuple[int, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    transfer_bytes: int


def _tensor_bytes(value: ValueInfoProto) -> int | None:
    """Returns a static tensor's packed byte count, or None when unknown."""
    if not value.type.has_tensor_type():
        return None
    tensor = value.type.tensor_type
    if not tensor.has_shape():
        return None
    bits = {
        TensorProto.FLOAT: 32,
        TensorProto.UINT8: 8,
        TensorProto.INT8: 8,
        TensorProto.UINT16: 16,
        TensorProto.INT16: 16,
        TensorProto.INT32: 32,
        TensorProto.INT64: 64,
        TensorProto.BOOL: 8,
        TensorProto.FLOAT16: 16,
        TensorProto.DOUBLE: 64,
        TensorProto.UINT32: 32,
        TensorProto.UINT64: 64,
        TensorProto.COMPLEX64: 64,
        TensorProto.COMPLEX128: 128,
        TensorProto.BFLOAT16: 16,
        TensorProto.FLOAT8E4M3FN: 8,
        TensorProto.FLOAT8E4M3FNUZ: 8,
        TensorProto.FLOAT8E5M2: 8,
        TensorProto.FLOAT8E5M2FNUZ: 8,
        TensorProto.UINT4: 4,
        TensorProto.INT4: 4,
        TensorProto.FLOAT4E2M1: 4,
        TensorProto.FLOAT8E8M0: 8,
        TensorProto.UINT2: 2,
        TensorProto.INT2: 2,
    }.get(tensor.elem_type)
    if bits is None or any(
        not dim.has_dim_value() or dim.dim_value < 0 for dim in tensor.shape.dim
    ):
        return None
    return (prod(dim.dim_value for dim in tensor.shape.dim) * bits + 7) // 8


def propose_operator_groups(
    model: ModelProto | GraphProto,
    *,
    max_boundary_bytes: int = 1024,
    min_group_nodes: int = 2,
    max_group_nodes: int = 128,
    value_sizes: Mapping[str, int] | None = None,
) -> list[OperatorGroup]:
    """Proposes disjoint function bodies minimizing boundary transfer cost.

    Searches intervals of the topologically ordered, non-Constant nodes, bounded
    by ``min_group_nodes`` and ``max_group_nodes``. Dynamic programming minimizes
    the total input/output bytes of proposed groups and remaining individual
    operators, rather than greedily accepting the first small interface. This is
    optimal within this interval search space, not over arbitrary node subsets.
    Equal-cost solutions prefer grouping more operators.

    A group's *combined* input/output cost must fit ``max_boundary_bytes``.
    Each boundary value is charged once per direction, regardless of fan-out.
    Standard-domain Constant outputs and dense/sparse initializers are free;
    initializers also declared as graph inputs remain chargeable because callers
    can override them. Free values remain explicit inputs to proposed bodies.

    Uses existing input/output/value_info descriptors, without executing the
    model or reading tensor payloads. Unknown, symbolic, string and non-tensor
    boundary sizes prevent a proposal; unknown internal sizes are allowed.
    ``value_sizes`` supplies nonnegative byte estimates, overriding descriptors.
    Unknown transfers of remaining operators are minimized before known bytes.
    Run shape inference first for useful proposals on unannotated models.

    The source is not mutated and no functions are installed. Returned indices
    and interfaces describe bodies suitable for subsequent function extraction.
    Intervals ensure that contracting all proposals preserves acyclic dataflow,
    including dependencies captured by graph attributes.

    Returns:
        Proposals in source order. Empty, constant-only and ineligible graphs
        yield an empty list.

    Raises:
        ValueError: If limits or size estimates are invalid, or the graph has
            duplicate definitions, undefined inputs or non-topological nodes.
    """
    for name, value, minimum in (
        ("max_boundary_bytes", max_boundary_bytes, 0),
        ("min_group_nodes", min_group_nodes, 2),
        ("max_group_nodes", max_group_nodes, min_group_nodes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    sizes: dict[str, int | None] = dict(value_sizes) if value_sizes is not None else {}
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in sizes.values()):
        raise ValueError("value_sizes must contain nonnegative integer byte counts.")

    graph = model.graph if isinstance(model, ModelProto) else model
    graph_inputs = {str(value.name) for value in graph.input}
    initializers = {str(value.name) for value in graph.initializer}
    initializers.update(str(value.values.name) for value in graph.sparse_initializer)
    constants = initializers - graph_inputs
    producers: dict[str, int] = {}
    node_inputs: list[tuple[str, ...]] = []
    node_outputs: list[tuple[str, ...]] = []
    indices: list[int] = []
    defined = graph_inputs | initializers
    for index, node in enumerate(graph.node):
        inputs = tuple(str(name) for name in collect_node_inputs(node))
        outputs = tuple(str(name) for name in node.output if str(name))
        if any(name not in defined for name in inputs):
            raise ValueError(f"Node {index} has undefined or non-topological inputs.")
        is_constant = (
            str(node.op_type) == "Constant"
            and str(node.domain) in ("", "ai.onnx", "onnx.ai")
            and not str(node.overload)
        )
        position = len(indices)
        for name in outputs:
            if name in defined:
                raise ValueError(f"Duplicate value definition {name!r}.")
            defined.add(name)
            if is_constant:
                constants.add(name)
            else:
                producers[name] = position
        if not is_constant:
            indices.append(index)
            node_inputs.append(inputs)
            node_outputs.append(outputs)

    for value in (*graph.value_info, *graph.input, *graph.output):
        name = str(value.name)
        if name not in sizes:
            sizes[name] = _tensor_bytes(value)

    count = len(indices)
    last_use: dict[str, int] = {}
    for position, inputs in enumerate(node_inputs):
        for name in inputs:
            last_use[name] = position
    for value in graph.output:
        name = str(value.name)
        if name not in defined:
            raise ValueError(f"Undefined graph output {name!r}.")
        last_use[name] = count

    def cost(inputs: Iterable[str], outputs: Iterable[str]) -> tuple[int, int]:
        unknown = known = 0
        for name in (*inputs, *outputs):
            if name in constants:
                continue
            size = sizes.get(name)
            if size is None:
                unknown += 1
            else:
                known += size
        return unknown, known

    # Each state minimizes (unknown transfers, known bytes, -grouped nodes).
    best: list[tuple[int, int, int] | None] = [None] * (count + 1)
    previous: list[tuple[int, OperatorGroup | None] | None] = [None] * (count + 1)
    best[0] = (0, 0, 0)
    for start in range(count):
        state = best[start]
        assert state is not None
        unknown, known, grouped = state
        single_outputs = tuple(
            name for name in node_outputs[start] if last_use.get(name, -1) > start
        )
        single_unknown, single_bytes = cost(node_inputs[start], single_outputs)
        score = (unknown + single_unknown, known + single_bytes, grouped)
        if best[start + 1] is None or score < best[start + 1]:
            best[start + 1] = score
            previous[start + 1] = (start, None)

        inputs: dict[str, None] = {}
        outputs: dict[str, None] = {}
        for end in range(start, min(count, start + max_group_nodes)):
            for name in node_inputs[end]:
                if producers.get(name, -1) < start:
                    inputs.setdefault(name, None)
                if last_use.get(name, -1) == end:
                    outputs.pop(name, None)
            for name in node_outputs[end]:
                if last_use.get(name, -1) > end:
                    outputs[name] = None
            input_unknown, input_bytes = cost(inputs, ())
            # External inputs cannot disappear as this interval grows forward.
            if input_unknown or input_bytes > max_boundary_bytes:
                break
            length = end - start + 1
            if length < min_group_nodes or not outputs:
                continue
            boundary_unknown, boundary_bytes = cost(inputs, outputs)
            if boundary_unknown or boundary_bytes > max_boundary_bytes:
                continue
            score = (unknown, known + boundary_bytes, grouped - length)
            if best[end + 1] is None or score < best[end + 1]:
                best[end + 1] = score
                previous[end + 1] = (
                    start,
                    OperatorGroup(
                        tuple(indices[start : end + 1]),
                        tuple(inputs),
                        tuple(outputs),
                        boundary_bytes,
                    ),
                )

    result: list[OperatorGroup] = []
    position = count
    while position:
        step = previous[position]
        assert step is not None
        position, group = step
        if group is not None:
            result.append(group)
    return result[::-1]
