"""Proposes operator groups with small function interfaces."""

from __future__ import annotations

from collections import ChainMap, Counter
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass

import onnx_light.onnx.helper as oh
from ..onnx import FunctionProto, GraphProto, ModelProto, NodeProto, ValueInfoProto
from ..onnx_py._onnxpycore import graph_partition as _C  # type: ignore[attr-defined]
from .expressions import simplify_expression

__all__ = [
    "OperatorGroup",
    "count_constant_inputs",
    "describe_operator_group_calls",
    "propose_operator_groups",
    "symbolic_byte_sizes",
]


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
    transfer_bytes: int | None


def symbolic_byte_sizes(values: Sequence[ValueInfoProto]) -> list[int | str | None]:
    """Returns packed logical byte sizes without reading tensor payloads.

    Reuses native memory-analysis expressions, including sub-byte packing.
    Returns ``None`` for unavailable shapes, unsupported types or expressions
    beyond the native optional-analysis complexity limit.
    """
    return _C.symbolic_byte_sizes(values)


def _local_constant_names(scope: GraphProto | FunctionProto) -> set[str]:
    names: set[str] = set()
    if not isinstance(scope, FunctionProto):
        names.update(str(value.name) for value in scope.initializer)
        names.update(str(value.values.name) for value in scope.sparse_initializer)
        names.update(str(value.name) for value in scope.extended_initializer)
        names.difference_update(str(value.name) for value in scope.input)
    for node in scope.node:
        if (
            node.op_type == "Constant"
            and node.domain in ("", "ai.onnx", "onnx.ai")
            and not node.overload
        ):
            names.update(str(name) for name in node.output if name)
    return names


def count_constant_inputs(model: ModelProto) -> int:
    """Counts distinct declared constant values across model scopes without payload I/O.

    Includes dense, sparse and extended initializers and standard Constant outputs.
    Excludes overridable initializers; captured values are counted only at their
    declaration, while independent scopes may reuse names.
    """
    count = 0
    scopes: list[GraphProto | FunctionProto] = [model.graph, *model.functions]
    while scopes:
        scope = scopes.pop()
        count += len(_local_constant_names(scope))
        attributes = [attribute for node in scope.node for attribute in node.attribute]
        if isinstance(scope, FunctionProto):
            attributes.extend(scope.attribute_proto)
        for attribute in attributes:
            if attribute.HasField("g"):
                scopes.append(attribute.g)
            scopes.extend(attribute.graphs)
    return count


def _sum_byte_sizes(sizes: Sequence[int | str | None]) -> str:
    numeric = sum(size for size in sizes if isinstance(size, int))
    symbolic = Counter(size for size in sizes if isinstance(size, str))
    terms = [str(numeric)] if numeric else []
    terms.extend(f"{count}*({size})" for size, count in symbolic.items())
    total = str(simplify_expression("+".join(terms))) if terms else "0"
    unknown = sum(size is None for size in sizes)
    if unknown:
        return f"{total} + unknown[{unknown}]" if total != "0" else f"unknown[{unknown}]"
    return total


def _describe_operator_group_call(
    node: NodeProto, values: Mapping[str, ValueInfoProto | None], constants: Mapping[str, bool]
) -> list[str]:
    descriptions = [f"call {str(node.name)!r}:"]
    for direction, names in (("inputs", node.input), ("outputs", node.output)):
        descriptors = []
        for name in names:
            descriptor = values.get(str(name))
            if descriptor is None:
                descriptor = ValueInfoProto()
                descriptor.name = name
            descriptors.append(descriptor)
        sizes = symbolic_byte_sizes(descriptors)
        transfers = [
            0 if constants.get(str(name), False) else size
            for name, size in zip(names, sizes, strict=True)
        ]
        descriptions.append(
            f"  {direction}: logical bytes={_sum_byte_sizes(sizes)}; "
            f"transfer bytes={_sum_byte_sizes(transfers)}"
        )
    return descriptions


def describe_operator_group_calls(
    model: ModelProto, functions: Sequence[FunctionProto]
) -> dict[tuple[str, str, str], list[list[str]]]:
    """Describes actual call interfaces for both extraction methods.

    Reports two per-call totals: inputs and outputs, without per-value details.
    Each total includes symbolic logical bytes and chargeable transfer bytes.
    Counts standard Constants and non-overridable initializers as free transfers,
    matching mincut accounting. Resolves nested captures without confusing
    function-local names, shadowed values or call-specific symbolic dimensions.
    Symbolic formulas describe interfaces; they do not change either algorithm.
    """
    reports: dict[tuple[str, str, str], list[list[str]]] = {
        (str(function.domain), str(function.name), str(function.overload)): []
        for function in functions
    }
    scopes: list[
        tuple[
            GraphProto | FunctionProto,
            MutableMapping[str, ValueInfoProto | None],
            MutableMapping[str, bool],
        ]
    ] = [(model.graph, {}, {}), *((function, {}, {}) for function in model.functions)]
    while scopes:
        scope, inherited_values, inherited_constants = scopes.pop()
        is_function = isinstance(scope, FunctionProto)
        input_names = (
            {str(name) for name in scope.input}
            if is_function
            else {str(value.name) for value in scope.input}
        )
        initializers = (
            {}
            if is_function
            else {
                str(tensor.name): oh.make_tensor_value_info(
                    str(tensor.name), tensor.data_type, list(tensor.dims)
                )
                for tensor in scope.initializer
            }
        )
        if not is_function:
            initializers.update(
                {
                    str(tensor.values.name): oh.make_tensor_value_info(
                        str(tensor.values.name), tensor.values.data_type, list(tensor.dims)
                    )
                    for tensor in scope.sparse_initializer
                }
            )
        local_names = input_names | set(initializers)
        if not is_function:
            local_names.update(str(value.name) for value in scope.extended_initializer)
        for node in scope.node:
            local_names.update(str(name) for name in node.output if name)
        local_values: dict[str, ValueInfoProto | None] = dict.fromkeys(local_names)
        values = ChainMap(local_values, inherited_values)
        values.update(initializers)
        values.update({str(value.name): value for value in scope.value_info})
        if not is_function:
            values.update({str(value.name): value for value in (*scope.input, *scope.output)})
        constants = ChainMap(dict.fromkeys(local_names, False), inherited_constants)
        constants.update(dict.fromkeys(_local_constant_names(scope), True))
        attributes = [attribute for node in scope.node for attribute in node.attribute]
        if is_function:
            attributes.extend(scope.attribute_proto)
        for node in scope.node:
            key = (str(node.domain), str(node.op_type), str(node.overload))
            if key in reports:
                reports[key].append(_describe_operator_group_call(node, values, constants))
        for attribute in attributes:
            if attribute.HasField("g"):
                scopes.append((attribute.g, values, constants))
            scopes.extend((nested, values, constants) for nested in attribute.graphs)
    return reports


def propose_operator_groups(
    model: ModelProto | GraphProto,
    *,
    max_boundary_bytes: int | None = None,
    min_group_nodes: int = 2,
    max_group_nodes: int = 128,
    value_sizes: Mapping[str, int] | None = None,
    max_constant_inputs: int | None = None,
) -> list[OperatorGroup]:
    """Proposes disjoint function bodies minimizing boundary transfer cost.

    Uses the C++ implementation to search intervals of the topologically ordered,
    non-Constant nodes, bounded by ``min_group_nodes`` and ``max_group_nodes``.
    Dynamic programming minimizes the total input/output bytes of proposed
    groups and remaining individual
    operators, rather than greedily accepting the first small interface. This is
    optimal within this interval search space, not over arbitrary node subsets.
    Equal-cost solutions prefer grouping more operators.

    No byte cap is applied by default. An explicit ``max_boundary_bytes`` requires
    fully known combined input/output costs within that cap.
    ``max_constant_inputs`` optionally caps distinct constant inputs before any
    small-constant internalization, independently of their size.
    Each boundary value is charged once per direction, regardless of fan-out.
    Standard-domain Constant outputs and dense/sparse initializers are free;
    initializers also declared as graph inputs remain chargeable because callers
    can override them. Free values remain explicit inputs to proposed bodies.

    Uses existing input/output/value_info descriptors, without executing the
    model or reading tensor payloads. Unknown, symbolic, string and non-tensor
    boundary sizes are supported without a byte cap and yield ``transfer_bytes=None``.
    ``value_sizes`` supplies nonnegative byte estimates, overriding descriptors.
    Unknown transfers of groups and remaining operators are minimized before known bytes.
    Run shape inference first for useful proposals on unannotated models.

    The source is not mutated and no functions are installed. Returned indices
    and interfaces describe bodies suitable for subsequent function extraction.
    Intervals ensure that contracting all proposals preserves acyclic dataflow,
    including dependencies captured by graph attributes.

    Returns:
        Proposals in source order. Empty, constant-only and ineligible graphs
        yield an empty list.

    Raises:
        ValueError: If limits or size estimates are invalid or exceed unsigned
            64-bit range, or the graph has duplicate definitions, undefined
            inputs or non-topological nodes, or accumulated costs overflow.
    """
    limits = [
        ("min_group_nodes", min_group_nodes, 2),
        ("max_group_nodes", max_group_nodes, min_group_nodes),
    ]
    if max_boundary_bytes is not None:
        limits.append(("max_boundary_bytes", max_boundary_bytes, 0))
    if max_constant_inputs is not None:
        limits.append(("max_constant_inputs", max_constant_inputs, 0))
    for name, value, minimum in limits:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    sizes = dict(value_sizes) if value_sizes is not None else {}
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in sizes.values()):
        raise ValueError("value_sizes must contain nonnegative integer byte counts.")
    if any(
        v is not None and v > (1 << 64) - 1
        for v in (
            max_boundary_bytes,
            min_group_nodes,
            max_group_nodes,
            max_constant_inputs,
            *sizes.values(),
        )
    ):
        raise ValueError("Limits and byte counts must fit in an unsigned 64-bit integer.")

    graph = model.graph if isinstance(model, ModelProto) else model
    return [
        OperatorGroup(
            tuple(group.node_indices),
            tuple(group.inputs),
            tuple(group.outputs),
            group.transfer_bytes,
        )
        for group in _C.propose_operator_groups(
            graph,
            max_boundary_bytes=max_boundary_bytes,
            min_group_nodes=min_group_nodes,
            max_group_nodes=max_group_nodes,
            value_sizes=sizes,
            max_constant_inputs=max_constant_inputs,
        )
    ]
