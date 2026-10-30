"""Proposes operator groups with small function interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..onnx import GraphProto, ModelProto
from ..onnx_py._onnxpycore import graph_partition as _C

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


def propose_operator_groups(
    model: ModelProto | GraphProto,
    *,
    max_boundary_bytes: int = 1024,
    min_group_nodes: int = 2,
    max_group_nodes: int = 128,
    value_sizes: Mapping[str, int] | None = None,
) -> list[OperatorGroup]:
    """Proposes disjoint function bodies minimizing boundary transfer cost.

    Uses the C++ implementation to search intervals of the topologically ordered,
    non-Constant nodes, bounded by ``min_group_nodes`` and ``max_group_nodes``.
    Dynamic programming minimizes the total input/output bytes of proposed
    groups and remaining individual
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
        ValueError: If limits or size estimates are invalid or exceed unsigned
            64-bit range, or the graph has duplicate definitions, undefined
            inputs or non-topological nodes, or accumulated costs overflow.
    """
    for name, value, minimum in (
        ("max_boundary_bytes", max_boundary_bytes, 0),
        ("min_group_nodes", min_group_nodes, 2),
        ("max_group_nodes", max_group_nodes, min_group_nodes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    sizes = dict(value_sizes) if value_sizes is not None else {}
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in sizes.values()):
        raise ValueError("value_sizes must contain nonnegative integer byte counts.")
    if any(
        v > (1 << 64) - 1
        for v in (max_boundary_bytes, min_group_nodes, max_group_nodes, *sizes.values())
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
        )
    ]
