from __future__ import annotations

from typing import TYPE_CHECKING

from ..onnx_py import _onnxpyprotolib as _C  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from ..onnx_py._onnxpyprotoop import GraphProto, ModelProto  # type: ignore[attr-defined]

_cpp = _C.compose


# Imported lazily to avoid circular imports at module load time.
def _check_model(model: ModelProto) -> None:  # type: ignore[name-defined]
    from . import checker  # noqa: PLC0415

    checker.check_model(model)


def check_overlapping_names(
    g1: GraphProto, g2: GraphProto, io_map: list[tuple[str, str]] | None = None
) -> list[tuple[str, list[str]]]:
    """Checks whether two graphs have overlapping names.

    Returns a list of ``(category, names)`` pairs for each category where
    names appear in both *g1* and *g2*.  Recognised categories are
    ``"edge"``, ``"value_info"``, ``"initializer"`` and
    ``"sparse_initializer"``.

    The optional *io_map* lists output/input pairs that are intentionally
    shared between the two graphs; those overlaps are excluded.
    """
    return _cpp.check_overlapping_names(g1, g2, io_map or [])


def add_prefix_graph(
    graph: GraphProto,
    prefix: str,
    rename_nodes: bool = True,
    rename_edges: bool = True,
    rename_inputs: bool = True,
    rename_outputs: bool = True,
    rename_initializers: bool = True,
    rename_value_infos: bool = True,
    inplace: bool = False,
    name_map: dict[str, str] | None = None,  # noqa: ARG001 – kept for API compatibility
) -> GraphProto:
    """Adds a prefix to names of elements in a graph.

    Applies *prefix* to nodes, edges, inputs, outputs, initializers, sparse
    initializers and value-infos as requested.  Empty names are never
    prefixed.

    Arguments:
        graph: The graph to prefix.
        prefix: Prefix string to prepend.
        rename_nodes: Whether to prefix node names.
        rename_edges: Whether to prefix node edge names.
        rename_inputs: Whether to prefix input names.
        rename_outputs: Whether to prefix output names.
        rename_initializers: Whether to prefix initializer names.
        rename_value_infos: Whether to prefix value-info names.
        inplace: If True, mutates *graph* in place; otherwise a copy is made.
        name_map: Ignored – kept for backward-compatibility with the old Python API.

    Returns:
        The (possibly new) GraphProto with prefixed names.
    """
    result = _cpp.add_prefix_graph(
        graph,
        prefix,
        rename_nodes=rename_nodes,
        rename_edges=rename_edges,
        rename_inputs=rename_inputs,
        rename_outputs=rename_outputs,
        rename_initializers=rename_initializers,
        rename_value_infos=rename_value_infos,
    )
    if inplace:
        graph.CopyFrom(result)
        return graph
    return result


def add_prefix(
    model: ModelProto,
    prefix: str,
    rename_nodes: bool = True,
    rename_edges: bool = True,
    rename_inputs: bool = True,
    rename_outputs: bool = True,
    rename_initializers: bool = True,
    rename_value_infos: bool = True,
    rename_functions: bool = True,
    inplace: bool = False,
) -> ModelProto:
    """Adds a prefix to names of elements in a model.

    Applies *prefix* to graph nodes, edges, inputs, outputs, initializers,
    sparse initializers, value-infos and local functions as requested.
    Empty names are never prefixed.

    Arguments:
        model: The model to prefix.
        prefix: Prefix string to prepend.
        rename_nodes: Whether to prefix node names.
        rename_edges: Whether to prefix node edge names.
        rename_inputs: Whether to prefix input names.
        rename_outputs: Whether to prefix output names.
        rename_initializers: Whether to prefix initializer names.
        rename_value_infos: Whether to prefix value-info names.
        rename_functions: Whether to prefix local function names.
        inplace: If True, mutates *model* in place; otherwise a copy is made.

    Returns:
        The (possibly new) ModelProto with prefixed names.
    """
    result = _cpp.add_prefix(
        model,
        prefix,
        rename_nodes=rename_nodes,
        rename_edges=rename_edges,
        rename_inputs=rename_inputs,
        rename_outputs=rename_outputs,
        rename_initializers=rename_initializers,
        rename_value_infos=rename_value_infos,
        rename_functions=rename_functions,
    )
    if inplace:
        model.CopyFrom(result)
        return model
    return result


def merge_graphs(
    g1: GraphProto,
    g2: GraphProto,
    io_map: list[tuple[str, str]],
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    prefix1: str = "",
    prefix2: str = "",
    name: str = "",
    doc_string: str = "",
) -> GraphProto:
    """Combines two ONNX graphs into a single one.

    Arguments:
        g1: First graph.
        g2: Second graph.
        io_map: Pairs ``[(out_name, in_name), …]`` mapping outputs of *g1*
                to inputs of *g2*.
        inputs: Optional list of inputs to include.
        outputs: Optional list of outputs to include.
        prefix1: Optional prefix for all names in *g1*.
        prefix2: Optional prefix for all names in *g2*.
        name: Optional name for the combined graph.
        doc_string: Optional doc-string for the combined graph.

    Returns:
        Combined GraphProto.
    """
    return _cpp.merge_graphs(
        g1,
        g2,
        io_map,
        inputs=inputs or [],
        outputs=outputs or [],
        prefix1=prefix1 or "",
        prefix2=prefix2 or "",
        name=name or "",
        doc_string=doc_string or "",
    )


def merge_models(
    m1: ModelProto,
    m2: ModelProto,
    io_map: list[tuple[str, str]],
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    prefix1: str = "",
    prefix2: str = "",
    name: str = "",
    doc_string: str = "",
    producer_name: str = "onnx_light.onnx.compose.merge_models",
    producer_version: str = "1.0",
    domain: str = "",
    model_version: int | None = 1,
) -> ModelProto:
    """Combines two ONNX models into a single one.

    Both models must share the same IR version and operator sets.

    Arguments:
        m1: First model.
        m2: Second model.
        io_map: Pairs ``[(out_name, in_name), …]`` mapping outputs of *m1*
                to inputs of *m2*.
        inputs: Optional list of inputs for the combined model.
        outputs: Optional list of outputs for the combined model.
        prefix1: Optional prefix for all names in *m1*.
        prefix2: Optional prefix for all names in *m2*.
        name: Optional name for the combined graph.
        doc_string: Optional doc-string for the combined graph.
        producer_name: Producer name for the combined model.
        producer_version: Producer version string.
        domain: Domain of the combined model.
        model_version: Version of the combined model.  Pass ``None`` to
                       leave the version unset.

    Returns:
        Combined ModelProto.
    """
    _model_version = -1 if model_version is None else int(model_version)
    result = _cpp.merge_models(
        m1,
        m2,
        io_map,
        inputs=inputs or [],
        outputs=outputs or [],
        prefix1=prefix1 or "",
        prefix2=prefix2 or "",
        name=name or "",
        doc_string=doc_string or "",
        producer_name=producer_name or "",
        producer_version=producer_version or "",
        domain=domain or "",
        model_version=_model_version,
    )
    _check_model(result)
    return result


def expand_out_dim_graph(graph: GraphProto, dim_idx: int, inplace: bool = False) -> GraphProto:
    """Inserts an extra dimension with extent 1 to each output in the graph.

    Appends an ``Unsqueeze`` node for each output.  Useful before merging
    graphs when the second graph expects a batch dimension.

    Arguments:
        graph: The graph to modify.
        dim_idx: Index at which the new dimension is inserted. Negative values
            count from the back of the existing dimension list.
        inplace: If True, mutates *graph* in place; otherwise a copy is made.

    Returns:
        The (possibly new) GraphProto with expanded output dimensions.
    """
    result = _cpp.expand_out_dim_graph(graph, dim_idx)
    if inplace:
        graph.CopyFrom(result)
        return graph
    return result


def expand_out_dim(model: ModelProto, dim_idx: int, inplace: bool = False) -> ModelProto:
    """Inserts an extra dimension with extent 1 to each output in the model.

    Arguments:
        model: The model to modify.
        dim_idx: Index at which the new dimension is inserted. Negative values
            count from the back of the existing dimension list.
        inplace: If True, mutates *model* in place; otherwise a copy is made.

    Returns:
        The (possibly new) ModelProto with expanded output dimensions.
    """
    result = _cpp.expand_out_dim(model, dim_idx)
    if inplace:
        model.CopyFrom(result)
        return model
    return result
