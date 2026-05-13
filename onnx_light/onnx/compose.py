# source: https://github.com/onnx/onnx/blob/main/onnx/compose.py
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from . import AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto
from . import helper
from . import pychecker

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from . import FunctionProto


def _s(v) -> str:
    """Converts a utils::String (or any object) to a Python str.

    RepeatedField elements are ``utils::String`` objects that are not Python
    ``str``.  Dict and set operations require proper Python strings, so this
    helper coerces any string-like value to ``str``.
    """
    return v if isinstance(v, str) else str(v)


def _update_repeated_str(field, name_map: dict[str, str]) -> None:
    """Updates a ``RepeatedField[String]`` in place using *name_map*.

    ``RepeatedField[String]`` does not expose ``__setitem__``, so we rebuild
    the field by calling ``clear()`` then ``extend()`` with the new values.
    """
    old = [_s(x) for x in field]
    new = [name_map.get(x, x) for x in old]
    if old != new:
        field.clear()
        field.extend(new)


class _Extractor:
    """Extracts a sub-graph from a ModelProto by specifying input/output names.

    Adapted from onnx.utils.Extractor.
    """

    def __init__(self, model: ModelProto) -> None:
        self.model = model
        self.graph = model.graph
        self.initializers: dict[str, TensorProto] = self._build_name2obj_dict(
            self.graph.initializer
        )
        self.value_infos: dict = self._build_name2obj_dict(self.graph.value_info)
        self.value_infos.update(self._build_name2obj_dict(self.graph.input))
        self.value_infos.update(self._build_name2obj_dict(self.graph.output))
        self.outmap: dict[str, int] = self._build_output_dict(self.graph)

    @staticmethod
    def _build_name2obj_dict(objs) -> dict:
        # obj.name is accessed via PYFIELD_STR which returns Python str.
        return {obj.name: obj for obj in objs}

    @staticmethod
    def _build_output_dict(graph: GraphProto) -> dict[str, int]:
        output_to_index: dict[str, int] = {}
        for index, node in enumerate(graph.node):
            for output_name in node.output:
                name = _s(output_name)
                if name == "":
                    continue
                output_to_index[name] = index
        return output_to_index

    def _dfs_search_reachable_nodes(
        self, node_output_name: str, graph_input_names: set[str], reachable: set[int]
    ) -> None:
        """Finds nodes reachable from the given output name via reverse DFS."""
        stack: list[str] = [node_output_name]
        while stack:
            current = stack.pop()
            if current in graph_input_names:
                continue
            if current in self.outmap:
                index = self.outmap[current]
                if index not in reachable:
                    reachable.add(index)
                    stack += [_s(inp) for inp in self.graph.node[index].input if _s(inp) != ""]

    def _collect_reachable_nodes(
        self, input_names: list[str], output_names: list[str]
    ) -> list[NodeProto]:
        """Returns topologically-ordered nodes needed to compute output_names."""
        input_set = set(input_names)
        reachable: set[int] = set()
        for name in output_names:
            self._dfs_search_reachable_nodes(name, input_set, reachable)
        return [self.graph.node[idx] for idx in sorted(reachable)]

    def _collect_referred_local_functions(self, nodes: list[NodeProto]) -> list[FunctionProto]:
        """Returns local functions transitively referenced by nodes."""
        function_map: dict[tuple[str, str], FunctionProto] = {}
        for fn in self.model.functions:
            function_map[(fn.name, fn.domain)] = fn
        referred: list[FunctionProto] = []
        queue: deque[NodeProto] = deque(nodes)
        while queue:
            node = queue.popleft()
            key = (node.op_type, node.domain)
            if key in function_map:
                fn = function_map.pop(key)
                referred.append(fn)
                queue.extend(fn.node)
        return referred

    def _collect_reachable_tensors(
        self, nodes: list[NodeProto]
    ) -> tuple[list[TensorProto], list]:
        """Returns initializers and value_infos referenced by nodes."""
        all_names: set[str] = set()
        for node in nodes:
            all_names.update(_s(x) for x in node.input)
            all_names.update(_s(x) for x in node.output)
        initializer = [t for name, t in self.initializers.items() if name in all_names]
        value_info = [v for name, v in self.value_infos.items() if name in all_names]
        return initializer, value_info

    def extract_model(self, input_names: list[str], output_names: list[str]) -> ModelProto:
        """Extracts a sub-model with the given input and output tensor names."""
        missing = [n for n in input_names + output_names if n not in self.value_infos]
        if missing:
            raise ValueError(
                f"The following names were not found in value_infos: {', '.join(missing)}"
            )
        inputs = [self.value_infos[n] for n in input_names]
        outputs = [self.value_infos[n] for n in output_names]
        nodes = self._collect_reachable_nodes(input_names, output_names)
        initializer, value_info = self._collect_reachable_tensors(nodes)
        local_functions = self._collect_referred_local_functions(nodes)
        name = f"Extracted from {{{self.graph.name}}}"
        graph = helper.make_graph(
            nodes, name, inputs, outputs, initializer=initializer, value_info=value_info
        )
        opset_imports = list(self.model.opset_import)
        model = helper.make_model(
            graph,
            ir_version=self.model.ir_version,
            opset_imports=opset_imports,
            producer_name="onnx_light.onnx.compose",
            functions=local_functions if local_functions else None,
        )
        return model


def check_overlapping_names(
    g1: GraphProto, g2: GraphProto, io_map: list[tuple[str, str]] | None = None
) -> list[tuple[str, list[str]]]:
    """Checks whether there are name collisions between two graphs.

    Returns a list of tuples where the first element represents the member
    containing overlapping names (one of: ``"edge"``, ``"value_info"``,
    ``"initializer"``, ``"sparse_initializer"``), and the second element
    contains a list of names that appear in both graphs under that category.

    Optionally takes an *io_map* representing the output/input pairs to be
    connected.  Overlaps present in the *io_map* are ignored.
    """
    if not isinstance(g1, GraphProto):
        raise TypeError("g1 argument is not an ONNX graph")
    if not isinstance(g2, GraphProto):
        raise TypeError("g2 argument is not an ONNX graph")

    def _overlapping(c1: list[str], c2: list[str]) -> list[str]:
        return list(set(c1) & set(c2))

    def _edge_names(graph: GraphProto, exclude: set[str] | None = None) -> list[str]:
        if exclude is None:
            exclude = set()
        edges = []
        for n in graph.node:
            for i in n.input:
                s = _s(i)
                if s != "" and s not in exclude:
                    edges.append(s)
            for o in n.output:
                s = _s(o)
                if s != "" and s not in exclude:
                    edges.append(s)
        return edges

    result: list[tuple[str, list[str]]] = []
    if not io_map:
        io_map = []
    io_map_inputs = {elem[1] for elem in io_map}

    overlap = _overlapping(_edge_names(g1), _edge_names(g2, exclude=io_map_inputs))
    if overlap:
        result.append(("edge", overlap))

    overlap = _overlapping([e.name for e in g1.value_info], [e.name for e in g2.value_info])
    if overlap:
        result.append(("value_info", overlap))

    overlap = _overlapping([e.name for e in g1.initializer], [e.name for e in g2.initializer])
    if overlap:
        result.append(("initializer", overlap))

    overlap = _overlapping(
        [e.values.name for e in g1.sparse_initializer],
        [e.values.name for e in g2.sparse_initializer],
    ) + _overlapping(
        [e.indices.name for e in g1.sparse_initializer],
        [e.indices.name for e in g2.sparse_initializer],
    )
    if overlap:
        result.append(("sparse_initializer", overlap))

    return result


def add_prefix_graph(
    graph: GraphProto,
    prefix: str,
    rename_nodes: bool | None = True,
    rename_edges: bool | None = True,
    rename_inputs: bool | None = True,
    rename_outputs: bool | None = True,
    rename_initializers: bool | None = True,
    rename_value_infos: bool | None = True,
    inplace: bool | None = False,
    name_map: dict[str, str] | None = None,
) -> GraphProto:
    """Adds a prefix to names of elements in a graph.

    Applies the prefix to nodes, edges, inputs, outputs, initializers,
    sparse initializers, and value infos as requested.  Empty names are
    not prefixed.

    Arguments:
        graph: The graph to prefix.
        prefix: Prefix string to prepend to each name.
        rename_nodes: Whether to prefix node names.
        rename_edges: Whether to prefix node edge names.
        rename_inputs: Whether to prefix input names.
        rename_outputs: Whether to prefix output names.
        rename_initializers: Whether to prefix initializer names.
        rename_value_infos: Whether to prefix value info names.
        inplace: If True, mutates *graph* in place; otherwise a copy is made.
        name_map: Shared name-map used when recursing into subgraphs.

    Returns:
        The (possibly new) GraphProto with prefixed names.
    """
    if not isinstance(graph, GraphProto):
        raise TypeError("graph argument is not an ONNX graph")

    if not inplace:
        g = GraphProto()
        g.CopyFrom(graph)
    else:
        g = graph

    def _pfx(name: str) -> str:
        return prefix + name if len(name) > 0 else name

    if name_map is None:
        name_map = {}

    if rename_edges:
        graph_output_names = {o.name for o in g.output}
        for n in g.node:
            for e in n.output:
                e_str = _s(e)
                if e_str not in graph_output_names:
                    name_map[e_str] = _pfx(e_str)

    if rename_inputs:
        for entry in g.input:
            name_map[entry.name] = _pfx(entry.name)
    if rename_outputs:
        for entry in g.output:
            name_map[entry.name] = _pfx(entry.name)

    if rename_nodes:
        for n in g.node:
            n.name = _pfx(n.name)
            for attribute in n.attribute:
                if attribute.HasField("g"):
                    add_prefix_graph(attribute.g, prefix, inplace=True, name_map=name_map)
                for sub_g in attribute.graphs:
                    add_prefix_graph(sub_g, prefix, inplace=True, name_map=name_map)

    if rename_initializers:
        for init in g.initializer:
            name_map[init.name] = _pfx(init.name)
        for sparse_init in g.sparse_initializer:
            name_map[sparse_init.values.name] = _pfx(sparse_init.values.name)
            name_map[sparse_init.indices.name] = _pfx(sparse_init.indices.name)

    if rename_value_infos:
        for entry in g.value_info:
            name_map[entry.name] = _pfx(entry.name)

    for n in g.node:
        _update_repeated_str(n.output, name_map)
        _update_repeated_str(n.input, name_map)

    for in_desc in g.input:
        if in_desc.name in name_map:
            in_desc.name = name_map[in_desc.name]
    for out_desc in g.output:
        if out_desc.name in name_map:
            out_desc.name = name_map[out_desc.name]

    for initializer in g.initializer:
        if initializer.name in name_map:
            initializer.name = name_map[initializer.name]
    for sparse_init in g.sparse_initializer:
        if sparse_init.values.name in name_map:
            sparse_init.values.name = name_map[sparse_init.values.name]
        if sparse_init.indices.name in name_map:
            sparse_init.indices.name = name_map[sparse_init.indices.name]

    for vi in g.value_info:
        if vi.name in name_map:
            vi.name = name_map[vi.name]

    return g


def add_prefix(
    model: ModelProto,
    prefix: str,
    rename_nodes: bool | None = True,
    rename_edges: bool | None = True,
    rename_inputs: bool | None = True,
    rename_outputs: bool | None = True,
    rename_initializers: bool | None = True,
    rename_value_infos: bool | None = True,
    rename_functions: bool | None = True,
    inplace: bool | None = False,
) -> ModelProto:
    """Adds a prefix to names of elements in a model.

    Applies the prefix to graph nodes, edges, inputs, outputs, initializers,
    sparse initializers, value infos, and local functions as requested.
    Empty names are not prefixed.

    Arguments:
        model: The model to prefix.
        prefix: Prefix string to prepend to each name.
        rename_nodes: Whether to prefix node names.
        rename_edges: Whether to prefix node edge names.
        rename_inputs: Whether to prefix input names.
        rename_outputs: Whether to prefix output names.
        rename_initializers: Whether to prefix initializer names.
        rename_value_infos: Whether to prefix value info names.
        rename_functions: Whether to prefix local function names.
        inplace: If True, mutates *model* in place; otherwise a copy is made.

    Returns:
        The (possibly new) ModelProto with prefixed names.
    """
    if not isinstance(model, ModelProto):
        raise TypeError("model argument is not an ONNX model")

    if not inplace:
        m = ModelProto()
        m.CopyFrom(model)
        model = m

    add_prefix_graph(
        model.graph,
        prefix,
        rename_nodes=rename_nodes,
        rename_edges=rename_edges,
        rename_inputs=rename_inputs,
        rename_outputs=rename_outputs,
        rename_initializers=rename_initializers,
        rename_value_infos=rename_value_infos,
        inplace=True,
    )

    if rename_functions:
        f_name_map: dict[str, str] = {}
        for f in model.functions:
            new_name = prefix + f.name
            f_name_map[f.name] = new_name
            f.name = new_name
        for f in model.functions:
            for n in f.node:
                if n.op_type in f_name_map:
                    n.op_type = f_name_map[n.op_type]
        for n in model.graph.node:
            if n.op_type in f_name_map:
                n.op_type = f_name_map[n.op_type]

    return model


def merge_graphs(
    g1: GraphProto,
    g2: GraphProto,
    io_map: list[tuple[str, str]],
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    prefix1: str | None = None,
    prefix2: str | None = None,
    name: str | None = None,
    doc_string: str | None = None,
) -> GraphProto:
    """Combines two ONNX graphs into a single one.

    The combined graph is defined by connecting the specified set of
    outputs/inputs.  Those inputs/outputs not specified in *io_map* will
    remain as inputs/outputs of the combined graph.

    Arguments:
        g1: First graph.
        g2: Second graph.
        io_map: Pairs ``[(out0, in0), ...]`` mapping outputs of *g1* to
                inputs of *g2* to be connected.
        inputs: Optional list of inputs to include in the combined graph.
                By default all inputs not present in *io_map* are included.
        outputs: Optional list of outputs to include in the combined graph.
                 By default all outputs not present in *io_map* are included.
        prefix1: Optional prefix added to all names in *g1*.
        prefix2: Optional prefix added to all names in *g2*.
        name: Optional name for the combined graph.
        doc_string: Optional docstring for the combined graph.

    Returns:
        Combined GraphProto.
    """
    if not isinstance(g1, GraphProto):
        raise TypeError("g1 argument is not an ONNX graph")
    if not isinstance(g2, GraphProto):
        raise TypeError("g2 argument is not an ONNX graph")

    if prefix1 or prefix2:
        if prefix1:
            g1_copy = GraphProto()
            g1_copy.CopyFrom(g1)
            g1 = add_prefix_graph(g1_copy, prefix=prefix1)
        if prefix2:
            g2_copy = GraphProto()
            g2_copy.CopyFrom(g2)
            g2 = add_prefix_graph(g2_copy, prefix=prefix2)
        io_map = [
            (prefix1 + io[0] if prefix1 else io[0], prefix2 + io[1] if prefix2 else io[1])
            for io in io_map
        ]

    io_map_g1_outs = {io[0] for io in io_map}
    io_map_g2_ins = {io[1] for io in io_map}
    reversed_io_map = {in_name: out_name for out_name, in_name in io_map}
    g1_outs = {o.name for o in g1.output}
    g2_ins = {i.name for i in g2.input}

    if inputs or outputs:
        if not inputs:
            g1_inputs = [i.name for i in g1.input]
            g2_inputs = [i.name for i in g2.input]
        else:
            input_set = set(inputs)
            g1_inputs = [i.name for i in g1.input if i.name in input_set]
            g2_inputs = [
                i.name for i in g2.input if i.name in input_set or i.name in io_map_g2_ins
            ]

        if not outputs:
            g1_outputs = [o.name for o in g1.output]
            g2_outputs = [o.name for o in g2.output]
        else:
            output_set = set(outputs)
            g1_outputs = [
                o.name for o in g1.output if o.name in output_set or o.name in io_map_g1_outs
            ]
            g2_outputs = [o.name for o in g2.output if o.name in output_set]

        if len(g1_inputs) < len(g1.input) or len(g1_outputs) < len(g1.output):
            e1 = _Extractor(helper.make_model(g1))
            g1 = e1.extract_model(g1_inputs, g1_outputs).graph

        if len(g2_inputs) < len(g2.input) or len(g2_outputs) < len(g2.output):
            e2 = _Extractor(helper.make_model(g2))
            g2 = e2.extract_model(g2_inputs, g2_outputs).graph

    for g1_out_name, g2_in_name in io_map:
        if g1_out_name not in g1_outs:
            raise ValueError(f"Output {g1_out_name} is not present in g1")
        if g2_in_name not in g2_ins:
            raise ValueError(f"Input {g2_in_name} is not present in g2")

    overlapping_names = check_overlapping_names(g1, g2, io_map)
    if len(overlapping_names) > 0:
        category, names = overlapping_names[0]
        raise ValueError(
            "Cant merge two graphs with overlapping names. "
            f"Found repeated {category} names: "
            + ", ".join(names)
            + "\n"
            + "Consider using ``onnx_light.onnx.compose.add_prefix`` to add a prefix "
            "to names in one of the graphs."
        )

    g = GraphProto()

    g.node.extend(g1.node)
    g2_nodes_begin = len(g.node)
    g.node.extend(g2.node)
    g2_nodes_end = len(g.node)

    def connect_io(sub_graph: GraphProto, start: int, end: int) -> None:
        for node_idx in range(start, end):
            node = sub_graph.node[node_idx]
            for attr in node.attribute:
                if attr.type == AttributeProto.GRAPH:
                    connect_io(attr.g, 0, len(attr.g.node))
                elif attr.type == AttributeProto.GRAPHS:
                    for sub_g in attr.graphs:
                        connect_io(sub_g, 0, len(sub_g.node))
            _update_repeated_str(node.input, reversed_io_map)

    connect_io(g, g2_nodes_begin, g2_nodes_end)

    if inputs:
        input_set = set(inputs)
        g.input.extend([i for i in g1.input if i.name in input_set])
        g.input.extend([i for i in g2.input if i.name in input_set])
    else:
        g.input.extend(g1.input)
        g.input.extend([i for i in g2.input if i.name not in io_map_g2_ins])

    if outputs:
        output_set = set(outputs)
        g.output.extend([o for o in g1.output if o.name in output_set])
        g.output.extend([o for o in g2.output if o.name in output_set])
    else:
        g.output.extend([o for o in g1.output if o.name not in io_map_g1_outs])
        g.output.extend(g2.output)

    g.initializer.extend(g1.initializer)
    g.initializer.extend([init for init in g2.initializer if init.name not in io_map_g2_ins])

    g.sparse_initializer.extend(g1.sparse_initializer)
    g.sparse_initializer.extend(
        [init for init in g2.sparse_initializer if init.values.name not in io_map_g2_ins]
    )

    g.value_info.extend(g1.value_info)
    g.value_info.extend([vi for vi in g2.value_info if vi.name not in io_map_g2_ins])
    value_info_names = {vi.name for vi in g.value_info}
    output_names = {o.name for o in g.output}
    g.value_info.extend(
        [
            o
            for o in g1.output
            if o.name in io_map_g1_outs
            and o.name not in value_info_names
            and o.name not in output_names
        ]
    )

    g.name = name if name is not None else f"{g1.name}_{g2.name}"

    if doc_string is None:
        doc_string = (
            f"Graph combining {g1.name} and {g2.name}\n"
            + g1.name
            + "\n\n"
            + g1.doc_string
            + "\n\n"
            + g2.name
            + "\n\n"
            + g2.doc_string
        )
    g.doc_string = doc_string

    return g


def merge_models(
    m1: ModelProto,
    m2: ModelProto,
    io_map: list[tuple[str, str]],
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    prefix1: str | None = None,
    prefix2: str | None = None,
    name: str | None = None,
    doc_string: str | None = None,
    producer_name: str | None = "onnx_light.onnx.compose.merge_models",
    producer_version: str | None = "1.0",
    domain: str | None = "",
    model_version: int | None = 1,
) -> ModelProto:
    """Combines two ONNX models into a single one.

    The combined model is defined by connecting the specified set of
    outputs/inputs.  Those inputs/outputs not specified in *io_map* will
    remain as inputs/outputs of the combined model.

    Both models must have the same IR version and the same operator sets
    imported.

    Arguments:
        m1: First model.
        m2: Second model.
        io_map: Pairs ``[(out0, in0), ...]`` mapping outputs of *m1* to
                inputs of *m2*.
        inputs: Optional list of inputs for the combined model.
        outputs: Optional list of outputs for the combined model.
        prefix1: Optional prefix for all names in *m1*.
        prefix2: Optional prefix for all names in *m2*.
        name: Optional name for the combined graph.
        doc_string: Optional docstring for the combined graph.
        producer_name: Producer name for the combined model.
        producer_version: Producer version for the combined model.
        domain: Domain of the combined model.
        model_version: Version of the combined model.

    Returns:
        Combined ModelProto.
    """
    if not isinstance(m1, ModelProto):
        raise TypeError("m1 argument is not an ONNX model")
    if not isinstance(m2, ModelProto):
        raise TypeError("m2 argument is not an ONNX model")

    if m1.ir_version != m2.ir_version:
        raise ValueError(
            f"IR version mismatch {m1.ir_version} != {m2.ir_version}."
            " Both models should have the same IR version"
        )
    ir_version = m1.ir_version

    opset_import_map: MutableMapping[str, int] = {}
    opset_imports = list(m1.opset_import) + list(m2.opset_import)

    for entry in opset_imports:
        if entry.domain in opset_import_map:
            found_version = opset_import_map[entry.domain]
            if entry.version != found_version:
                raise ValueError(
                    "Can't merge two models with different operator set ids for a given domain. "
                    f"Got: {list(m1.opset_import)} and {list(m2.opset_import)}"
                )
        else:
            opset_import_map[entry.domain] = entry.version

    if prefix1 or prefix2:
        if prefix1:
            m1_copy = ModelProto()
            m1_copy.CopyFrom(m1)
            m1 = add_prefix(m1_copy, prefix=prefix1)
        if prefix2:
            m2_copy = ModelProto()
            m2_copy.CopyFrom(m2)
            m2 = add_prefix(m2_copy, prefix=prefix2)
        io_map = [
            (prefix1 + io[0] if prefix1 else io[0], prefix2 + io[1] if prefix2 else io[1])
            for io in io_map
        ]

    graph = merge_graphs(
        m1.graph,
        m2.graph,
        io_map,
        inputs=inputs,
        outputs=outputs,
        name=name,
        doc_string=doc_string,
    )
    model = helper.make_model(
        graph,
        ir_version=ir_version,
        opset_imports=opset_imports,
        producer_name=producer_name,
        producer_version=producer_version,
    )
    if domain is not None:
        model.domain = domain
    if model_version is not None:
        model.model_version = model_version

    model_props: dict[str, str] = {}
    for meta_entry in m1.metadata_props:
        model_props[meta_entry.key] = meta_entry.value
    for meta_entry in m2.metadata_props:
        if meta_entry.key in model_props:
            value = model_props[meta_entry.key]
            if value != meta_entry.value:
                raise ValueError(
                    "Can't merge models with different values for the same"
                    f" model metadata property. Found: property = {meta_entry.key},"
                    f" with values {value} and {meta_entry.value}."
                )
        else:
            model_props[meta_entry.key] = meta_entry.value
    helper.set_model_props(model, model_props)

    function_overlap = list({f.name for f in m1.functions} & {f.name for f in m2.functions})
    if function_overlap:
        raise ValueError(
            "Can't merge models with overlapping local function names."
            " Found in both graphs: " + ", ".join(function_overlap)
        )
    model.functions.extend(m1.functions)
    model.functions.extend(m2.functions)

    pychecker.check_model(model)
    return model


def expand_out_dim_graph(
    graph: GraphProto, dim_idx: int, inplace: bool | None = False
) -> GraphProto:
    """Inserts an extra dimension with extent 1 to each output in the graph.

    Inserts an ``Unsqueeze`` node for each output.  Useful before merging
    graphs when the second graph expects a batch dimension.

    Arguments:
        graph: The graph to modify.
        dim_idx: Index of the dimension to insert.  Negative values count
                 from the back.
        inplace: If True, mutates *graph* in place; otherwise a copy is made.

    Returns:
        The (possibly new) GraphProto with expanded output dimensions.
    """
    if not isinstance(graph, GraphProto):
        raise TypeError("graph argument is not an ONNX graph")

    if not inplace:
        g = GraphProto()
        g.CopyFrom(graph)
    else:
        g = graph

    orig_out_names = [output.name for output in g.output]
    collapsed_map = {name: name + f"_collapsed_dim_{dim_idx}" for name in orig_out_names}

    for n in g.node:
        _update_repeated_str(n.output, collapsed_map)
        _update_repeated_str(n.input, collapsed_map)

    expand_dim_k = g.name + "_expand_out_dim_idx"
    g.node.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=[expand_dim_k],
            name=f"{expand_dim_k}-constant",
            value=helper.make_tensor(
                name=f"{expand_dim_k}-value",
                data_type=TensorProto.INT64,
                dims=[1],
                vals=[dim_idx],
            ),
        )
    )

    # Collect original outputs before clearing so we can rebuild them.
    orig_outputs = list(g.output)
    new_outputs = []
    for o in orig_outputs:
        prev_output = o.name + f"_collapsed_dim_{dim_idx}"
        g.node.append(
            helper.make_node(
                "Unsqueeze",
                inputs=[prev_output, expand_dim_k],
                outputs=[o.name],
                name=f"unsqueeze-{o.name}",
            )
        )
        new_shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
        new_shape.insert(dim_idx, 1)
        new_outputs.append(
            helper.make_tensor_value_info(o.name, o.type.tensor_type.elem_type, new_shape)
        )
    # Replace all outputs atomically.
    del g.output[:]
    g.output.extend(new_outputs)
    return g


def expand_out_dim(model: ModelProto, dim_idx: int, inplace: bool | None = False) -> ModelProto:
    """Inserts an extra dimension with extent 1 to each output in the model.

    Inserts an ``Unsqueeze`` node for each output.  Useful before merging
    models when the second one expects a batch dimension.

    Arguments:
        model: The model to modify.
        dim_idx: Index of the dimension to insert.  Negative values count
                 from the back.
        inplace: If True, mutates *model* in place; otherwise a copy is made.

    Returns:
        The (possibly new) ModelProto with expanded output dimensions.
    """
    if not isinstance(model, ModelProto):
        raise TypeError("model argument is not an ONNX model")

    if not inplace:
        m = ModelProto()
        m.CopyFrom(model)
        model = m

    expand_out_dim_graph(model.graph, dim_idx, inplace=True)
    return model
