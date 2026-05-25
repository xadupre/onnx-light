# Copyright (c) ONNX Project Contributors
# Adapted from https://github.com/onnx/onnx/blob/main/onnx/utils.py
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import tarfile
from collections import deque
from typing import TYPE_CHECKING

from . import helper
from . import checker as _checker_mod
from .io_helper import load, save

__all__ = [
    "Extractor",
    "MAXIMUM_PROTOBUF",
    "extract_model",
]

if TYPE_CHECKING:
    from . import FunctionProto, ModelProto, NodeProto, TensorProto, ValueInfoProto

#: Maximum protobuf size in bytes (2 GB).
MAXIMUM_PROTOBUF = 2 * 1024**3


class Extractor:
    """Extracts a sub-model from an ONNX model by specifying input and output tensor names."""

    def __init__(self, model: ModelProto) -> None:
        self.model = model
        self.graph = self.model.graph
        self.initializers: dict[str, TensorProto] = self._build_name2obj_dict(
            self.graph.initializer
        )
        self.value_infos: dict[str, ValueInfoProto] = self._build_name2obj_dict(
            self.graph.value_info
        )
        # Add input and output values (not included in the value_info for intermediate values)
        self.value_infos.update(self._build_name2obj_dict(self.graph.input))
        self.value_infos.update(self._build_name2obj_dict(self.graph.output))
        self.outmap: dict[str, int] = self._build_output_dict(self.graph)

    @staticmethod
    def _build_name2obj_dict(objs) -> dict:
        return {str(obj.name): obj for obj in objs}

    @staticmethod
    def _build_output_dict(graph) -> dict[str, int]:
        output_to_index: dict[str, int] = {}
        for index, node in enumerate(graph.node):
            for output_name in node.output:
                output_name_str = str(output_name)
                if output_name_str == "":
                    continue
                if output_name_str in output_to_index:
                    raise ValueError(
                        f"Duplicate output name {output_name_str!r} found in graph nodes."
                    )
                output_to_index[output_name_str] = index
        return output_to_index

    def _collect_new_io(self, io_names_to_extract: list[str]) -> list[ValueInfoProto]:
        # Validate that all names exist in self.value_infos
        missing_names = [name for name in io_names_to_extract if name not in self.value_infos]
        if missing_names:
            raise ValueError(
                f"The following names were not found in value_infos: {', '.join(missing_names)}"
            )
        return [self.value_infos[name] for name in io_names_to_extract]

    def _dfs_search_reachable_nodes(
        self, node_output_name: str, graph_input_names: set[str], reachable: set[int]
    ) -> None:
        """Searches for nodes connected to an output via depth-first traversal.

        Args:
            node_output_name: The name of the output.
            graph_input_names: The names of all inputs of the graph.
            reachable: The set of indexes to reachable nodes in ``nodes``.
        """
        stack = [node_output_name]
        while stack:
            current_output_name = stack.pop()
            # finish search at inputs
            if current_output_name in graph_input_names:
                continue
            # find nodes connected to this output
            if current_output_name in self.outmap:
                index = self.outmap[current_output_name]
                if index not in reachable:
                    # add nodes connected to this output to sets
                    reachable.add(index)
                    stack += [
                        str(input_name)
                        for input_name in self.graph.node[index].input
                        if str(input_name) != ""
                    ]

    def _collect_reachable_nodes(
        self, input_names: list[str], output_names: list[str]
    ) -> list[NodeProto]:
        _input_names = set(input_names)
        reachable: set[int] = set()
        for name in output_names:
            self._dfs_search_reachable_nodes(name, _input_names, reachable)
        # needs to be topologically sorted
        return [self.graph.node[index] for index in sorted(reachable)]

    def _collect_referred_local_functions(self, nodes: list[NodeProto]) -> list[FunctionProto]:
        """Finds functions referred by graph nodes and by nodes used to define functions.

        Returns:
            List of FunctionProto for all functions reachable from ``nodes``.
        """
        function_map: dict[tuple[str, str], FunctionProto] = {}
        for function in self.model.functions:
            function_map[(str(function.name), str(function.domain))] = function
        referred_local_functions: list[FunctionProto] = []
        queue = deque(nodes)
        while queue:
            node = queue.popleft()
            # check if the node is a function op
            key = (str(node.op_type), str(node.domain))
            if key in function_map:
                function = function_map.pop(key)
                referred_local_functions.append(function)
                queue.extend(function.node)
        # needs to be topologically sorted
        return referred_local_functions

    def _collect_reachable_tensors(
        self, nodes: list[NodeProto]
    ) -> tuple[list[TensorProto], list[ValueInfoProto]]:
        all_tensors_names: set[str] = set()
        for node in nodes:
            all_tensors_names.update(str(n) for n in node.input)
            all_tensors_names.update(str(n) for n in node.output)
        initializer = [self.initializers[t] for t in self.initializers if t in all_tensors_names]
        value_info = [self.value_infos[t] for t in self.value_infos if t in all_tensors_names]
        len_sparse_initializer = len(self.graph.sparse_initializer)
        if len_sparse_initializer != 0:
            raise ValueError(f"len_sparse_initializer is {len_sparse_initializer}, it must be 0.")
        if self.graph.has_quantization_annotation():
            raise ValueError("quantization_annotation must be empty for sub-model extraction.")
        return initializer, value_info

    def _make_model(
        self,
        nodes: list[NodeProto],
        inputs: list[ValueInfoProto],
        outputs: list[ValueInfoProto],
        initializer: list[TensorProto],
        value_info: list[ValueInfoProto],
        local_functions: list[FunctionProto],
    ) -> ModelProto:
        name = "Extracted from {" + str(self.graph.name) + "}"
        graph = helper.make_graph(
            nodes, name, inputs, outputs, initializer=initializer, value_info=value_info
        )
        meta = {
            "ir_version": self.model.ir_version,
            "opset_imports": list(self.model.opset_import),
            "producer_name": "onnx_light.utils.extract_model",
            "functions": local_functions,
        }
        return helper.make_model(graph, **meta)

    def extract_model(self, input_names: list[str], output_names: list[str]) -> ModelProto:
        """Extracts a sub-model defined by *input_names* and *output_names*.

        Returns:
            A new :class:`ModelProto` containing only the nodes reachable from
            *output_names* given *input_names* as graph inputs.
        """
        inputs = self._collect_new_io(input_names)
        outputs = self._collect_new_io(output_names)
        nodes = self._collect_reachable_nodes(input_names, output_names)
        initializer, value_info = self._collect_reachable_tensors(nodes)
        local_functions = self._collect_referred_local_functions(nodes)
        return self._make_model(nodes, inputs, outputs, initializer, value_info, local_functions)


def extract_model(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    input_names: list[str],
    output_names: list[str],
    check_model: bool = True,
    infer_shapes: bool = False,
) -> None:
    """Extracts sub-model from an ONNX model.

    The sub-model is defined by the names of the input and output tensors *exactly*.

    Note: For control-flow operators, e.g. If and Loop, the boundary of sub-model,
    which is defined by the input and output tensors, should not cut through the
    subgraph that is connected to the main graph as attributes of these operators.

    Args:
        input_path: The path to original ONNX model.
        output_path: The path to save the extracted ONNX model.
        input_names: The names of the input tensors that to be extracted.
        output_names: The names of the output tensors that to be extracted.
        check_model: Whether to run model checker on the original model and the extracted model.
        infer_shapes: Whether to infer the shapes of the original model. Not yet supported;
            raises :exc:`NotImplementedError` when ``True``.
    """
    if not os.path.exists(input_path):
        raise ValueError(f"Invalid input model path: {input_path}")
    if not output_path:
        raise ValueError("Output model path shall not be empty!")
    if not input_names:
        raise ValueError("Input tensor names shall not be empty!")
    if not output_names:
        raise ValueError("Output tensor names shall not be empty!")

    if len(input_names) != len(set(input_names)):
        raise ValueError("Duplicate names found in the input tensor names.")
    if len(output_names) != len(set(output_names)):
        raise ValueError("Duplicate names found in the output tensor names.")

    if infer_shapes:
        raise NotImplementedError(
            "infer_shapes is not yet supported in onnx_light.utils.extract_model."
        )

    model = load(str(input_path))

    if check_model:
        _checker_mod.check_model(model)

    e = Extractor(model)
    extracted = e.extract_model(input_names, output_names)

    save(extracted, str(output_path))

    if check_model:
        _checker_mod.check_model(extracted)


def _tar_members_filter(tar: tarfile.TarFile, base: str | os.PathLike) -> list[tarfile.TarInfo]:
    """Checks that the content of ``tar`` will be extracted safely.

    Args:
        tar: The tarball file.
        base: The directory where the tarball will be extracted.

    Returns:
        List of tarball members.
    """
    result = []
    abs_base = os.path.abspath(base)
    for member in tar:
        member_path = os.path.join(base, member.name)
        abs_member = os.path.abspath(member_path)
        try:
            is_within_base = os.path.commonpath([abs_base, abs_member]) == abs_base
        except ValueError:
            is_within_base = False
        if not is_within_base:
            raise RuntimeError(
                f"The tarball member {member_path} in downloading model contains "
                f"directory traversal sequence which may contain harmful payload."
            )
        if member.issym() or member.islnk():
            raise RuntimeError(
                f"The tarball member {member_path} in downloading model contains "
                f"symbolic links which may contain harmful payload."
            )
        result.append(member)
    return result


def _extract_model_safe(
    model_tar_path: str | os.PathLike, local_model_with_data_dir_path: str | os.PathLike
) -> None:
    """Safely extracts a tar file to a specified directory.

    This function mitigates directory traversal vulnerabilities by validating
    paths within the tar file and provides compatibility for different versions
    of the tarfile module.

    Args:
        model_tar_path: The path to the tar file to be extracted.
        local_model_with_data_dir_path: The directory path where the tar file
            contents will be extracted to.
    """
    with tarfile.open(model_tar_path) as model_with_data_zipped:
        # Mitigate tarball directory traversal risks
        if hasattr(tarfile, "data_filter"):
            model_with_data_zipped.extractall(path=local_model_with_data_dir_path, filter="data")
        else:
            model_with_data_zipped.extractall(  # noqa: S202
                path=local_model_with_data_dir_path,
                members=_tar_members_filter(
                    model_with_data_zipped, local_model_with_data_dir_path
                ),
            )
