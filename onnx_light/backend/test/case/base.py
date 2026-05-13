import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
from .... import onnx


@dataclass
class TestCase:
    """
    Defines a test case.
    """

    name: str
    model_name: str
    url: str | None
    model_dir: str | None
    model: onnx.ModelProto | None
    data_sets: Sequence[tuple[Sequence[np.ndarray], Sequence[np.ndarray]]] | None
    kind: str
    rtol: float
    atol: float
    # Tell PyTest this isn't a real test.
    __test__: bool = False


class Base:
    """Base class for all tests."""

    pass


ALL_TESTS: dict[str, TestCase] = {}


def expect(
    node_op: onnx.NodeProto,
    inputs: Sequence[np.ndarray | onnx.TensorProto],
    outputs: Sequence[np.ndarray | onnx.TensorProto],
    name: str,
    **kwargs: Any,
) -> None:
    """
    In the case of ops with optional inputs and outputs, node_op.input and node_op.output indicate
    which inputs/outputs are present and which are omitted. However, the parameter inputs
    and outputs of this function include values only for inputs/outputs that are present.
    E.g., for an op with 3 inputs, if the second parameter is optional and we wish to omit it,
    node_op.inputs would look like ["Param1", "", "Param3"], while inputs would look like
    [input-1-value, input-3-value]
    Instead of creating model with latest version, it now generates models for since_version
    by default. Thus it can make every model uses the same opset version after every opset
    change. Besides, user can specify "use_max_opset_version" to generate models for
    the latest opset version that supports before targeted opset version.
    """
    import onnx as _onnx_ref

    # retrieve the specifications for this node
    op_type = node_op.op_type
    domain = node_op.domain
    try:
        schema = _onnx_ref.defs.get_schema(op_type, domain=domain)
        since_version = schema.since_version
    except _onnx_ref.defs.SchemaError:
        since_version = _onnx_ref.defs.onnx_opset_version()

    present_inputs = [x for x in node_op.input if x != ""]
    present_outputs = [x for x in node_op.output if x != ""]

    # build value infos using onnx_ref (since node_op may be from the onnx package)
    def _extract_vi_ref(arr, arr_name):
        if isinstance(arr, _onnx_ref.TensorProto):
            elem_type = arr.data_type
            shape = tuple(arr.dims)
            return _onnx_ref.helper.make_tensor_value_info(arr_name, elem_type, shape)
        if isinstance(arr, list):
            elem_type = _onnx_ref.helper.np_dtype_to_tensor_dtype(arr[0].dtype)
            return _onnx_ref.helper.make_tensor_sequence_value_info(arr_name, elem_type, None)
        elem_type = _onnx_ref.helper.np_dtype_to_tensor_dtype(arr.dtype)
        return _onnx_ref.helper.make_tensor_value_info(arr_name, elem_type, list(arr.shape))

    inputs_vi = [_extract_vi_ref(arr, n) for arr, n in zip(inputs, present_inputs)]
    outputs_vi = [_extract_vi_ref(arr, n) for arr, n in zip(outputs, present_outputs)]

    # create a model based on that specification
    if "opset_imports" not in kwargs:
        opset_imports = [_onnx_ref.helper.make_opsetid(domain, since_version)]
    else:
        opset_imports = kwargs.pop("opset_imports")

    graph = _onnx_ref.helper.make_graph(
        nodes=[node_op], name=name, inputs=inputs_vi, outputs=outputs_vi
    )
    ref_model = _onnx_ref.helper.make_model(
        graph,
        opset_imports=opset_imports,
        producer_name=kwargs.pop("producer_name", "backend-test"),
        **kwargs,
    )

    # deserialize into onnx_light ModelProto
    model_bytes = ref_model.SerializeToString()
    model = onnx.ModelProto()
    model.ParseFromString(model_bytes)

    # create a dictionary of inputs
    inputs_dict = dict(zip(present_inputs, inputs))

    # create a dictionary of outputs
    outputs_dict = dict(zip(present_outputs, outputs))

    # add this information into a dictionary
    ALL_TESTS[name] = TestCase(
        name=name,
        model_name=name,
        url=None,
        model_dir=None,
        model=model,
        data_sets=[(list(inputs_dict.values()), list(outputs_dict.values()))],
        kind="node",
        rtol=1e-3,
        atol=1e-7,
    )


def collect_test_case() -> dict[str, TestCase]:
    """
    Collects all test cases by running all export methods on Base subclasses.

    Returns:
        A dictionary mapping test case names to TestCase instances.
    """
    global ALL_TESTS

    # empty ALL_TESTS before collecting
    ALL_TESTS.clear()

    # walk through all node submodules and import them so their classes are registered
    from . import node as _node_pkg

    def _ignore_import_error(name: str) -> None:  # noqa: ARG001
        pass  # silently skip modules that fail to import

    for _finder, _modname, _ispkg in pkgutil.walk_packages(
        path=_node_pkg.__path__, prefix=_node_pkg.__name__ + ".", onerror=_ignore_import_error
    ):
        importlib.import_module(_modname)

    # call all export methods on Base subclasses (metaclass already did this, but
    # re-running here ensures ALL_TESTS is populated after clearing)
    for subclass in Base.__subclasses__():
        for attr_name in dir(subclass):
            if attr_name.startswith("export"):
                method = getattr(subclass, attr_name)
                if callable(method):
                    method()

    # copy ALL_TESTS and reset it
    result = dict(ALL_TESTS)
    ALL_TESTS.clear()
    return result
