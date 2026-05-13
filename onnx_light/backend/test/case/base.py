import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import numpy as np
from .... import onnx
from ....onnx import defs as onnx_defs
from ....onnx import helper as onnx_helper


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

    def run(self, rt: Callable) -> Sequence[np.ndarray]:
        """Runs a function taking multiple inputs and returns multiple outputs."""
        assert (
            self.data_sets and len(self.data_sets) > 0
        ), f"There is no stored data_sets for test={self!r}"
        return rt(*self.data_sets[0][0])

    def __repr__(self) -> str:
        "usual"
        return f"{self.__class__.__name__}(name={self.name!r}, kind={self.kind!r})"

    def assert_allclose(self, rt: callable, atol: float | None = None, rtol: float | None = None):
        """
        Checks that the outputs match the expected outputs.
        Uses atol, rtol from the class or overwritten values.
        """
        outputs = self.run(rt)
        expected_outputs = self.data_sets[0][1]
        assert len(outputs) == len(expected_outputs), (
            f"Unexpected number of outputs {len(outputs)} "
            f"(expected {len(expected_outputs)}) for test {self.name!r}"
        )
        # compares all outputs using atol, rtol
        use_atol = atol if atol is not None else self.atol
        use_rtol = rtol if rtol is not None else self.rtol
        for i, (output, expected) in enumerate(zip(outputs, expected_outputs)):
            np.testing.assert_allclose(
                output,
                expected,
                rtol=use_rtol,
                atol=use_atol,
                err_msg=f"Output {i} mismatch for test {self.name!r}",
            )


class Base:
    """Base class for all tests."""


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
    # retrieve the specifications for this node
    op_type = node_op.op_type
    domain = node_op.domain

    schema = onnx_defs.get_schema(op_type, domain=domain)
    since_version = schema.since_version

    present_inputs = [x for x in node_op.input if x != ""]
    present_outputs = [x for x in node_op.output if x != ""]

    # build value infos using onnx_light helper
    def _extract_vi(arr, arr_name):
        if isinstance(arr, onnx.TensorProto):
            elem_type = arr.data_type
            shape = tuple(arr.dims)
            return onnx_helper.make_tensor_value_info(arr_name, elem_type, shape)
        if isinstance(arr, list):
            elem_type = onnx_helper.np_dtype_to_tensor_dtype(arr[0].dtype)
            return onnx_helper.make_tensor_sequence_value_info(arr_name, elem_type, None)
        elem_type = onnx_helper.np_dtype_to_tensor_dtype(arr.dtype)
        return onnx_helper.make_tensor_value_info(arr_name, elem_type, list(arr.shape))

    inputs_vi = [_extract_vi(arr, n) for arr, n in zip(inputs, present_inputs)]
    outputs_vi = [_extract_vi(arr, n) for arr, n in zip(outputs, present_outputs)]

    # create a model based on that specification
    if "opset_imports" not in kwargs:
        opset_imports = [onnx_helper.make_opsetid(domain, since_version)]
    else:
        opset_imports = kwargs.pop("opset_imports")

    graph = onnx_helper.make_graph(
        nodes=[node_op], name=name, inputs=inputs_vi, outputs=outputs_vi
    )
    model = onnx_helper.make_model(
        graph,
        opset_imports=opset_imports,
        producer_name=kwargs.pop("producer_name", "backend-test"),
        **kwargs,
    )

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


def make_test_class(
    rt: Callable,
    include_regex: Sequence[str] | None = None,
    exclude_regex: Sequence[str] | None = None,
    atols: dict[str, float] | None = None,
    rtols: dict[str, float] | None = None,
):
    """
    Collects all test cases with collect_test_case.
    Keeps or removes tests based on include_regex and exclude_regex.
    Creates a test class which has a test method per test, like ``test_{name}``.
    Compares outputs.
    """
    # TODO: implement
