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
    # TODO
    # retrieve the specifications for this node
    # create a model based on that specification
    # create a dictionary of inputs
    # create a dictionary of outputs
    # add this information into a dictionary


def collect_test_case() -> dict[str, TestCase]:
    """
    Collects all test cases.
    """
    # empty TEST_CASE
    # walk through all test cases
    # call all expect methods
    # copy ALL_TESTS
    # delete ALL_TESTS
    # return the copy
