"""Backward-compatibility shim: ``onnx_light.onnx`` re-exports ``onnx_light.onnx_lib``.

The library was reorganised so that
- ``onnx_light.onnx_proto`` contains the standalone proto C++ library and the
  ``_onnxpy`` C-extension,
- ``onnx_light.onnx_lib`` contains the main Python API (helper, checker, …),
- ``onnx_light.onnx_py`` contains the C++ source files for the Python bindings.

This shim keeps ``import onnx_light.onnx`` and all
``import onnx_light.onnx.<submodule>`` imports working unchanged.
"""

from __future__ import annotations

import sys

from .. import onnx_lib  # noqa: F401
from ..onnx_lib import (  # noqa: F401
    AttributeProto,
    DeviceConfigurationProto,
    FunctionProto,
    GraphProto,
    IR_VERSION,
    IntIntListEntryProto,
    MapProto,
    Message,
    ModelProto,
    NodeDeviceConfigurationProto,
    NodeProto,
    OperatorSetIdProto,
    OperatorStatus,
    OptionalProto,
    ParseOptions,
    PrintOptions,
    SequenceProto,
    SerializeOptions,
    ShardedDimProto,
    ShardingSpecProto,
    SimpleShardedDimProto,
    SparseTensorProto,
    StringStringEntryProto,
    TensorAnnotation,
    TensorBufferOptions,
    TensorProto,
    TensorShapeProto,
    TypeProto,
    ValueInfoProto,
    checker,
    compose,
    consolidate_tensors_to_buffer,
    defs,
    helper,
    inliner,
    io_helper,
    numpy_helper,
    parser,
    shape_inference,
    utils,
    version_converter,
)
from ..onnx_lib.io_helper import (  # noqa: F401
    load,
    load_encrypted,
    load_encrypted_string,
    save,
    save_encrypted,
    save_encrypted_string,
)

# Register sub-modules in sys.modules so that
# ``import onnx_light.onnx.<name>`` resolves correctly.
_SUBMODULE_NAMES = [
    "checker",
    "compose",
    "defs",
    "helper",
    "inliner",
    "io_helper",
    "numpy_helper",
    "parser",
    "shape_inference",
    "utils",
    "version_converter",
]

for _name in _SUBMODULE_NAMES:
    _key = f"onnx_light.onnx.{_name}"
    if _key not in sys.modules:
        sys.modules[_key] = sys.modules[f"onnx_light.onnx_lib.{_name}"]
