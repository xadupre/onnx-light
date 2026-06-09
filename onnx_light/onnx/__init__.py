from __future__ import annotations

import sys

from .. import onnx_lib  # noqa: F401
from ..onnx_lib import (  # noqa: F401
    AttributeProto,
    DeviceConfigurationProto,
    FileLoadMode,
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
    align_external_data_streaming,
    consolidate_tensors_to_buffer,
    save_model_with_shared_external_data,
    utils_onnx_read_varint64,
    defs,
    external_data_helper,
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

# Re-export the onnx-light specific Python sub-packages so that they are
# also reachable from the ``onnx_light.onnx`` namespace.  ``onnx_light.onnx``
# is the API entry point that mirrors the upstream :mod:`onnx` package and
# is what compatibility helpers (see
# :mod:`onnx_light.compatibility.api_compare`) walk through.
from .. import backend, backend_test, tools  # noqa: F401

# Register sub-modules in sys.modules so that
# ``import onnx_light.onnx.<name>`` resolves correctly.
_SUBMODULE_NAMES = [
    "checker",
    "compose",
    "defs",
    "external_data_helper",
    "helper",
    "inliner",
    "io_helper",
    "numpy_helper",
    "parser",
    "shape_inference",
    "utils",
    "version_converter",
]

# TODO: module sys should be avoided.
for _name in _SUBMODULE_NAMES:
    _key = f"onnx_light.onnx.{_name}"
    if _key not in sys.modules:
        sys.modules[_key] = sys.modules[f"onnx_light.onnx_lib.{_name}"]

# Same trick for the onnx-light-only sub-packages so that
# ``import onnx_light.onnx.<name>`` resolves to the existing
# ``onnx_light.<name>`` package.
for _name in ("backend", "backend_test", "tools"):
    _key = f"onnx_light.onnx.{_name}"
    if _key not in sys.modules:
        sys.modules[_key] = sys.modules[f"onnx_light.{_name}"]
