from __future__ import annotations

import sys  # needed for package-level sys.modules registrations below

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
from .. import backend, backend_test, fuzz, tools  # noqa: F401

# Register the onnx-light-only sub-packages in sys.modules so that
# ``import onnx_light.onnx.<name>`` resolves to the existing
# ``onnx_light.<name>`` package.  A proxy .py file cannot be used here
# because these are full packages (not single-file modules) and the
# import identity check ``assertIs(mod, sys.modules["onnx_light.<name>"])``
# must hold.
for _name in ("backend", "backend_test", "fuzz", "tools"):
    _key = f"onnx_light.onnx.{_name}"
    if _key not in sys.modules:
        sys.modules[_key] = sys.modules[f"onnx_light.{_name}"]
