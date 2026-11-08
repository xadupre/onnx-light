"""Exposes exact versioned GGUF block payloads, not a .gguf container loader.

GPTQModel adaptation accepts extracted UINT8 SafeTensors qweight matrices and
explicit subtype and logical shape metadata. This stage rejects partial blocks.
Imports return EncodedValueProto with native GGML raw_data, understood by the
standard dequantization APIs. Former incompatible GGML-named profiles are rejected.
"""

from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

LAYOUT_VERSION = runtime.gguf_layout_version
UPSTREAM_REVISION = runtime.gguf_upstream_revision
REFERENCE_REVISION = runtime.gguf_reference_revision
DEFAULT_MAX_BYTES = 1 << 30
GgufBlockLayout = runtime.GgufBlockLayout
GgufBlockParameters = runtime.GgufBlockParameters
GptqModelGgufWeights = runtime.GptqModelGgufWeights
gguf_block_layouts = runtime.gguf_block_layouts
import_gguf_tensor_payload = runtime.import_gguf_tensor_payload
export_gguf_tensor_payload = runtime.export_gguf_tensor_payload
dequantize_gguf_tensor_payload = runtime.dequantize_gguf_tensor_payload
read_gguf_block_parameters = runtime.read_gguf_block_parameters
import_gptqmodel_gguf_weights = runtime.import_gptqmodel_gguf_weights
export_gptqmodel_gguf_weights = runtime.export_gptqmodel_gguf_weights

__all__ = [
    "DEFAULT_MAX_BYTES",
    "LAYOUT_VERSION",
    "REFERENCE_REVISION",
    "UPSTREAM_REVISION",
    "GgufBlockLayout",
    "GgufBlockParameters",
    "GptqModelGgufWeights",
    "dequantize_gguf_tensor_payload",
    "export_gguf_tensor_payload",
    "export_gptqmodel_gguf_weights",
    "gguf_block_layouts",
    "import_gguf_tensor_payload",
    "import_gptqmodel_gguf_weights",
    "read_gguf_block_parameters",
]
