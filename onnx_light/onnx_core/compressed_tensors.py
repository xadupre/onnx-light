"""Exposes the native compressed-tensors checkpoint adapter."""

from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

UPSTREAM_REVISION = runtime.compressed_tensors_upstream_revision
DEFAULT_MAX_BYTES = 1 << 30
CompressedTensorsWeights = runtime.CompressedTensorsWeights
import_compressed_tensors_weights = runtime.import_compressed_tensors_weights
export_compressed_tensors_weights = runtime.export_compressed_tensors_weights

__all__ = [
    "DEFAULT_MAX_BYTES",
    "UPSTREAM_REVISION",
    "CompressedTensorsWeights",
    "export_compressed_tensors_weights",
    "import_compressed_tensors_weights",
]
