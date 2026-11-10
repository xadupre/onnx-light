"""Exposes lossless native GPTQModel AWQ checkpoint adapters.

The explicit layouts are separate from the unchanged portable ``awq`` profile.
Imports borrow C-contiguous, native-endian NumPy arrays during validation, then
retain an independent encoded snapshot. No GPTQModel runtime is required.
"""

from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

UPSTREAM_REVISION = runtime.awq_upstream_revision
DEFAULT_MAX_BYTES = 1 << 30
import_awq_weights = runtime.import_awq_weights
export_awq_weights = runtime.export_awq_weights
unpack_awq_codes = runtime.unpack_awq_codes
reconstruct_awq_weights = runtime.reconstruct_awq_weights

__all__ = [
    "DEFAULT_MAX_BYTES",
    "UPSTREAM_REVISION",
    "export_awq_weights",
    "import_awq_weights",
    "reconstruct_awq_weights",
    "unpack_awq_codes",
]
