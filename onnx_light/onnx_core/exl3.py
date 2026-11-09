"""Exposes lossless GPTQModel EXL3 checkpoint conversion and reference reconstruction.

The native contract is separate from the unchanged portable ``exl3`` affine profile.
No upstream runtime or accelerator is required.
"""

from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

UPSTREAM_REVISION = runtime.exl3_upstream_revision
DEFAULT_MAX_BYTES = 1 << 30
import_exl3_weights = runtime.import_exl3_weights
export_exl3_weights = runtime.export_exl3_weights
reconstruct_exl3_weights = runtime.reconstruct_exl3_weights
unpack_exl3_states = runtime.unpack_exl3_states

__all__ = [
    "DEFAULT_MAX_BYTES",
    "UPSTREAM_REVISION",
    "export_exl3_weights",
    "import_exl3_weights",
    "reconstruct_exl3_weights",
    "unpack_exl3_states",
]
