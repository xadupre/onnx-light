"""Exposes the pinned QQQ tiled weight ABI, not QQQ activation execution.

Packing accepts native nibble codes in logical ``[N, K]`` order and the original
permuted ``s_channel`` and ``s_group`` tensors. It never requantizes or combines
scales. Reconstruction follows the upstream Torch weight reference, including
group rounding and saturation before channel scaling.
"""

from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

UPSTREAM_REVISION = runtime.qqq_upstream_revision
DEFAULT_MAX_BYTES = 1 << 30
import_qqq_weights = runtime.import_qqq_weights
export_qqq_weights = runtime.export_qqq_weights
pack_qqq_weights = runtime.pack_qqq_weights
unpack_qqq_codes = runtime.unpack_qqq_codes
reconstruct_qqq_weights = runtime.reconstruct_qqq_weights

__all__ = [
    "DEFAULT_MAX_BYTES",
    "UPSTREAM_REVISION",
    "export_qqq_weights",
    "import_qqq_weights",
    "pack_qqq_weights",
    "reconstruct_qqq_weights",
    "unpack_qqq_codes",
]
