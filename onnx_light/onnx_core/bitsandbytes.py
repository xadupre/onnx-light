"""Exposes exact GPTQModel bitsandbytes NF4, FP4 and INT8 checkpoint conversion.

Uses GPTQModel's module-local weight_* buffers, not Hugging Face's dotted keys.
Preserves native codes, JSON and nested scale state without requantization.
Keeps the portable low-nibble-first NF4 profile unchanged.
"""

from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

UPSTREAM_REVISION = runtime.bitsandbytes_upstream_revision
DEFAULT_MAX_BYTES = 1 << 30
import_bitsandbytes_weights = runtime.import_bitsandbytes_weights
export_bitsandbytes_weights = runtime.export_bitsandbytes_weights
unpack_bitsandbytes_codes = runtime.unpack_bitsandbytes_codes
reconstruct_bitsandbytes_weights = runtime.reconstruct_bitsandbytes_weights

__all__ = [
    "DEFAULT_MAX_BYTES",
    "UPSTREAM_REVISION",
    "export_bitsandbytes_weights",
    "import_bitsandbytes_weights",
    "reconstruct_bitsandbytes_weights",
    "unpack_bitsandbytes_codes",
]
