"""Exposes explicitly identified GPTQModel row-packed BitBLAS checkpoints.

``row-packed-v1`` names the INT8 row streams accepted by the pinned MLX
checkpoint reader, not arbitrary target-specific BitBLAS weight transforms.
Packing preserves raw unsigned code bit patterns and native side tensors.
GPTQ symmetric codes use two's complement; AWQ always retains packed zeros.
"""

from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

UPSTREAM_REVISION = runtime.gptqmodel_upstream_revision
DEFAULT_MAX_BYTES = 1 << 30
import_bitblas_weights = runtime.import_bitblas_weights
export_bitblas_weights = runtime.export_bitblas_weights
pack_bitblas_weights = runtime.pack_bitblas_weights
unpack_bitblas_codes = runtime.unpack_bitblas_codes
reconstruct_bitblas_weights = runtime.reconstruct_bitblas_weights

__all__ = [
    "DEFAULT_MAX_BYTES",
    "UPSTREAM_REVISION",
    "export_bitblas_weights",
    "import_bitblas_weights",
    "pack_bitblas_weights",
    "reconstruct_bitblas_weights",
    "unpack_bitblas_codes",
]
