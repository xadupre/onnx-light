"""Exposes exact GPTQModel checkpoint packing independently of portable GPTQ.

``gptq`` (v1), ``gptq_v2`` and ``gptq_p`` name upstream checkpoint ABIs,
not the portable ``quantization`` module's unchanged ``gptq`` profile.
Imports retain an independent encoded snapshot of all four native tensors.
"""

from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

UPSTREAM_REVISION = runtime.gptqmodel_upstream_revision
DEFAULT_MAX_BYTES = 1 << 30
import_gptq_weights = runtime.import_gptq_weights
export_gptq_weights = runtime.export_gptq_weights
pack_gptq_weights = runtime.pack_gptq_weights
unpack_gptq_codes = runtime.unpack_gptq_codes
unpack_gptq_zeros = runtime.unpack_gptq_zeros
reconstruct_gptq_weights = runtime.reconstruct_gptq_weights

__all__ = [
    "DEFAULT_MAX_BYTES",
    "UPSTREAM_REVISION",
    "export_gptq_weights",
    "import_gptq_weights",
    "pack_gptq_weights",
    "reconstruct_gptq_weights",
    "unpack_gptq_codes",
    "unpack_gptq_zeros",
]
