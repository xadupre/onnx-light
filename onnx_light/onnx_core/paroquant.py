"""Exposes lossless native GPTQModel ParoQuant checkpoint adapters.

Preserves exact AWQ-GEMM weights and required input rotation/channel-scale
state from the pinned upstream revision. Reconstruction returns weights in
the transformed basis, not a stand-alone replacement for the linear operator.
``apply_paroquant_linear`` applies the complete operator using FLOAT arithmetic;
it retains the pinned dense consumer's FP16/BF16 weight materialization, but
does not emulate low-precision activations, accelerator kernels or runtime adapters.
"""

from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

UPSTREAM_REVISION = runtime.paroquant_upstream_revision
DEFAULT_MAX_BYTES = 1 << 30
import_paroquant_weights = runtime.import_paroquant_weights
export_paroquant_weights = runtime.export_paroquant_weights
unpack_paroquant_codes = runtime.unpack_paroquant_codes
reconstruct_paroquant_weights = runtime.reconstruct_paroquant_weights
apply_paroquant_linear = runtime.apply_paroquant_linear

__all__ = [
    "DEFAULT_MAX_BYTES",
    "UPSTREAM_REVISION",
    "apply_paroquant_linear",
    "export_paroquant_weights",
    "import_paroquant_weights",
    "reconstruct_paroquant_weights",
    "unpack_paroquant_codes",
]
