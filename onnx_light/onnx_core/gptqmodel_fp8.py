"""Exposes lossless GPTQModel FP8 checkpoints with original FP32 inverse scales."""

from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

UPSTREAM_REVISION = runtime.gptqmodel_fp8_upstream_revision
DEFAULT_MAX_BYTES = 1 << 30
import_gptqmodel_fp8_weights = runtime.import_gptqmodel_fp8_weights
export_gptqmodel_fp8_weights = runtime.export_gptqmodel_fp8_weights
reconstruct_gptqmodel_fp8_weights = runtime.reconstruct_gptqmodel_fp8_weights

__all__ = [
    "DEFAULT_MAX_BYTES",
    "UPSTREAM_REVISION",
    "export_gptqmodel_fp8_weights",
    "import_gptqmodel_fp8_weights",
    "reconstruct_gptqmodel_fp8_weights",
]
