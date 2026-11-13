"""Imports the historical Marlin loading token without inventing a tiled ABI.

At the pinned revision, GPTQ and AWQ Marlin holders load ordinary GPTQ-v1 and
AWQ GEMM tensors, respectively. Their post_init methods perform runtime tile
repacking. The explicit source layout is mandatory; transformed runtime buffers
are not accepted as checkpoints. Export uses the ordinary GPTQ/AWQ APIs, not a
new Marlin quantization path.
"""

import numpy

from . import awq, gptq

UPSTREAM_REVISION = gptq.UPSTREAM_REVISION
DEFAULT_MAX_BYTES = gptq.DEFAULT_MAX_BYTES


def import_marlin_weights(
    tensors,
    *,
    method,
    layout,
    in_features,
    out_features,
    group_size=128,
    bits=4,
    upstream_revision=UPSTREAM_REVISION,
    max_bytes=DEFAULT_MAX_BYTES,
):
    """Imports an explicitly identified historical, pre-runtime Marlin checkpoint."""
    if group_size not in (-1, in_features, 32, 64, 128):
        raise ValueError("Historical Marlin requires group size -1, K, 32, 64 or 128.")
    if method == "gptq" and layout == "gptq-v1":
        if bits not in (4, 8):
            raise ValueError("Historical GPTQ Marlin requires four or eight bits.")
        if in_features % 32 or out_features % 32:
            raise ValueError("Historical GPTQ Marlin row reader requires 32-aligned dimensions.")
        value = gptq.import_gptq_weights(
            tensors,
            format="gptq",
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            bits=bits,
            upstream_revision=upstream_revision,
            max_bytes=max_bytes,
        )
        effective_group = in_features if group_size == -1 else group_size
        if tensors["scales"].dtype.name not in ("float16", "bfloat16"):
            raise ValueError("Historical Marlin requires FP16/BF16 scales.")
        if in_features % effective_group:
            raise ValueError("Historical GPTQ Marlin requires complete groups.")
        if not numpy.all(gptq.unpack_gptq_zeros(value, max_bytes) == (1 << (bits - 1))):
            raise ValueError("Historical GPTQ Marlin requires symmetric midpoint zeros.")
        if not numpy.array_equal(
            tensors["g_idx"], numpy.arange(in_features, dtype=numpy.int64) // effective_group
        ):
            raise ValueError("Historical GPTQ Marlin requires contiguous groups.")
        return value
    if method == "awq" and layout == "awq-gemm":
        return awq.import_awq_weights(
            tensors,
            variant="gemm",
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            bits=bits,
            upstream_revision=upstream_revision,
            max_bytes=max_bytes,
        )
    raise ValueError("Unsupported or ambiguous historical Marlin method/layout.")


__all__ = ["DEFAULT_MAX_BYTES", "UPSTREAM_REVISION", "import_marlin_weights"]
