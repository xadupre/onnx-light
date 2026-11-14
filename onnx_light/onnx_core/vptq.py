"""Exposes the versioned native GPTQModel VPTQ checkpoint contract.

Preserves packed vector indices and auxiliary tensors without requantization.
This format is distinct from the portable AQLM and QuIP codebook profiles.
"""

from ..onnx_py._onnxpykernels import runtime  # type: ignore[missing-import]

UPSTREAM_REVISION = runtime.vptq_upstream_revision
SCHEMA_VERSION = 1
DEFAULT_MAX_BYTES = 1 << 30
export_vptq_weights = runtime.export_vptq_weights
export_vptq_config = runtime.export_vptq_config
reconstruct_vptq_weights = runtime.reconstruct_vptq_weights


def import_vptq_weights(
    tensors,
    *,
    upstream_revision=UPSTREAM_REVISION,
    schema_version=SCHEMA_VERSION,
    max_bytes=DEFAULT_MAX_BYTES,
    **config,
):
    """Imports loaded native tensors with explicit VPTQ constructor metadata.

    Requires ``in_features`` and ``out_features`` in ``config``. Other keys use
    the pinned ``VPTQLinear`` defaults: ``bits=4``, ``vector_lens=(8,8)``,
    ``num_centroids=(4096,4096)``, ``num_res_centroids=(-1,-1)``, ``npercent=0``,
    ``group_size=-1``, ``group_num=-1``, ``enable_norm=False``, ``norm_dim=0``,
    ``enable_perm=False``, ``kmeans_mode=""``, ``kmeans_iters=100``,
    ``kmeans_tol=1e-5``, ``kmeans_seed=0``, ``block_size=128``, ``percdamp=.01``.
    No tensor shapes or backend names are used to infer missing configuration.

    Returns:
        EncodedValueProto: Independent, losslessly encoded native checkpoint.
    """
    if type(schema_version) is not int:
        raise ValueError("VPTQ schema_version must be an integer.")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("VPTQ max_bytes must be a positive integer.")
    return runtime.import_vptq_weights(
        tensors, config, upstream_revision, schema_version, max_bytes
    )


__all__ = [
    "DEFAULT_MAX_BYTES",
    "SCHEMA_VERSION",
    "UPSTREAM_REVISION",
    "export_vptq_config",
    "export_vptq_weights",
    "import_vptq_weights",
    "reconstruct_vptq_weights",
]
