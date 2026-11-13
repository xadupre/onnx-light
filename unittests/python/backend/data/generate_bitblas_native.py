# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Regenerates row-packed fixtures against fingerprinted GPTQModel CPU readers.

Pass gptqmodel/utils/mlx_packing.py from GPTQModel revision
c1976918cfe3703fbcf6d1ce2440a82b26394375. No MLX, Torch, GPU or JIT is needed.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy

SOURCE_SHA256 = "f89cce983006dcf424367b4e3db63d5843d2cc78072e6c586a6655d6ebdd424f"


def array(value):
    """Returns exact little-endian fixture storage."""
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "hex": value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes().hex(),
    }


def generate(path):
    """Writes native bytes and independently transcoded reference weights."""
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected the pinned GPTQModel mlx_packing.py source.")
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_pack_rows", "repack_gptq_bitblas", "repack_awq_bitblas"}
    ]
    namespace = {"np": numpy}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    records = []
    for method, bits, symmetric, k, n, group in (
        ("gptq", 2, False, 64, 32, 32),
        ("gptq", 4, True, 128, 64, 32),
        ("gptq", 8, True, 32, 32, -1),
        ("gptq", 8, False, 64, 32, 64),
        ("awq", 4, False, 64, 32, 32),
        ("awq", 4, True, 32, 64, -1),
        ("awq", 2, False, 32, 32, 32),
        ("awq", 3, False, 64, 32, 32),
        ("awq", 5, False, 32, 32, -1),
        ("awq", 6, False, 32, 32, -1),
        ("awq", 7, False, 32, 32, -1),
        ("awq", 8, False, 32, 32, -1),
        ("awq", 3, False, 32, 16, -1),
    ):
        groups = k // (k if group == -1 else group)
        row, col = numpy.indices((n, k))
        codes = ((row * 13 + col * 7 + col // 8) % (1 << bits)).astype(numpy.uint8)
        zeros = ((numpy.arange(groups)[:, None] * 5 + numpy.arange(n)) % (1 << bits)).astype(
            numpy.uint8
        )
        scales = numpy.resize(
            numpy.array([0.125, 0.5, 2, 0.3125, 1.75], dtype=numpy.float16), (n, groups)
        )

        def pack(values, width=bits):
            code_bits = numpy.unpackbits(values[..., None], axis=-1, bitorder="little")[
                ..., :width
            ]
            return numpy.packbits(
                code_bits.reshape(values.shape[0], -1), axis=-1, bitorder="little"
            ).view(numpy.int8)

        packed_zeros = (
            numpy.empty(0, dtype=numpy.int8) if method == "gptq" and symmetric else pack(zeros)
        )
        tensors = {"qweight": pack(codes), "scales": scales, "qzeros": packed_zeros}
        reader = namespace[f"repack_{method}_bitblas"]
        args = (tensors["qweight"], packed_zeros, scales, k, n, bits)
        weight, reference_scales, bias = (
            reader(*args, symmetric) if method == "gptq" else reader(*args)
        )
        target_bits = 8 if bits == 7 else bits
        stream = numpy.unpackbits(
            numpy.ascontiguousarray(weight, dtype="<u4").view(numpy.uint8),
            axis=-1,
            bitorder="little",
        )
        affine_codes = (
            stream.reshape(n, k, target_bits).astype(numpy.uint32)
            << numpy.arange(target_bits, dtype=numpy.uint32)
        ).sum(axis=-1)
        expected = affine_codes.astype(numpy.float32) * numpy.repeat(
            reference_scales.astype(numpy.float32), k // groups, axis=1
        ) + numpy.repeat(bias, k // groups, axis=1)
        records.append(
            {
                "layout": "row-packed-v1",
                "method": method,
                "bits": bits,
                "symmetric": symmetric,
                "in_features": k,
                "out_features": n,
                "group_size": group,
                "upstream_revision": "c1976918cfe3703fbcf6d1ce2440a82b26394375",
                "source_sha256": SOURCE_SHA256,
                "tensors": {name: array(value) for name, value in tensors.items()},
                "codes": array(codes),
                "expected": array(expected),
            }
        )
    Path(__file__).with_name("bitblas_native_v1.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    generate(parser.parse_args().source)
