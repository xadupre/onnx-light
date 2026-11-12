# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Regenerates native reference state with bitsandbytes 0.50.2 (not a test dependency).

GPTQModel c1976918cfe3703fbcf6d1ce2440a82b26394375's
nn_modules/qlinear/bitsandbytes.py uses these same quantize/dequantize calls and
maps QuantState.as_dict(packed=True) keys to module-local weight_* buffers.
The bitsandbytes tag pins 08a9956b939857ae2c4fbd56753960cc370fa7cf.
"""

import json
from pathlib import Path

import bitsandbytes
import torch


def array(value):
    """Returns exact little-endian storage, without decimal float conversion."""
    value = value.cpu().contiguous().numpy()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "hex": value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes().hex(),
    }


def generate():
    """Writes independent native codes, packed metadata and reconstructed weights."""
    if bitsandbytes.__version__ != "0.50.2":
        raise RuntimeError("Reference fixtures require exactly bitsandbytes 0.50.2.")
    records = []
    for quant_type in ("nf4", "fp4"):
        table = bitsandbytes.functional.get_4bit_type(quant_type, device="cpu")
        # Covers all table entries, non-row-aligned blocks, odd size and partial blocks.
        source = table[torch.arange(99) % 16]
        source *= torch.tensor([1.0, 2.0, 0.5, 4.0])[torch.arange(99) // 32]
        source = source.reshape(3, 33)
        for nested in (False, True):
            packed, state = bitsandbytes.functional.quantize_4bit(
                source, blocksize=32, quant_type=quant_type, compress_statistics=nested
            )
            tensors = {"weight": array(packed)}
            for name, value in state.as_dict(packed=True).items():
                key = (
                    "weight_quant_state" if name.startswith("quant_state.") else f"weight_{name}"
                )
                tensors[key] = array(value)
            expected = bitsandbytes.functional.dequantize_4bit(packed, state)
            records.append(
                {
                    "name": f"{quant_type}-nested-{nested}",
                    "tensors": tensors,
                    "expected": array(expected),
                }
            )
            if quant_type == "fp4" and not nested:
                # The quantizer chooses code 0 for both zeros. A native decoder
                # fixture must also exercise the distinct stored code 8.
                logical_codes = torch.arange(100, dtype=torch.uint8) % 16
                all_codes = ((logical_codes[::2] << 4) | logical_codes[1::2]).reshape(-1, 1)
                tensors = dict(tensors)
                tensors["weight"] = array(all_codes)
                expected = bitsandbytes.functional.dequantize_4bit(all_codes, state)
                records.append(
                    {"name": "fp4-all-codes", "tensors": tensors, "expected": array(expected)}
                )
    source = torch.linspace(-3, 4, 99).reshape(3, 33).half()
    codes, scales, outliers = bitsandbytes.functional.int8_vectorwise_quant(source, threshold=0.0)
    if outliers is not None and outliers.numel():
        raise RuntimeError("The direct INT8 fixture must not use outlier routing.")
    # GPTQModel _dequantize_int8_weight, with FP32 division before multiplication.
    expected = codes.float() * (scales.float() / 127.0).unsqueeze(1)
    records.append(
        {
            "name": "int8",
            "tensors": {"weight": array(codes), "weight_scb": array(scales)},
            "expected": array(expected),
        }
    )
    target = Path(__file__).with_name("bitsandbytes_native_v1.json")
    target.write_text(json.dumps(records, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    generate()
