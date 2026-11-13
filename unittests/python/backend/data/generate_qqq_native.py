# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Regenerates QQQ fixtures using unchanged, fingerprinted upstream methods.

Requires Torch only for regeneration. Pass the local qqq.py downloaded from
ModelCloud/GPTQModel c1976918cfe3703fbcf6d1ce2440a82b26394375 at
gptqmodel/nn_modules/qlinear/qqq.py. Extracting the weight-only methods avoids
importing GPTQModel's unrelated activation kernels and optional dependencies.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy
import torch

SOURCE_SHA256 = "2237a6b5e8c8b58866faf9bdffddcf6fc4955fb06f562d06f6b52ff0696d2042"


def upstream_class(path):
    """Loads only the fingerprinted producer and weight-reference methods."""
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected the exact pinned GPTQModel qqq.py source.")
    tree = ast.parse(source)
    functions = []
    methods = []
    wanted = {
        "QQQLinear": {"_get_perms", "pack"},
        "QQQTorchLinear": {
            "_unpack_weight_codes",
            "_unpermute_scales",
            "_dequantize_weight_for_torch",
        },
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_invert_permutation",
            "_unpack_uint4",
        }:
            functions.append(node)
        if isinstance(node, ast.ClassDef) and node.name in wanted:
            methods.extend(
                member
                for member in node.body
                if isinstance(member, ast.FunctionDef) and member.name in wanted[node.name]
            )
    namespace = {
        "torch": torch,
        "np": numpy,
        "CPU": torch.device("cpu"),
        "_INT4_DIVISORS": tuple(16**i for i in range(8)),
    }
    # The complete source is verified above before any extracted code executes.
    exec(
        compile(ast.Module(body=functions + methods, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return type(
        "PinnedQQQ",
        (torch.nn.Module,),
        {method.name: namespace[method.name] for method in methods},
    )


def array(value):
    """Returns exact little-endian tensor storage."""
    value = value.detach().cpu().contiguous().numpy()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "hex": value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes().hex(),
    }


def generate(path):
    """Writes native packed bytes, separate scales, codes and reference weights."""
    reference = upstream_class(path)
    records = []
    for k, n, group in ((64, 128, -1), (128, 128, 128), (256, 64, 128), (256, 128, 128)):
        module = reference()
        module.in_features, module.out_features = k, n
        module.group_size = k if group == -1 else group
        module.tile, module.max_par, module.bits, module.maxq = 16, 16, 4, 15
        module.bias = None
        (
            module._perm,
            module._perm_i,
            module._scale_perm,
            module._scale_perm_i,
            module._scale_perm_single,
            module._scale_perm_single_i,
        ) = module._get_perms()
        grouped = module.group_size != k
        row, col = torch.meshgrid(torch.arange(k), torch.arange(n), indexing="ij")
        signed = ((row * 5 + col * 3 + row // 16 + col // 16) % 16 - 8).float()
        channel = ((torch.arange(n) % 31 + 1).float() / 128).reshape(1, n)
        if grouped:
            ratios = (torch.arange(k // 128 * n) % 17 + 1).float() / 2
            raw_scales = ratios.reshape(k // 128, n) * channel
        else:
            raw_scales = channel * 16
        linear = torch.nn.Linear(k, n, bias=False, dtype=torch.float16)
        linear.weight.data.copy_(
            (signed * raw_scales.repeat_interleave(module.group_size, dim=0)).T
        )
        module.pack(linear, raw_scales.T, channel if grouped else None)
        for extreme in (False, True):
            if extreme:
                # Exercises scale underflow, ties, int8 saturation and large finite
                # channels without using the adapter to produce expected values.
                logical_channel = torch.tensor(
                    [2.0**-140, 0.3, 2.0**100, 2.0**-100], dtype=torch.float32
                ).repeat(n // 4)
                module.s_channel = logical_channel.reshape(-1, 32)[
                    :, module._scale_perm_single
                ].reshape(1, n)
                if grouped:
                    logical_group = torch.tensor(
                        [0, 2.0**-24, 0.5, 1.5, 2.5, 18.5, 65504, 0.2142333984375],
                        dtype=torch.float16,
                    ).repeat(k // 128 * n // 8)
                    module.s_group = logical_group.reshape(-1, 64)[:, module._scale_perm].reshape(
                        k // 128, n
                    )
            weight, s_channel = module._dequantize_weight_for_torch()
            records.append(
                {
                    "name": f"k{k}-n{n}-g{group}-extreme{extreme}",
                    "in_features": k,
                    "out_features": n,
                    "group_size": group,
                    "tensors": {
                        name: array(getattr(module, name))
                        for name in ("B", "s_channel", "s_group")
                    },
                    "codes": array(module._unpack_weight_codes().T.to(torch.uint8)),
                    "expected": array((weight * s_channel).T),
                }
            )
        if k == 64:
            tiny_scales = torch.full((1, n), 2.0**-24, dtype=torch.float16)
            linear.weight.data.copy_((signed * tiny_scales).T)
            module.pack(linear, tiny_scales.T)
            weight, s_channel = module._dequantize_weight_for_torch()
            records.append(
                {
                    "name": "channel-fp16-division-underflow",
                    "in_features": k,
                    "out_features": n,
                    "group_size": group,
                    "tensors": {
                        name: array(getattr(module, name))
                        for name in ("B", "s_channel", "s_group")
                    },
                    "codes": array(module._unpack_weight_codes().T.to(torch.uint8)),
                    "expected": array((weight * s_channel).T),
                }
            )
    for record in records:
        record["upstream_revision"] = "c1976918cfe3703fbcf6d1ce2440a82b26394375"
        record["source_sha256"] = SOURCE_SHA256
        record["torch_version"] = torch.__version__
    target = Path(__file__).with_name("qqq_native_v1.json")
    target.write_text(json.dumps(records, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    generate(parser.parse_args().source)
