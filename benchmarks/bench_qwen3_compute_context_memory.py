"""Builds a random Qwen3-0.6B (4 layers), converts it with yobx, and plots memory."""

from __future__ import annotations

import argparse
import inspect

import matplotlib.pyplot as plt
import pandas
import torch
from torch.utils._pytree import tree_flatten
from transformers import AutoConfig, AutoModelForCausalLM
from yobx.torch import to_onnx
from yobx.torch.export_options import ExportOptions

import onnx_light.onnx.defs as defs
from onnx_light.onnx import load as ol_load
from onnx_light.onnx_optim.expressions import evaluate_expression
from onnx_light.onnx_optim.shape_inference import (
    ComputeContext,
    NODE_MEMORY_TOTAL_BYTES_KEY,
    ShapesContext,
    apply_inferred_shapes_to_model,
    compute_shape_model,
)

TICK_INTERVAL = 10


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the benchmark.

    Returns:
        The parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B", help="Hugging Face model id.")
    parser.add_argument("--layers", type=int, default=4, help="Number of hidden layers.")
    parser.add_argument("--batch", type=int, default=1, help="Input batch size.")
    parser.add_argument("--sequence-length", type=int, default=16, help="Input sequence length.")
    parser.add_argument(
        "--past-sequence-length",
        type=int,
        default=8,
        help="Past key-value cache sequence length.",
    )
    parser.add_argument(
        "--output-prefix",
        default="bench_qwen3_compute_context_memory",
        help="Output file prefix for ONNX, PNG, and XLSX artifacts.",
    )
    return parser.parse_args()


def evaluate_memory_scalar(value: int | str, assignment: dict[str, int]) -> int:
    if isinstance(value, int):
        return value
    return evaluate_expression(value, assignment)


def flatten_with_pytree(data: object) -> tuple[list[object], object]:
    return tree_flatten(data)


def make_past_key_values(config: AutoConfig, batch_size: int, past_sequence_length: int) -> tuple:
    num_attention_heads = int(config.num_attention_heads)
    num_key_value_heads = (
        int(config.num_key_value_heads)
        if hasattr(config, "num_key_value_heads")
        else num_attention_heads
    )
    head_dim = (
        int(config.head_dim)
        if hasattr(config, "head_dim")
        else int(config.hidden_size // num_attention_heads)
    )
    dtype = torch.float32
    cache: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(config.num_hidden_layers):
        key = torch.randn(
            (batch_size, num_key_value_heads, past_sequence_length, head_dim), dtype=dtype
        )
        value = torch.randn(
            (batch_size, num_key_value_heads, past_sequence_length, head_dim), dtype=dtype
        )
        cache.append((key, value))
    return tuple(cache)


def make_export_options() -> ExportOptions:
    export_options_signature = inspect.signature(ExportOptions)
    kwargs: dict[str, object] = {}
    if "strategy" in export_options_signature.parameters:
        kwargs["strategy"] = "transformers"
    if "dynamic_shapes" in export_options_signature.parameters:
        kwargs["dynamic_shapes"] = True
    if "patches" in export_options_signature.parameters:
        kwargs["patches"] = "transformers"
    if "flattening_function" in export_options_signature.parameters:
        kwargs["flattening_function"] = flatten_with_pytree
    return ExportOptions(**kwargs)


def make_tick_label(output_name: str, node_type: str) -> str:
    return f"{str(output_name)[:5]}-{node_type}"


def main() -> None:
    args = parse_args()

    defs.register_onnx_operator_set_schema()

    config = AutoConfig.from_pretrained(args.model_id)
    config.num_hidden_layers = args.layers
    if hasattr(config, "use_cache"):
        config.use_cache = True
    model = AutoModelForCausalLM.from_config(config).eval()

    past_key_values = make_past_key_values(config, args.batch, args.past_sequence_length)
    sample_inputs = {
        "input_ids": torch.randint(
            0, config.vocab_size, (args.batch, args.sequence_length), dtype=torch.int64
        ),
        "attention_mask": torch.ones(
            (args.batch, args.past_sequence_length + args.sequence_length), dtype=torch.int64
        ),
        "past_key_values": past_key_values,
        "use_cache": True,
    }
    dynamic_shapes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "cache_plus_sequence_length"},
        "past_key_values": [
            (
                {0: "batch_size", 2: "past_sequence_length"},
                {0: "batch_size", 2: "past_sequence_length"},
            )
            for _ in range(config.num_hidden_layers)
        ],
    }
    export_options = make_export_options()
    to_onnx_signature = inspect.signature(to_onnx)
    to_onnx_kwargs: dict[str, object] = {
        "kwargs": sample_inputs,
        "export_options": export_options,
    }
    if "dynamic_shapes" in to_onnx_signature.parameters:
        to_onnx_kwargs["dynamic_shapes"] = dynamic_shapes
    if "flattening_function" in to_onnx_signature.parameters:
        to_onnx_kwargs["flattening_function"] = flatten_with_pytree
    if "patches" in to_onnx_signature.parameters:
        to_onnx_kwargs["patches"] = "transformers"
    artifact = to_onnx(model, **to_onnx_kwargs)
    filename = f"{args.output_prefix}.onnx"
    artifact.save(filename)
    onnx_model = ol_load(filename, load_external_data=False)

    shape_context = ShapesContext()
    compute_shape_model(shape_context, onnx_model)
    apply_inferred_shapes_to_model(shape_context, onnx_model)

    compute_context = ComputeContext()
    value_tags, _ = compute_context.compute_value_and_node_tags(onnx_model.graph)
    compute_context.compute_inplace_reuse_graph(
        onnx_model.graph, shape_context, value_tags=value_tags
    )

    assignment = {
        "batch_size": args.batch,
        "sequence_length": args.sequence_length,
        "past_sequence_length": args.past_sequence_length,
    }
    total_bytes = [
        evaluate_memory_scalar(profile[NODE_MEMORY_TOTAL_BYTES_KEY], assignment)
        for profile in compute_context.memory
    ]
    node_indices = list(range(len(total_bytes)))
    event_key = "event"
    extra_keys = sorted(
        key
        for key in {k for profile in compute_context.memory for k in profile}
        if key not in {NODE_MEMORY_TOTAL_BYTES_KEY, event_key}
    )
    memory_rows: list[dict[str, object]] = []
    for index, profile in enumerate(compute_context.memory):
        node = onnx_model.graph.node[index] if index < len(onnx_model.graph.node) else None
        row: dict[str, object] = {
            "node index": index,
            "node type": node.op_type if node else "",
            "input": ", ".join(map(str, node.input)) if node else "",
            "output": ", ".join(map(str, node.output)) if node else "",
            "memory": total_bytes[index],
            "event": profile.get(event_key, ""),
        }
        row.update({key: profile.get(key, "") for key in extra_keys})
        memory_rows.append(row)

    print(f"Converted model with {len(onnx_model.graph.node)} nodes.")
    print(f"Peak ComputeContext total bytes: {max(total_bytes):,}")
    pandas.DataFrame(memory_rows).to_excel(f"{args.output_prefix}.xlsx", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(node_indices, total_bytes, linewidth=1)
    tick_indices = [
        index
        for index in range(0, len(node_indices), TICK_INTERVAL)
        if index < len(onnx_model.graph.node)
    ]
    tick_labels = []
    for index in tick_indices:
        node = onnx_model.graph.node[index]
        tick_labels.append(make_tick_label(node.output[0] if node.output else "", node.op_type))
    ax.set_xticks(tick_indices, labels=tick_labels, rotation=45, ha="right")
    ax.set_title(f"ComputeContext total bytes ({args.model_id}, layers={args.layers})")
    ax.set_xlabel("node index")
    ax.set_ylabel("total bytes")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.output_prefix}.png")


if __name__ == "__main__":
    main()
