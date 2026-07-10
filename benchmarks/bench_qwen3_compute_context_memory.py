"""Builds a random Qwen3-0.6B (4 layers), converts it with yobx, and plots memory."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from yobx.torch import to_onnx

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


def make_tick_label(output_name: str, node_type: str) -> str:
    return f"{str(output_name)[:5]}-{node_type}"


def main() -> None:
    """Performs export/shape profiling and writes ONNX, XLSX, and PNG artifacts."""

    args = parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    config.num_hidden_layers = args.layers
    if hasattr(config, "use_cache"):
        config.use_cache = True
    model = AutoModelForCausalLM.from_config(config).eval().to(torch.float16)

    from yobx.torch import apply_patches_for_model, register_flattening_functions
    from yobx.torch.in_transformers.cache_helper import make_dynamic_cache

    num_attention_heads = int(config.num_attention_heads)
    shape = (
        args.batch,
        (
            int(config.num_key_value_heads)
            if hasattr(config, "num_key_value_heads")
            else num_attention_heads
        ),
        args.past_sequence_length,
        (
            int(config.head_dim)
            if hasattr(config, "head_dim")
            else int(config.hidden_size // num_attention_heads)
        ),
    )
    sample_inputs = {
        "input_ids": torch.randint(
            0, config.vocab_size, (args.batch, args.sequence_length), dtype=torch.int64
        ),
        "attention_mask": torch.ones(
            (args.batch, args.past_sequence_length + args.sequence_length), dtype=torch.int64
        ),
        "past_key_values": make_dynamic_cache(
            [
                (torch.rand(shape, dtype=torch.float16), torch.rand(shape, dtype=torch.float16))
                for _ in range(config.num_hidden_layers)
            ]
        ),
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
    with (
        register_flattening_functions(patch_transformers=True),
        apply_patches_for_model(patch_transformers=True, patch_torch=False),
    ):
        artifact = to_onnx(model, kwargs=sample_inputs, dynamic_shapes=dynamic_shapes)
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
