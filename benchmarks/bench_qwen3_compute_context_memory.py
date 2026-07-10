"""Builds a random Qwen3-0.6B (4 layers), converts it with yobx, and plots memory."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from yobx.torch import to_onnx
from yobx.torch.export_options import ExportOptions

import onnx_light.onnx.defs as defs
from onnx_light.onnx_optim.expressions import evaluate_expression
from onnx_light.onnx_optim.shape_inference import (
    ComputeContext,
    NODE_MEMORY_TOTAL_BYTES_KEY,
    ShapesContext,
    apply_inferred_shapes_to_model,
    compute_shape_model,
)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B", help="Hugging Face model id.")
    parser.add_argument("--layers", type=int, default=4, help="Number of hidden layers.")
    parser.add_argument("--batch", type=int, default=1, help="Input batch size.")
    parser.add_argument("--sequence-length", type=int, default=16, help="Input sequence length.")
    return parser.parse_args()


def evaluate_memory_scalar(value: int | str, assignment: dict[str, int]) -> int:
    """Evaluates one memory scalar.

    Returns:
        The evaluated integer value.
    """

    if isinstance(value, int):
        return value
    return evaluate_expression(value, assignment)


def main() -> None:
    """Runs the benchmark."""

    args = parse_args()

    defs.register_onnx_operator_set_schema()

    config = AutoConfig.from_pretrained(args.model_id)
    config.num_hidden_layers = args.layers
    model = AutoModelForCausalLM.from_config(config).eval()

    sample_inputs = {
        "input_ids": torch.randint(
            0, config.vocab_size, (args.batch, args.sequence_length), dtype=torch.int64
        ),
        "attention_mask": torch.ones((args.batch, args.sequence_length), dtype=torch.int64),
    }

    artifact = to_onnx(
        model, kwargs=sample_inputs, export_options=ExportOptions(strategy="transformers")
    )
    onnx_model = artifact.get_proto()

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
        "past_sequence_length": 0,
    }
    total_bytes = [
        evaluate_memory_scalar(profile[NODE_MEMORY_TOTAL_BYTES_KEY], assignment)
        for profile in compute_context.memory
    ]
    node_indices = list(range(len(total_bytes)))

    print(f"Converted model with {len(onnx_model.graph.node)} nodes.")
    print(f"Peak ComputeContext total bytes: {max(total_bytes):,}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(node_indices, total_bytes, linewidth=1)
    ax.set_title(f"ComputeContext total bytes ({args.model_id}, layers={args.layers})")
    ax.set_xlabel("node index")
    ax.set_ylabel("total bytes")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
