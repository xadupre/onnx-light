"""
.. _l-example-plot-qwen3-compute-context-memory:

Qwen3-like ComputeContext memory profile
========================================

This example builds a random-weight Qwen3-like model aligned with
``test_cc_shape_inference_big_qwen3_4_layers_like`` by retrieving it from
backend test cases through :func:`onnx_light.onnx.backend.collect_test_case`,
computes :class:`~onnx_light.onnx_core.shape_inference.ComputeContext` memory
events, saves them to Excel, and prints the same profile as a table.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas
from onnx_light.onnx import load as ol_load, save as ol_save, inliner
from onnx_light.onnx.backend import collect_test_cases_by_name
from onnx_light.onnx_core.expressions import evaluate_expression
from onnx_light.onnx_core.shape_inference import (
    ComputeContext,
    NODE_MEMORY_INITIALIZERS_KEY,
    NODE_MEMORY_TOTAL_BYTES_KEY,
)

TICK_INTERVAL = 10
TEST_CASE_NAME = "test_cc_shape_inference_big_qwen3_4_layers_like"


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the benchmark.

    Returns:
        parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num_hidden_layers",
        "-l",
        type=int,
        default=4,
        help="Number of hidden num_hidden_layers.",
    )
    parser.add_argument("--batch", "-b", type=int, default=1, help="Input batch size.")
    parser.add_argument(
        "--sequence-length", "-s", type=int, default=16, help="Input sequence length."
    )
    parser.add_argument(
        "--past-sequence-length",
        "-p",
        type=int,
        default=8,
        help="Past key-value cache sequence length.",
    )
    parser.add_argument(
        "--output-prefix",
        "-o",
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


def make_plot_assignments(args: argparse.Namespace) -> list[tuple[str, dict[str, int]]]:
    """Returns the four configurations displayed on the memory plot.

    Args:
        args: Parsed arguments with ``batch``, ``sequence_length``, and
            ``past_sequence_length`` attributes.

    Returns:
        A list of ``(label, assignment)`` tuples.
    """

    return [
        (
            f"current (past={args.past_sequence_length}, seq={args.sequence_length})",
            {
                "batch_size": args.batch,
                "sequence_length": args.sequence_length,
                "past_sequence_length": args.past_sequence_length,
                "total_sequence_length": args.sequence_length + args.past_sequence_length,
            },
        ),
        (
            "past=0, seq=128",
            {
                "batch_size": args.batch,
                "sequence_length": 128,
                "past_sequence_length": 0,
                "total_sequence_length": 128,
            },
        ),
        (
            "past=129, seq=1",
            {
                "batch_size": args.batch,
                "sequence_length": 1,
                "past_sequence_length": 129,
                "total_sequence_length": 130,
            },
        ),
        (
            "past=256, seq=1",
            {
                "batch_size": args.batch,
                "sequence_length": 1,
                "past_sequence_length": 256,
                "total_sequence_length": 257,
            },
        ),
    ]


def model_to_onnx(output_prefix, num_hidden_layers, batch, sequence_length, past_sequence_length):
    import torch
    from yobx.torch import to_onnx
    from yobx.torch import apply_patches_for_model, register_flattening_functions
    from yobx.torch.in_transformers.cache_helper import make_dynamic_cache
    from transformers import AutoConfig, AutoModelForCausalLM

    TEST_CASE_MODEL_TYPE = "qwen2"
    TEST_CASE_HIDDEN_SIZE = 1024
    TEST_CASE_INTERMEDIATE_SIZE = 3072
    TEST_CASE_NUM_ATTENTION_HEADS = 16
    TEST_CASE_NUM_KEY_VALUE_HEADS = 8
    TEST_CASE_HEAD_DIM = 128
    TEST_CASE_VOCAB_SIZE = 32000

    config = AutoConfig.for_model(
        TEST_CASE_MODEL_TYPE,
        vocab_size=TEST_CASE_VOCAB_SIZE,
        hidden_size=TEST_CASE_HIDDEN_SIZE,
        intermediate_size=TEST_CASE_INTERMEDIATE_SIZE,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=TEST_CASE_NUM_ATTENTION_HEADS,
        num_key_value_heads=TEST_CASE_NUM_KEY_VALUE_HEADS,
        head_dim=TEST_CASE_HEAD_DIM,
        use_cache=True,
    )
    model = AutoModelForCausalLM.from_config(config).eval().to(torch.float16)

    num_attention_heads = int(config.num_attention_heads)
    shape = (
        batch,
        (
            int(config.num_key_value_heads)
            if hasattr(config, "num_key_value_heads")
            else num_attention_heads
        ),
        past_sequence_length,
        (
            int(config.head_dim)
            if hasattr(config, "head_dim")
            else int(config.hidden_size // num_attention_heads)
        ),
    )
    sample_inputs = {
        "input_ids": torch.randint(
            0, TEST_CASE_VOCAB_SIZE, (batch, sequence_length), dtype=torch.int64
        ),
        "attention_mask": torch.ones(
            (batch, past_sequence_length + sequence_length), dtype=torch.int64
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
        "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
        "past_key_values": [
            {0: "batch_size", 2: "past_sequence_length"}
            for _ in range(config.num_hidden_layers * 2)
        ],
    }
    print("-- converts the model")
    with (
        register_flattening_functions(patch_transformers=True),
        apply_patches_for_model(patch_transformers=True, patch_torch=False),
    ):
        artifact = to_onnx(model, kwargs=sample_inputs, dynamic_shapes=dynamic_shapes)
    filename = f"{output_prefix}.onnx"

    print("-- saves the onnx model")
    artifact.save(filename)
    return filename


def get_big_qwen3_test_case_model(output_prefix):
    """Returns the ONNX model from the backend test case collection.

    Returns:
        The ``ModelProto`` from ``test_cc_shape_inference_big_qwen3_4_layers_like``.
    """

    cases = collect_test_cases_by_name(f".*{TEST_CASE_NAME}.*", include_big=True)
    if not cases:
        raise ValueError(f"{TEST_CASE_NAME!r} was not found in backend test cases.")
    filename = f"{output_prefix}.onnx"
    ol_save(cases[0].model, filename)
    return filename


def main() -> None:
    """Performs export/profiling, writes artifacts, and prints the XLSX profile table."""

    args = parse_args()
    if args.num_hidden_layers != 4:
        print("-- create the onnx model with transformers")
        filename = model_to_onnx(
            args.output_prefix,
            args.num_hidden_layers,
            args.batch,
            args.sequence_length,
            args.past_sequence_length,
        )
    else:
        print("-- get the model from the backend tests")
        filename = get_big_qwen3_test_case_model(args.output_prefix)

    onnx_model = ol_load(filename, load_external_data=False)
    onnx_model = inliner.inline_local_functions(onnx_model)
    del onnx_model.graph.value_info[:]

    print("-- run every analysis (shapes, tags, in-place reuse/release, peak memory)")
    compute_context = ComputeContext()
    compute_context.compute(onnx_model)
    print("-- write inferred shapes and annotations back to the model")
    compute_context.write_to_model(onnx_model)
    print("-- saves the model again")
    ol_save(onnx_model, filename, save_as_external_data=True)

    print("-- create export")
    plot_assignments = make_plot_assignments(args)
    assignment = plot_assignments[0][1]
    total_bytes = [
        evaluate_memory_scalar(profile[NODE_MEMORY_TOTAL_BYTES_KEY], assignment)
        for profile in compute_context.memory
    ]
    initializer_bytes_per_node = [
        sum(
            evaluate_memory_scalar(v, assignment)
            for v in profile.get(NODE_MEMORY_INITIALIZERS_KEY, {}).values()
        )
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
            "memory_without_initializers": total_bytes[index] - initializer_bytes_per_node[index],
            "event": profile.get(event_key, ""),
        }
        row.update({key: profile.get(key, "") for key in extra_keys})
        memory_rows.append(row)

    print(f"Converted model with {len(onnx_model.graph.node)} nodes.")
    print(f"Peak ComputeContext total bytes: {max(total_bytes):,}")
    memory_df = pandas.DataFrame(memory_rows)
    memory_df.to_excel(f"{args.output_prefix}.xlsx", index=False)
    with pandas.option_context("display.max_rows", None, "display.max_columns", None):
        print(memory_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, plot_assignment in plot_assignments:
        plot_total_bytes = [
            evaluate_memory_scalar(profile[NODE_MEMORY_TOTAL_BYTES_KEY], plot_assignment)
            for profile in compute_context.memory
        ]
        ax.plot(node_indices, plot_total_bytes, linewidth=1, label=label)
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
    ax.set_title(
        f"ComputeContext total bytes (qwen3_4_layers_like, "
        f"num_hidden_layers={args.num_hidden_layers})"
    )
    ax.set_xlabel("node index")
    ax.set_ylabel("total bytes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{args.output_prefix}.png")


if __name__ == "__main__":
    main()
