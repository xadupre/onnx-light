"""
.. _l-example-plot-qwen3-compute-context-memory:

Qwen3-like ComputeContext memory profile
========================================

This example builds a random-weight Qwen3-like model aligned with
``test_cc_shape_inference_big_qwen3_4_layers_like`` by retrieving it from
backend test cases through :func:`onnx_light.onnx.backend.collect_test_case`,
computes :class:`~onnx_light.onnx_optim.shape_inference.ComputeContext` memory
events, saves them to Excel, and prints the same profile as a table.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas

from onnx_light.onnx import load as ol_load, save as ol_save, inliner
from onnx_light.onnx.backend import collect_test_case
from onnx_light.onnx_optim.expressions import evaluate_expression
from onnx_light.onnx_optim.shape_inference import (
    ComputeContext,
    NODE_MEMORY_TOTAL_BYTES_KEY,
    ShapesContext,
    apply_inferred_shapes_to_model,
    compute_shape_model,
)

TICK_INTERVAL = 10
TEST_CASE_NAME = "test_cc_shape_inference_big_qwen3_4_layers_like"


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the benchmark.

    Returns:
        parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
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


def get_big_qwen3_test_case_model():
    """Returns the ONNX model from the backend test case collection.

    Returns:
        The ``ModelProto`` from ``test_cc_shape_inference_big_qwen3_4_layers_like``.
    """

    # collect_test_case() returns a mapping: test case name -> backend TestCase.
    cases = collect_test_case()
    if TEST_CASE_NAME not in cases:
        available_qwen_cases = ", ".join(name for name in sorted(cases) if "qwen" in name.lower())
        raise ValueError(
            f"{TEST_CASE_NAME!r} was not found in backend test cases. "
            f"Available qwen-like names: {available_qwen_cases}"
        )
    return cases[TEST_CASE_NAME].model


def main() -> None:
    """Profiles a backend test-case model and writes ONNX/XLSX/PNG artifacts."""

    args = parse_args()

    onnx_model = get_big_qwen3_test_case_model()
    filename = f"{args.output_prefix}.onnx"

    print("-- saves the onnx model")
    ol_save(onnx_model, filename, save_as_external_data=True)
    onnx_model = ol_load(filename, load_external_data=True)
    onnx_model = inliner.inline_local_functions(onnx_model)
    del onnx_model.graph.value_info[:]

    print("-- infer shapes")
    shape_context = ShapesContext()
    compute_shape_model(shape_context, onnx_model)
    apply_inferred_shapes_to_model(shape_context, onnx_model)
    print("-- saves the model again")
    ol_save(onnx_model, filename, save_as_external_data=True)

    print("-- compute value and node tags")
    compute_context = ComputeContext()
    value_tags, _ = compute_context.compute_value_and_node_tags(onnx_model.graph, verbose=10)
    print("-- compute inplace")
    compute_context.compute_inplace_reuse_graph(
        onnx_model.graph, shape_context, value_tags=value_tags, verbose=10
    )

    print("-- create export")
    assignment = {
        "batch_size": args.batch,
        "sequence_length": args.sequence_length,
        "past_sequence_length": args.past_sequence_length,
        "total_sequence_length": args.sequence_length + args.past_sequence_length,
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
    memory_df = pandas.DataFrame(memory_rows)
    memory_df.to_excel(f"{args.output_prefix}.xlsx", index=False)
    with pandas.option_context("display.max_rows", None, "display.max_columns", None):
        print(memory_df.to_string(index=False))

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
    ax.set_title(f"ComputeContext total bytes ({TEST_CASE_NAME})")
    ax.set_xlabel("node index")
    ax.set_ylabel("total bytes")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.output_prefix}.png")


if __name__ == "__main__":
    main()
