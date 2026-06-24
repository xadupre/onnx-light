"""Entry point for ``python -m onnx_light``.

Supported subcommands
---------------------
fillshape
    Fills a model's ``graph.value_info`` and ``graph.output`` with the shapes
    inferred by ``onnx_optim`` shape inference.

    Usage::

        python -m onnx_light fillshape model.onnx [options]

    Options:

    ``--output OUTPUT`` / ``-o OUTPUT``
        Write the result to *OUTPUT* instead of overwriting the input file.
    ``--keep``
        Seed the inference context from any shapes already present in
        ``graph.value_info`` / ``graph.output`` (i.e. call
        ``infer_shapes_model`` with ``prefill_with_value_info_output=True``).
        Existing non-conflicting shapes are preferred over newly inferred
        ones.
    ``--show``
        Print the inferred shapes to stdout; do **not** save the model.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_fillshape(args: argparse.Namespace) -> int:
    """Implements the ``fillshape`` subcommand.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    from .onnx import load, save
    from .onnx_optim.shape_inference import infer_shapes_model
    from .tools.pretty_print import pretty_onnx

    model_path: str = args.model
    output_path: str | None = args.output
    keep: bool = args.keep
    show: bool = args.show

    try:
        model = load(model_path)
    except Exception as exc:
        print(f"error: could not load {model_path!r}: {exc}", file=sys.stderr)
        return 1

    try:
        infer_shapes_model(model, prefill_with_value_info_output=keep)
    except Exception as exc:
        print(f"error: shape inference failed: {exc}", file=sys.stderr)
        return 1

    if show:
        print(pretty_onnx(model))
        return 0

    dest = output_path if output_path is not None else model_path
    try:
        save(model, dest)
    except Exception as exc:
        print(f"error: could not save model to {dest!r}: {exc}", file=sys.stderr)
        return 1

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m onnx_light", description="onnx-light command-line utilities."
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # --- fillshape -----------------------------------------------------------
    fillshape_parser = subparsers.add_parser(
        "fillshape",
        help="Fill a model with optimized shape-inference information.",
        description=(
            "Runs onnx_optim shape inference on a model and writes the inferred "
            "shapes back into graph.value_info and graph.output."
        ),
    )
    fillshape_parser.add_argument("model", help="Path to the input ONNX model file.")
    fillshape_parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="OUTPUT",
        help="Path to write the result to. When omitted the input file is overwritten in place.",
    )
    fillshape_parser.add_argument(
        "--keep",
        action="store_true",
        default=False,
        help=(
            "Seed the inference context from shapes already present in the model. "
            "Existing non-conflicting shapes are kept as anchors."
        ),
    )
    fillshape_parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Print inferred shapes to stdout; do not save the model.",
    )
    fillshape_parser.set_defaults(func=_cmd_fillshape)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parses *argv* and dispatches to the appropriate subcommand.

    Returns:
        Exit code: 0 on success, non-zero on failure.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
