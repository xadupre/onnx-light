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
        When the model stores weights in a separate file (external data), the
        output is placed in the **same directory as the input model** so that
        the relative weight-file paths encoded in the proto remain valid.  The
        weight file itself is never written again.
    ``--keep``
        Seed the inference context from any shapes already present in
        ``graph.value_info`` / ``graph.output`` (i.e. call
        ``infer_shapes_model`` with ``prefill_with_value_info_output=True``).
        Existing non-conflicting shapes are preferred over newly inferred
        ones.
    ``--inplace-info``
        After shape inference, also compute in-place buffer-reuse
        opportunities and write them into each node's ``metadata_props``
        under the key ``onnx_light.inplace_reuse``.
    ``--show``
        Print the inferred shapes to stdout; do **not** save the model.
"""

from __future__ import annotations

import argparse
import os


def _cmd_fillshape(args: argparse.Namespace) -> None:
    """Implements the ``fillshape`` subcommand."""
    from .onnx import load, save
    from .onnx_lib.external_data_helper import uses_external_data
    from .onnx_optim.shape_inference import (
        ShapesContext,
        apply_inferred_shapes_to_model,
        compute_shape_model,
        infer_shapes_model,
        write_inplace_reuse_to_metadata,
    )
    from .tools.pretty_print import pretty_onnx

    model_path: str = args.model
    output_path: str | None = args.output
    keep: bool = args.keep
    inplace_info: bool = args.inplace_info
    show: bool = args.show

    # Load without fetching external tensor bytes – shape inference only
    # needs type/shape metadata, not the actual weight values.
    model = load(model_path, load_external_data=False)

    # Detect whether the model references weights stored in a separate file.
    has_external_data = any(uses_external_data(init) for init in model.graph.initializer)

    if inplace_info:
        # Retain the ShapesContext so the in-place reuse analysis can
        # reuse the already-inferred shape data.
        ctx = ShapesContext()
        compute_shape_model(ctx, model, keep)
        apply_inferred_shapes_to_model(ctx, model)
        write_inplace_reuse_to_metadata(ctx, model.graph)
    else:
        infer_shapes_model(model, prefill_with_value_info_output=keep)

    if show:
        print(pretty_onnx(model))
        return

    if output_path is not None and has_external_data:
        # The weight-file paths encoded in the proto are relative to the model
        # directory. Place the output file next to the original model (beside
        # the existing weight files) so those paths remain valid. The weight
        # files are NOT written again.
        model_dir = os.path.dirname(os.path.abspath(model_path))
        dest = os.path.join(model_dir, os.path.basename(output_path))
    elif output_path is not None:
        dest = output_path
    else:
        dest = model_path

    save(model, dest)


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
        "--inplace-info",
        action="store_true",
        default=False,
        dest="inplace_info",
        help=(
            "Also compute in-place buffer-reuse opportunities and record them "
            "in each node's metadata_props under the key "
            "``onnx_light.inplace_reuse``."
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


def main(argv: list[str] | None = None) -> None:
    """Parses *argv* and dispatches to the appropriate subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
