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
    ``--shape-tag``
        After shape inference, infer semantic ``shape``/``axes``/``weight``
        tags for every value and node in the graph and record them in
        ``metadata_props`` (keys ``onnx_light.value_tags`` and
        ``onnx_light.node_tag``).
    ``--show``
        Print the inferred shapes to stdout; do **not** save the model.
    ``--verbose [LEVEL]``
        Print shape-inference progress information. With no level, defaults
        to ``1`` (summary). With ``2``, also prints per-event details.

run
    Generates random inputs and runs a model through the onnx-light runtime.

    Usage::

        python -m onnx_light run model.onnx [options]

    Options:

    ``--dim NAME=VALUE``
        Override a symbolic (dynamic) dimension by name.  May be specified
        multiple times.  For example, ``--dim batch=4 --dim seq=16`` sets the
        symbolic dims named ``batch`` and ``seq`` to 4 and 16 respectively.
        Dimensions that are neither concrete nor covered by ``--dim`` fall
        back to 1.
    ``--seed SEED``
        Integer seed for the deterministic pseudo-random input generator.
        Defaults to 0.
    ``--verbose [LEVEL]`` / ``-v [LEVEL]``
        Print run progress information. With no level, defaults to ``1``
        (summary: loading, input shapes, output shapes). With ``2``, also
        prints per-node execution details.
    ``--dump PATH``
        Write all inputs and outputs to *PATH* as an ONNX model whose
        ``graph.initializer`` list contains one ``TensorProto`` per input
        and per output tensor (in that order).  Only ``numpy.ndarray``
        values are stored; non-array outputs (e.g. sequences) are skipped.
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .onnx_proto._helper import TypeProto as _TypeProto

# Action string used in RuntimeEvent records for node-dispatch events.
_EVENT_ACTION_RUN_NODE = "run_node"


def _print_shape_inference_events(events: list) -> None:
    """Prints a compact summary of shape-inference events."""
    print(f"[fillshape] shape inference events: {len(events)}")


def _print_shape_inference_events_detailed(events: list) -> None:
    """Prints detailed shape-inference events."""
    for ev in events:
        d = ev.as_dict()
        op = f"{d['op_domain']}::{d['op_type']}" if d["op_type"] else "-"
        print(
            f"[fillshape] node={d['node_index']:<3d} "
            f"action={d['action']:<12s} op={op:<20s} "
            f"name={d['name'] or '-':<16s} shape={d['shape']}"
        )


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
        write_value_and_node_tags_to_metadata,
    )
    from .tools.pretty_print import pretty_onnx

    model_path: str = args.model
    output_path: str | None = args.output
    keep: bool = args.keep
    inplace_info: bool = args.inplace_info
    shape_tag: bool = args.shape_tag
    show: bool = args.show
    verbose: int = args.verbose

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path!r}")

    # Load without fetching external tensor bytes – shape inference only
    # needs type/shape metadata, not the actual weight values.
    model = load(model_path, load_external_data=False)

    # Detect whether the model references weights stored in a separate file.
    has_external_data = any(uses_external_data(init) for init in model.graph.initializer)

    if inplace_info or verbose > 0:
        # Retains the ShapesContext so in-place reuse analysis and verbose
        # event logging can reuse the already-inferred shape data.
        ctx = ShapesContext()
        ctx.events_enabled = verbose > 0
        compute_shape_model(ctx, model, keep)
        apply_inferred_shapes_to_model(ctx, model)
        if verbose > 0:
            events = ctx.events()
            _print_shape_inference_events(events)
            if verbose >= 2:
                _print_shape_inference_events_detailed(events)
        if inplace_info:
            write_inplace_reuse_to_metadata(ctx, model.graph)
    else:
        infer_shapes_model(model, prefill_with_value_info_output=keep)

    if shape_tag:
        write_value_and_node_tags_to_metadata(model.graph)

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


def _resolve_input_shape(
    vi_type: _TypeProto | Any, dim_overrides: dict[str, int], input_name: str
) -> list[int]:
    """Resolves the concrete shape for a tensor graph input.

    Args:
        vi_type: The ``TypeProto`` of the ``ValueInfoProto`` for this input.
        dim_overrides: Mapping from symbolic dim parameter name to concrete
            integer value supplied via ``--dim``.
        input_name: Input name, used only for diagnostic messages.

    Returns:
        List of non-negative integer dimension sizes.

    Raises:
        ValueError: When the type is not a tensor type or a dimension cannot
            be resolved.
    """
    if not vi_type.has_tensor_type():
        raise ValueError(
            f"Input {input_name!r} is not a tensor type; only tensor inputs are "
            "supported by the run subcommand."
        )
    tensor_type = vi_type.tensor_type
    if not tensor_type.has_shape():
        raise ValueError(
            f"Input {input_name!r} has no shape information. "
            "Use --dim to specify the size of each dynamic dimension."
        )
    shape: list[int] = []
    for dim in tensor_type.shape.dim:
        if dim.has_dim_value() and dim.dim_value > 0:
            shape.append(int(dim.dim_value))
        elif dim.has_dim_param() and dim.dim_param:
            param = dim.dim_param
            if param in dim_overrides:
                shape.append(dim_overrides[param])
            else:
                # Fall back to 1 for unspecified symbolic dims.
                shape.append(1)
        else:
            # Completely unknown dimension (0 or no annotation).
            shape.append(1)
    return shape


def _make_random_input(elem_type: int, shape: list[int], seed: int) -> object:
    """Creates a random NumPy array for an ONNX tensor input.

    Uses the same deterministic pseudo-random generators as the onnx-light
    runtime (``onnx_light.onnx_lib.backend.random``).

    Args:
        elem_type: ``TensorProto`` data-type integer (e.g.
            ``TensorProto.FLOAT``).
        shape: Concrete non-negative integer dimensions.
        seed: Integer seed forwarded to the random generator.

    Returns:
        A ``numpy.ndarray`` of the appropriate dtype and shape.

    Raises:
        NotImplementedError: For unsupported element types (e.g. STRING).
    """
    import numpy as np

    from .onnx_lib.backend.random import rand, randint
    from .onnx_proto._helper import tensor_dtype_to_np_dtype

    try:
        from .onnx_py._onnxpyprotoop import TensorProto  # type: ignore[attr-defined]
    except ImportError:
        from .onnx import TensorProto  # type: ignore[assignment]

    _FLOAT_TYPES = frozenset(
        {
            int(TensorProto.FLOAT),
            int(TensorProto.DOUBLE),
            int(TensorProto.FLOAT16),
            int(TensorProto.BFLOAT16),
            int(TensorProto.FLOAT8E4M3FN),
            int(TensorProto.FLOAT8E4M3FNUZ),
            int(TensorProto.FLOAT8E5M2),
            int(TensorProto.FLOAT8E5M2FNUZ),
            int(TensorProto.FLOAT8E8M0),
            int(TensorProto.FLOAT4E2M1),
            int(TensorProto.COMPLEX64),
            int(TensorProto.COMPLEX128),
        }
    )
    _INT_TYPES = frozenset(
        {
            int(TensorProto.INT8),
            int(TensorProto.INT16),
            int(TensorProto.INT32),
            int(TensorProto.INT64),
            int(TensorProto.UINT8),
            int(TensorProto.UINT16),
            int(TensorProto.UINT32),
            int(TensorProto.UINT64),
            int(TensorProto.INT4),
            int(TensorProto.UINT4),
            int(TensorProto.INT2),
            int(TensorProto.UINT2),
        }
    )

    np_dtype = tensor_dtype_to_np_dtype(elem_type)

    if elem_type == int(TensorProto.BOOL):
        values = randint(0, 2, size=shape, seed=seed, dtype=np.int32)
        return values.astype(bool)

    if elem_type in _FLOAT_TYPES:
        if elem_type in (int(TensorProto.COMPLEX64), int(TensorProto.COMPLEX128)):
            real = rand(*shape, seed=seed)
            imag = rand(*shape, seed=seed + 1)
            return (real + 1j * imag).astype(np_dtype)
        values = rand(*shape, seed=seed)
        return values.astype(np_dtype)

    if elem_type in _INT_TYPES:
        return randint(0, 10, size=shape, seed=seed, dtype=np_dtype)

    if elem_type == int(TensorProto.STRING):
        raise NotImplementedError(
            "STRING inputs are not supported by the run subcommand's random input generator."
        )

    raise NotImplementedError(
        f"Unsupported element type {elem_type} for random input generation."
    )


def _dump_tensors_as_model(
    tensors: dict[str, object], dump_path: str, *, ir_version: int = 8
) -> None:
    """Writes *tensors* to *dump_path* as an ONNX model with initializers.

    Creates a ``ModelProto`` whose ``graph.initializer`` list contains one
    ``TensorProto`` per entry in *tensors*.  Silently skips non-``numpy.ndarray``
    values.

    Args:
        tensors: Ordered mapping from tensor name to value.  Skips entries
            that are not ``numpy.ndarray``.
        dump_path: Filesystem path where this function writes the resulting
            ``.onnx`` file.
        ir_version: IR version to set on the model (default: 8).
    """
    import numpy as np

    from .onnx import save
    from .onnx.helper import make_graph, make_model, make_opsetid
    from .onnx.numpy_helper import from_array

    initializers = [
        from_array(value, name=name)
        for name, value in tensors.items()
        if isinstance(value, np.ndarray)
    ]
    graph = make_graph([], "dump", [], [], initializer=initializers)
    model = make_model(graph, opset_imports=[make_opsetid("", 21)])
    model.ir_version = ir_version
    save(model, dump_path)


def _cmd_run(args: argparse.Namespace) -> None:
    """Implements the ``run`` subcommand."""
    import numpy as np

    from .onnx import load
    from .onnx.reference import ReferenceEvaluator

    model_path: str = args.model
    dim_overrides: dict[str, int] = {}
    for entry in args.dim or []:
        if "=" not in entry:
            raise ValueError(
                f"--dim value {entry!r} must be in NAME=VALUE format (e.g. --dim batch=4)."
            )
        name, _, value_str = entry.partition("=")
        dim_overrides[name.strip()] = int(value_str.strip())
    seed: int = args.seed
    verbose: int = args.verbose
    dump_path: str | None = args.dump

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path!r}")

    if verbose >= 1:
        print(f"[run] loading {model_path!r}")
    model = load(model_path)

    initializer_names = {init.name for init in model.graph.initializer}

    # Collect every symbolic dim_param used across all non-initializer inputs.
    used_dim_params: set[str] = set()
    for vi in model.graph.input:
        if vi.name in initializer_names:
            continue
        if vi.type.has_tensor_type() and vi.type.tensor_type.has_shape():
            for dim in vi.type.tensor_type.shape.dim:
                if dim.has_dim_param() and dim.dim_param:
                    used_dim_params.add(dim.dim_param)

    # Warn about --dim overrides that do not match any symbolic dim in the inputs.
    for dim_name in dim_overrides:
        if dim_name not in used_dim_params:
            warnings.warn(
                f"--dim {dim_name!r} does not match any symbolic dimension in the model inputs.",
                UserWarning,
                stacklevel=2,
            )

    feed: dict[str, object] = {}
    for vi in model.graph.input:
        if vi.name in initializer_names:
            continue
        elem_type = int(vi.type.tensor_type.elem_type)
        shape = _resolve_input_shape(vi.type, dim_overrides, vi.name)
        tensor = _make_random_input(elem_type, shape, seed)
        feed[vi.name] = tensor
        if verbose >= 1 and isinstance(tensor, np.ndarray):
            print(f"[run]   input {vi.name!r}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

    if verbose >= 1:
        print("[run] running inference")
    sess = ReferenceEvaluator(model, events_enabled=(verbose >= 2))
    outputs = sess.run(None, feed)

    if verbose >= 2:
        # Events include tensor add/replace records as well as node-dispatch
        # ("run_node") records. Only run_node events are shown here because
        # they identify which operator was executed and at which graph position.
        for ev in sess.events():
            d = ev.as_dict()
            if d["action"] == _EVENT_ACTION_RUN_NODE:
                op = f"{d['op_domain']}::{d['op_type']}" if d["op_domain"] else d["op_type"]
                print(f"[run]   node={d['node_index']:<3d} op={op}")

    for name, value in zip(sess.output_names, outputs):
        if isinstance(value, np.ndarray):
            print(f"[run] output {name!r}: shape={list(value.shape)}, dtype={value.dtype}")
            if verbose >= 1:
                print(f"[run]   values: {value}")
        elif isinstance(value, list):
            print(f"[run] output {name!r}: sequence of {len(value)} element(s)")
            if verbose >= 1:
                for i, elem in enumerate(value):
                    print(f"[run]   [{i}]: {elem}")
        else:
            print(f"[run] output {name!r}: {value!r}")

    if dump_path is not None:
        all_tensors: dict[str, object] = dict(feed)
        for name, value in zip(sess.output_names, outputs):
            all_tensors[name] = value
        _dump_tensors_as_model(all_tensors, dump_path)


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
        "--shape-tag",
        action="store_true",
        default=False,
        dest="shape_tag",
        help=(
            "After shape inference, infer semantic shape/axes/weight tags for "
            "every value and node in the graph and record them in metadata_props "
            "(keys ``onnx_light.value_tags`` and ``onnx_light.node_tag``)."
        ),
    )
    fillshape_parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Print inferred shapes to stdout; do not save the model.",
    )
    fillshape_parser.add_argument(
        "--verbose",
        nargs="?",
        const=1,
        default=0,
        metavar="LEVEL",
        type=int,
        help=(
            "Print shape-inference progress; default level is 1 when the option "
            "is present. Level 2 also prints per-event details."
        ),
    )
    fillshape_parser.set_defaults(func=_cmd_fillshape)

    # --- run -----------------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Generate random inputs and run a model.",
        description=(
            "Generates random inputs for every graph input using the onnx-light "
            "deterministic pseudo-random generators and runs the model through "
            "the onnx-light runtime."
        ),
    )
    run_parser.add_argument("model", help="Path to the input ONNX model file.")
    run_parser.add_argument(
        "--dim",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help=(
            "Override a symbolic (dynamic) dimension by name. "
            "May be given multiple times (e.g. --dim batch=4 --dim seq=16). "
            "Dimensions that have no concrete value and are not covered by --dim "
            "default to 1."
        ),
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="SEED",
        help="Integer seed for the random input generator (default: 0).",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        nargs="?",
        const=1,
        default=0,
        metavar="LEVEL",
        type=int,
        help=(
            "Print run progress; default level is 1 when the option is present. "
            "Level 1 prints loading progress, input shapes and output shapes. "
            "Level 2 also prints per-node execution details."
        ),
    )
    run_parser.add_argument(
        "--dump",
        default=None,
        metavar="PATH",
        help=(
            "Write all inputs and outputs to PATH as an ONNX model whose "
            "graph.initializer contains one TensorProto per tensor. "
            "Non-array outputs (e.g. sequences) are skipped."
        ),
    )
    run_parser.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parses *argv* and dispatches to the appropriate subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
