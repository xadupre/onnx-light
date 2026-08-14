"""Entry point for ``python -m onnx_light``.

Supported subcommands
---------------------
fillshape
    Fills a model's ``graph.value_info`` and ``graph.output`` with the shapes
    inferred by ``onnx_shapes`` shape inference.

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
    ``--release-info``
        After shape inference, compute last-use release hints and write
        them into each node's ``metadata_props`` under the key
        ``onnx_light.release_after``.
    ``--shape-tag``
        After shape inference, infer semantic ``shape``/``axes``/``weight``/``ambiguous``
        tags for every value and node in the graph and record them in
        ``metadata_props`` (per-value key ``onnx_light.value_tag`` and
        per-node key ``onnx_light.node_tag``).
    ``--token NAME=LOW:HIGH``
        Bind a symbolic dimension token to an inclusive integer range before
        running shape inference.  May be specified multiple times.
        For example, ``--token seq=1:128`` treats ``seq`` as having range
        ``[1, 128]`` (the lower bound is used for shape propagation).
        Symbolic dims not covered by ``--token`` remain symbolic.
    ``--show``
        Print the inferred shapes to stdout; do **not** save the model.
    ``--verbose [LEVEL]``
        Print shape-inference progress information. With no level, defaults
        to ``1`` (summary). With ``2``, also prints per-event details.

show
    Prints a human-readable, Mermaid or SVG rendering of an ONNX model.

    Usage::

        python -m onnx_light show model.onnx [options]

    Options:

    ``--format FORMAT`` / ``-f FORMAT``
        Output format: ``pretty`` (default), ``mermaid`` or ``svg``.
    ``--output OUTPUT`` / ``-o OUTPUT``
        Write the rendered text to *OUTPUT* instead of printing to stdout.
    ``--shape-inference``
        Run onnx-light shape inference on the model before rendering.
    ``--include-shapes / --no-shapes``
        Include (or suppress) shape annotations in the rendered output.
        Enabled by default; only has effect for the ``mermaid`` and ``svg``
        formats.
    ``--include-attributes``
        Include node attributes in the rendered output.
    ``--include-inplace``
        Show in-place buffer-reuse annotations (``onnx_light.inplace_reuse``
        metadata).
    ``--include-release``
        Show post-execution release hints (``onnx_light.release_after``
        metadata).  Only used by the ``pretty`` format.
    ``--include-node-tags``
        Show semantic ``shape``/``axes``/``weight``/``ambiguous`` node-tag annotations
        (``onnx_light.node_tag`` metadata).  Only used by the ``pretty``
        format.
    ``--no-initializers``
        Exclude initializer nodes from the rendered graph.  Only used by the
        ``mermaid`` and ``svg`` formats.
    ``--direction DIRECTION``
        Flowchart direction for the ``mermaid`` and ``svg`` formats.
        One of ``TB`` (top-to-bottom, default), ``LR`` (left-to-right),
        ``TD`` or ``BT`` (``mermaid`` only).
    ``--layout LAYOUT``
        Box positioning strategy for the ``svg`` format.  One of
        ``layered`` (default) or ``umap`` (requires ``umap-learn``).

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

tune-kernels
    Proposes calibration updates for tuning keys missing from the local cache.
    It is read-only unless ``--apply`` is specified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .onnx_proto._helper import TypeProto as _TypeProto

# Action string used in RuntimeEvent records for node-dispatch events.
_EVENT_ACTION_RUN_NODE = "run_node"

# Maximum byte size of an external-data tensor that is loaded inline for shape
# inference.  Tensors at or above this threshold remain as external-data
# references and are not read from disk.  Tensors below the threshold are
# loaded because shape inference may need their values (e.g. the ``shape``
# input of a Reshape node).
_FILLSHAPE_TINY_TENSOR_THRESHOLD = 128


def _parse_kernel_element_type(value: str) -> int:
    """Parses an ONNX element-type integer or enum name."""
    try:
        return int(value)
    except ValueError:
        from .onnx import TensorProto

        name = value.upper()
        if not hasattr(TensorProto, name):
            raise argparse.ArgumentTypeError(
                f"unknown ONNX element type {value!r}; use an integer or name such as FLOAT"
            ) from None
        return int(getattr(TensorProto, name))


def _cmd_tune_kernels(args: argparse.Namespace) -> None:
    """Proposes or applies local kernel tuning cache updates."""
    from . import kernel_tuning

    common = {
        "kernels": args.kernel,
        "element_types": args.element_type,
        "library": args.library,
        "implementation": args.implementation,
        "path": args.cache,
    }
    if args.apply:
        report = kernel_tuning.apply_kernel_tuning_updates(
            **common,
            maximum_duration_ms=args.maximum_duration_ms,
            maximum_memory_bytes=args.maximum_memory_mb << 20,
        )
        summary = report["remaining"]
    else:
        report = kernel_tuning.propose_kernel_tuning_updates(**common)
        summary = report

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    action = "remaining" if args.apply else "proposed"
    print(
        f"cache={summary['cache_path']} selected={summary['selected']} "
        f"covered={summary['covered']} missing={len(summary['missing'])}"
    )
    print(f"{action} calibrations: {len(summary['calibratable'])}")
    for item in summary["calibratable"]:
        print(
            f"  {item['library']}/{item['kernel']}/{item['implementation']} "
            f"dtype={item['element_type']} abi={item['tuning_abi']}"
        )
    if summary["unsupported"]:
        print(f"missing without calibration callback: {len(summary['unsupported'])}")
        for item in summary["unsupported"]:
            print(
                f"  {item['library']}/{item['kernel']}/{item['implementation']} "
                f"dtype={item['element_type']} abi={item['tuning_abi']}"
            )


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


def _parse_token_spec(token_str: str) -> tuple[str, int, int]:
    """Parses a ``--token`` specification into ``(name, low, high)``.

    The only accepted format is ``NAME=LOW:HIGH`` — an inclusive range where
    ``LOW <= HIGH``.

    Args:
        token_str: The raw argument string, e.g. ``"seq=1:128"``.

    Returns:
        A ``(name, low, high)`` triple where *low* and *high* are
        non-negative integers.

    Raises:
        ValueError: When the format is invalid or the bounds are
            inconsistent.
    """
    _fmt_error = (
        f"--token value {token_str!r} must be in NAME=LOW:HIGH format "
        "(e.g. --token seq=1:128)."
    )
    if "=" not in token_str:
        raise ValueError(_fmt_error)
    name, _, range_str = token_str.partition("=")
    name = name.strip()
    range_str = range_str.strip()
    if ":" not in range_str:
        raise ValueError(_fmt_error)
    low_str, _, high_str = range_str.partition(":")
    low = int(low_str.strip())
    high = int(high_str.strip())
    if low > high:
        raise ValueError(f"--token {name!r}: lower bound {low} must be <= upper bound {high}.")
    return name, low, high


def _seed_context_with_token_ranges(
    ctx: Any, model: Any, token_ranges: dict[str, tuple[int, int]]
) -> None:
    """Seeds the ShapesContext with concrete dim values for symbolic tokens.

    For each graph input whose shape contains a symbolic dim that matches a
    key in *token_ranges*, that dim is replaced with the lower bound of the
    corresponding range before shape inference runs.  This makes the
    shape-inference engine propagate concrete integer shapes through the
    graph for those dimensions.

    Args:
        ctx: The :class:`ShapesContext` to seed.
        model: The ``ModelProto`` whose graph inputs are inspected.
        token_ranges: Mapping from symbolic dim name to ``(low, high)``
            where *low* is used as the concrete substitute value.
    """
    from .onnx_core.shape_inference import SymTensor

    initializer_names = {init.name for init in model.graph.initializer}
    for vi in model.graph.input:
        vi_name = vi.name
        if vi_name in initializer_names:
            continue
        if not vi.type.has_tensor_type():
            continue
        tensor_type = vi.type.tensor_type
        dtype = int(tensor_type.elem_type)
        if not tensor_type.has_shape():
            continue
        dims: list[int | str] = []
        for dim in tensor_type.shape.dim:
            if dim.has_dim_value() and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            elif dim.has_dim_param() and dim.dim_param in token_ranges:
                # Substitute the symbolic token with its lower bound so that
                # shape inference can propagate a concrete integer dimension.
                dims.append(token_ranges[dim.dim_param][0])
            elif dim.has_dim_param() and dim.dim_param:
                dims.append(str(dim.dim_param))
            else:
                # Fully dynamic (no name, no value): leave as 0.
                dims.append(0)
        ctx.set(vi_name, SymTensor(dtype, dims))


def _remove_node_metadata_key(graph: Any, key: str) -> None:
    """Removes a metadata key from all nodes in a graph and nested subgraphs."""

    for node in graph.node:
        props = node.metadata_props
        kept = [(entry.key, entry.value) for entry in props if entry.key != key]
        if len(kept) != len(props):
            del props[:]
            for k, v in kept:
                entry = props.add()
                entry.key = k
                entry.value = v
        for attr in node.attribute:
            if attr.has_g():
                _remove_node_metadata_key(attr.g, key)
            for subgraph in attr.graphs:
                _remove_node_metadata_key(subgraph, key)


def _cmd_fillshape(args: argparse.Namespace) -> None:
    """Implements the ``fillshape`` subcommand."""
    from .onnx import load, save
    from .onnx_lib.external_data_helper import uses_external_data
    from .onnx_core.shape_inference import (
        ComputeContext,
        Device,
        ShapesContext,
        apply_inferred_shapes_to_model,
        compute_shape_model,
        infer_shapes_model,
        write_peak_memory_to_metadata,
        write_value_and_node_tags_to_metadata,
    )
    from .tools.pretty_print import pretty_onnx

    model_path: str = args.model
    output_path: str | None = args.output
    keep: bool = args.keep
    inplace_info: bool = args.inplace_info
    release_info: bool = args.release_info
    peak_memory: bool = args.peak_memory
    shape_tag: bool = args.shape_tag
    show: bool = args.show
    verbose: int = args.verbose

    # Parse --token specifications into a name → (low, high) mapping.
    token_ranges: dict[str, tuple[int, int]] = {}
    for spec in args.token or []:
        name, low, high = _parse_token_spec(spec)
        token_ranges[name] = (low, high)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path!r}")

    if verbose:
        print(f"[fillshape] load {model_path!r}")

    # Load without fetching large external tensor bytes. Shape inference only
    # needs the values of small shape-driving tensors (e.g. Reshape's shape
    # input), so parsing inlines only tiny external tensors and leaves larger
    # weights as external references.
    model = load(
        model_path,
        load_external_data=False,
        tiny_external_data_threshold=_FILLSHAPE_TINY_TENSOR_THRESHOLD,
    )

    # Detect whether the model references weights stored in a separate file
    # (must happen before the tiny-tensor step, which inlines small tensors).
    has_external_data = any(uses_external_data(init) for init in model.graph.initializer)

    if inplace_info or release_info or peak_memory or verbose > 0 or token_ranges:
        # Retains the ShapesContext so in-place reuse analysis, peak memory
        # estimation, verbose event logging, and token-range substitution can
        # reuse the already-inferred shape data.
        ctx = ShapesContext()
        ctx.events_enabled = verbose > 0
        if token_ranges:
            # Pre-seed the context with concrete dim values for the specified
            # symbolic tokens so that shape inference propagates concrete
            # shapes for those dimensions.
            _seed_context_with_token_ranges(ctx, model, token_ranges)
        if verbose:
            print("[fillshape] shape inference")
        compute_shape_model(ctx, model, keep)
        apply_inferred_shapes_to_model(ctx, model)
        if verbose:
            events = ctx.events()
            _print_shape_inference_events(events)
            if verbose >= 2:
                _print_shape_inference_events_detailed(events)
        if inplace_info or release_info:
            what = (
                "inplace/release"
                if inplace_info and release_info
                else ("inplace" if inplace_info else "release")
            )
            if verbose:
                print(f"[fillshape] compute {what} info")
            inplace_context = ComputeContext()
            inplace_context.compute_inplace_reuse_graph(model.graph, ctx)
            if verbose:
                print(f"[fillshape] write {what} info in the model")
            inplace_context.write_to_metadata(model.graph)
            if release_info and not inplace_info:
                _remove_node_metadata_key(model.graph, "onnx_light.inplace_reuse")
        if peak_memory:
            if verbose:
                print("[fillshape] write peak memory info in the model")
            write_peak_memory_to_metadata(ctx, model.graph, Device.kUndefined)
    else:
        if verbose:
            print("[fillshape] shape inference only")
        infer_shapes_model(model, prefill_with_value_info_output=keep)

    if shape_tag:
        tag_context = ComputeContext()
        if verbose:
            print("[fillshape] compute shape tags")
        value_tags, node_tags = tag_context.compute_value_and_node_tags(model.graph)
        if verbose:
            tagged_nodes = sum(1 for tag in node_tags if tag)
            print(
                f"[fillshape] computed {len(value_tags)} value tag(s) "
                f"and {tagged_nodes} node tag(s)"
            )
        if verbose:
            print("[fillshape] write shape tags in the model")
        write_value_and_node_tags_to_metadata(model.graph)

    if show:
        print(
            pretty_onnx(
                model,
                include_inplace=inplace_info,
                include_release=(inplace_info or release_info),
                include_node_tags=shape_tag,
            )
        )
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

    if verbose:
        print(f"[fillshape] save into {dest!r}")
    save(model, dest)


def _cmd_show(args: argparse.Namespace) -> None:
    """Implements the ``show`` subcommand."""
    import sys

    from .onnx import load

    model_path: str = args.model
    fmt: str = args.format
    output_path: str | None = args.output
    run_shape_inference: bool = args.shape_inference
    include_shapes: bool = args.include_shapes
    include_attributes: bool = args.include_attributes
    include_inplace: bool = args.include_inplace
    include_release: bool = args.include_release
    include_node_tags: bool = args.include_node_tags
    include_initializers: bool = args.include_initializers
    direction: str = args.direction
    graphviz_format: str | None = args.graphviz

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path!r}")

    model = load(model_path, load_external_data=False)

    if run_shape_inference:
        from .onnx_core.shape_inference import (
            ShapesContext,
            apply_inferred_shapes_to_model,
            compute_shape_model,
        )

        ctx = ShapesContext()
        compute_shape_model(ctx, model)
        apply_inferred_shapes_to_model(ctx, model)

    if fmt == "pretty":
        from .tools.pretty_print import pretty_onnx

        text = pretty_onnx(
            model,
            with_attributes=include_attributes,
            include_node_tags=include_node_tags,
            include_inplace=include_inplace,
            include_release=include_release,
        )
    elif fmt == "mermaid":
        from .tools.mermaid import to_mermaid

        text = to_mermaid(
            model,
            direction=direction,
            include_initializers=include_initializers,
            include_shapes=include_shapes,
            include_attributes=include_attributes,
            include_inplace=include_inplace,
        )
    elif fmt == "svg":
        from .tools.svg import to_svg

        text = to_svg(
            model,
            direction=direction,
            layout=args.layout,
            include_initializers=include_initializers,
            include_shapes=include_shapes,
            include_attributes=include_attributes,
            include_inplace=include_inplace,
        )
    elif fmt == "dot":
        from .tools.dot import to_dot

        dot_text = to_dot(
            model,
            direction=direction,
            include_initializers=include_initializers,
            include_shapes=include_shapes,
            include_attributes=include_attributes,
            include_inplace=include_inplace,
        )

        if graphviz_format is not None:
            import subprocess

            # Validates the format string to prevent command injection.
            # Only allows alphanumeric characters, dots, underscores, and hyphens
            # which covers all legitimate Graphviz output formats (e.g. "png",
            # "svg", "pdf", "xdot1.2").
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", graphviz_format):
                raise ValueError(
                    f"Invalid Graphviz output format {graphviz_format!r}. "
                    "The format must start with a letter or digit and contain only "
                    "alphanumeric characters, dots, underscores, or hyphens "
                    "(e.g. 'png', 'svg', 'pdf', 'xdot1.2')."
                )

            result = subprocess.run(
                ["dot", f"-T{graphviz_format}"],
                input=dot_text.encode(),
                capture_output=True,
                check=True,
            )
            if output_path is not None:
                with open(output_path, "wb") as f:
                    f.write(result.stdout)
            else:
                sys.stdout.buffer.write(result.stdout)
            return

        text = dot_text
    else:
        # argparse enforces choices=['pretty', 'mermaid', 'svg', 'dot'], so
        # this branch is unreachable in normal usage.  It acts as a defensive
        # guard for programmatic callers that bypass the argument parser.
        raise ValueError(
            f"Unknown format {fmt!r}; expected one of 'pretty', 'mermaid', 'svg', 'dot'."
        )

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


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
    from .onnx.tools import make_random_input

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
        tensor = make_random_input(elem_type, shape, seed)
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
            "Runs onnx_shapes shape inference on a model and writes the inferred "
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
        "--release-info",
        action="store_true",
        default=False,
        dest="release_info",
        help=(
            "Also compute last-use release hints and record them in each "
            "node's metadata_props under the keys "
            "``onnx_light.release_after`` and ``onnx_light.not_used_after``."
        ),
    )
    fillshape_parser.add_argument(
        "--peak-memory",
        action="store_true",
        default=False,
        dest="peak_memory",
        help=(
            "Also compute the estimated peak scratch memory for each node and "
            "record it in each node's metadata_props under the key "
            "``onnx_light.peak_memory``. Only nodes with a registered "
            "peak-memory function and fully concrete input shapes produce a "
            "non-zero entry; all other nodes are left untouched."
        ),
    )
    fillshape_parser.add_argument(
        "--shape-tag",
        action="store_true",
        default=False,
        dest="shape_tag",
        help=(
            "After shape inference, infer semantic shape/axes/weight/ambiguous tags for "
            "every value and node in the graph and record them in metadata_props "
            "(per-value key ``onnx_light.value_tag`` and per-node key "
            "``onnx_light.node_tag``)."
        ),
    )
    fillshape_parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Print inferred shapes to stdout; do not save the model.",
    )
    fillshape_parser.add_argument(
        "--token",
        action="append",
        default=None,
        metavar="NAME=LOW:HIGH",
        help=(
            "Bind a symbolic dimension token to an inclusive integer range "
            "before running shape inference. May be given multiple times "
            "(e.g. --token batch=1:8 --token seq=1:128). "
            "The lower bound is used for shape propagation. "
            "Symbolic dims not covered by --token remain symbolic."
        ),
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

    # --- show ----------------------------------------------------------------
    show_parser = subparsers.add_parser(
        "show",
        help="Print a human-readable, Mermaid or SVG rendering of a model.",
        description=(
            "Loads an ONNX model and renders it as plain text (pretty), a Mermaid "
            "flowchart or an SVG image. The result is written to stdout by default."
        ),
    )
    show_parser.add_argument("model", help="Path to the input ONNX model file.")
    show_parser.add_argument(
        "--format",
        "-f",
        default="pretty",
        choices=["pretty", "mermaid", "svg", "dot"],
        dest="format",
        help=(
            "Output format: 'pretty' (default) for a compact text listing, "
            "'mermaid' for a Mermaid flowchart, 'svg' for an SVG image, "
            "or 'dot' for Graphviz DOT source."
        ),
    )
    show_parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="OUTPUT",
        help="Write the rendered output to OUTPUT instead of printing to stdout.",
    )
    show_parser.add_argument(
        "--shape-inference",
        action="store_true",
        default=False,
        dest="shape_inference",
        help="Run onnx-light shape inference on the model before rendering.",
    )
    show_parser.add_argument(
        "--no-shapes",
        action="store_false",
        default=True,
        dest="include_shapes",
        help=(
            "Suppress shape annotations in the rendered output "
            "(applies to 'mermaid' and 'svg' formats)."
        ),
    )
    show_parser.add_argument(
        "--include-attributes",
        action="store_true",
        default=False,
        dest="include_attributes",
        help="Include node attributes in the rendered output.",
    )
    show_parser.add_argument(
        "--include-inplace",
        action="store_true",
        default=False,
        dest="include_inplace",
        help="Show in-place buffer-reuse annotations (onnx_light.inplace_reuse metadata).",
    )
    show_parser.add_argument(
        "--include-release",
        action="store_true",
        default=False,
        dest="include_release",
        help=(
            "Show post-execution release hints "
            "(onnx_light.release_after and onnx_light.not_used_after metadata). "
            "Only used by the 'pretty' format."
        ),
    )
    show_parser.add_argument(
        "--include-node-tags",
        action="store_true",
        default=False,
        dest="include_node_tags",
        help=(
            "Show semantic shape/axes/weight/ambiguous node-tag annotations "
            "(onnx_light.node_tag metadata). Only used by the 'pretty' format."
        ),
    )
    show_parser.add_argument(
        "--no-initializers",
        action="store_false",
        default=True,
        dest="include_initializers",
        help=(
            "Exclude initializer nodes from the rendered graph "
            "(applies to 'mermaid', 'svg' and 'dot' formats)."
        ),
    )
    show_parser.add_argument(
        "--direction",
        default="TB",
        metavar="DIRECTION",
        help=(
            "Flowchart direction for 'mermaid', 'svg' and 'dot' formats. "
            "One of TB (default), LR, TD or BT (mermaid only)."
        ),
    )
    show_parser.add_argument(
        "--layout",
        default="layered",
        metavar="LAYOUT",
        choices=("layered", "umap"),
        help=(
            "Box positioning strategy for the 'svg' format. "
            "One of layered (default) or umap (requires the 'umap-learn' package)."
        ),
    )
    show_parser.add_argument(
        "--graphviz",
        default=None,
        metavar="GRAPHVIZ_FORMAT",
        dest="graphviz",
        help=(
            "Invoke the Graphviz 'dot' executable on the generated DOT source "
            "and write the result in GRAPHVIZ_FORMAT (e.g. 'png', 'svg', 'pdf'). "
            "Only used when --format dot is given. "
            "Requires Graphviz to be installed and 'dot' to be on PATH."
        ),
    )
    show_parser.set_defaults(func=_cmd_show)

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

    # --- tune-kernels --------------------------------------------------------
    tuning_parser = subparsers.add_parser(
        "tune-kernels",
        help="Propose calibration updates for locally uncovered kernel tuning keys.",
    )
    tuning_parser.add_argument(
        "--kernel",
        action="append",
        help="Restrict to a kernel name; may be specified multiple times.",
    )
    tuning_parser.add_argument(
        "--element-type",
        action="append",
        type=_parse_kernel_element_type,
        help="Restrict to an ONNX element-type integer or name; may be repeated.",
    )
    tuning_parser.add_argument(
        "--library", default="onnx_light", help="Tuning library identifier."
    )
    tuning_parser.add_argument("--implementation", help="Restrict to one implementation.")
    tuning_parser.add_argument("--cache", help="Use this cache instead of the platform default.")
    tuning_parser.add_argument(
        "--apply",
        action="store_true",
        help="Calibrate and persist proposed keys; without this flag the command is read-only.",
    )
    tuning_parser.add_argument(
        "--maximum-duration-ms",
        type=int,
        default=0,
        help="Per-key calibration duration budget; 0 uses the callback default.",
    )
    tuning_parser.add_argument(
        "--maximum-memory-mb",
        type=int,
        default=0,
        help="Calibration memory budget in MiB; 0 uses the callback default.",
    )
    tuning_parser.add_argument(
        "--json", action="store_true", help="Print the complete JSON report."
    )
    tuning_parser.set_defaults(func=_cmd_tune_kernels)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parses *argv* and dispatches to the appropriate subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
