"""Discovery of ONNX backend node tests as on-disk model directories.

Recent versions of ``onnx`` build the node backend tests entirely in memory:
:func:`onnx.backend.test.loader.load_model_tests` returns
:class:`~onnx.backend.test.case.test_case.TestCase` objects whose ``model_dir``
is ``None`` while the ``ModelProto`` and the input/output values live in the
``model`` and ``data_sets`` attributes. Older versions materialised the same
cases under ``onnx/backend/test/data/node`` and exposed them through
``model_dir``.

Both ``onnx`` vs ``onnx_light`` backend test modules were written against the
on-disk layout (a ``model.onnx`` file next to ``test_data_set_*`` directories of
serialised ``TensorProto`` / ``SequenceProto`` / ``OptionalProto`` values). This
module bridges the two: :func:`iter_node_model_dirs` yields ``(name,
model_dir)`` pairs, materialising the in-memory cases into a temporary directory
in that exact layout so the callers keep reading models and data from disk.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.backend.test.loader import load_model_tests

# In-memory node test cases are materialised under this directory, created once
# per process and reused across modules and test methods. It is intentionally
# not cleaned up: the process-wide temporary directory is reclaimed by the OS.
_MATERIALISED_ROOT: str | None = None


def _materialised_root() -> str:
    global _MATERIALISED_ROOT
    if _MATERIALISED_ROOT is None:
        _MATERIALISED_ROOT = tempfile.mkdtemp(prefix="onnxl_backend_node_")
    return _MATERIALISED_ROOT


def _value_info_kind(value_info) -> str:
    """Returns the boundary kind of one graph input/output value info.

    One of ``tensor``, ``sequence``, ``optional_tensor`` or
    ``optional_sequence``, matching the on-disk serialisation used for the
    corresponding value.
    """
    type_proto = value_info.type
    which = type_proto.WhichOneof("value")
    if which == "tensor_type":
        return "tensor"
    if which == "sequence_type":
        return "sequence"
    if which == "optional_type":
        elem = type_proto.optional_type.elem_type
        if elem.WhichOneof("value") == "sequence_type":
            return "optional_sequence"
        return "optional_tensor"
    raise AssertionError(f"Unexpected value info type {which!r}.")


def _tensor_proto(value: object) -> onnx.TensorProto:
    """Coerces a tensor-like backend value to a ``TensorProto``."""
    if isinstance(value, onnx.TensorProto):
        return value
    return numpy_helper.from_array(np.asarray(value))


def _sequence_proto(value) -> onnx.SequenceProto:
    """Coerces a list of tensor-like backend values to a ``SequenceProto``."""
    sequence = onnx.SequenceProto()
    sequence.elem_type = onnx.SequenceProto.TENSOR
    for element in value:
        sequence.tensor_values.add().CopyFrom(_tensor_proto(element))
    return sequence


def _write_value(path: str, kind: str, value: object) -> None:
    """Serialises one backend input/output value to ``path``.

    Mirrors the on-disk format produced by older ``onnx`` releases: the value is
    written as a ``TensorProto``, ``SequenceProto`` or ``OptionalProto`` (an
    empty optional being ``None``) according to the declared boundary ``kind``.
    """
    if kind == "tensor":
        payload = _tensor_proto(value).SerializeToString()
    elif kind == "sequence":
        payload = _sequence_proto(value).SerializeToString()
    elif kind in ("optional_tensor", "optional_sequence"):
        optional = onnx.OptionalProto()
        if value is None:
            optional.elem_type = onnx.OptionalProto.UNDEFINED
        elif kind == "optional_tensor":
            optional.elem_type = onnx.OptionalProto.TENSOR
            optional.tensor_value.CopyFrom(_tensor_proto(value))
        else:
            optional.elem_type = onnx.OptionalProto.SEQUENCE
            optional.sequence_value.CopyFrom(_sequence_proto(value))
        payload = optional.SerializeToString()
    else:
        raise AssertionError(f"Unexpected backend value kind {kind!r}.")
    with open(path, "wb") as f:
        f.write(payload)


def _materialise(test) -> str:
    """Writes an in-memory node ``TestCase`` to disk and returns its directory."""
    model_dir = os.path.join(_materialised_root(), test.name)
    model_file = os.path.join(model_dir, "model.onnx")
    if os.path.exists(model_file):
        return model_dir
    os.makedirs(model_dir, exist_ok=True)
    onnx.save_model(test.model, model_file)
    input_kinds = [_value_info_kind(vi) for vi in test.model.graph.input]
    output_kinds = [_value_info_kind(vi) for vi in test.model.graph.output]
    for set_index, (inputs, outputs) in enumerate(test.data_sets or []):
        data_dir = os.path.join(model_dir, f"test_data_set_{set_index}")
        os.makedirs(data_dir, exist_ok=True)
        for i, value in enumerate(inputs):
            _write_value(os.path.join(data_dir, f"input_{i}.pb"), input_kinds[i], value)
        for i, value in enumerate(outputs):
            _write_value(os.path.join(data_dir, f"output_{i}.pb"), output_kinds[i], value)
    return model_dir


def iter_node_model_dirs() -> Iterator[tuple[str, str]]:
    """Yields ``(name, model_dir)`` for every ONNX backend node test.

    ``name`` is the test's directory basename (e.g. ``test_abs``) and
    ``model_dir`` a directory containing ``model.onnx`` and, when available,
    ``test_data_set_*`` subdirectories. In-memory test cases are materialised on
    disk on first use; on-disk test cases are yielded as-is.
    """
    for test in load_model_tests(kind="node"):
        if test.model_dir is not None:
            model_dir = test.model_dir
            if not os.path.exists(os.path.join(model_dir, "model.onnx")):
                continue
            name = os.path.basename(model_dir)
        elif test.model is not None:
            model_dir = _materialise(test)
            name = test.name
        else:
            continue
        yield name, model_dir
