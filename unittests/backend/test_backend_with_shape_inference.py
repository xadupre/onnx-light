import unittest

import onnx_light.onnx as onnxl
import onnx_light.onnx.shape_inference as shape_inference
from onnx_light.backend.test.case import make_test_class


def _dims_of(tt: onnxl.TypeProto.Tensor) -> list[int]:
    """Returns the dims of a tensor type as ``int``s, using ``-1`` for unknown
    (symbolic or unset) dimensions. Returns ``[]`` when no shape is set.
    """
    if not tt.has_shape():
        return []
    return [d.dim_value if d.has_dim_value() else -1 for d in tt.shape.dim]


def _snapshot_and_strip(value_infos):
    """For each ``ValueInfoProto`` in ``value_infos`` carrying a plain
    ``tensor_type``, records ``(name, elem_type, had_shape, shape)`` and
    rebuilds the entry in place with one that only carries ``name`` and
    ``elem_type`` (no shape) so :func:`shape_inference.infer_shapes` must
    recover it. Entries with non-tensor types (sequence/optional/map) are
    snapshotted (and serialized so they can be restored after clearing the
    repeated field) but the snapshot tuple is flagged so the post-check
    only verifies the entry is preserved by name.
    """
    # Snapshot first (the items become dangling once the repeated field is
    # cleared, since they are references into the underlying C++ storage).
    snapshot = []
    serialized: list[bytes | None] = []
    for vi in value_infos:
        name = vi.name
        is_plain_tensor = vi.has_type() and vi.type.has_tensor_type()
        if is_plain_tensor:
            tt = vi.type.tensor_type
            elem_type = int(tt.elem_type)
            had_shape = tt.has_shape()
            shape = _dims_of(tt)
            snapshot.append((name, elem_type, had_shape, shape, True))
            serialized.append(None)
        else:
            snapshot.append((name, 0, False, [], False))
            serialized.append(vi.SerializeToString())

    value_infos.clear()
    for (name, elem_type, _had_shape, _shape, is_plain_tensor), data in zip(snapshot, serialized):
        new_vi = value_infos.add()
        if is_plain_tensor:
            new_vi.name = name
            new_tt = new_vi.add_type().add_tensor_type()
            new_tt.elem_type = elem_type
        else:
            assert data is not None
            new_vi.ParseFromString(data)
    return snapshot


def _check_match(
    value_infos, snapshot, *, label: str, case_name: str, strict_shape: bool
) -> None:
    """Checks that each entry in ``snapshot`` is present in ``value_infos``
    after shape inference with the same elem_type and a compatible shape
    (concrete dims must match; symbolic / unknown dims are tolerated).

    When ``strict_shape`` is True, an entry that originally had a shape must
    have one after inference. When False, a missing shape is tolerated (some
    operators cannot recover output shapes without data propagation).
    """
    by_name = {vi.name: vi for vi in value_infos}
    for name, elem_type, had_shape, expected_shape, is_plain_tensor in snapshot:
        assert (
            name in by_name
        ), f"{label} {name!r} missing from graph after shape inference (case {case_name!r})"
        vi = by_name[name]
        assert vi.has_type(), f"{label} {name!r} missing type (case {case_name!r})"
        if not is_plain_tensor:
            # Non-tensor types are passed through unchanged; nothing more to
            # check against the snapshot.
            continue
        assert (
            vi.type.has_tensor_type()
        ), f"{label} {name!r} lost its tensor_type after inference (case {case_name!r})"
        tt = vi.type.tensor_type
        assert int(tt.elem_type) == elem_type, (
            f"elem_type mismatch on {label} {name!r} (case {case_name!r}): "
            f"got {int(tt.elem_type)}, expected {elem_type}"
        )
        if not tt.has_shape():
            if had_shape and strict_shape:
                raise AssertionError(
                    f"{label} {name!r} has no shape after inference "
                    f"(expected rank {len(expected_shape)}, case {case_name!r})"
                )
            continue
        inferred = _dims_of(tt)
        if had_shape:
            assert len(inferred) == len(expected_shape), (
                f"rank mismatch on {label} {name!r} (case {case_name!r}): "
                f"got {len(inferred)}, expected {len(expected_shape)}"
            )
            for d, (got, exp) in enumerate(zip(inferred, expected_shape)):
                if got != -1 and exp != -1:
                    assert got == exp, (
                        f"dim[{d}] mismatch on {label} {name!r} (case {case_name!r}): "
                        f"got {got}, expected {exp}"
                    )


def shape_inference_check(model: onnxl.ModelProto, *inputs):
    """Validates :func:`shape_inference.infer_shapes` for a backend test case.

    Mirrors the C++ ``BackendTestCaseShapeInference.AllCollectedCasesInferOutputShapes``
    gtest: snapshots and strips the recorded graph output shapes, runs
    ``infer_shapes`` on a fresh copy of the model, and checks the recovered
    shapes match. When the graph also declares intermediate ``value_info``
    entries (the ``kind == "model"`` cases), those intermediate shapes are
    validated the same way; node-kind cases have an empty ``value_info`` so
    the check is a no-op.
    """
    # Operate on a fresh copy so subsequent dataset iterations of the same
    # ``TestCase`` still see the original (un-stripped) shapes.
    work = onnxl.ModelProto()
    work.CopyFrom(model)

    name = work.graph.name or "<unnamed>"
    has_intermediate_value_info = len(work.graph.value_info) > 0

    output_snapshot = _snapshot_and_strip(work.graph.output)
    value_info_snapshot = (
        _snapshot_and_strip(work.graph.value_info) if has_intermediate_value_info else []
    )

    shape_inference.infer_shapes(work)

    _check_match(
        work.graph.output, output_snapshot, label="output", case_name=name, strict_shape=False
    )
    if has_intermediate_value_info:
        _check_match(
            work.graph.value_info,
            value_info_snapshot,
            label="value_info",
            case_name=name,
            strict_shape=True,
        )

    assert len(inputs) == len(model.graph.input)
    # No output values to compare; this is a model-level check.
    return None


TestShapeInferenceBackend = make_test_class(shape_inference_check)


if __name__ == "__main__":
    unittest.main(verbosity=2)
