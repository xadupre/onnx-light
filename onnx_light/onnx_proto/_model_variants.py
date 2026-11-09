"""File-backed model variants using native rewriting protos and replay."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
from typing import Any

from ..onnx_py._onnxpyprotoop import FileLoadMode, ModelProto
from ._path_security import validate_external_data_path


def _reference_path(location: str, directory: Path) -> Path:
    """Validates a contained reference without permitting link indirection."""
    candidate = Path(validate_external_data_path(location, str(directory)))
    current = directory
    for component in Path(location).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"Model variant reference contains a symbolic link: {location!r}.")
    return candidate


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    """Returns the identity and mutation counters of an artifact."""
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"Model variant base is not a regular file: {path}.")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _file_digest(path: Path) -> str:
    """Hashes a model file incrementally, without loading its external weights."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _rebase_external_locations(model: ModelProto, source: Path, destination: Path) -> None:
    """Rebases locations in all scopes without changing offsets or lengths."""
    rebased: dict[str, str] = {}
    for tensor in model.external_tensors():
        entries = {entry.key: entry.value for entry in tensor.external_data}
        if len(entries) != len(tensor.external_data):
            raise ValueError("An external payload must not contain duplicate metadata keys.")
        ranges = {}
        for key in ("offset", "length", "size"):
            if key not in entries:
                continue
            text = entries[key]
            if len(text) > 19 or not text.isascii() or not text.isdecimal():
                raise ValueError("Invalid external payload byte range.")
            ranges[key] = int(text)
        if "length" in ranges and "size" in ranges:
            raise ValueError("An external payload must not specify both length and size.")
        if ranges.get("offset", 0) + ranges.get("length", ranges.get("size", 0)) > 2**63 - 1:
            raise ValueError("External payload byte range overflows int64.")
        locations = [entry for entry in tensor.external_data if entry.key == "location"]
        if len(locations) != 1:
            raise ValueError("An external tensor must have exactly one location.")
        location = locations[0]
        if location.value not in rebased:
            resolved = _reference_path(location.value, source)
            relative = os.path.relpath(resolved, destination)
            _reference_path(relative, destination)
            rebased[location.value] = Path(relative).as_posix()
        location.value = rebased[location.value]


def _validate_envelope(model: ModelProto) -> str:
    """Validates the single-plan, graph-free variant representation."""
    if not model.base_model or not re.fullmatch("[0-9a-f]{64}", model.base_model_sha256):
        raise ValueError("A model variant requires a base_model and its SHA-256 digest.")
    if len(model.rewritings) != 1:
        raise ValueError("A model variant must contain exactly one named rewriting plan.")
    name = next(iter(model.rewritings))
    expected = ModelProto()
    if model.HasField("ir_version"):
        expected.ir_version = model.ir_version
    expected.base_model = model.base_model
    expected.base_model_sha256 = model.base_model_sha256
    expected.add_rewriting(name, model.rewritings[name])
    if not model.Equals(expected):
        raise ValueError(
            "A model variant envelope must not contain a graph or other declarations."
        )
    return name


def resolve_model_variant(
    model: ModelProto,
    source_path: str | Path,
    *,
    load_external_data: bool = False,
    _seen: tuple[tuple[int, int], ...] = (),
    **load_options: Any,
) -> ModelProto:
    """Resolves a graph-free envelope with native, hash-validated replay.

    Relative references stay within the envelope's directory tree. Chains are
    limited to 32 references, and cycles, links, missing files, changed bases and
    malformed envelopes are errors. Returned external locations are relative to
    ``source_path``; their byte offsets and lengths are preserved.
    """
    from ..onnx_core.optimization import apply_rewritings_to_model
    from ._io_helper import load

    name = _validate_envelope(model)
    if len(_seen) >= 32:
        raise ValueError("Model variant reference depth exceeds 32.")
    source = Path(source_path).absolute()
    source_identity = _file_identity(source)
    key = source_identity[:2]
    if key in _seen:
        raise ValueError("Model variant reference cycle detected.")
    seen = (*_seen, key)
    _rebase_external_locations(model, source.parent, source.parent)
    base_path = _reference_path(model.base_model, source.parent)
    identity = _file_identity(base_path)
    if identity[:2] in seen:
        raise ValueError("Model variant reference cycle detected.")
    # Parse metadata before resolving the chain so even malformed cyclic
    # envelopes report the reference cycle rather than a digest mismatch.
    base = load(base_path, resolve_references=False, load_external_data=False, **load_options)
    if not isinstance(base, ModelProto):
        raise ValueError("A model variant cannot reference a pipeline.")
    if base.HasField("base_model") or base.HasField("base_model_sha256"):
        base = resolve_model_variant(
            base, base_path, load_external_data=False, _seen=seen, **load_options
        )
    if _file_digest(base_path) != model.base_model_sha256:
        raise ValueError("Model variant base SHA-256 mismatch.")
    if _file_identity(base_path) != identity:
        raise ValueError("Model variant base changed while resolving.")
    _rebase_external_locations(base, base_path.parent, source.parent)
    result = apply_rewritings_to_model(base, model.rewritings[name])
    result.clear_rewritings()
    if load_external_data:
        from ..onnx_lib.external_data_helper import load_external_data_for_model

        load_external_data_for_model(result, str(source.parent))
    return result


def save_model_variant(
    base_model: str | Path,
    target: ModelProto | str | Path,
    path: str | Path,
    *,
    name: str,
    target_dir: str | Path | None = None,
) -> ModelProto:
    """Writes only a canonical-file reference and a native target delta.

    The canonical model is not rewritten. Target files may be removed after
    export; referenced weight shards must remain in the artifact directory.
    ``target_dir`` locates external weights of an in-memory target. A target
    path supplies that directory automatically. Unchanged external references
    require no payload reads. Different references are compared by content in
    native code; only changed tensors are included in the plan.

    Returns:
        Serialized graph-free variant envelope.
    """
    from ..onnx_core.optimization import diff_models
    from ._io_helper import load, save

    if not name:
        raise ValueError("A model variant name must not be empty.")
    destination = Path(path).absolute()
    base_path = Path(base_model).absolute()
    relative = os.path.relpath(base_path, destination.parent)
    base_path = _reference_path(relative, destination.parent)
    _reference_path(destination.name, destination.parent)
    if destination.resolve() == base_path:
        raise ValueError("A model variant must not overwrite its canonical model.")
    identity = _file_identity(base_path)
    digest = _file_digest(base_path)
    canonical = load(base_path, file_load_mode=FileLoadMode.MMAP)
    if not isinstance(canonical, ModelProto):
        raise ValueError("A model variant cannot reference a pipeline.")
    if isinstance(target, (str, Path)):
        if target_dir is not None:
            raise ValueError("target_dir is only supported with an in-memory target.")
        target_path = Path(target).absolute()
        target_dir = target_path.parent
        target = load(target_path, file_load_mode=FileLoadMode.MMAP)
    if not isinstance(target, ModelProto):
        raise TypeError("A model variant target must be a ModelProto or model path.")
    target_directory = (
        Path(target_dir).absolute() if target_dir is not None else destination.parent
    )
    plan = diff_models(
        canonical, target, base_dir=str(base_path.parent), target_dir=str(target_directory)
    )
    if _file_identity(base_path) != identity:
        raise ValueError("Model variant base changed while constructing the delta.")
    envelope = ModelProto()
    envelope.ir_version = canonical.ir_version
    envelope.base_model = Path(relative).as_posix()
    envelope.base_model_sha256 = digest
    envelope.add_rewriting(name, plan)
    if envelope.has_deferred_weights():
        envelope.load_weights()
    _rebase_external_locations(envelope, target_directory, destination.parent)
    save(envelope, destination)
    return envelope
