import os
from pathlib import Path
from typing import IO, Any, Callable, Optional, cast
from ..onnx_py._onnxpyprotoop import (  # type: ignore
    FileLoadMode,
    ModelProto,
    ParseOptions,
    SerializeOptions,
    TensorProto,
)

# ParseOptions used by :func:`_find_external_location` to skip copying raw tensor
# bytes when scanning a model file purely for external-data ``location`` entries.
# Without this, the scan would parse and copy every tensor's raw data and roughly
# double the apparent ``load`` time when a caller passes ``load_external_data=True``
# without an explicit ``location``.
_FIND_EXTERNAL_PARSE_OPTS = ParseOptions()
_FIND_EXTERNAL_PARSE_OPTS.skip_raw_data = True

# Serialization formats understood by :func:`load` and :func:`save`. ``"protobuf"``
# is the binary ONNX format; ``"textproto"`` is the protobuf text format handled
# by :mod:`onnx_light.onnx_proto._text_format`.
_SUPPORTED_FORMATS = frozenset({"protobuf", "textproto"})


def _infer_format(f: object) -> str:
    """Infers the serialization format from a file path extension.

    Returns ``"textproto"`` for paths ending in ``.textproto`` and ``"protobuf"``
    otherwise. Non-path inputs default to ``"protobuf"``.
    """
    if isinstance(f, (str, Path)) and os.path.splitext(str(f))[-1] == ".textproto":
        return "textproto"
    return "protobuf"


def _find_external_location(model_path: str) -> str:
    """Scans a model file's structure to find the primary external data location.

    Parses the ONNX protobuf structure without loading external tensor data,
    then inspects the initializers in all graphs (including nested sub-graphs
    found inside node attributes) for the first ``location`` entry in their
    ``external_data`` metadata.

    The discovered location is validated against path traversal and symlink
    escape before being returned.

    :param model_path: Absolute or relative path to the ``.onnx`` model file.
    :return: Absolute path to the primary external data file, or ``""`` if no
        external data references are found.
    :raises ValueError: If the location would escape the model directory.
    """
    from ._path_security import validate_external_data_path

    struct_model = ModelProto()
    struct_model.ParseFromFile(model_path, _FIND_EXTERNAL_PARSE_OPTS)
    model_dir = os.path.dirname(os.path.abspath(model_path))
    if not struct_model.has_graph():
        return ""
    # BFS over all graphs (top-level + nested sub-graphs inside node attributes).
    # Index-based access is used for node and attribute lists because the Python
    # iterator raises a TypeError for RepeatedProtoField when sub-graph
    # attributes are present.
    queue = [struct_model.graph]
    while queue:
        graph = queue.pop()
        for i in range(len(graph.initializer)):
            init = graph.initializer[i]
            if int(init.data_location) == 1:  # 1 == TensorProto.EXTERNAL
                for j in range(len(init.external_data)):
                    entry = init.external_data[j]
                    if entry.key == "location" and entry.value:
                        return validate_external_data_path(str(entry.value), model_dir)
        for i in range(len(graph.node)):
            node = graph.node[i]
            for j in range(len(node.attribute)):
                attr = node.attribute[j]
                if attr.has_g():
                    queue.append(attr.g)
                for k in range(len(attr.graphs)):
                    queue.append(attr.graphs[k])
    return ""


def save(
    proto: ModelProto,
    f: str | Path | IO[bytes],
    format: str | None = None,
    *,
    save_as_external_data: bool = False,
    all_tensors_to_one_file: bool = True,
    location: str | None = None,
    size_threshold: int = 1024,
    convert_attribute: bool = False,
    num_threads: int = -1,
    min_block_size: int = 0,
    max_external_file_size: int = 0,
    raw_data_callback: Callable[[TensorProto, Any | None, bool], int] | None = None,
) -> None:
    """
    Saves the ModelProto to the specified path and optionally,
    serializes tensors with raw data as external data before saving.
    When external data is used, serialization writes through a temporary view,
    so the input in-memory ModelProto is left unchanged.

    :param proto: should be a in-memory ModelProto
    :param f: can be a file-like object (has "write" function) or a string containing
        a file name or a pathlike object
    :param format: The serialization format, either ``"protobuf"`` (binary ONNX)
        or ``"textproto"`` (protobuf text format). When it is not specified, it is
        inferred from the file extension (``.textproto`` selects ``"textproto"``,
        otherwise ``"protobuf"``). Text-format files are written using "utf-8".
    :param save_as_external_data: If true, save tensors to external file(s).
        all_tensors_to_one_file: Effective only if save_as_external_data is True.
        If true, save all tensors to one external file specified by location.
        If false, save each tensor to a file named with the tensor name.
    :param all_tensors_to_one_file: if `save_as_external_data` is True,
        then saves all tensors into one file instead of a file per tensor
    :param location: Effective only if `save_as_external_data` is true.
        Specify the external file that all tensors to save to.
        If an absolute path is given it is used as-is; the value stored in
        the ONNX metadata will be the path relative to the model file.
        If not specified, defaults to ``str(f) + ".data"`` (next to the model
        file with a ``.data`` suffix).
    :param size_threshold: Effective only if save_as_external_data is True.
        Threshold for size of data. Only when tensor's data
        is >= the size_threshold it will be converted
        to external data. To convert every tensor with raw data
        to external data set size_threshold=0.
    :param convert_attribute: Effective only if save_as_external_data is True.
        If true, convert all tensors to external data
        If false, convert only non-attribute tensors to external data
    :param num_threads: number of threads to use for parallel serialization.
        ``1`` disables parallelization, ``> 1`` uses exactly that
        many worker threads, and any negative value (``-1`` is the
        default) picks a sensible value based on the number of
        available CPU cores.
    :param min_block_size: minimum raw-data block size in bytes to write in parallel
        when ``num_threads != 1``; tensor blocks smaller than this threshold are
        written on the calling thread to avoid thread-pool overhead.
        A value of 0 (default) parallelizes all blocks.
    :param max_external_file_size: maximum size in bytes for one external
        weight file when saving with external data. A value of 0 (default)
        means no limit.
    :param raw_data_callback: optional callable invoked as
        ``fn(tensor, buffer, size_only)`` for every tensor carrying
        ``raw_data`` immediately before serialization. It is first called with
        ``buffer=None`` and ``size_only=True`` and must return the number of
        bytes it will write. onnx-light then allocates a writable buffer of
        that size and calls it again with ``size_only=False`` so it can fill
        the bytes and update tensor metadata before serialization.
    """
    assert isinstance(proto, ModelProto), f"Unexpected type {type(proto)} for proto."
    assert isinstance(f, (str, Path)) or hasattr(
        f, "write"
    ), f"Expected str, Path, or file-like object with write() method, got {type(f)}."
    if hasattr(f, "write"):
        # File-like object path: external data is not supported.
        if save_as_external_data or location:
            raise ValueError(
                "save_as_external_data and location are not supported when f is a "
                "file-like object. Pass a file path instead."
            )
        resolved_format = format if format is not None else "protobuf"
        if resolved_format not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format={resolved_format!r}; onnx-light only supports "
                f"{sorted(_SUPPORTED_FORMATS)!r}."
            )
        if resolved_format == "textproto":
            if raw_data_callback is not None:
                raise ValueError("raw_data_callback is not supported for the 'textproto' format.")
            from ._text_format import serialize_to_textproto

            text = serialize_to_textproto(proto)
            cast(IO[bytes], f).write(text.encode("utf-8"))
            return
        proto.SerializeToOstream(f)
        return
    if format is None:
        format = _infer_format(f)
    if format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format={format!r}; onnx-light only supports "
            f"{sorted(_SUPPORTED_FORMATS)!r}."
        )
    if format == "textproto":
        if save_as_external_data or location or raw_data_callback is not None:
            raise ValueError(
                "save_as_external_data, location, and raw_data_callback are not "
                "supported for the 'textproto' format."
            )
        from ._text_format import serialize_to_textproto

        with open(str(f), "w", encoding="utf-8") as handle:
            handle.write(serialize_to_textproto(proto))
        return
    assert (
        all_tensors_to_one_file
    ), f"all_tensors_to_one_file={all_tensors_to_one_file} is not implemented"
    if save_as_external_data or location:
        if location is None:
            location = str(f) + ".data"
        # Validate that the external data location does not escape the model
        # directory via path traversal (CVE-2025-51480) and does not point at
        # a symlink or hardlink (GHSA-8qff-7g33-75mx, TOCTOU symlink attack on
        # save_external_data).  The C++ write path performs a second check at
        # open time; both layers together close the TOCTOU window.
        from ._path_security import validate_external_data_path

        model_dir = os.path.dirname(os.path.abspath(str(f)))
        validate_external_data_path(location, model_dir, allow_absolute=os.path.isabs(location))
        opts = SerializeOptions()
        opts.raw_data_threshold = size_threshold
        opts.num_threads = num_threads
        opts.min_parallel_block_size = min_block_size
        opts.max_external_file_size = max_external_file_size
        opts.raw_data_callback = raw_data_callback
        proto.SerializeToFile(str(f), opts, str(location))
    elif num_threads > 1 or num_threads < 0 or raw_data_callback is not None:
        opts = SerializeOptions()
        opts.raw_data_threshold = size_threshold
        opts.num_threads = num_threads
        opts.min_parallel_block_size = min_block_size
        opts.max_external_file_size = max_external_file_size
        opts.raw_data_callback = raw_data_callback
        proto.SerializeToFile(str(f), opts)
    else:
        proto.SerializeToFile(str(f))


def save_to_file_descriptor(
    proto: ModelProto, fd: int, *, num_threads: int = 1, min_block_size: int = 0
) -> None:
    """Saves the ModelProto to an open file descriptor without closing it.

    Writes the serialized protobuf bytes directly into the open file descriptor
    *fd*. The descriptor is not closed after writing; the caller is responsible
    for closing it.

    :param proto: should be an in-memory ModelProto.
    :param fd: open writable file descriptor (integer). Must be a valid file
        descriptor opened for writing.
    :param num_threads: number of threads to use for serialization.
        ``1`` (default) disables parallelization, ``> 1`` uses exactly that
        many worker threads. Negative values are not supported for file
        descriptor writes and are treated as ``1``.
    :param min_block_size: minimum raw-data block size in bytes to write in
        parallel when ``num_threads != 1``; tensor blocks smaller than this
        threshold are written on the calling thread to avoid thread-pool
        overhead. A value of 0 (default) parallelizes all blocks.
    """
    assert isinstance(proto, ModelProto), f"Unexpected type {type(proto)} for proto."
    if num_threads != 1 or min_block_size != 0:
        opts = SerializeOptions()
        opts.num_threads = num_threads
        opts.min_parallel_block_size = min_block_size
        proto.SerializeToFileDescriptor(fd, opts)
    else:
        proto.SerializeToFileDescriptor(fd)


def load(
    f: str | Path,
    skip_raw_data: bool = False,
    raw_data_threshold: int = 1024,
    tiny_external_data_threshold: int = -1,
    load_external_data: Optional[bool] = None,
    num_threads: int = -1,
    location: str = "",
    min_block_size: int = 0,
    no_copy: bool = False,
    touch_raw_data_pages: bool = False,
    file_load_mode: FileLoadMode | str = FileLoadMode.AUTO,
    format: Optional[str] = None,
) -> ModelProto:
    """
    Loads a serialized ModelProto into memory.

    When *f* is a file path, the file is memory-mapped (``mmap`` on POSIX,
    ``CreateFileMapping`` on Windows) and parsed directly out of the mapped
    region.  The OS page cache is exposed as contiguous memory, so no
    per-byte system call buffering is required and parsing is comparable to
    parsing from an already-in-memory bytes object.  The same mapping
    strategy applies to the *external weights* file when a model is stored
    with external data — each weights file is mapped once into a shared
    buffer that all tensors point into.  When ``no_copy=True`` is requested
    with a single-file model the loader still copies inline ``raw_data``
    so that the parsed model does not depend on the lifetime of the mmap
    region; zero-copy of inline raw data is reserved for ``bytes`` inputs
    (where the caller owns the buffer) and for external weights files.
    When *f* is a :class:`bytes` object, it is parsed in-place using a
    :class:`StringStream` with no additional copy.

    :param f: path or bytes
    :param skip_raw_data: skips the raw data of every tensor, this can be used
        to load only the architecture of the model even if the model is stored in
        one unique file
    :param raw_data_threshold: if `skip_raw_data` is True, still keeps the tensors
        smaller than this size (in bytes)
    :param tiny_external_data_threshold: when parsing from a model file path without
        ``external_data_file`` (that is, with ``load_external_data=False``), loads
        tensors marked as external if their declared external ``length``/``size`` is
        strictly below this threshold (in bytes), then inlines them into ``raw_data``.
        A negative value (default) disables this behavior.
    :param load_external_data: Whether to load the external data.
            Set to True if the data is under the same directory of the model.
    :param num_threads: number of threads to use for parallel parsing.
        ``1`` disables parallelization, ``> 1`` uses exactly that
        many worker threads, and any negative value (``-1`` is the
        default) picks a sensible value based on the number of
        available CPU cores.
    :param location: location of the external weights
        (can be different from the value stored in the main model).
        When ``load_external_data`` is ``True`` and this parameter is omitted,
        the primary external data file is auto-discovered from the tensor
        metadata stored in the model file.
    :param min_block_size: minimum raw-data block size in bytes to read in parallel
        when ``num_threads != 1``; tensor blocks smaller than this threshold are read
        on the calling thread to avoid thread-pool overhead for tiny tensors.
        A value of 0 (default) parallelizes all blocks.
    :param no_copy: if True, raw tensor data is **not** copied into per-tensor owned buffers.
        Inline protobuf ``raw_data`` then points directly into the source bytes buffer.
        For models with external tensor data, each external weights file is loaded once
        into a shared model-owned buffer and every tensor points into that buffer.
        This avoids one memory allocation + copy per tensor and can be significantly
        faster for large models.

        .. warning::
            When *f* is a :class:`bytes` object, the caller **must** keep that original
            bytes object alive for the entire lifetime of the returned model.
            Modifying or releasing the bytes while the model is still in use leads
            to undefined behaviour. External-data files do not have this lifetime
            requirement because onnx-light keeps the shared file buffers alive.
    :param touch_raw_data_pages: if True, touches one byte per memory page in each non-empty
        tensor ``raw_data`` buffer (plus the last byte) after parsing. This forces lazy page
        faults (for example mmap-backed no-copy buffers) to happen during load timing.
    :param file_load_mode: selects the file-backed stream implementation used when *f* is a
        file path. Accepts either a :class:`FileLoadMode` value or its name as a string
        (``"AUTO"``, ``"MMAP"`` or ``"IFSTREAM"``, case-insensitive). ``FileLoadMode.AUTO``
        (default) lets onnx-light pick the fastest implementation compatible with the other
        options — currently :class:`MmapFileStream` for the main model file, except when
        ``no_copy=True`` is set on a single-file model (in which case the buffered
        :class:`FileStream` is used so borrowed pointers do not outlive the stream).
        ``FileLoadMode.MMAP`` forces memory-mapped I/O and ``FileLoadMode.IFSTREAM`` forces
        the buffered ``std::ifstream``-based reader. Ignored when *f* is a :class:`bytes`
        object. When an external weights file is provided via ``location``, the main model
        file is always read through the buffered reader (``TwoFilesStream``); ``MMAP`` then
        applies to the separate weights file, which is memory-mapped on the ``no_copy=True``
        path, so all modes are honoured.
    :param format: The serialization format, either ``"protobuf"`` (binary ONNX)
        or ``"textproto"`` (protobuf text format). When it is not specified, it is
        inferred from the file extension (``.textproto`` selects ``"textproto"``,
        otherwise ``"protobuf"``). Text-format inputs are decoded as "utf-8".
    :return: Loaded in-memory ModelProto.
    """
    assert isinstance(f, (str, bytes, Path)), f"Unexpected type {type(f)} for f."
    if isinstance(f, Path):
        f = str(f)
    inferred_format = format is None
    if format is None:
        format = _infer_format(f)
    if format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format={format!r}; onnx-light only supports "
            f"{sorted(_SUPPORTED_FORMATS)!r}."
        )
    if format == "textproto":
        from ._text_format import parse_from_textproto

        if isinstance(f, bytes):
            text = f.decode("utf-8")
        else:
            with open(f, encoding="utf-8") as handle:
                text = handle.read()
        return parse_from_textproto(text, ModelProto())
    if isinstance(file_load_mode, str):
        try:
            file_load_mode = FileLoadMode.__members__[file_load_mode.upper()]
        except KeyError as e:
            raise ValueError(
                f"Unknown file_load_mode={file_load_mode!r}; expected one of "
                f"{sorted(FileLoadMode.__members__)}"
            ) from e
    assert isinstance(
        file_load_mode, FileLoadMode
    ), f"Unexpected type {type(file_load_mode)} for file_load_mode."
    if load_external_data is None:
        load_external_data = bool(location)
    assert (
        not location or load_external_data
    ), f"'load_external_data' must be True if location={location!r}"
    if inferred_format:
        assert not isinstance(f, str) or os.path.splitext(f)[-1] in {
            ".onnx"
        }, f"File name must have the extension .onnx to be loaded but f={f!r}"
    if load_external_data and not location and isinstance(f, str):
        location = _find_external_location(f)
    model = ModelProto()
    if (
        skip_raw_data
        or (num_threads > 1 or num_threads < 0)
        or tiny_external_data_threshold >= 0
        or no_copy
        or touch_raw_data_pages
        or file_load_mode != FileLoadMode.AUTO
    ):
        opts = ParseOptions()
        opts.skip_raw_data = skip_raw_data
        opts.raw_data_threshold = raw_data_threshold
        opts.num_threads = num_threads
        opts.min_parallel_block_size = min_block_size
        opts.tiny_external_data_threshold = tiny_external_data_threshold
        opts.no_copy = no_copy
        opts._touch_raw_data_pages = touch_raw_data_pages
        opts.file_load_mode = file_load_mode
        if isinstance(f, bytes):
            model.ParseFromString(f, opts)
        elif location:
            model.ParseFromFile(f, opts, external_data_file=location)
        else:
            model.ParseFromFile(f, opts)
    else:
        if isinstance(f, bytes):
            model.ParseFromString(f)
        elif location:
            model.ParseFromFile(f, external_data_file=location)
        else:
            model.ParseFromFile(f)
    return model


def save_encrypted(
    proto: ModelProto,
    f: str | Path,
    key: str | bytes,
    *,
    encryption: str = "AES-256-CBC",
    num_threads: int = -1,
    size_threshold: int = 1024,
    min_block_size: int = 0,
) -> None:
    """Serializes and encrypts a ModelProto to a single encrypted file.

    The model is first serialized to an in-memory buffer and then encrypted
    using the selected cipher. The passphrase is stretched to a 32-byte key via
    PBKDF2-HMAC-SHA256 (100 000 iterations).  The output file is a
    self-contained binary that can only be loaded by :func:`load_encrypted`
    with the same key.

    .. note::
        This function requires that onnx-light was built with OpenSSL support
        (``ONNX_LIGHT_HAS_OPENSSL`` compile-time flag).

    :param proto: The ModelProto to save.
    :param f: Destination file path (str or :class:`pathlib.Path`).
    :param key: Passphrase or raw bytes used to derive the encryption key.
        When *key* is :class:`bytes` it is decoded as ``latin-1`` before
        PBKDF2 so that arbitrary byte values are preserved faithfully.
    :param encryption: Encryption algorithm. Supported values are
        ``"AES-256-CBC"`` (ONNXCRY1) and ``"ChaCha20-Poly1305"`` (ONNXCRY2).
    :param num_threads: Number of threads to use for parallel serialization.
        ``1`` disables parallelization, ``> 1`` uses exactly that many
        worker threads, and any negative value (``-1`` is the default)
        picks a sensible value based on the number of available CPU cores.
    :param size_threshold: Minimum tensor raw-data size (bytes) that is
        considered "large" for the purposes of parallelisation.
    :param min_block_size: Minimum raw-data block size (bytes) parallelised
        when ``num_threads != 1``.
    :raises RuntimeError: On OpenSSL errors or I/O failures.
    :raises NotImplementedError: When OpenSSL support is not compiled in.
    """
    assert isinstance(proto, ModelProto), f"Unexpected type {type(proto)} for proto."
    assert isinstance(f, (str, Path)), f"Unexpected type {type(f)} for f."
    if isinstance(key, bytes):
        key = key.decode("latin-1")
    if not hasattr(proto, "SerializeToEncryptedFile"):
        raise NotImplementedError(
            "onnx-light was not built with OpenSSL support.  "
            "Recompile with OpenSSL available to use save_encrypted."
        )
    opts = SerializeOptions()
    opts.raw_data_threshold = size_threshold
    opts.num_threads = num_threads
    opts.min_parallel_block_size = min_block_size
    proto.SerializeToEncryptedFile(str(f), key, opts, encryption)


def load_encrypted(
    f: str | Path, key: str | bytes, *, num_threads: int = -1, min_block_size: int = 0
) -> ModelProto:
    """Decrypts and parses an encrypted ONNX model.

    The file must have been produced by :func:`save_encrypted` with the
    same key.  Decryption supports ONNXCRY1 (AES-256-CBC) and ONNXCRY2
    (ChaCha20-Poly1305) formats using a key
    derived from the passphrase via PBKDF2-HMAC-SHA256.

    .. note::
        This function requires that onnx-light was built with OpenSSL support
        (``ONNX_LIGHT_HAS_OPENSSL`` compile-time flag).

    :param f: Source file path (str or :class:`pathlib.Path`).
    :param key: Passphrase or raw bytes (must match the one used to save).
        :class:`bytes` values are decoded as ``latin-1``.
    :param num_threads: Number of threads to use for parallel parsing.
        ``1`` disables parallelization, ``> 1`` uses exactly that many
        worker threads, and any negative value (``-1`` is the default)
        picks a sensible value based on the number of available CPU cores.
    :param min_block_size: Minimum block size (bytes) to parallelise
        when ``num_threads != 1``.
    :return: The decrypted and parsed :class:`ModelProto`.
    :raises RuntimeError: On decryption failures or I/O errors.
    :raises NotImplementedError: When OpenSSL support is not compiled in.
    """
    assert isinstance(f, (str, Path)), f"Unexpected type {type(f)} for f."
    if isinstance(key, bytes):
        key = key.decode("latin-1")
    model = ModelProto()
    if not hasattr(model, "ParseFromEncryptedFile"):
        raise NotImplementedError(
            "onnx-light was not built with OpenSSL support.  "
            "Recompile with OpenSSL available to use load_encrypted."
        )
    opts = ParseOptions()
    opts.num_threads = num_threads
    opts.min_parallel_block_size = min_block_size
    model.ParseFromEncryptedFile(str(f), key, opts)
    return model


def save_encrypted_string(
    proto: ModelProto,
    key: str | bytes,
    *,
    encryption: str = "AES-256-CBC",
    num_threads: int = -1,
    size_threshold: int = 1024,
    min_block_size: int = 0,
) -> bytes:
    """Serializes and encrypts a ModelProto to an in-memory bytes object.

    Equivalent to :func:`save_encrypted` but returns the ciphertext as
    :class:`bytes` instead of writing it to a file.  The returned bytes are
    in ONNXCRY1 or ONNXCRY2 format and can be decrypted with :func:`load_encrypted_string`
    (or :func:`load_encrypted` after writing the bytes to a file).

    .. note::
        This function requires that onnx-light was built with OpenSSL support
        (``ONNX_LIGHT_HAS_OPENSSL`` compile-time flag).

    :param proto: The ModelProto to save.
    :param key: Passphrase or raw bytes used to derive the encryption key.
        When *key* is :class:`bytes` it is decoded as ``latin-1`` before
        PBKDF2 so that arbitrary byte values are preserved faithfully.
    :param encryption: Encryption algorithm. Supported values are
        ``"AES-256-CBC"`` (ONNXCRY1) and ``"ChaCha20-Poly1305"`` (ONNXCRY2).
    :param num_threads: Number of threads to use for parallel serialization.
        ``1`` disables parallelization, ``> 1`` uses exactly that many
        worker threads, and any negative value (``-1`` is the default)
        picks a sensible value based on the number of available CPU cores.
    :param size_threshold: Minimum tensor raw-data size (bytes) that is
        considered "large" for the purposes of parallelisation.
    :param min_block_size: Minimum raw-data block size (bytes) parallelised
        when ``num_threads != 1``.
    :return: Encrypted model bytes in ONNXCRY1 or ONNXCRY2 format.
    :raises RuntimeError: On OpenSSL errors.
    :raises NotImplementedError: When OpenSSL support is not compiled in.
    """
    assert isinstance(proto, ModelProto), f"Unexpected type {type(proto)} for proto."
    if isinstance(key, bytes):
        key = key.decode("latin-1")
    if not hasattr(proto, "SerializeToEncryptedString"):
        raise NotImplementedError(
            "onnx-light was not built with OpenSSL support.  "
            "Recompile with OpenSSL available to use save_encrypted_string."
        )
    opts = SerializeOptions()
    opts.raw_data_threshold = size_threshold
    opts.num_threads = num_threads
    opts.min_parallel_block_size = min_block_size
    return proto.SerializeToEncryptedString(key, opts, encryption)


def load_encrypted_string(
    data: bytes, key: str | bytes, *, num_threads: int = -1, min_block_size: int = 0
) -> ModelProto:
    """Decrypts and parses an in-memory encrypted ONNX model.

    Equivalent to :func:`load_encrypted` but takes a :class:`bytes` object
    instead of a file path.  The bytes must be in ONNXCRY1 or ONNXCRY2 format as
    produced by :func:`save_encrypted_string` (or :func:`save_encrypted`).

    .. note::
        This function requires that onnx-light was built with OpenSSL support
        (``ONNX_LIGHT_HAS_OPENSSL`` compile-time flag).

    :param data: Encrypted model bytes in ONNXCRY1 or ONNXCRY2 format.
    :param key: Passphrase or raw bytes (must match the one used to encrypt).
        :class:`bytes` values are decoded as ``latin-1``.
    :param num_threads: Number of threads to use for parallel parsing.
        ``1`` disables parallelization, ``> 1`` uses exactly that many
        worker threads, and any negative value (``-1`` is the default)
        picks a sensible value based on the number of available CPU cores.
    :param min_block_size: Minimum block size (bytes) to parallelise
        when ``num_threads != 1``.
    :return: The decrypted and parsed :class:`ModelProto`.
    :raises RuntimeError: On decryption failures.
    :raises NotImplementedError: When OpenSSL support is not compiled in.
    """
    assert isinstance(data, (bytes, bytearray)), f"Unexpected type {type(data)} for data."
    if isinstance(key, bytes):
        key = key.decode("latin-1")
    model = ModelProto()
    if not hasattr(model, "ParseFromEncryptedString"):
        raise NotImplementedError(
            "onnx-light was not built with OpenSSL support.  "
            "Recompile with OpenSSL available to use load_encrypted_string."
        )
    opts = ParseOptions()
    opts.num_threads = num_threads
    opts.min_parallel_block_size = min_block_size
    model.ParseFromEncryptedString(bytes(data), key, opts)
    return model
