import os
from pathlib import Path
from typing import Optional
from . import ModelProto, ParseOptions, SerializeOptions


def _find_external_location(model_path: str) -> str:
    """Scans a model file's structure to find the primary external data location.

    Parses the ONNX protobuf structure without loading external tensor data,
    then inspects the initializers in all graphs (including nested sub-graphs
    found inside node attributes) for the first ``location`` entry in their
    ``external_data`` metadata.

    :param model_path: Absolute or relative path to the ``.onnx`` model file.
    :return: Absolute path to the primary external data file, or ``""`` if no
        external data references are found.
    """
    struct_model = ModelProto()
    struct_model.ParseFromFile(model_path)
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
                        return os.path.join(model_dir, entry.value)
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
    f: str | Path,
    format: str = "protobuf",
    *,
    save_as_external_data: bool = False,
    all_tensors_to_one_file: bool = True,
    location: str | None = None,
    size_threshold: int = 1024,
    convert_attribute: bool = False,
    parallel: bool = False,
    num_threads: int = -1,
    min_block_size: int = 0,
    max_external_file_size: int = 0,
) -> None:
    """
    Saves the ModelProto to the specified path and optionally,
    serializes tensors with raw data as external data before saving.
    When external data is used, serialization writes through a temporary view,
    so the input in-memory ModelProto is left unchanged.

    :param proto: should be a in-memory ModelProto
    :param f: can be a file-like object (has "write" function) or a string containing
        a file name or a pathlike object
    :param format: The serialization format. When it is not specified, it is inferred
        from the file extension when ``f`` is a path. If not specified _and_
        ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
        be "utf-8" when the format is a text format.
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
    :param parallel: parallelize writing of large raw-data blocks
    :param num_threads: number of threads to use, -1 means the number of cores
    :param min_block_size: minimum raw-data block size in bytes to write in parallel
        when `parallel` is True; tensor blocks smaller than this threshold are
        written on the calling thread to avoid thread-pool overhead.
        A value of 0 (default) parallelizes all blocks.
    :param max_external_file_size: maximum size in bytes for one external
        weight file when saving with external data. A value of 0 (default)
        means no limit.
    """
    assert isinstance(proto, ModelProto), f"Unexpected type {type(proto)} for proto."
    assert isinstance(f, (str, Path)), f"Unexpected type {type(f)} for f."
    assert format == "protobuf", f"Unsupported format={format!r}"
    assert (
        all_tensors_to_one_file
    ), f"all_tensors_to_one_file={all_tensors_to_one_file} is not implemented"
    if save_as_external_data or location:
        if location is None:
            location = str(f) + ".data"
        opts = SerializeOptions()
        opts.raw_data_threshold = size_threshold
        opts.parallel = parallel
        opts.num_threads = num_threads
        opts.min_parallel_block_size = min_block_size
        opts.max_external_file_size = max_external_file_size
        proto.SerializeToFile(str(f), opts, str(location))
    elif parallel:
        opts = SerializeOptions()
        opts.raw_data_threshold = size_threshold
        opts.parallel = parallel
        opts.num_threads = num_threads
        opts.min_parallel_block_size = min_block_size
        opts.max_external_file_size = max_external_file_size
        proto.SerializeToFile(str(f), opts)
    else:
        proto.SerializeToFile(str(f))


def load(
    f: str | Path,
    skip_raw_data: bool = False,
    raw_data_threshold: int = 1024,
    load_external_data: Optional[bool] = None,
    parallel: bool = False,
    num_threads: int = -1,
    location: str = "",
    min_block_size: int = 0,
    no_copy: bool = False,
    touch_raw_data_pages: bool = False,
) -> ModelProto:
    """
    Loads a serialized ModelProto into memory.

    When *f* is a file path, the file is loaded via memory-mapping (``mmap``
    on POSIX, ``CreateFileMapping`` on Windows), which avoids per-byte system
    calls and lets the OS prefetch pages transparently.  This makes file
    loading nearly as fast as parsing from an already-in-memory bytes object.
    When *f* is a :class:`bytes` object, it is parsed in-place using a
    :class:`StringStream` with no additional copy.

    :param f: path or bytes
    :param skip_raw_data: skips the raw data of every tensor, this can be used
        to load only the architecture of the model even if the model is stored in
        one unique file
    :param raw_data_threshold: if `skip_raw_data` is True, still keeps the tensors
        smaller than this size (in bytes)
    :param load_external_data: Whether to load the external data.
            Set to True if the data is under the same directory of the model.
    :param parallel: parallelize the loading of the tensors
    :param num_threads: number of threads to use, -1 means the number of cores
    :param location: location of the external weights
        (can be different from the value stored in the main model).
        When ``load_external_data`` is ``True`` and this parameter is omitted,
        the primary external data file is auto-discovered from the tensor
        metadata stored in the model file.
    :param min_block_size: minimum raw-data block size in bytes to read in parallel
        when `parallel` is True; tensor blocks smaller than this threshold are read
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
    :return: Loaded in-memory ModelProto.
    """
    assert isinstance(f, (str, bytes, Path)), f"Unexpected type {type(f)} for f."
    if load_external_data is None:
        load_external_data = bool(location)
    assert (
        not location or load_external_data
    ), f"'load_external_data' must be True if location={location!r}"
    if isinstance(f, Path):
        f = str(f)
    assert not isinstance(f, str) or os.path.splitext(f)[-1] in {
        ".onnx"
    }, f"File name must have the extension .onnx to be loaded but f={f!r}"
    if load_external_data and not location and isinstance(f, str):
        location = _find_external_location(f)
    model = ModelProto()
    if skip_raw_data or parallel or no_copy or touch_raw_data_pages:
        opts = ParseOptions()
        opts.skip_raw_data = skip_raw_data
        opts.raw_data_threshold = raw_data_threshold
        opts.parallel = parallel
        opts.num_threads = num_threads
        opts.min_parallel_block_size = min_block_size
        opts.no_copy = no_copy
        opts.touch_raw_data_pages = touch_raw_data_pages
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
    parallel: bool = False,
    num_threads: int = -1,
    size_threshold: int = 1024,
    min_block_size: int = 0,
) -> None:
    """Serializes and encrypts a ModelProto to a single AES-256-CBC file.

    The model is first serialized to an in-memory buffer and then encrypted
    using AES-256-CBC.  The passphrase is stretched to a 32-byte key via
    PBKDF2-HMAC-SHA256 (100 000 iterations).  The output file is a
    self-contained binary that can only be loaded by :func:`load_encrypted`
    with the same key.

    .. note::
        This function requires that onnx-light was built with OpenSSL support
        (``ONNX_LIGHT_HAS_OPENSSL`` compile-time flag).

    :param proto: The ModelProto to save.
    :param f: Destination file path (str or :class:`pathlib.Path`).
    :param key: Passphrase or raw bytes used to derive the AES-256 key.
        When *key* is :class:`bytes` it is decoded as ``latin-1`` before
        PBKDF2 so that arbitrary byte values are preserved faithfully.
    :param parallel: Enable parallel serialization of large tensor blocks.
    :param num_threads: Number of threads to use for parallel serialization
        (``-1`` means use the number of available CPU cores).
    :param size_threshold: Minimum tensor raw-data size (bytes) that is
        considered "large" for the purposes of parallelisation.
    :param min_block_size: Minimum raw-data block size (bytes) parallelised
        when *parallel* is ``True``.
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
    opts.parallel = parallel
    opts.num_threads = num_threads
    opts.min_parallel_block_size = min_block_size
    proto.SerializeToEncryptedFile(str(f), key, opts)


def load_encrypted(
    f: str | Path,
    key: str | bytes,
    *,
    parallel: bool = False,
    num_threads: int = -1,
    min_block_size: int = 0,
) -> ModelProto:
    """Decrypts and parses an AES-256-CBC encrypted ONNX model.

    The file must have been produced by :func:`save_encrypted` with the
    same key.  Decryption is performed with AES-256-CBC using a key
    derived from the passphrase via PBKDF2-HMAC-SHA256.

    .. note::
        This function requires that onnx-light was built with OpenSSL support
        (``ONNX_LIGHT_HAS_OPENSSL`` compile-time flag).

    :param f: Source file path (str or :class:`pathlib.Path`).
    :param key: Passphrase or raw bytes (must match the one used to save).
        :class:`bytes` values are decoded as ``latin-1``.
    :param parallel: Enable parallel parsing of large tensor blocks.
    :param num_threads: Number of threads to use (``-1`` = number of cores).
    :param min_block_size: Minimum block size (bytes) to parallelise.
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
    opts.parallel = parallel
    opts.num_threads = num_threads
    opts.min_parallel_block_size = min_block_size
    model.ParseFromEncryptedFile(str(f), key, opts)
    return model


def save_encrypted_string(
    proto: ModelProto,
    key: str | bytes,
    *,
    parallel: bool = False,
    num_threads: int = -1,
    size_threshold: int = 1024,
    min_block_size: int = 0,
) -> bytes:
    """Serializes and encrypts a ModelProto to an in-memory AES-256-CBC bytes object.

    Equivalent to :func:`save_encrypted` but returns the ciphertext as
    :class:`bytes` instead of writing it to a file.  The returned bytes are
    in ONNXCRY1 format and can be decrypted with :func:`load_encrypted_string`
    (or :func:`load_encrypted` after writing the bytes to a file).

    .. note::
        This function requires that onnx-light was built with OpenSSL support
        (``ONNX_LIGHT_HAS_OPENSSL`` compile-time flag).

    :param proto: The ModelProto to save.
    :param key: Passphrase or raw bytes used to derive the AES-256 key.
        When *key* is :class:`bytes` it is decoded as ``latin-1`` before
        PBKDF2 so that arbitrary byte values are preserved faithfully.
    :param parallel: Enable parallel serialization of large tensor blocks.
    :param num_threads: Number of threads to use for parallel serialization
        (``-1`` means use the number of available CPU cores).
    :param size_threshold: Minimum tensor raw-data size (bytes) that is
        considered "large" for the purposes of parallelisation.
    :param min_block_size: Minimum raw-data block size (bytes) parallelised
        when *parallel* is ``True``.
    :return: Encrypted model bytes in ONNXCRY1 format.
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
    opts.parallel = parallel
    opts.num_threads = num_threads
    opts.min_parallel_block_size = min_block_size
    return proto.SerializeToEncryptedString(key, opts)


def load_encrypted_string(
    data: bytes,
    key: str | bytes,
    *,
    parallel: bool = False,
    num_threads: int = -1,
    min_block_size: int = 0,
) -> ModelProto:
    """Decrypts and parses an in-memory AES-256-CBC encrypted ONNX model.

    Equivalent to :func:`load_encrypted` but takes a :class:`bytes` object
    instead of a file path.  The bytes must be in ONNXCRY1 format as
    produced by :func:`save_encrypted_string` (or :func:`save_encrypted`).

    .. note::
        This function requires that onnx-light was built with OpenSSL support
        (``ONNX_LIGHT_HAS_OPENSSL`` compile-time flag).

    :param data: Encrypted model bytes in ONNXCRY1 format.
    :param key: Passphrase or raw bytes (must match the one used to encrypt).
        :class:`bytes` values are decoded as ``latin-1``.
    :param parallel: Enable parallel parsing of large tensor blocks.
    :param num_threads: Number of threads to use (``-1`` = number of cores).
    :param min_block_size: Minimum block size (bytes) to parallelise.
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
    opts.parallel = parallel
    opts.num_threads = num_threads
    opts.min_parallel_block_size = min_block_size
    model.ParseFromEncryptedString(bytes(data), key, opts)
    return model
