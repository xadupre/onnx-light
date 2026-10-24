.. _l-api-onnx-onnx-proto-onnx-crypt:

onnx_crypt.h
============

Encrypted external weights
--------------------------

With OpenSSL enabled, the native API supports all four storage choices:
ordinary ONNX serialization (embedded or plain external weights),
``SaveEncryptedModel`` (encrypted embedded weights), and
``SaveEncryptedModelWithExternalData`` (encrypted model and encrypted external weights).

.. code-block:: cpp

    // model already contains its tensor data in memory.
    // Caps each encrypted weight file at 64 MiB, including encryption overhead.
    SaveEncryptedModelWithExternalData(
        model, "model.onnxc", password, 64ULL * 1024 * 1024, 1024);
    ModelProto loaded;
    LoadEncryptedModel(loaded, "model.onnxc", password);

The new export API always uses AES-256-GCM (ONNXCRY3) for **every** file,
with an independent random salt and nonce and the same nonempty password.
It packs eligible ``raw_data`` tensors in traversal order, without splitting
tensors; typed tensor fields and tensors below the threshold stay embedded.
This includes sparse initializer values/indices and sparse tensor attributes
in nested graphs and functions. Borrowed input buffers are read without
requiring mutable access.
The maximum external-file size includes the 52-byte encryption envelope.
If an eligible tensor alone exceeds the remaining payload capacity of an empty
shard, export throws **before writing any files**. Increase the limit to save it.
The default cap is 1 GiB; supported caps range from 53 to ``INT_MAX + 52`` bytes.

External tensors must already have their plaintext data loaded in memory before
export (for plain input, use ``LoadExternalDataForModel``). Export leaves the
input model unchanged and never creates plaintext temporary weight files.
The working copy duplicates metadata, not safely shareable tensor storage:
it uses the existing copy-on-write snapshot mechanism. Ownerless borrowed
buffers and buffers with active exports still require copies. Packing and
encryption allocate additional per-shard buffers; export is not zero-copy.
Callers must synchronize access to the source model and reacquire existing
storage views after export, as with other shared-storage snapshots.
External encoded initializers and external tensor references in struct
declarations or their encoder/decoder functions are currently unsupported:
export rejects them before writing any files, including references in inline
declarations, nested codec graphs and function defaults. Load those codec
tensors into embedded storage first; embedded struct/codec tensors remain
inside the encrypted model rather than being sharded.
Generated shards are siblings of the model, with unique names of the form
``<model-name>.weights.<random-128-bit-hex>.<index>.onnxc``.
Copy the model and its referenced shards together; unrelated/old shards are not
deleted automatically. Shards are written first and the model last. Export is
not transactional: I/O failures may leave encrypted orphan shards or an
incomplete model file.

Format and loading
~~~~~~~~~~~~~~~~~~

The encrypted ModelProto contains standard external-data ``location``,
``offset`` and ``length`` entries, plus ``onnx_light.encryption=ONNXCRY3`` and
``onnx_light.encrypted_sha256`` (lowercase hex SHA-256 of the complete encrypted
shard). Offsets and lengths refer to **decrypted** weight bytes. The model's
authentication protects these entries, including the digest binding each shard
to this export; replacing a shard with another valid encrypted file is rejected.
Each shard uses the existing ONNXCRY3 envelope, with raw weight bytes rather than
a protobuf message as its plaintext.

``LoadEncryptedModel`` automatically authenticates and decrypts marked shards
relative to the model directory, then embeds their bytes and removes their
external-data metadata. Missing files, wrong passwords, altered/truncated or
substituted shards, unsafe paths/links and invalid ranges raise errors.
Metadata validation errors identify the tensor, shard location when available,
and offending field; shard size, digest and range errors also include context.
Plain native external-data readers reject marked encrypted weights rather than
using ciphertext as tensor bytes. Unmarked plain external references retain
their existing behavior.

For a shard limit other than the default 1 GiB, decrypt the model with
``LoadEncryptedModelFromString`` and then call
``LoadEncryptedExternalDataForModel(model, base_dir, password, parse_options, limit)``.
String loading alone leaves external references unresolved because it has no
base directory. Only pass trusted/authenticated model metadata to the external
loader. It applies ``max_tensor_size_bytes`` to tensor slices and verifies the
complete shard before exposing any of its plaintext. On failure, a model may
be partially populated; discard it. Decrypted buffers are owned in memory, not
memory-mapped from disk, even with ``no_copy`` enabled.
The loader groups references by location and releases each decrypted shard
after populating its tensors, rather than retaining all shards in memory.
Resolved tensor buffers honor ``ParseOptions.alignment``. The
``raw_data_callback`` runs once for each resolved tensor with its owning graph
(or ``nullptr`` for function-level tensors), and its returned cleanup callable
is attached to the tensor buffer, just as for embedded tensors.

These multi-file APIs require updated onnx-light readers; they are not standard
plain ONNX external-data files. Existing single-file ONNXCRY1/2/3 APIs and plain
ONNX external-data support remain available.

.. doxygenfile:: onnx_crypt.h
   :project: onnx-light
