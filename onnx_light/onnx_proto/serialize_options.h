#pragma once

#include "stream.h"
#include <cstdint>
#include <functional>

namespace ONNX_LIGHT_NAMESPACE {

// Forward declaration: ParseOptions::raw_data_callback references TensorProto, which is
// defined later in onnx.h. Only the declaration is needed for the std::function signature.
class TensorProto;
class NodeProto;
class GraphProto;
class ModelProto;
struct SerializeOptions;
void ApplySerializeRawDataCallback(ModelProto &model, const SerializeOptions &options);

/**
 * Common options shared by tensor buffer operations: in-place consolidation
 * (ConsolidateTensorsToBuffer), serialization (SerializeOptions) and parsing (ParseOptions).
 */
struct TensorBufferOptions {
  /** Specifies the minimum raw_data size (in bytes) to include in buffer operations.
   *  Tensors whose raw_data is smaller than this threshold are left in-place. */
  int64_t raw_data_threshold = kSmallTensorDataThresholdBytes;
  /** Controls the alignment boundary for tensor offsets within the buffer.
   *  If > 0, each tensor's offset is padded to a multiple of this many bytes.
   *  0 disables alignment.  Use 4096 for mmap-friendly page-aligned offsets. */
  int64_t alignment = 0;
};

/** Selects which file-backed BinaryStream implementation is used when parsing
 *  a model from a file path (for example via ``ModelProto::ParseFromFile``).
 *  - ``kAuto`` (default): pick the fastest implementation that is compatible
 *    with the other options.  Today that means ``MmapFileStream`` except when
 *    ``no_copy`` is true with a single-file model — see ``ParseFromFile`` for
 *    the precise selection rules.
 *  - ``kMmap``: force usage of ``MmapFileStream`` (memory-mapped file).
 *  - ``kFileStream``: force usage of ``FileStream`` (buffered ``std::ifstream``). */
enum class FileLoadMode : int32_t {
  kAuto = 0,
  kMmap = 1,
  kFileStream = 2,
};

/** Selects the on-disk serialization format used when parsing or serializing a
 *  ``ModelProto``. ``kOnnx`` is the default ONNX protobuf format. ``kOrtFlatbuffers``
 *  selects the flatbuffer-based format used by ``onnxruntime`` (``*.ort`` files). */
enum class SerializeFormat : int32_t {
  kOnnx = 0,
  kOrtFlatbuffers = 1,
};

/** Controls behavior when parsing ONNX protobuf messages from a stream or string. */
struct ParseOptions : TensorBufferOptions {
  /** Selects the on-disk serialization format expected when parsing.
   *  ``SerializeFormat::kOnnx`` (default) parses the ONNX protobuf wire format;
   *  ``SerializeFormat::kOrtFlatbuffers`` parses the onnxruntime flatbuffer
   *  format (``.ort`` files). The flatbuffer path is not yet implemented and
   *  raises an error when used. */
  SerializeFormat format = SerializeFormat::kOnnx;
  /** if true, raw data will not be read but skipped, tensors are not valid in that case  but the
   * model structure is still available */
  bool skip_raw_data = false;
  /** Number of threads to use for parallel reading of big blocks.
   *  - ``1`` (default): no parallelization, everything runs on the calling thread.
   *  - ``> 1``: use exactly this many worker threads.
   *  - ``< 0``: choose a sensible value based on the number of available CPU cores
   *    (``std::thread::hardware_concurrency()``).
   *  - ``0``: treated the same as ``1`` (no parallelization) for the purposes of
   *    :cpp:func:`is_parallel`. */
  int32_t num_threads = 1;
  /** minimum raw-data block size in bytes to submit to the thread pool when parallel reading is
   * enabled (``num_threads != 1``); blocks smaller than this value are read on the main thread
   * to avoid thread-pool overhead */
  int64_t min_parallel_block_size = 0;
  /** Returns true when parallel reading should be enabled, i.e. when
   *  ``num_threads`` is greater than 1 or negative.  ``num_threads == 0`` and
   *  ``num_threads == 1`` both disable parallelization. */
  inline bool is_parallel() const { return num_threads > 1 || num_threads < 0; }
  /** If true, raw_data blocks are not copied into a new buffer.  Inline protobuf raw_data
   * borrows directly from the source bytes buffer (for example the bytes passed to
   * ParseFromString), so the caller MUST keep that buffer alive for as long as any
   * TensorProto references it.  For external-data files, onnx-light loads each weights file
   * once into a shared model-owned buffer and each tensor borrows a view into that buffer. */
  bool no_copy = false;
  /** If true, parses all tensors normally and then touches one byte per memory page in
   * each non-empty raw_data buffer (plus the last byte). This forces lazy page faults
   * (for example mmap-backed no-copy buffers) to occur within the parse timing window. */
  bool _touch_raw_data_pages = false;
  /** Loads tiny external-data tensors inline during parsing when reading a model
   *  file without an explicit external weights stream.
   *  - ``< 0`` (default): disabled.
   *  - ``>= 0``: if a tensor is marked ``EXTERNAL`` and its external metadata
   *    declares ``length``/``size`` below this threshold (in bytes), parsing
   *    loads it from disk into ``raw_data`` and clears ``data_location`` and
   *    ``external_data``. */
  int64_t tiny_external_data_threshold = -1;
  /** Selects the file-backed BinaryStream implementation used when parsing a model
   *  from a file path (e.g. ``ModelProto::ParseFromFile``).  See ``FileLoadMode``
   *  for the semantics of each value.  Ignored when parsing from bytes/streams. */
  FileLoadMode file_load_mode = FileLoadMode::kAuto;
  /** Maximum nesting depth of protobuf sub-messages accepted while parsing.
   *  Protects the parser against stack overflow / out-of-memory caused by
   *  maliciously or accidentally deeply nested messages. Parsing raises an error
   *  when a message nests deeper than this value. The default is deliberately
   *  more conservative than protobuf's limit of 100: the recursive-descent
   *  parser uses large per-message stack frames (especially in debug builds), so
   *  a lower limit guarantees the guard rejects the message before the recursion
   *  exhausts the platform's default thread stack (e.g. 1 MB on Windows). It is
   *  still far above any realistic ONNX message nesting. */
  int32_t max_recursion_depth = 50;
  /** Internal counter tracking the current sub-message nesting depth while
   *  parsing. Managed automatically by the parser through a scoped guard; it is
   *  not a user-facing setting and is reset to 0 once a top-level parse
   *  completes. */
  int32_t _recursion_depth = 0;
  /** Maximum number of bytes that may be allocated for a single tensor's raw
   *  data (or packed repeated-field payload) during parsing.  This guards
   *  against OOM caused by maliciously or accidentally large size prefixes in
   *  the wire format.
   *  - ``0`` (default): no limit — any allocation is allowed.
   *  - ``> 0``: parsing raises an error when the declared byte count for a
   *    single tensor allocation exceeds this value.
   *  The check fires before the allocation, so the process is never asked to
   *  commit memory larger than this threshold.  Set this to a value comfortably
   *  above the largest legitimate tensor you expect, e.g. 2 GB for most models:
   *  ``options.max_tensor_size_bytes = 2LL * 1024 * 1024 * 1024;`` */
  int64_t max_tensor_size_bytes = 0;
  /** Holds an optional callback invoked for each TensorProto once its ``raw_data`` has been
   *  parsed (including external-data tensors, after their bytes have been resolved).  The
   *  callback receives the freshly parsed TensorProto and returns a deleter — a zero-argument
   *  callable invoked once when the tensor's ``raw_data`` is released (the tensor and all copies
   *  sharing the same buffer go out of scope, or the buffer is overwritten/cleared).
   *
   *  This lets callers take custom ownership of tensor data and register the matching cleanup,
   *  regardless of whether the bytes live on disk (no_copy borrowed view of an mmap or external
   *  file) or in CPU memory (owned buffer): the returned deleter is attached on top of the
   *  existing storage without moving the bytes.  Return an empty ``std::function`` to leave the
   *  tensor's ownership unchanged.
   *
   *  The callback also receives the parent GraphProto (the graph the tensor belongs to) as a
   *  pointer, or ``nullptr`` when the tensor is parsed on its own (for example
   *  ``TensorProto::ParseFromString``) rather than as part of a graph.
   *
   *  By default it is empty (no callback) and parsing behaves exactly as before. */
  std::function<std::function<void()>(TensorProto &, GraphProto *)> raw_data_callback = {};
  /** Internal transient pointer to the GraphProto currently being parsed, used only to pass the
   *  parent graph to ``raw_data_callback``. It is set and restored automatically while parsing a
   *  GraphProto, is never serialized, and is not exposed in the Python bindings. */
  GraphProto *_current_graph = nullptr;
  /** Holds an optional callback invoked for each NodeProto once it has been fully parsed.
   *
   *  The callback receives the freshly parsed NodeProto and its parent GraphProto (the graph the
   *  node belongs to) by reference and may inspect or modify the node in place.  The parent graph
   *  lets the callback read graph-level metadata or the surrounding nodes.
   *
   *  By default it is empty (no callback) and parsing behaves exactly as before. */
  std::function<void(NodeProto &, GraphProto &)> node_callback = {};
};

/** Controls behavior when serializing ONNX protobuf messages to a stream or string. */
struct SerializeOptions : TensorBufferOptions {
  /** Constructs a SerializeOptions instance with the default raw_data_threshold. */
  SerializeOptions() { raw_data_threshold = kSmallTensorDataThresholdBytes; }
  /** Selects the on-disk serialization format produced when serializing.
   *  ``SerializeFormat::kOnnx`` (default) writes the ONNX protobuf wire format;
   *  ``SerializeFormat::kOrtFlatbuffers`` writes the onnxruntime flatbuffer
   *  format (``.ort`` files). The flatbuffer path is not yet implemented and
   *  raises an error when used. */
  SerializeFormat format = SerializeFormat::kOnnx;
  /** if true, raw data will not be written but skipped, tensors are not valid in that case but the
   * model structure is still available */
  bool skip_raw_data = false;
  /** Number of threads to use for parallel writing of big blocks.
   *  - ``1`` (default): no parallelization, everything runs on the calling thread.
   *  - ``> 1``: use exactly this many worker threads.
   *  - ``< 0``: choose a sensible value based on the number of available CPU cores
   *    (``std::thread::hardware_concurrency()``).
   *  - ``0``: treated the same as ``1`` (no parallelization) for the purposes of
   *    :cpp:func:`is_parallel`. */
  int32_t num_threads = 1;
  /** minimum raw-data block size in bytes to submit to the thread pool when parallel writing is
   * enabled (``num_threads != 1``); blocks smaller than this value are written on the main thread
   * to avoid thread-pool overhead */
  int64_t min_parallel_block_size = 0;
  /** Returns true when parallel writing should be enabled, i.e. when
   *  ``num_threads`` is greater than 1 or negative.  ``num_threads == 0`` and
   *  ``num_threads == 1`` both disable parallelization. */
  inline bool is_parallel() const { return num_threads > 1 || num_threads < 0; }
  /** if true, tensors already marked with data_location=EXTERNAL are serialized using their
   * external_data metadata location (can target multiple weights files). */
  bool use_external_data_location = true;
  /** Maximum serialized size in bytes allowed for one serialization operation.
   *  The limit applies to the total output size (protobuf payload + external data).
   *  - ``0`` (default): no limit.
   *  - ``> 0``: serialization returns ``false`` when the computed size exceeds this limit.
   */
  int64_t max_serialized_size_bytes = 0;
  /** maximum size in bytes for one external weights file when saving with external data;
   * 0 means no limit (single weights file) */
  int64_t max_external_file_size = 0;
  /** Holds an optional callback invoked for each TensorProto carrying ``raw_data`` immediately
   *  before serialization.
   *
   *  The callback also receives the parent GraphProto (the graph the tensor belongs to) by
   *  pointer, taken from a working copy of the model, so the parent graph lets the callback locate
   *  the tensor's surrounding graph.
   *
   *  Serialization calls the callback twice per tensor:
   *
   *  - size pass: ``fn(tensor, graph, nullptr, 0, true)`` must return the number of bytes that
   *    the callback will serialize for that tensor.
   *  - fill pass: onnx-light allocates a buffer of that size, then calls
   *    ``fn(tensor, graph, buffer, buffer_size, false)``. The callback may update the tensor
   *    metadata in place (for example dims or data_type), must fill ``buffer`` with exactly that
   *    many bytes, and must return the same size again.
   *
   *  When the tensor was previously marked with ``data_location=EXTERNAL`` and still carries
   *  ``raw_data`` (for example after ``load_external_data``), serialization regenerates the
   *  external-data metadata after the callback so the stored ``length`` and ``offset`` reflect
   *  the rewritten bytes.
   *
   *  By default it is empty (no callback) and serialization behaves exactly as before. */
  std::function<int64_t(TensorProto &, GraphProto *, uint8_t *, size_t, bool)> raw_data_callback =
      {};
  /** Holds an optional callback invoked for each NodeProto immediately before it is serialized.
   *
   *  The callback receives the NodeProto and its parent GraphProto (both from a working copy of
   *  the model, so edits never alter the caller's model) by reference and may inspect or modify
   *  the node in place.  The parent graph lets the callback locate the node's surrounding graph.
   *
   *  By default it is empty (no callback) and serialization behaves exactly as before. */
  std::function<void(NodeProto &, GraphProto &)> node_callback = {};
};

/** Enforces ``SerializeOptions::max_serialized_size_bytes`` for a computed serialized size. */
bool EnforceMaxSerializedSize(const SerializeSizeResult &total_size,
                              const SerializeOptions &options, const char *context);

} // namespace ONNX_LIGHT_NAMESPACE
