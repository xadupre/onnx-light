#pragma once

#include "simple_span.h"
#include "simple_string.h"
#include "stream.h"
#include <functional>
#include <sstream>
#include <type_traits>
#include <utility>

#define FIELD_VARINT 0
#define FIELD_FIXED64 1
#define FIELD_FIXED_SIZE 2
#define FIELD_FIXED32 5 // deprecated value but used in old files

/** Serialization/parsing API declaration macro for generated proto classes. */
#define SERIALIZATION_METHOD()                                                                     \
  SerializeSizeResult SerializeSize() const;                                                       \
  size_t ByteSizeLong() const;                                                                     \
  bool ParseFromString(const std::string &raw);                                                    \
  bool ParseFromString(const std::string &raw, ParseOptions &opts);                                \
  /** Parses from a raw byte buffer (protobuf compat). */                                          \
  inline bool ParseFromArray(const void *data, int size) {                                         \
    return ParseFromString(                                                                        \
        std::string(static_cast<const char *>(data), static_cast<size_t>(size)));                  \
  }                                                                                                \
  bool ParseFromZeroCopyStream(utils::BinaryStream *stream);                                       \
  bool ParseFromZeroCopyStream(utils::BinaryStream *stream, ParseOptions &opts);                   \
  bool ParseFromIstream(std::istream *input);                                                      \
  std::string SerializeAsString() const;                                                           \
  bool SerializeToArray(void *data, int size) const;                                               \
  bool SerializeToOstream(std::ostream *output) const;                                             \
  bool SerializeToOStream(std::ostream *output) const;                                             \
  /** Serializes to a zero-copy output stream (protobuf compat). */                                \
  inline bool SerializeToZeroCopyStream(utils::BinaryWriteStream *output) const {                  \
    std::string buf;                                                                               \
    if (!SerializeToString(buf))                                                                   \
      return false;                                                                                \
    output->write_raw_bytes(reinterpret_cast<const uint8_t *>(buf.data()), buf.size());            \
    return true;                                                                                   \
  }                                                                                                \
  bool SerializeToString(std::string &out) const;                                                  \
  /** Pointer overload for protobuf compatibility. */                                              \
  inline bool SerializeToString(std::string *out) const { return SerializeToString(*out); }        \
  bool SerializeToString(std::string &out, SerializeOptions &opts) const;                          \
  bool SerializeToFileDescriptor(int fd) const;                                                    \
  bool SerializeToFileDescriptor(int fd, SerializeOptions &opts) const;                            \
  bool SaveToFileDescriptor(int fd) const;                                                         \
  bool SaveToFileDescriptor(int fd, SerializeOptions &opts) const;                                 \
  bool ParseFromFileDescriptor(int fd);                                                            \
  SerializeSizeResult SerializeSize(utils::BinaryWriteStream &stream, SerializeOptions &opts)      \
      const;                                                                                       \
  bool ParseFromStream(utils::BinaryStream &stream, ParseOptions &options);                        \
  /** Parses from a BinaryStream with default options. */                                          \
  inline bool ParseFromStream(utils::BinaryStream &stream) {                                       \
    ParseOptions opts;                                                                             \
    return ParseFromStream(stream, opts);                                                          \
  }                                                                                                \
  void SerializeToStream(utils::BinaryWriteStream &stream, SerializeOptions &options) const;       \
  /** Serializes to a BinaryWriteStream with default options. */                                   \
  inline void SerializeToStream(utils::BinaryWriteStream &stream) const {                          \
    SerializeOptions opts;                                                                         \
    SerializeToStream(stream, opts);                                                               \
  }                                                                                                \
  void PrintToStringStream(std::stringstream &ss, utils::PrintOptions &options) const;

/** Macro for beginning a generated proto class with a default constructor. */
#define BEGIN_PROTO(cls, doc)                                                                      \
  class cls : public Message {                                                                     \
  public:                                                                                          \
    static inline constexpr const char *DOC = doc;                                                 \
    explicit inline cls() {}                                                                       \
    /** Resets this message to default state (protobuf compat). */                                 \
    inline void Clear() {                                                                          \
      this->~cls();                                                                                \
      new (this) cls();                                                                            \
    }                                                                                              \
    void CopyFrom(const cls &proto);

/** Macro for beginning a generated proto class without adding a default constructor. */
#define BEGIN_PROTO_NOINIT(cls, doc)                                                               \
  class cls : public Message {                                                                     \
  public:                                                                                          \
    static inline constexpr const char *DOC = doc;                                                 \
    void CopyFrom(const cls &proto);

/** Macro for ending a generated proto class and injecting the serialization/parsing API. */
#define END_PROTO()                                                                                \
  SERIALIZATION_METHOD()                                                                           \
  }                                                                                                \
  ;

#if defined(FIELD)
#pragma error("macro FIELD is already defined.")
#endif

#define FIELD(type, name, order, doc)                                                              \
public:                                                                                            \
  inline type &ref_##name() { return name##_; }                                                    \
  inline const type &ref_##name() const { return name##_; }                                        \
  /** Compatibility accessor - equivalent to ref_##name(). */                                      \
  inline type &name() { return name##_; }                                                          \
  /** Compatibility accessor - equivalent to ref_##name() const. */                                \
  inline const type &name() const { return name##_; }                                              \
  inline const type *ptr_##name() const { return &name##_; }                                       \
  inline bool has_##name() const { return _has_field_(name##_); }                                  \
  inline void set_##name(const type &v) { name##_ = v; }                                           \
  /** Compatibility accessor returning a mutable pointer to the field. */                          \
  inline type *mutable_##name() { return &name##_; }                                               \
  inline int order_##name() const { return order; }                                                \
  static inline constexpr const char *_name_##name = #name;                                        \
  static inline constexpr const char *DOC_##name = doc;                                            \
  type name##_;                                                                                    \
  using name##_t = type;

/** Like FIELD but for ByteSpan (protobuf ``bytes``) fields.                                       \
 *  ``mutable_##name()`` returns ``std::string*`` (via ByteSpan::mutable_value()) and              \
 *  the const accessor returns ``const std::string&`` (via ByteSpan::value()) so that              \
 *  consuming code written against protobuf ``bytes`` fields compiles unchanged. */
#define FIELD_BYTES(name, order, doc)                                                              \
public:                                                                                            \
  inline utils::ByteSpan &ref_##name() { return name##_; }                                         \
  inline const utils::ByteSpan &ref_##name() const { return name##_; }                             \
  /** Returns a const std::string reference (protobuf bytes-field compat). */                      \
  inline const std::string &name() const { return name##_.value(); }                               \
  inline const utils::ByteSpan *ptr_##name() const { return &name##_; }                            \
  inline bool has_##name() const { return _has_field_(name##_); }                                  \
  inline void set_##name(const utils::ByteSpan &v) { name##_ = v; }                                \
  /** Returns a mutable std::string pointer to the owned data (protobuf bytes-field compat).       \
   *  Mutations are written directly into the ByteSpan's owned storage. */                         \
  inline std::string *mutable_##name() { return name##_.mutable_value(); }                         \
  inline int order_##name() const { return order; }                                                \
  static inline constexpr const char *_name_##name = #name;                                        \
  static inline constexpr const char *DOC_##name = doc;                                            \
  utils::ByteSpan name##_;                                                                         \
  using name##_t = utils::ByteSpan;

#define FIELD_DEFAULT(type, name, order, default_value, doc)                                       \
public:                                                                                            \
  inline type &ref_##name() { return name##_; }                                                    \
  inline const type &ref_##name() const { return name##_; }                                        \
  /** Compatibility accessor - equivalent to ref_##name(). */                                      \
  inline type &name() { return name##_; }                                                          \
  /** Compatibility accessor - equivalent to ref_##name() const. */                                \
  inline const type &name() const { return name##_; }                                              \
  inline const type *ptr_##name() const { return &name##_; }                                       \
  inline bool has_##name() const { return _has_field_(name##_); }                                  \
  inline void set_##name(const type &v) { name##_ = v; }                                           \
  inline int order_##name() const { return order; }                                                \
  static inline constexpr const char *_name_##name = #name;                                        \
  static inline constexpr const char *DOC_##name = doc;                                            \
  type name##_ = default_value;                                                                    \
  using name##_t = type;

#define FIELD_STR(name, order, doc)                                                                \
  FIELD(utils::OptionalString, name, order, doc)                                                   \
  inline void clear_##name() { name##_.reset(); }                                                  \
  inline void set_##name(const char *v) { name##_ = v; }                                           \
  inline void set_##name(const char *data, size_t len) { name##_ = std::string(data, len); }       \
  inline void set_##name(const std::string &v) { name##_ = v; }                                    \
  inline void set_##name(const utils::RefString &v) { name##_ = v; }

#define FIELD_REPEATED_BASE(type, repeated_type, name, order, doc)                                 \
public:                                                                                            \
  inline repeated_type &ref_##name() { return name##_; }                                           \
  inline const repeated_type &ref_##name() const { return name##_; }                               \
  /** Compatibility accessor - equivalent to ref_##name(). */                                      \
  inline repeated_type &name() { return name##_; }                                                 \
  inline repeated_type *mutable_##name() { return &name##_; }                                      \
  inline type *mutable_##name(size_t i) { return &name##_[i]; }                                    \
  inline const type &name(size_t i) const { return name##_[i]; }                                   \
  inline const repeated_type &name() const { return name##_; }                                     \
  inline const repeated_type *ptr_##name() const { return &name##_; }                              \
  inline type *add_##name() { return &name##_.add(); }                                             \
  inline type *add_##name(const type &v) {                                                         \
    name##_.push_back(v);                                                                          \
    return &name##_.back();                                                                        \
  }                                                                                                \
  inline void add_##name(const std::vector<type> &v) { name##_.extend(v); }                        \
  inline void extend_##name(const std::vector<type> &v) { name##_.extend(v); }                     \
  inline bool has_##name() const { return _has_field_(name##_) && !name##_.empty(); }              \
  inline int order_##name() const { return order; }                                                \
  inline void clr_##name() { name##_.clear(); }                                                    \
  inline void clear_##name() { name##_.clear(); }                                                  \
  inline int name##_size() const { return static_cast<int>(name##_.size()); }                      \
  static inline constexpr const char *DOC_##name = doc;                                            \
  static inline constexpr const char *_name_##name = #name;                                        \
  inline bool packed_##name() const { return false; }                                              \
  repeated_type name##_;                                                                           \
  using name##_t = type;

#define FIELD_REPEATED(type, name, order, doc)                                                     \
  FIELD_REPEATED_BASE(type, utils::RepeatedField<type>, name, order, doc)

#define FIELD_REPEATED_STR(type, name, order, doc)                                                 \
  FIELD_REPEATED_BASE(type, utils::RepeatedStringField, name, order, doc)                          \
  inline void add_##name(const char *v) { name##_.push_back(utils::String(v)); }                   \
  inline void add_##name(const std::string &v) { name##_.push_back(utils::String(v)); }            \
  inline void add_##name(const utils::RefString &v) { name##_.push_back(utils::String(v)); }

#define FIELD_REPEATED_PROTO(type, name, order, doc)                                               \
public:                                                                                            \
  inline utils::RepeatedProtoField<type> &ref_##name() { return name##_; }                         \
  inline const utils::RepeatedProtoField<type> &ref_##name() const { return name##_; }             \
  /** Compatibility accessor - equivalent to ref_##name(). */                                      \
  inline utils::RepeatedProtoField<type> &name() { return name##_; }                               \
  inline const type &name(size_t i) const { return name##_[i]; }                                   \
  inline utils::RepeatedProtoField<type> *mutable_##name() { return &name##_; }                    \
  inline type *mutable_##name(size_t i) { return &name##_[i]; }                                    \
  inline const utils::RepeatedProtoField<type> &name() const { return name##_; }                   \
  inline const utils::RepeatedProtoField<type> *ptr_##name() const { return &name##_; }            \
  inline type *add_##name() { return &name##_.add(); }                                             \
  inline type *add_##name(type &&v) {                                                              \
    name##_.push_back(std::move(v));                                                               \
    return &name##_.back();                                                                        \
  }                                                                                                \
  inline type *add_##name(const type &v) {                                                         \
    name##_.push_back(v);                                                                          \
    return &name##_.back();                                                                        \
  }                                                                                                \
  inline bool has_##name() const { return _has_field_(name##_) && !name##_.empty(); }              \
  inline int order_##name() const { return order; }                                                \
  inline void clr_##name() { name##_.clear(); }                                                    \
  inline void clear_##name() { name##_.clear(); }                                                  \
  inline int name##_size() const { return static_cast<int>(name##_.size()); }                      \
  static inline constexpr const char *DOC_##name = doc;                                            \
  static inline constexpr const char *_name_##name = #name;                                        \
  inline bool packed_##name() const { return false; }                                              \
  utils::RepeatedProtoField<type> name##_;                                                         \
  using name##_t = type;

#define FIELD_REPEATED_PACKED(type, name, order, doc)                                              \
public:                                                                                            \
  inline utils::RepeatedField<type> &ref_##name() { return name##_; }                              \
  inline const utils::RepeatedField<type> &ref_##name() const { return name##_; }                  \
  /** Compatibility accessor - equivalent to ref_##name(). */                                      \
  inline utils::RepeatedField<type> &name() { return name##_; }                                    \
  /** Compatibility accessor - equivalent to ref_##name() const. */                                \
  inline const utils::RepeatedField<type> &name() const { return name##_; }                        \
  inline const utils::RepeatedField<type> *ptr_##name() const { return &name##_; }                 \
  inline type *add_##name() { return &name##_.add(); }                                             \
  inline type *add_##name(const type &v) {                                                         \
    name##_.push_back(v);                                                                          \
    return &name##_.back();                                                                        \
  }                                                                                                \
  inline void add_##name(const std::vector<type> &v) { name##_.extend(v); }                        \
  inline bool has_##name() const { return _has_field_(name##_) && !name##_.empty(); }              \
  inline int order_##name() const { return order; }                                                \
  inline void clr_##name() { name##_.clear(); }                                                    \
  inline void clear_##name() { name##_.clear(); }                                                  \
  inline int name##_size() const { return static_cast<int>(name##_.size()); }                      \
  inline const type &name(size_t i) const { return name##_[i]; }                                   \
  inline type *mutable_##name(size_t i) { return &name##_[i]; }                                    \
  /** Returns a mutable pointer to the repeated field (protobuf compat). */                        \
  inline utils::RepeatedField<type> *mutable_##name() { return &name##_; }                         \
  static inline constexpr const char *DOC_##name = doc;                                            \
  static inline constexpr const char *_name_##name = #name;                                        \
  inline bool packed_##name() const { return true; }                                               \
  utils::RepeatedField<type> name##_;                                                              \
  using name##_t = type;

#define _FIELD_OPTIONAL(type, name, order, doc)                                                    \
public:                                                                                            \
  inline type &ref_##name() {                                                                      \
    if (!has_##name()) {                                                                           \
      add_##name();                                                                                \
    }                                                                                              \
    return *name##_;                                                                               \
  }                                                                                                \
  inline const type &ref_##name() const {                                                          \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value in " #type ".");   \
    return *name##_;                                                                               \
  }                                                                                                \
  /** Compatibility accessor - equivalent to ref_##name(). */                                      \
  inline type &name() { return ref_##name(); }                                                     \
  /** Compatibility accessor - equivalent to ref_##name() const. */                                \
  inline const type &name() const { return ref_##name(); }                                         \
  inline type *mutable_##name() { return &ref_##name(); }                                          \
  inline const type *ptr_##name() const {                                                          \
    return has_##name() ? &(*name##_) : static_cast<type *>(nullptr);                              \
  }                                                                                                \
  inline utils::OptionalField<type> &name##_optional() { return name##_; }                         \
  inline const utils::OptionalField<type> &name##_optional() const {                               \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value in " #type ".");   \
    return name##_;                                                                                \
  }                                                                                                \
  inline type *add_##name() {                                                                      \
    name##_.set_empty_value();                                                                     \
    return &(*name##_);                                                                            \
  }                                                                                                \
  inline void set_##name(const type &v) { name##_ = v; }                                           \
  inline void set_##name(type &&v) { name##_ = std::move(v); }                                     \
  inline void reset_##name() { name##_.reset(); }                                                  \
  inline void clear_##name() { name##_.reset(); }                                                  \
  inline bool has_##name() const { return name##_.has_value(); }                                   \
  inline int order_##name() const { return order; }                                                \
  static inline constexpr const char *DOC_##name = doc;                                            \
  static inline constexpr const char *_name_##name = #name;                                        \
  utils::OptionalField<type> name##_;                                                              \
  using name##_t = type;

#define FIELD_OPTIONAL(type, name, order, doc)                                                     \
  _FIELD_OPTIONAL(type, name, order, doc)                                                          \
  inline bool has_oneof_##name() const { return has_##name(); }

#define FIELD_OPTIONAL_ONEOF(type, name, order, oneof, doc)                                        \
  _FIELD_OPTIONAL(type, name, order, doc)                                                          \
  inline bool has_oneof_##name() const { return has_##oneof(); }

#define FIELD_OPTIONAL_ENUM(type, name, order, doc)                                                \
public:                                                                                            \
  inline type &ref_##name() {                                                                      \
    if (!has_##name()) {                                                                           \
      add_##name();                                                                                \
    }                                                                                              \
    return *name##_;                                                                               \
  }                                                                                                \
  inline const type &ref_##name() const {                                                          \
    EXT_ENFORCE(name##_.has_value(), "Optional enum field '", #name,                               \
                "' has no value in " #type ".");                                                   \
    return *name##_;                                                                               \
  }                                                                                                \
  /** Compatibility accessor - equivalent to ref_##name(). */                                      \
  inline type &name() { return ref_##name(); }                                                     \
  /** Compatibility accessor - returns default enum value when unset (protobuf compat). */         \
  inline type name() const { return has_##name() ? *name##_ : type(0); }                           \
  inline const type *ptr_##name() const {                                                          \
    return has_##name() ? &(*name##_) : static_cast<type *>(nullptr);                              \
  }                                                                                                \
  inline utils::OptionalEnumField<type> &name##_optional() { return name##_; }                     \
  inline const utils::OptionalEnumField<type> &name##_optional() const {                           \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value in " #type ".");   \
    return name##_;                                                                                \
  }                                                                                                \
  inline type *add_##name() {                                                                      \
    name##_.set_empty_value();                                                                     \
    return &(*name##_);                                                                            \
  }                                                                                                \
  inline void set_##name(const type &v) { name##_ = v; }                                           \
  inline void reset_##name() { name##_.reset(); }                                                  \
  inline void clear_##name() { name##_.reset(); }                                                  \
  inline bool has_##name() const { return name##_.has_value(); }                                   \
  inline int order_##name() const { return order; }                                                \
  static inline constexpr const char *DOC_##name = doc;                                            \
  static inline constexpr const char *_name_##name = #name;                                        \
  utils::OptionalEnumField<type> name##_;                                                          \
  using name##_t = type;

namespace ONNX_LIGHT_NAMESPACE {

// Forward declaration: ParseOptions::raw_data_callback references TensorProto, which is
// defined later in onnx.h. Only the declaration is needed for the std::function signature.
class TensorProto;
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
  /** Constructs a ParseOptions instance with the default raw_data_threshold of 1024 bytes. */
  ParseOptions() { raw_data_threshold = 1024; }
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
   *  By default it is empty (no callback) and parsing behaves exactly as before. */
  std::function<std::function<void()>(TensorProto &)> raw_data_callback = {};
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
   *  Serialization calls the callback twice per tensor:
   *
   *  - size pass: ``fn(tensor, nullptr, 0, true)`` must return the number of bytes that the
   *    callback will serialize for that tensor.
   *  - fill pass: onnx-light allocates a buffer of that size, then calls
   *    ``fn(tensor, buffer, buffer_size, false)``. The callback may update the tensor metadata
   *    in place (for example dims or data_type), must fill ``buffer`` with exactly that many
   *    bytes, and must return the same size again.
   *
   *  When the tensor was previously marked with ``data_location=EXTERNAL`` and still carries
   *  ``raw_data`` (for example after ``load_external_data``), serialization regenerates the
   *  external-data metadata after the callback so the stored ``length`` and ``offset`` reflect
   *  the rewritten bytes.
   *
   *  By default it is empty (no callback) and serialization behaves exactly as before. */
  std::function<int64_t(TensorProto &, uint8_t *, size_t, bool)> raw_data_callback = {};
};

/** Enforces ``SerializeOptions::max_serialized_size_bytes`` for a computed serialized size. */
bool EnforceMaxSerializedSize(const SerializeSizeResult &total_size,
                              const SerializeOptions &options, const char *context);

using utils::offset_t;

/** Returns true if the field holds a non-default value (always true for scalar types other
 *  than String and raw-bytes vectors, which have their own specializations).
 */
template <typename T> inline bool _has_field_(const T &) { return true; }
/** Returns true if the string field is non-empty. */
template <> inline bool _has_field_(const utils::String &field) { return !field.empty(); }
/** Returns true if the optional string field is present (proto2 optional-string presence). */
template <> inline bool _has_field_(const utils::OptionalString &field) {
  return field.has_value();
}
/** Returns true if the raw-bytes field is non-empty. */
template <> inline bool _has_field_(const std::vector<uint8_t> &field) { return !field.empty(); }
/** Returns true if the ByteSpan field is non-empty (owned or borrowed). */
template <> inline bool _has_field_(const utils::ByteSpan &field) { return !field.empty(); }

/** Copies all fields from src into dest. Generated for every proto class. */
template <typename T> void CopyProtoFrom(T &dest, const T &src);

/** Base class for generated ONNX proto messages. */
class Message {
public:
  /** Constructs an empty message base object. */
  explicit inline Message() {}
  /** Throws an exception as a placeholder; generated classes provide their own operator==. */
  inline bool operator==(const Message &) const {
    EXT_THROW("operator == not implemented for a Message");
  }
};

/** ADL-visible swap for generated proto messages so that unqualified
 *  ``swap(a, b)`` calls in consuming code (e.g. onnxruntime) resolve, mirroring
 *  the friend ``swap`` that protobuf generates for every message class. */
template <typename T, typename = std::enable_if_t<std::is_base_of<Message, T>::value>>
inline void swap(T &a, T &b) noexcept {
  T tmp(std::move(a));
  a = std::move(b);
  b = std::move(tmp);
}

} // namespace ONNX_LIGHT_NAMESPACE
