#pragma once

#include "simple_span.h"
#include "simple_string.h"
#include "stream.h"

#define FIELD_VARINT 0
// #define FIELD_FIXED64 1
#define FIELD_FIXED_SIZE 2
#define FIELD_FIXED32 5 // deprecated value but used in old files

/** Serialization/parsing API declaration macro for generated proto classes. */
#define SERIALIZATION_METHOD()                                                                     \
  uint64_t SerializeSize() const;                                                                  \
  void ParseFromString(const std::string &raw);                                                    \
  void ParseFromString(const std::string &raw, ParseOptions &opts);                                \
  void SerializeToString(std::string &out) const;                                                  \
  void SerializeToString(std::string &out, SerializeOptions &opts) const;                          \
  uint64_t SerializeSize(utils::BinaryWriteStream &stream, SerializeOptions &opts) const;          \
  void ParseFromStream(utils::BinaryStream &stream, ParseOptions &options);                        \
  void SerializeToStream(utils::BinaryWriteStream &stream, SerializeOptions &options) const;       \
  std::vector<std::string> PrintToVectorString(utils::PrintOptions &options) const;

/** Macro for beginning a generated proto class with a default constructor. */
#define BEGIN_PROTO(cls, doc)                                                                      \
  class cls : public Message {                                                                     \
  public:                                                                                          \
    static inline constexpr const char *DOC = doc;                                                 \
    explicit inline cls() {}                                                                       \
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
  inline const type *ptr_##name() const { return &name##_; }                                       \
  inline bool has_##name() const { return _has_field_(name##_); }                                  \
  inline void set_##name(const type &v) { name##_ = v; }                                           \
  inline int order_##name() const { return order; }                                                \
  static inline constexpr const char *_name_##name = #name;                                        \
  static inline constexpr const char *DOC_##name = doc;                                            \
  type name##_;                                                                                    \
  using name##_t = type;

#define FIELD_DEFAULT(type, name, order, default_value, doc)                                       \
public:                                                                                            \
  inline type &ref_##name() { return name##_; }                                                    \
  inline const type &ref_##name() const { return name##_; }                                        \
  inline const type *ptr_##name() const { return &name##_; }                                       \
  inline bool has_##name() const { return _has_field_(name##_); }                                  \
  inline void set_##name(const type &v) { name##_ = v; }                                           \
  inline int order_##name() const { return order; }                                                \
  static inline constexpr const char *_name_##name = #name;                                        \
  static inline constexpr const char *DOC_##name = doc;                                            \
  type name##_ = default_value;                                                                    \
  using name##_t = type;

#define FIELD_STR(name, order, doc)                                                                \
  FIELD(utils::String, name, order, doc)                                                           \
  inline void set_##name(const std::string &v) { name##_ = v; }                                    \
  inline void set_##name(const utils::RefString &v) { name##_ = v; }

#define FIELD_REPEATED(type, name, order, doc)                                                     \
public:                                                                                            \
  inline utils::RepeatedField<type> &ref_##name() { return name##_; }                              \
  inline const utils::RepeatedField<type> &ref_##name() const { return name##_; }                  \
  inline const utils::RepeatedField<type> *ptr_##name() const { return &name##_; }                 \
  inline type &add_##name() { return name##_.add(); }                                              \
  inline type &add_##name(type &&v) {                                                              \
    name##_.emplace_back(v);                                                                       \
    return name##_.back();                                                                         \
  }                                                                                                \
  inline bool has_##name() const { return _has_field_(name##_) && !name##_.empty(); }              \
  inline int order_##name() const { return order; }                                                \
  inline void clr_##name() { name##_.clear(); }                                                    \
  static inline constexpr const char *DOC_##name = doc;                                            \
  static inline constexpr const char *_name_##name = #name;                                        \
  inline bool packed_##name() const { return false; }                                              \
  utils::RepeatedField<type> name##_;                                                              \
  using name##_t = type;

#define FIELD_REPEATED_PROTO(type, name, order, doc)                                               \
public:                                                                                            \
  inline utils::RepeatedProtoField<type> &ref_##name() { return name##_; }                         \
  inline const utils::RepeatedProtoField<type> &ref_##name() const { return name##_; }             \
  inline const utils::RepeatedProtoField<type> *ptr_##name() const { return &name##_; }            \
  inline type &add_##name() { return name##_.add(); }                                              \
  inline type &add_##name(const type &v) {                                                         \
    name##_.push_back(v);                                                                          \
    return name##_.back();                                                                         \
  }                                                                                                \
  inline bool has_##name() const { return _has_field_(name##_) && !name##_.empty(); }              \
  inline int order_##name() const { return order; }                                                \
  inline void clr_##name() { name##_.clear(); }                                                    \
  static inline constexpr const char *DOC_##name = doc;                                            \
  static inline constexpr const char *_name_##name = #name;                                        \
  inline bool packed_##name() const { return false; }                                              \
  utils::RepeatedProtoField<type> name##_;                                                         \
  using name##_t = type;

#define FIELD_REPEATED_PACKED(type, name, order, doc)                                              \
public:                                                                                            \
  inline utils::RepeatedField<type> &ref_##name() { return name##_; }                              \
  inline const utils::RepeatedField<type> &ref_##name() const { return name##_; }                  \
  inline const utils::RepeatedField<type> *ptr_##name() const { return &name##_; }                 \
  inline type &add_##name() { return name##_.add(); }                                              \
  inline type &add_##name(const type &v) {                                                         \
    name##_.push_back(v);                                                                          \
    return name##_.back();                                                                         \
  }                                                                                                \
  inline bool has_##name() const { return _has_field_(name##_) && !name##_.empty(); }              \
  inline int order_##name() const { return order; }                                                \
  inline void clr_##name() { name##_.clear(); }                                                    \
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
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value.");                \
    return *name##_;                                                                               \
  }                                                                                                \
  inline const type *ptr_##name() const {                                                          \
    return has_##name() ? &(*name##_) : static_cast<type *>(nullptr);                              \
  }                                                                                                \
  inline utils::OptionalField<type> &name##_optional() { return name##_; }                         \
  inline const utils::OptionalField<type> &name##_optional() const {                               \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value.");                \
    return name##_;                                                                                \
  }                                                                                                \
  inline type &add_##name() {                                                                      \
    name##_.set_empty_value();                                                                     \
    return *name##_;                                                                               \
  }                                                                                                \
  inline void set_##name(const type &v) { name##_ = v; }                                           \
  inline void reset_##name() { name##_.reset(); }                                                  \
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
    EXT_ENFORCE(name##_.has_value(), "Optional enum field '", #name, "' has no value.");           \
    return *name##_;                                                                               \
  }                                                                                                \
  inline const type *ptr_##name() const {                                                          \
    return has_##name() ? &(*name##_) : static_cast<type *>(nullptr);                              \
  }                                                                                                \
  inline utils::OptionalEnumField<type> &name##_optional() { return name##_; }                     \
  inline const utils::OptionalEnumField<type> &name##_optional() const {                           \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value.");                \
    return name##_;                                                                                \
  }                                                                                                \
  inline type &add_##name() {                                                                      \
    name##_.set_empty_value();                                                                     \
    return *name##_;                                                                               \
  }                                                                                                \
  inline void set_##name(const type &v) { name##_ = v; }                                           \
  inline void reset_##name() { name##_.reset(); }                                                  \
  inline bool has_##name() const { return name##_.has_value(); }                                   \
  inline int order_##name() const { return order; }                                                \
  static inline constexpr const char *DOC_##name = doc;                                            \
  static inline constexpr const char *_name_##name = #name;                                        \
  utils::OptionalEnumField<type> name##_;                                                          \
  using name##_t = type;

namespace onnx {

/** Controls behavior when parsing ONNX protobuf messages from a stream or string. */
struct ParseOptions {
  /** if true, raw data will not be read but skipped, tensors are not valid in that case  but the
   * model structure is still available */
  bool skip_raw_data = false;
  /** if skip_raw_data is true, raw data will be read only if it is larger than the threshold */
  int64_t raw_data_threshold = 1024;
  /** parallelizes the reading of the big blocks */
  bool parallel = false;
  /** number of threads to run in parallel if parallel is true, -1 for as many threads as the number
   * of cores */
  int32_t num_threads = -1;
  /** minimum raw-data block size in bytes to submit to the thread pool when parallel is true;
   * blocks smaller than this value are read on the main thread to avoid thread-pool overhead */
  int64_t min_parallel_block_size = 0;
  /** If true, raw_data blocks are not copied into a new buffer.  Instead, the tensor's
   * raw_data_ field is set to borrowed mode pointing into the source byte buffer (e.g. the
   * bytes passed to ParseFromString).  The caller MUST keep that buffer alive for as
   * long as any TensorProto that references it.  Ignored for file-backed streams. */
  bool no_copy = false;
};

/** Controls behavior when serializing ONNX protobuf messages to a stream or string. */
struct SerializeOptions {
  /** if true, raw data will not be written but skipped, tensors are not valid in that case but the
   * model structure is still available */
  bool skip_raw_data = false;
  /** if skip_raw_data is true, raw data will be written only if it is larger than the threshold */
  int64_t raw_data_threshold = 1024;
  /** parallelizes the writing of the big blocks */
  bool parallel = false;
  /** number of threads to run in parallel if parallel is true, -1 for as many threads as the number
   * of cores */
  int32_t num_threads = -1;
  /** minimum raw-data block size in bytes to submit to the thread pool when parallel is true;
   * blocks smaller than this value are written on the main thread to avoid thread-pool overhead */
  int64_t min_parallel_block_size = 0;
  /** if true, tensors already marked with data_location=EXTERNAL are serialized using their
   * external_data metadata location (can target multiple weights files). */
  bool use_external_data_location = true;
  /** maximum size in bytes for one external weights file when saving with external data;
   * 0 means no limit (single weights file) */
  int64_t max_external_file_size = 0;
};

using utils::offset_t;

/** Returns true if the field holds a non-default value (always true for scalar types other
 *  than String and raw-bytes vectors, which have their own specializations).
 */
template <typename T> inline bool _has_field_(const T &) { return true; }
/** Returns true if the string field is non-empty. */
template <> inline bool _has_field_(const utils::String &field) { return !field.empty(); }
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

} // namespace onnx
