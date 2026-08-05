#pragma once

#include "serialize_options.h"
#include "simple_span.h"
#include "simple_string.h"
#include "stream.h"
#include <functional>
#include <sstream>
#include <type_traits>
#include <utility>

namespace onnx_light {
namespace proto_default_detail {

/// Returns a const reference to a default-constructed instance of T.
/// Uses a template to defer instantiation until the type is complete.
template <typename T> inline const T &default_proto_instance() {
  static const T instance{};
  return instance;
}

} // namespace proto_default_detail
} // namespace onnx_light

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
 *  Bytes are exposed as a ``utils::ByteSpan`` (breaking change).  The old protobuf-style          \
 *  ``name()``/``mutable_##name()`` accessors returned ``const std::string&``/``std::string*``,    \
 *  which is unsafe here: a ByteSpan may hold borrowed (zero-copy) or aligned-owned bytes that     \
 *  are not backed by a std::string, so exposing them as a std::string would require a hidden copy \
 *  and could not preserve alignment or borrowed semantics.  ``name()``/``mutable_##name()`` are   \
 *  kept but now return ``utils::ByteSpan``, equivalent to ref_##name(); use the ByteSpan API      \
 *  (data()/size()/assign()/...) to read or mutate bytes. */
#define FIELD_BYTES(name, order, doc)                                                              \
public:                                                                                            \
  inline utils::ByteSpan &ref_##name() { return name##_; }                                         \
  inline const utils::ByteSpan &ref_##name() const { return name##_; }                             \
  /** Compatibility accessor - equivalent to ref_##name(); returns a ByteSpan. */                  \
  inline utils::ByteSpan &name() { return name##_; }                                               \
  /** Compatibility accessor - equivalent to ref_##name() const; returns a ByteSpan. */            \
  inline const utils::ByteSpan &name() const { return name##_; }                                   \
  inline const utils::ByteSpan *ptr_##name() const { return &name##_; }                            \
  inline bool has_##name() const { return name##_.data() != nullptr; }                             \
  inline void set_##name(const utils::ByteSpan &v) {                                               \
    name##_ = v;                                                                                   \
    name##_.set_empty();                                                                           \
  }                                                                                                \
  /** Compatibility accessor returning a mutable pointer to the ByteSpan field. */                 \
  inline utils::ByteSpan *mutable_##name() {                                                       \
    name##_.set_empty();                                                                           \
    return &name##_;                                                                               \
  }                                                                                                \
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
public:                                                                                            \
  inline utils::OptionalString &ref_##name() { return name##_; }                                   \
  inline const utils::OptionalString &ref_##name() const { return name##_; }                       \
  /** Compatibility accessor - equivalent to ref_##name(). */                                      \
  inline utils::OptionalString &name() { return name##_; }                                         \
  /** Compatibility accessor - equivalent to ref_##name() const. */                                \
  inline const utils::OptionalString &name() const { return name##_; }                             \
  /** Returns the stored string, or a shared empty string when unset (never throws). */            \
  inline const std::string &str_##name() const { return name##_.value(); }                         \
  inline const utils::OptionalString *ptr_##name() const { return &name##_; }                      \
  inline bool has_##name() const { return _has_field_(name##_); }                                  \
  inline void set_##name(const utils::OptionalString &v) { name##_ = v; }                          \
  /** Returns a mutable std::string pointer (protobuf compat). */                                  \
  inline std::string *mutable_##name() { return name##_.mutable_ptr(); }                           \
  inline int order_##name() const { return order; }                                                \
  static inline constexpr const char *_name_##name = #name;                                        \
  static inline constexpr const char *DOC_##name = doc;                                            \
  utils::OptionalString name##_;                                                                   \
  using name##_t = utils::OptionalString;                                                          \
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
  inline void add_##name(const utils::RefString &v) { name##_.push_back(utils::String(v)); }       \
  inline void add_##name(const utils::OptionalString &v) {                                         \
    name##_.push_back(utils::String(v.value()));                                                   \
  }

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
    if (!name##_.has_value()) {                                                                    \
      return ::onnx_light::proto_default_detail::default_proto_instance<type>();                   \
    }                                                                                              \
    return *name##_;                                                                               \
  }                                                                                                \
  /** Compatibility accessor - returns a const reference like protobuf, so read access on a        \
   *  non-const message never auto-creates the sub-message; use mutable_##name() to create it. */  \
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
