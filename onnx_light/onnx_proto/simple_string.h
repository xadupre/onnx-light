#pragma once

#include "onnx_light_helpers.h"
#include <cstring>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace utils {

class String;

/**
 * String view used by binary readers.
 * It references existing memory and does not own it.
 */
class RefString {
private:
  const char *ptr_;
  size_t size_;

public:
  /** Initializes a view by copying pointer and size from another view. */
  explicit inline RefString(const RefString &copy) : ptr_(copy.ptr_), size_(copy.size_) {}
  /** Initializes a view from a pointer and an explicit size. */
  explicit inline RefString(const char *ptr, size_t size) : ptr_(ptr), size_(size) {}
  /** Assigns the pointer and size from another view. */
  inline RefString &operator=(const RefString &v) {
    ptr_ = v.ptr_;
    size_ = v.size_;
    return *this;
  }
  /** Assigns the pointer and size from an owning string (non-owning view). */
  RefString &operator=(const String &v);
  /** Returns the number of characters in the view. */
  inline size_t size() const { return size_; }
  /** Returns the underlying pointer. */
  inline const char *c_str() const { return ptr_; }
  /** Returns the underlying pointer without ownership. */
  inline const char *data() const { return ptr_; }
  /** Returns a string_view. */
  inline const std::string_view sv() const {
    return ptr_ == nullptr ? std::string_view() : std::string_view(ptr_, size_);
  }
  /** Indicates whether the view is empty. */
  inline bool empty() const { return size_ == 0; }
  /** Returns the character at the specified index. */
  inline char operator[](size_t i) const { return ptr_[i]; }
  /** Compares this view with another view. */
  bool operator==(const RefString &other) const;
  /** Compares this view with an owning string. */
  bool operator==(const String &other) const;
  /** Compares this view with a standard string. */
  bool operator==(const std::string &other) const;
  /** Compares this view with a null-terminated string. */
  bool operator==(const char *other) const;
  /** Converts the view into an owning standard string. */
  inline operator std::string() const {
    return ptr_ == nullptr ? std::string() : std::string(ptr_, size_);
  }
  /** Parses the content as a signed 64-bit integer. */
  int64_t toint64() const;
};

/**
 * Owning string type used for repeated ONNX-light protobuf string fields
 * (for example NodeProto inputs/outputs). It is a thin std::string subclass so it is a
 * drop-in std::string with a few protobuf-friendly helpers.
 */
class String : public std::string {
public:
  using std::string::string;
  using std::string::operator=;
  /** Initializes an empty string. */
  String() = default;
  /** Copy/move follow std::string semantics. */
  String(const String &) = default;
  String(String &&) = default;
  String &operator=(const String &) = default;
  String &operator=(String &&) = default;
  /** Initializes by copying a standard string. */
  String(const std::string &s) : std::string(s) {}
  /** Initializes by taking ownership from a standard string. */
  String(std::string &&s) noexcept : std::string(std::move(s)) {}
  /** Initializes by copying content from a non-owning string view. */
  explicit String(const RefString &s)
      : std::string(s.data() == nullptr ? std::string() : std::string(s.data(), s.size())) {}
  /** Assigns from a non-owning string view. */
  String &operator=(const RefString &s) {
    if (s.data() == nullptr) {
      clear();
    } else {
      assign(s.data(), s.size());
    }
    return *this;
  }
  /** Returns a const reference to the underlying std::string (protobuf compatibility). */
  inline const std::string &value() const { return *this; }
  /** Returns a mutable pointer to the underlying std::string (protobuf compatibility). */
  inline std::string *mutable_ptr() { return this; }
  /** Returns a string_view over the content. */
  inline std::string_view sv() const { return std::string_view(data(), size()); }
  /** Parses the content as a signed 64-bit integer. */
  inline int64_t toint64() const { return RefString(data(), size()).toint64(); }
  /** Returns a shared empty String, usable to bind a reference to a default value. */
  static inline const String &empty_string() {
    static const String kEmpty;
    return kEmpty;
  }
};

/**
 * Owning optional string type used for singular ONNX-light protobuf string fields, mirroring
 * proto2 ``optional string`` presence semantics. It is a thin std::optional<std::string>
 * subclass with protobuf-friendly read helpers so it behaves like a string when set.
 */
class OptionalString : public std::optional<std::string> {
public:
  using base = std::optional<std::string>;
  /** Initializes an unset (absent) value. */
  OptionalString() = default;
  OptionalString(const OptionalString &) = default;
  OptionalString(OptionalString &&) = default;
  OptionalString &operator=(const OptionalString &) = default;
  OptionalString &operator=(OptionalString &&) = default;
  /** Initializes from an absent marker. */
  OptionalString(std::nullopt_t) : base(std::nullopt) {}
  /** Initializes from a standard optional. */
  OptionalString(const base &o) : base(o) {}
  OptionalString(base &&o) : base(std::move(o)) {}
  /** Initializes to a present value from a null-terminated string (nullptr is absent). */
  OptionalString(const char *s) : base(s == nullptr ? base(std::nullopt) : base(std::string(s))) {}
  /** Initializes to a present value from a standard string. */
  OptionalString(const std::string &s) : base(s) {}
  OptionalString(std::string &&s) : base(std::move(s)) {}
  /** Initializes to a present value from an owning repeated string. */
  OptionalString(const String &s) : base(static_cast<const std::string &>(s)) {}
  /** Initializes to a present value from a non-owning string view. */
  OptionalString(const RefString &s)
      : base(s.data() == nullptr ? base(std::nullopt) : base(std::string(s.data(), s.size()))) {}

  /** Assigns a present value from a null-terminated string (nullptr clears the value). */
  OptionalString &operator=(const char *s) {
    if (s == nullptr) {
      reset();
    } else {
      emplace(s);
    }
    return *this;
  }
  /** Assigns a present value from a standard string. */
  OptionalString &operator=(const std::string &s) {
    emplace(s);
    return *this;
  }
  OptionalString &operator=(std::string &&s) {
    emplace(std::move(s));
    return *this;
  }
  /** Assigns a present value from a non-owning string view. */
  OptionalString &operator=(const RefString &s) {
    if (s.data() == nullptr) {
      reset();
    } else {
      emplace(s.data(), s.size());
    }
    return *this;
  }
  /** Assigns a present value from an owning repeated string. */
  OptionalString &operator=(const String &s) {
    emplace(static_cast<const std::string &>(s));
    return *this;
  }

  /** Indicates whether the field is unset (absent). */
  inline bool null() const { return !has_value(); }
  /** Returns the number of characters (0 when unset). */
  inline size_t size() const { return has_value() ? value().size() : 0; }
  /** Returns the number of characters (0 when unset). */
  inline size_t length() const { return size(); }
  /** Indicates whether the value is unset or empty. */
  inline bool empty() const { return !has_value() || value().empty(); }
  /** Returns the underlying pointer (nullptr when unset). */
  inline const char *data() const { return has_value() ? value().data() : nullptr; }
  /** Returns a null-terminated C string (never nullptr; empty when unset). */
  inline const char *c_str() const { return has_value() ? value().c_str() : ""; }
  /** Returns a string_view over the content (empty when unset). */
  inline std::string_view sv() const {
    return has_value() ? std::string_view(value()) : std::string_view();
  }
  /** Returns a shared empty string, usable to bind a reference to a default value. */
  static inline const std::string &empty_value() {
    static const std::string kEmpty;
    return kEmpty;
  }
  /** Returns the stored string, or a shared empty string when unset (never throws). */
  inline const std::string &value() const { return has_value() ? base::value() : empty_value(); }
  /** Returns the character at the specified index (requires a value). */
  inline char operator[](size_t i) const { return value()[i]; }
  /** Parses the content as a signed 64-bit integer. */
  inline int64_t toint64() const { return RefString(data(), size()).toint64(); }
  /** Implicit conversion to const std::string& (shared empty when unset).
   *  Allows binding to const std::string& in call sites that expect protobuf semantics. */
  inline operator const std::string &() const { return value(); }
  /** Implicit conversion to string_view (empty when unset). */
  inline operator std::string_view() const {
    return has_value() ? std::string_view(value()) : std::string_view();
  }
  /** Assigns content from a character range, matching protobuf's assign(ptr, len) API. */
  inline void assign(const char *s, size_t len) { emplace(s, len); }
  /** Assigns content from a standard string. */
  inline void assign(const std::string &s) { emplace(s); }
  /** Returns a mutable reference to the underlying string, creating it if absent. */
  inline std::string &mutable_ref() {
    if (!has_value())
      emplace();
    return base::value();
  }
  /** Returns a mutable pointer to the underlying string, creating it if absent. */
  inline std::string *mutable_ptr() { return &mutable_ref(); }
  /** Delegates std::string::find to the underlying value (npos if absent). */
  inline size_t find(const char *s, size_t pos = 0) const {
    return has_value() ? value().find(s, pos) : std::string::npos;
  }
  inline size_t find(const std::string &s, size_t pos = 0) const {
    return has_value() ? value().find(s, pos) : std::string::npos;
  }
  inline size_t find(char c, size_t pos = 0) const {
    return has_value() ? value().find(c, pos) : std::string::npos;
  }
  inline int compare(const std::string &other) const {
    const std::string &s = *this;
    return s.compare(other);
  }
  inline int compare(const char *other) const {
    const std::string &s = *this;
    return s.compare(other);
  }
  /** Delegates std::string::substr. Returns empty string if absent. */
  inline std::string substr(size_t pos = 0, size_t len = std::string::npos) const {
    return has_value() ? value().substr(pos, len) : std::string();
  }
  /** Compares two OptionalString values for equality (disambiguation for C++20 heterogeneous
   * optional overloads). */
  inline bool operator==(const OptionalString &other) const {
    return static_cast<const base &>(*this) == static_cast<const base &>(other);
  }
  inline bool operator!=(const OptionalString &other) const { return !(*this == other); }
  /** Compares against a null-terminated string (nullptr matches absent). */
  inline bool operator==(const char *other) const {
    if (other == nullptr)
      return !has_value();
    return has_value() && value() == other;
  }
  inline bool operator!=(const char *other) const { return !(*this == other); }
  /** Compares against a standard string (unset never equals any std::string). */
  inline bool operator==(const std::string &other) const { return has_value() && value() == other; }
  inline bool operator!=(const std::string &other) const { return !(*this == other); }

  /** String concatenation with standard strings and C strings. */
  friend inline std::string operator+(const OptionalString &lhs, const char *rhs) {
    return std::string(lhs) + rhs;
  }
  friend inline std::string operator+(const char *lhs, const OptionalString &rhs) {
    return lhs + std::string(rhs);
  }
  friend inline std::string operator+(const OptionalString &lhs, const std::string &rhs) {
    return std::string(lhs) + rhs;
  }
  friend inline std::string operator+(const std::string &lhs, const OptionalString &rhs) {
    return lhs + std::string(rhs);
  }
  friend inline std::string operator+(const OptionalString &lhs, const OptionalString &rhs) {
    return std::string(lhs) + std::string(rhs);
  }
};

/** Assigns a non-owning view from an owning string. */
inline RefString &RefString::operator=(const String &v) {
  ptr_ = v.data();
  size_ = v.size();
  return *this;
}

/** Concatenates rows with a delimiter. */
std::string join_string(const std::vector<std::string> &rows, const char *delimiter = "\n");

/** Returns a copy of the text, optionally wrapped in double quotes (used by printers). */
inline std::string quote_string(std::string_view text, bool quote = true) {
  if (!quote) {
    return std::string(text);
  }
  std::string result;
  result.reserve(text.size() + 2);
  result.push_back('"');
  result.append(text);
  result.push_back('"');
  return result;
}

/** Streams a RefString to an output stream. */
inline std::ostream &operator<<(std::ostream &os, const RefString &s) {
  os.write(s.data(), static_cast<std::streamsize>(s.size()));
  return os;
}

/** Streams an OptionalString to an output stream (empty when unset). */
inline std::ostream &operator<<(std::ostream &os, const OptionalString &s) {
  if (s.has_value()) {
    os.write(s.data(), static_cast<std::streamsize>(s.size()));
  }
  return os;
}

} // namespace utils
} // namespace ONNX_LIGHT_NAMESPACE

/** Specializes std::hash for OptionalString and String so they can be used as
 * keys in unordered containers (and with absl hashing). */
template <> struct std::hash<ONNX_LIGHT_NAMESPACE::utils::OptionalString> {
  size_t operator()(const ONNX_LIGHT_NAMESPACE::utils::OptionalString &s) const noexcept {
    return s.has_value() ? std::hash<std::string>{}(s.value()) : 0;
  }
};
template <> struct std::hash<ONNX_LIGHT_NAMESPACE::utils::String> {
  size_t operator()(const ONNX_LIGHT_NAMESPACE::utils::String &s) const noexcept {
    return std::hash<std::string>{}(static_cast<const std::string &>(s));
  }
};
