#pragma once

#include "onnx_light_helpers.h"
#include <cstring>
#include <ostream>
#include <string>
#include <string_view>

namespace ONNX_LIGHT_NAMESPACE {
namespace utils {

class String;

/**
 * String view used by binary readers.
 * It references existing memory and may keep short values in an internal buffer.
 */
class RefString {
private:
  static constexpr size_t kInlineCapacity = 15;
  const char *ptr_;
  size_t size_;
  bool has_inline_;
  char inline_data_[kInlineCapacity];

public:
  /** Initializes a view by copying pointer and size from another view. */
  explicit inline RefString(const RefString &copy)
      : ptr_(copy.ptr_), size_(copy.size_), has_inline_(copy.has_inline_) {
    if (has_inline_) {
      EXT_ENFORCE(size_ <= kInlineCapacity, "RefString inline copy exceeds capacity.");
      memcpy(inline_data_, copy.inline_data_, size_);
      ptr_ = inline_data_;
    }
  }
  /** Initializes a view from a pointer and an explicit size. */
  explicit inline RefString(const char *ptr, size_t size)
      : ptr_(ptr), size_(size), has_inline_(false) {}
  /** Assigns the pointer and size from another view. */
  inline RefString &operator=(const RefString &v) {
    size_ = v.size_;
    has_inline_ = v.has_inline_;
    if (has_inline_) {
      EXT_ENFORCE(size_ <= kInlineCapacity, "RefString inline assignment exceeds capacity.");
      memcpy(inline_data_, v.inline_data_, size_);
      ptr_ = inline_data_;
    } else {
      ptr_ = v.ptr_;
    }
    return *this;
  }
  /** Assigns the pointer and size from an owning string. */
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
  /** Returns whether this view differs from another view. */
  bool operator!=(const RefString &other) const;
  /** Returns whether this view differs from an owning string. */
  bool operator!=(const String &other) const;
  /** Returns whether this view differs from a standard string. */
  bool operator!=(const std::string &other) const;
  /** Returns whether this view differs from a null-terminated string. */
  bool operator!=(const char *other) const;
  /** Converts the view into a standard string. */
  std::string as_string(bool quote = false) const;
  /** Parses the content as a signed 64-bit integer. */
  int64_t toint64() const;
};

/**
 * Owning string type used by ONNX-light protobuf fields.
 * It keeps short values in an internal buffer and uses heap storage for larger values.
 */
class String {
private:
  // The value is stored in a std::string so it can be exposed by reference
  // (see str()) as a drop-in for protobuf string fields, which are std::string.
  // std::string already provides small-string optimization, so this keeps the
  // previous inline-storage behavior for short values without a custom buffer.
  std::string data_;

public:
  /** Releases owned memory. */
  inline ~String() = default;
  /** Resets the instance to an empty state and frees owned memory. */
  inline void clear() { data_.clear(); }
  /** Initializes an empty string. */
  explicit inline String() = default;
  /** Initializes by copying content from a non-owning string view. */
  explicit inline String(const RefString &s) { set(s.data(), s.size()); }
  /** Initializes by copying a pointer and explicit size. */
  explicit inline String(const char *ptr, size_t size) { set(ptr, size); }
  /** Initializes by copying a standard string. */
  explicit String(const std::string &s) { set(s.data(), s.size()); }
  /** Initializes by copying another owning string. */
  explicit String(const String &s) : data_(s.data_) {}
  /** Initializes by taking ownership from another instance. */
  explicit String(String &&other) noexcept : data_(std::move(other.data_)) { other.data_.clear(); }
  /** Returns the number of characters. */
  inline size_t size() const { return data_.size(); }
  /** Returns the number of characters. */
  inline size_t length() const { return data_.size(); }
  /** Returns the underlying pointer. */
  inline const char *data() const { return data_.data(); }
  /** Returns a null-terminated C string (never nullptr). */
  inline const char *c_str() const { return data_.c_str(); }
  /** Indicates whether the string is empty. */
  inline bool empty() const { return data_.empty(); }
  /** Returns a string_view. */
  inline const std::string_view sv() const { return std::string_view(data_); }
  /** Returns the owned value as a standard string reference. This makes the type
   *  a zero-copy drop-in for protobuf string fields (which are std::string) in
   *  consuming code that needs a ``const std::string&``. */
  inline const std::string &str() const noexcept { return data_; }
  /** Indicates whether the string is empty and has no allocated buffer. */
  inline bool null() const { return data_.empty(); }
  /** Returns the character at the specified index. */
  inline char operator[](size_t i) const { return data_[i]; }
  /** Assigns from a null-terminated string. */
  String &operator=(const char *s);
  /** Assigns by taking ownership from another instance. */
  String &operator=(String &&other) noexcept;
  /** Assigns from a non-owning string view. */
  String &operator=(const RefString &s);
  /** Assigns from another owning string. */
  String &operator=(const String &s);
  /** Assigns from a standard string. */
  String &operator=(const std::string &s);
  /** Compares with a standard string. */
  bool operator==(const std::string &other) const;
  /** Compares with another owning string. */
  bool operator==(const String &other) const;
  /** Compares with a non-owning string view. */
  bool operator==(const RefString &other) const;
  /** Compares with a null-terminated string. */
  bool operator==(const char *other) const;
  /** Returns whether this string differs from a standard string. */
  bool operator!=(const std::string &other) const;
  /** Returns whether this string differs from another owning string. */
  bool operator!=(const String &other) const;
  /** Returns whether this string differs from a non-owning string view. */
  bool operator!=(const RefString &other) const;
  /** Returns whether this string differs from a null-terminated string. */
  bool operator!=(const char *other) const;
  /** Returns whether this string is lexicographically less than a standard string. */
  bool operator<(const std::string &other) const;
  /** Returns whether this string is lexicographically less than another owning string. */
  bool operator<(const String &other) const;
  /** Returns whether this string is lexicographically less than a non-owning string view. */
  bool operator<(const RefString &other) const;
  /** Returns whether this string is lexicographically less than a null-terminated string. */
  bool operator<(const char *other) const;
  /** Returns whether this string is lexicographically greater than a standard string. */
  bool operator>(const std::string &other) const;
  /** Returns whether this string is lexicographically greater than another owning string. */
  bool operator>(const String &other) const;
  /** Returns whether this string is lexicographically greater than a non-owning string view. */
  bool operator>(const RefString &other) const;
  /** Returns whether this string is lexicographically greater than a null-terminated string. */
  bool operator>(const char *other) const;
  /** Converts the value into a standard string. */
  std::string as_string(bool quote = false) const;
  /** Implicit conversion to a standard string so the type is a drop-in for
   *  protobuf string fields (which are std::string) in consuming code. */
  inline operator std::string() const { return data_; }
  /** Implicit conversion to a string view (drop-in for protobuf string fields). */
  inline operator std::string_view() const { return std::string_view(data_); }
  /** Parses the content as a signed 64-bit integer. */
  inline int64_t toint64() const { return RefString(data(), size()).toint64(); }

private:
  /** Replaces the content with a copy of the provided buffer. */
  void set(const char *ptr, size_t size);
};

/** Assigns a non-owning view from an owning string. */
inline RefString &RefString::operator=(const String &v) {
  size_ = v.size();
  if (size_ > 0 && size_ <= kInlineCapacity) {
    EXT_ENFORCE(size_ <= kInlineCapacity, "RefString inline conversion exceeds capacity.");
    memcpy(inline_data_, v.data(), size_);
    ptr_ = inline_data_;
    has_inline_ = true;
  } else {
    ptr_ = v.data();
    has_inline_ = false;
  }
  return *this;
}

/** Concatenates rows with a delimiter. */
std::string join_string(const std::vector<std::string> &rows, const char *delimiter = "\n");

/** Streams a RefString to an output stream. */
inline std::ostream &operator<<(std::ostream &os, const RefString &s) {
  os.write(s.data(), static_cast<std::streamsize>(s.size()));
  return os;
}

/** Streams a String to an output stream. */
inline std::ostream &operator<<(std::ostream &os, const String &s) {
  os.write(s.data(), static_cast<std::streamsize>(s.size()));
  return os;
}

// --- Concatenation helpers ---------------------------------------------------
// std::operator+ for basic_string is a function template, so the String ->
// std::string implicit conversion is not considered during template argument
// deduction.  These non-template overloads make ``String`` concatenate with
// ``std::string`` and C strings the same way a protobuf std::string field would.

/** Concatenates an owning string followed by a standard string. */
inline std::string operator+(const String &a, const std::string &b) {
  return std::string(a.sv()) + b;
}
/** Concatenates a standard string followed by an owning string. */
inline std::string operator+(const std::string &a, const String &b) {
  return a + std::string(b.sv());
}
/** Concatenates an owning string followed by a null-terminated string. */
inline std::string operator+(const String &a, const char *b) { return std::string(a.sv()) + b; }
/** Concatenates a null-terminated string followed by an owning string. */
inline std::string operator+(const char *a, const String &b) { return a + std::string(b.sv()); }
/** Concatenates two owning strings. */
inline std::string operator+(const String &a, const String &b) {
  return std::string(a.sv()) + std::string(b.sv());
}

} // namespace utils
} // namespace ONNX_LIGHT_NAMESPACE
