#pragma once

#include "onnx_light_helpers.h"
#include <cassert>
#include <cstring>
#include <new>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE {
namespace utils {

class String;

/**
 * Non-owning string view used by binary readers.
 * It references existing memory that must remain valid for the lifetime of this view.
 */
class RefString {
private:
  const char *ptr_;
  size_t size_;

public:
  /** Initializes a view by copying pointer and size from another view. */
  RefString(const RefString &copy) = default;
  /** Initializes a view from a pointer and an explicit size. */
  explicit inline RefString(const char *ptr, size_t size) : ptr_(ptr), size_(size) {}
  /** Assigns the pointer and size from another view. */
  RefString &operator=(const RefString &v) = default;
  /** Assigns the pointer and size from an owning string. */
  RefString &operator=(const String &v);
  /** Returns the number of characters in the view. */
  inline size_t size() const { return size_; }
  /** Returns the underlying null-terminated pointer (asserts null-termination in debug builds). */
  inline const char *c_str() const {
#ifndef NDEBUG
    assert(ptr_ == nullptr || ptr_[size_] == '\0');
#endif
    return ptr_;
  }
  /** Returns the underlying pointer without ownership. */
  inline const char *data() const { return ptr_; }
  /** Returns a string_view. */
  inline const std::string_view sv() const {
    return ptr_ == nullptr ? std::string_view() : std::string_view(ptr_, size_);
  }
  /** Materializes the view into an owning string. */
  inline operator std::string() const {
    return ptr_ == nullptr ? std::string() : std::string(ptr_, size_);
  }
  /** Materializes the view into a string_view. */
  inline operator std::string_view() const { return sv(); }
  /** Indicates whether the view is empty. */
  inline bool empty() const { return size_ == 0; }
  /** Finds a substring inside the view. */
  inline size_t find(std::string_view needle, size_t pos = 0) const {
    return sv().find(needle, pos);
  }
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
  /** Parses the content as a signed 64-bit integer. */
  int64_t toint64() const;
};

/** Shared empty string returned by null String conversions (avoids a per-call static guard). */
inline const std::string kEmptyString;

// The "null" (unset) state is stored without a dedicated bool on standard library
// implementations where a live std::string never has an all-zero object representation
// (verified for libstdc++ and the Microsoft STL). There, String overlaps the std::string
// with a byte buffer and treats all-zero storage as the unset state, so sizeof(String) ==
// sizeof(std::string). Other implementations (e.g. libc++, whose empty string can be
// all-zero) fall back to an explicit bool flag.
#if !defined(ONNX_LIGHT_STRING_NULL_VIA_BYTES)
#if defined(__GLIBCXX__) || defined(_MSVC_STL_VERSION)
#define ONNX_LIGHT_STRING_NULL_VIA_BYTES 1
#else
#define ONNX_LIGHT_STRING_NULL_VIA_BYTES 0
#endif
#endif

/**
 * Owning string type used by ONNX-light protobuf fields.
 * It keeps short values in an internal buffer and uses heap storage for larger values.
 */
class String {
private:
#if ONNX_LIGHT_STRING_NULL_VIA_BYTES
  static_assert(sizeof(std::string) % 8 == 0,
                "null-via-bytes String assumes std::string size is a multiple of 8 bytes.");
  // value_ is alive iff the object representation is not all-zero (see null()).
  union {
    std::string value_;
    unsigned char raw_[sizeof(std::string)];
  };
#else
  std::string value_;
  bool null_;
#endif

  /** Ensures value_ holds a live (non-null) std::string and returns it. */
  inline std::string &mutable_value_() {
#if ONNX_LIGHT_STRING_NULL_VIA_BYTES
    if (null())
      new (&value_) std::string();
#else
    null_ = false;
#endif
    return value_;
  }
  /** Resets the instance to the null (unset) state, releasing any owned memory. */
  inline void reset_null_() {
#if ONNX_LIGHT_STRING_NULL_VIA_BYTES
    if (!null())
      value_.~basic_string();
    std::memset(static_cast<void *>(raw_), 0, sizeof(raw_));
#else
    value_.clear();
    null_ = true;
#endif
  }
  /** Strips a single trailing NUL left by a direct std::string assignment. */
  static inline void normalize_std_string_value(std::string &v) {
    if (!v.empty() && v.back() == 0)
      v.pop_back();
  }
  /** Replaces the content with a copy of the provided buffer. */
  void set(const char *ptr, size_t size);

public:
#if ONNX_LIGHT_STRING_NULL_VIA_BYTES
  /** Indicates whether the string is unset (distinct from an empty value). */
  inline bool null() const {
    // On the supported standard libraries a live std::string never has an all-zero object
    // representation, so all-zero storage uniquely encodes the unset state.
    const unsigned char *bytes = reinterpret_cast<const unsigned char *>(this);
    for (size_t i = 0; i < sizeof(std::string); ++i) {
      if (bytes[i] != 0)
        return false;
    }
    return true;
  }
  /** Releases owned memory. */
  inline ~String() {
    if (!null())
      value_.~basic_string();
  }
  /** Initializes an empty (unset) string. */
  explicit inline String() : raw_{} {}
  /** Initializes by copying another owning string. Explicit to keep accidental deep copies
   *  visible at call sites (use references or std::move for cheap passing). */
  explicit inline String(const String &s) : raw_{} {
    if (!s.null())
      new (&value_) std::string(s.value_);
  }
  /** Initializes by taking ownership from another instance. */
  inline String(String &&other) noexcept : raw_{} {
    if (!other.null()) {
      new (&value_) std::string(std::move(other.value_));
      other.reset_null_();
    }
  }
#else
  /** Indicates whether the string is unset (distinct from an empty value). */
  inline bool null() const { return null_; }
  /** Releases owned memory. */
  inline ~String() = default;
  /** Initializes an empty (unset) string. */
  explicit inline String() : value_(), null_(true) {}
  /** Initializes by copying another owning string. Explicit to keep accidental deep copies
   *  visible at call sites (use references or std::move for cheap passing). */
  explicit inline String(const String &s) : value_(s.value_), null_(s.null_) {}
  /** Initializes by taking ownership from another instance. */
  inline String(String &&other) noexcept : value_(std::move(other.value_)), null_(other.null_) {
    other.reset_null_();
  }
#endif

  /** Resets the instance to an empty state and frees owned memory. */
  inline void clear() { reset_null_(); }
  /** Initializes by copying content from a non-owning string view. */
  explicit inline String(const RefString &s) : String() { set(s.data(), s.size()); }
  /** Initializes by copying a pointer and explicit size. */
  explicit inline String(const char *ptr, size_t size) : String() { set(ptr, size); }
  /** Initializes by copying a standard string. */
  explicit inline String(const std::string &s) : String() { *this = s; }
  /** Initializes by taking ownership from a standard string. */
  explicit inline String(std::string &&s) noexcept : String() { *this = std::move(s); }
  /** Returns the number of characters. */
  inline size_t size() const { return null() ? 0 : value_.size(); }
  /** Returns the number of characters. */
  inline size_t length() const { return size(); }
  /** Returns the underlying pointer. */
  inline const char *data() const { return null() ? nullptr : value_.data(); }
  /** Returns a null-terminated C string (never nullptr). */
  inline const char *c_str() const { return null() ? "" : value_.c_str(); }
  /** Indicates whether the string is empty. */
  inline bool empty() const { return null() || value_.empty(); }
  /** Returns a string_view. */
  inline const std::string_view sv() const {
    return null() ? std::string_view() : std::string_view(value_.data(), value_.size());
  }
  /** Returns a const reference to the underlying std::string (empty string for null). */
  inline operator const std::string &() const { return null() ? kEmptyString : value_; }
  /** Returns a shared empty std::string usable as a zero-copy reference fallback, e.g. the
   *  else-branch of a conditional bound to a const std::string&. */
  static inline const std::string &empty_string() { return kEmptyString; }
  /** Materializes the string into a string_view. */
  inline operator std::string_view() const { return sv(); }
  /** Finds a substring inside the string. */
  inline size_t find(std::string_view needle, size_t pos = 0) const {
    return sv().find(needle, pos);
  }
  /** Returns the character at the specified index. */
  inline char operator[](size_t i) const { return null() ? char(0) : value_[i]; }
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
  /** Assigns by taking ownership from a standard string. */
  String &operator=(std::string &&s) noexcept;
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
  /** Implicit conversion to a standard string so the type is a drop-in for
   *  protobuf string fields (which are std::string) in consuming code. */
  /** Parses the content as a signed 64-bit integer. */
  inline int64_t toint64() const { return RefString(data(), size()).toint64(); }
};

/** Assigns a non-owning view from an owning string. */
inline RefString &RefString::operator=(const String &v) {
  ptr_ = v.data();
  size_ = v.size();
  return *this;
}

/** Concatenates rows with a delimiter. */
std::string join_string(const std::vector<std::string> &rows, const char *delimiter = "\n");

/** Quotes a string view for debug and error messages. */
inline std::string quote_string(std::string_view s) { return "\"" + std::string(s) + "\""; }
/** Quotes a RefString for debug and error messages. */
inline std::string quote_string(const RefString &s) { return quote_string(s.sv()); }
/** Quotes an owning string for debug and error messages. */
inline std::string quote_string(const String &s) { return quote_string(s.sv()); }

/** Appends an owning string to a MakeString stream without copying. */
inline void MakeStringInternalElement(onnx_light_helpers::StringStream &ss, const String &s) {
  ss.append_string(static_cast<const std::string &>(s));
}

/** Appends a non-owning string view to a MakeString stream. */
inline void MakeStringInternalElement(onnx_light_helpers::StringStream &ss, const RefString &s) {
  onnx_light_helpers::MakeStringInternalElement(ss, s.sv());
}

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
