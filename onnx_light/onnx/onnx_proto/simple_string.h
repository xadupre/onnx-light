#pragma once

#include "onnx_light_helpers.h"
#include <cstring>
#include <ostream>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace utils {

class String;

/**
 * Non-owning string view used by binary readers.
 * It references existing memory and never allocates or frees storage.
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
  /** Assigns the pointer and size from an owning string. */
  RefString &operator=(const String &v);
  /** Returns the number of characters in the view. */
  inline size_t size() const { return size_; }
  /** Returns the underlying pointer. */
  inline const char *c_str() const { return ptr_; }
  /** Returns the underlying pointer without ownership. */
  inline const char *data() const { return ptr_; }
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
 * It manages a heap-allocated character buffer with explicit copy and move support.
 */
class String {
private:
  char *ptr_;
  size_t size_;

public:
  /** Releases owned memory. */
  inline ~String() { clear(); }
  /** Resets the instance to an empty state and frees owned memory. */
  inline void clear() {
    if (ptr_ != nullptr) {
      delete[] ptr_;
      ptr_ = nullptr;
    }
    size_ = 0;
  }
  /** Initializes an empty string. */
  explicit inline String() : ptr_(nullptr), size_(0) {}
  /** Initializes by copying content from a non-owning string view. */
  explicit inline String(const RefString &s) { set(s.data(), s.size()); }
  /** Initializes by copying a pointer and explicit size. */
  explicit inline String(const char *ptr, size_t size) { set(ptr, size); }
  /** Initializes by copying a standard string. */
  explicit String(const std::string &s) { set(s.data(), s.size()); }
  /** Initializes by copying another owning string. */
  explicit String(const String &s) { set(s.data(), s.size()); }
  /** Initializes by taking ownership from another instance. */
  explicit String(String &&other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  /** Returns the number of characters. */
  inline size_t size() const { return size_; }
  /** Returns the underlying pointer. */
  inline const char *data() const { return ptr_; }
  /** Indicates whether the string is empty. */
  inline bool empty() const { return size_ == 0; }
  /** Indicates whether the string is empty and has no allocated buffer. */
  inline bool null() const { return size_ == 0 && ptr_ == nullptr; }
  /** Returns the character at the specified index. */
  inline char operator[](size_t i) const { return ptr_[i]; }
  /** Assigns from a null-terminated string. */
  String &operator=(const char *s);
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
  /** Parses the content as a signed 64-bit integer. */
  inline int64_t toint64() const { return RefString(ptr_, size_).toint64(); }

private:
  /** Replaces the content with a copy of the provided buffer. */
  void set(const char *ptr, size_t size);
};

/** Assigns a non-owning view from an owning string. */
inline RefString &RefString::operator=(const String &v) {
  size_ = v.size();
  ptr_ = v.data();
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

} // namespace utils
} // namespace ONNX_LIGHT_NAMESPACE
