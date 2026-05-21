#include "simple_string.h"
#include <charconv>
#include <sstream>

namespace ONNX_LIGHT_NAMESPACE {
namespace utils {

bool RefString::operator==(const char *other) const {
  if (size_ == 0)
    return other == nullptr || other[0] == 0;
  if (other == nullptr)
    return false;
  size_t i;
  for (i = 0; i < size_ && ptr_[i] == other[i] && other[i] != 0; ++i)
    ;
  return i == size_ && other[i] == 0;
}

bool RefString::operator==(const RefString &other) const {
  if (size() != other.size())
    return false;
  if (size() == 0)
    return true;
  if (data() == other.data())
    return true;
  size_t i;
  for (i = 0; i < size_ && ptr_[i] == other[i]; ++i)
    ;
  return i == size_;
}

bool RefString::operator==(const std::string &other) const {
  return *this == RefString(other.data(), other.size());
}

bool RefString::operator==(const String &other) const {
  return *this == RefString(other.data(), other.size());
}

bool RefString::operator!=(const std::string &other) const { return !(*this == other); }
bool RefString::operator!=(const String &other) const { return !(*this == other); }
bool RefString::operator!=(const RefString &other) const { return !(*this == other); }
bool RefString::operator!=(const char *other) const { return !(*this == other); }

std::string RefString::as_string(bool quote) const {
  if (empty())
    return quote ? std::string("\"\"") : std::string();
  auto s = std::string(data(), size());
  return quote ? std::string("\"") + s + std::string("\"") : s;
}

int64_t RefString::toint64() const {
  // Use std::from_chars for fast, allocation-free integer parsing.
  int64_t result = 0;
  auto [ptr, ec] = std::from_chars(ptr_, ptr_ + size_, result);
  EXT_ENFORCE(ec == std::errc{} && ptr == ptr_ + size_,
              "Invalid integer string passed to toint64(): '", as_string(), "'");
  return result;
}

void String::set(const char *ptr, size_t size) {
  size_t effective_size = size;
  if (size == SIZE_MAX) {
    if (ptr == nullptr) {
      effective_size = 0;
    } else {
      const char *p = ptr;
      effective_size = 0;
      for (; *p != 0; ++p, ++effective_size)
        ;
    }
  }
  if (ptr == nullptr) {
    ptr_ = nullptr;
    size_ = 0;
    is_inline_ = false;
    return;
  }
  if (effective_size > 0 && ptr[effective_size - 1] == 0) {
    --effective_size;
  }
  if (effective_size <= kInlineCapacity) {
    EXT_ENFORCE(effective_size <= kInlineCapacity, "String inline storage exceeds capacity.");
    if (effective_size > 0) {
      memcpy(inline_data_, ptr, effective_size);
    }
    ptr_ = inline_data_;
    size_ = effective_size;
    is_inline_ = true;
  } else {
    ptr_ = new char[effective_size];
    memcpy(ptr_, ptr, effective_size);
    size_ = effective_size;
    is_inline_ = false;
  }
}

bool String::operator==(const char *other) const {
  if (size_ == 0)
    return other == nullptr || other[0] == 0;
  if (other == nullptr)
    return false;
  size_t i;
  for (i = 0; i < size_ && ptr_[i] == other[i] && other[i] != 0; ++i)
    ;
  return i == size_ && other[i] == 0;
}

bool String::operator==(const RefString &other) const {
  if (size() != other.size())
    return false;
  if (size() == 0)
    return true;
  size_t i;
  for (i = 0; i < size_ && ptr_[i] == other[i]; ++i)
    ;
  return i == size_;
}

bool String::operator==(const String &other) const {
  return *this == RefString(other.data(), other.size());
}

bool String::operator==(const std::string &other) const {
  return *this == RefString(other.data(), other.size());
}

bool String::operator!=(const std::string &other) const { return !(*this == other); }
bool String::operator!=(const String &other) const { return !(*this == other); }
bool String::operator!=(const RefString &other) const { return !(*this == other); }
bool String::operator!=(const char *other) const { return !(*this == other); }

bool String::operator<(const RefString &other) const {
  size_t min_size = size_ < other.size() ? size_ : other.size();
  if (min_size > 0) {
    int cmp = memcmp(ptr_, other.data(), min_size);
    if (cmp != 0)
      return cmp < 0;
  }
  return size_ < other.size();
}

bool String::operator<(const std::string &other) const {
  return *this < RefString(other.data(), other.size());
}

bool String::operator<(const String &other) const {
  return *this < RefString(other.data(), other.size());
}

bool String::operator<(const char *other) const {
  if (other == nullptr)
    return false;
  if (ptr_ == nullptr)
    return other[0] != 0;
  for (size_t i = 0; i < size_; ++i) {
    if (other[i] == 0)
      return false;
    if (static_cast<unsigned char>(ptr_[i]) < static_cast<unsigned char>(other[i]))
      return true;
    if (static_cast<unsigned char>(ptr_[i]) > static_cast<unsigned char>(other[i]))
      return false;
  }
  return other[size_] != 0;
}

bool String::operator>(const std::string &other) const {
  return *this > RefString(other.data(), other.size());
}

bool String::operator>(const String &other) const {
  return *this > RefString(other.data(), other.size());
}

bool String::operator>(const RefString &other) const {
  size_t min_size = other.size() < size_ ? other.size() : size_;
  if (min_size > 0) {
    int cmp = memcmp(other.data(), ptr_, min_size);
    if (cmp != 0)
      return cmp < 0;
  }
  return other.size() < size_;
}

bool String::operator>(const char *other) const {
  if (ptr_ == nullptr || size_ == 0)
    return false;
  if (other == nullptr)
    return true;
  for (size_t i = 0; i < size_; ++i) {
    if (other[i] == 0)
      return true;
    if (static_cast<unsigned char>(ptr_[i]) > static_cast<unsigned char>(other[i]))
      return true;
    if (static_cast<unsigned char>(ptr_[i]) < static_cast<unsigned char>(other[i]))
      return false;
  }
  return false;
}

std::string String::as_string(bool quote) const {
  if (empty())
    return quote ? std::string("\"\"") : std::string();
  auto s = std::string(data(), size());
  return quote ? std::string("\"") + s + std::string("\"") : s;
}

std::string join_string(const std::vector<std::string> &elements, const char *delimiter) {
  std::stringstream oss;
  auto it = elements.begin();
  if (it != elements.end()) {
    oss << *it;
    ++it;
  }
  while (it != elements.end()) {
    oss << delimiter << *it;
    ++it;
  }
  return oss.str();
}

String &String::operator=(const char *s) {
  EXT_ENFORCE(s != data(), "Cannot assign to self.");
  clear();
  set(s, SIZE_MAX);
  return *this;
}

String &String::operator=(const RefString &s) {
  if (data() == s.data() && size_ == s.size())
    return *this; // no change
  EXT_ENFORCE(s.data() != data(), "Cannot assign to self when size is different.");
  clear();
  set(s.data(), s.size());
  return *this;
}

String &String::operator=(const String &s) {
  if (data() == s.data() && size_ == s.size())
    return *this; // no change
  EXT_ENFORCE(s.data() != data(), "Cannot assign to self when size is different.");
  clear();
  set(s.data(), s.size());
  return *this;
}

String &String::operator=(const std::string &s) {
  clear();
  set(s.data(), s.size());
  return *this;
}

} // namespace utils
} // namespace ONNX_LIGHT_NAMESPACE
