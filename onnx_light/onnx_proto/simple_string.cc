#include "simple_string.h"
#include <charconv>
#include <cstring>

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

int64_t RefString::toint64() const {
  // Use std::from_chars for fast, allocation-free integer parsing.
  int64_t result = 0;
  auto [ptr, ec] = std::from_chars(ptr_, ptr_ + size_, result);
  EXT_ENFORCE(ec == std::errc{} && ptr == ptr_ + size_,
              "Invalid integer string passed to toint64(): '", std::string(*this), "'");
  return result;
}

void String::set(const char *ptr, size_t size) {
  if (ptr == nullptr) {
    reset_null_();
    return;
  }
  size_t effective_size = size;
  if (size == SIZE_MAX) {
    const char *p = ptr;
    effective_size = 0;
    for (; *p != 0; ++p, ++effective_size)
      ;
  }
  if (effective_size > 0 && ptr[effective_size - 1] == 0) {
    --effective_size;
  }
  std::string &v = mutable_value_();
  if (!v.empty() && ptr >= v.data() && ptr < v.data() + v.size()) {
    std::string copy(ptr, effective_size);
    v = std::move(copy);
  } else {
    v.assign(ptr, effective_size);
  }
}

bool String::operator==(const char *other) const {
  if (empty())
    return other == nullptr || other[0] == 0;
  if (other == nullptr)
    return false;
  size_t i;
  for (i = 0; i < size() && data()[i] == other[i] && other[i] != 0; ++i)
    ;
  return i == size() && other[i] == 0;
}

bool String::operator==(const RefString &other) const { return sv() == other.sv(); }

bool String::operator==(const String &other) const { return sv() == other.sv(); }

bool String::operator==(const std::string &other) const { return sv() == std::string_view(other); }

bool String::operator!=(const std::string &other) const { return !(*this == other); }
bool String::operator!=(const String &other) const { return !(*this == other); }
bool String::operator!=(const RefString &other) const { return !(*this == other); }
bool String::operator!=(const char *other) const { return !(*this == other); }

bool String::operator<(const RefString &other) const { return sv() < other.sv(); }

bool String::operator<(const std::string &other) const {
  return *this < RefString(other.data(), other.size());
}

bool String::operator<(const String &other) const {
  return *this < RefString(other.data(), other.size());
}

bool String::operator<(const char *other) const {
  if (other == nullptr)
    return false;
  return sv() < std::string_view(other);
}

bool String::operator>(const std::string &other) const {
  return *this > RefString(other.data(), other.size());
}

bool String::operator>(const String &other) const {
  return *this > RefString(other.data(), other.size());
}

bool String::operator>(const RefString &other) const { return sv() > other.sv(); }

bool String::operator>(const char *other) const {
  if (empty())
    return false;
  if (other == nullptr)
    return true;
  return sv() > std::string_view(other);
}

std::string join_string(const std::vector<std::string> &elements, const char *delimiter) {
  if (elements.empty())
    return std::string();
  size_t delim_len = std::strlen(delimiter);
  size_t total = 0;
  for (const auto &e : elements)
    total += e.size();
  total += delim_len * (elements.size() - 1);
  std::string result;
  result.reserve(total);
  auto it = elements.begin();
  result.append(*it);
  ++it;
  while (it != elements.end()) {
    result.append(delimiter, delim_len);
    result.append(*it);
    ++it;
  }
  return result;
}

String &String::operator=(const char *s) {
  set(s, SIZE_MAX);
  return *this;
}

String &String::operator=(String &&other) noexcept {
  if (this == &other)
    return *this;
  if (other.null()) {
    reset_null_();
  } else {
    mutable_value_() = std::move(other.value_);
    other.reset_null_();
  }
  return *this;
}

String &String::operator=(const RefString &s) {
  set(s.data(), s.size());
  return *this;
}

String &String::operator=(const String &s) {
  if (this == &s)
    return *this;
  if (s.null()) {
    reset_null_();
  } else {
    mutable_value_() = s.value_;
  }
  return *this;
}

String &String::operator=(const std::string &s) {
  std::string &v = mutable_value_();
  v = s;
  normalize_std_string_value(v);
  return *this;
}

String &String::operator=(std::string &&s) noexcept {
  std::string &v = mutable_value_();
  v = std::move(s);
  normalize_std_string_value(v);
  return *this;
}

} // namespace utils
} // namespace ONNX_LIGHT_NAMESPACE
