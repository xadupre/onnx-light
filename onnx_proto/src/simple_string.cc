#include "simple_string.h"
#include <charconv>
#include <cstring>

namespace ONNX_LIGHT_NAMESPACE::utils {

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

int64_t RefString::toint64() const {
  // Use std::from_chars for fast, allocation-free integer parsing.
  int64_t result = 0;
  auto [ptr, ec] = std::from_chars(ptr_, ptr_ + size_, result);
  EXT_ENFORCE(ec == std::errc{} && ptr == ptr_ + size_,
              "Invalid integer string passed to toint64(): '", std::string(ptr_, size_), "'");
  return result;
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

} // namespace ONNX_LIGHT_NAMESPACE::utils
