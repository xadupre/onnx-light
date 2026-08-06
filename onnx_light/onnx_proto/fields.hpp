#pragma once

#include "onnx_light_helpers.h"
#include <cstddef>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::utils {

template <typename T>
void RepeatedField<T>::PrintToStringStream(std::stringstream &ss,
                                           utils::PrintOptions &options) const {
  // Keep string items quoted to match protobuf-like debug text output.
  constexpr bool quote_strings = true;
  ss << "[ ";
  for (const auto &p : values_) {
    if constexpr (requires(const T &value) { value.PrintToStringStream(ss, options); }) {
      p.PrintToStringStream(ss, options);
    } else if constexpr (std::is_same_v<T, utils::String>) {
      ss << ::ONNX_LIGHT_NAMESPACE::utils::quote_string((p).sv(), quote_strings);
    } else {
      ss << p;
    }
    ss << " ";
  }
  ss << "]";
}

template <typename T> void RepeatedProtoField<T>::clear() { values_.clear(); }

template <typename T> void RepeatedProtoField<T>::resize(size_t n) {
  if (n < values_.size()) {
    values_.resize(n);
  } else {
    values_.reserve(n);
    while (values_.size() < n) {
      values_.emplace_back(std::make_shared<T>());
    }
  }
}

template <typename T> inline T &RepeatedProtoField<T>::operator[](size_t index) {
  return *values_[index];
}

template <typename T> inline const T &RepeatedProtoField<T>::operator[](size_t index) const {
  return *values_[index];
}

template <typename T> void RepeatedProtoField<T>::push_back(const T &v) { add().CopyFrom(v); }

template <typename T> void RepeatedProtoField<T>::push_back(T &&v) {
  values_.emplace_back(std::make_shared<T>(std::move(v)));
}

template <typename T> void RepeatedProtoField<T>::extend(const std::vector<T> &v) {
  values_.reserve(values_.size() + v.size());
  for (const auto &value : v) {
    push_back(value);
  }
}

template <typename T> void RepeatedProtoField<T>::extend(std::vector<T> &&v) {
  values_.reserve(values_.size() + v.size());
  for (auto &value : v) {
    push_back(std::move(value));
  }
  v.clear();
}

template <typename T> void RepeatedProtoField<T>::extend(const RepeatedProtoField<T> &v) {
  values_.reserve(values_.size() + v.values_.size());
  for (size_t i = 0; i < v.size(); ++i)
    push_back(v[i]);
}

template <typename T> void RepeatedProtoField<T>::extend(RepeatedProtoField<T> &&v) {
  // Steal ownership of each shared_ptr without allocating new placeholders or
  // performing per-element copies. std::make_move_iterator turns the source
  // std::shared_ptr<T> into rvalues so std::vector::insert calls the move
  // constructor and leaves v.values_ in a valid (emptied) state.
  values_.reserve(values_.size() + v.values_.size());
  values_.insert(values_.end(), std::make_move_iterator(v.values_.begin()),
                 std::make_move_iterator(v.values_.end()));
  v.values_.clear();
}

template <typename T> T &RepeatedProtoField<T>::add() {
  values_.emplace_back(std::make_shared<T>());
  return back();
}

template <typename T> T &RepeatedProtoField<T>::back() {
  EXT_ENFORCE(!values_.empty(), "Cannot call back() on an empty RepeatedField.");
  return *values_.back();
}

template <typename T>
void RepeatedProtoField<T>::PrintToStringStream(std::stringstream &ss,
                                                utils::PrintOptions &options) const {
  ss << "[ ";
  for (const auto &p : values_) {
    p->PrintToStringStream(ss, options);
    ss << " ";
  }
  ss << "]";
}

template <typename T> void OptionalField<T>::reset() { value_.reset(); }

template <typename T> void OptionalField<T>::set_empty_value() { value_.reset(new T); }

template <typename T> T &OptionalField<T>::operator*() {
  EXT_ENFORCE(has_value(), "Optional field has no value in ", typeid(T).name(), ".");
  return *value_;
}

template <typename T> const T &OptionalField<T>::operator*() const {
  EXT_ENFORCE(has_value(), "Optional field has no value in ", typeid(T).name(), ".");
  return *value_;
}

template <typename T> OptionalField<T> &OptionalField<T>::operator=(const T &v) {
  // Reset to a fresh empty value so that CopyFrom (which internally calls
  // ParseFromStream) starts from a clean state. Without this, repeated fields
  // in an existing value would be appended to rather than replaced.
  set_empty_value();
  value_->CopyFrom(v);
  return *this;
}

template <typename T> OptionalField<T> &OptionalField<T>::operator=(T &&v) {
  // Steal the contents of v instead of copying them: the field owns a single
  // pointer, so a move-construct into a fresh T avoids the CopyFrom above.
  value_.reset(new T(std::move(v)));
  return *this;
}

template <typename T> OptionalField<T> &OptionalField<T>::operator=(const OptionalField<T> &v) {
  // Guard against self-assignment: reset() below would otherwise destroy our own
  // value before we could copy it.
  if (this == &v) {
    return *this;
  }
  // We make a copy.
  reset();
  if (v.has_value()) {
    set_empty_value();
    value_->CopyFrom(*v);
  }
  return *this;
}

} // namespace ONNX_LIGHT_NAMESPACE::utils
