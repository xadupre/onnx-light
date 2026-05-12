// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Forwarding header: maps onnx/defs/tensor_proto_util.h to onnx_light tensor_util.h.
// Also provides ToTensor<T> for creating TensorProto values from scalars/vectors.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "onnx/defs/tensor_util.h"

namespace ONNX_LIGHT_NAMESPACE {

// Creates a scalar TensorProto (rank-0) from a single value.
template <typename T>
TensorProto ToTensor(const T &value);

// Creates a TensorProto (rank-1, length = values.size()) from a vector.
template <typename T>
TensorProto ToTensor(const std::vector<T> &values);

// ---------------------------------------------------------------------------
// Specializations
// ---------------------------------------------------------------------------

template <>
inline TensorProto ToTensor<float>(const float &value) {
  TensorProto tp;
  tp.set_data_type(static_cast<int>(TensorProto::DataType::FLOAT));
  tp.add_float_data(value);
  return tp;
}

template <>
inline TensorProto ToTensor<double>(const double &value) {
  TensorProto tp;
  tp.set_data_type(static_cast<int>(TensorProto::DataType::DOUBLE));
  tp.add_double_data(value);
  return tp;
}

template <>
inline TensorProto ToTensor<int32_t>(const int32_t &value) {
  TensorProto tp;
  tp.set_data_type(static_cast<int>(TensorProto::DataType::INT32));
  tp.add_int32_data(value);
  return tp;
}

template <>
inline TensorProto ToTensor<int64_t>(const int64_t &value) {
  TensorProto tp;
  tp.set_data_type(static_cast<int>(TensorProto::DataType::INT64));
  tp.add_int64_data(value);
  return tp;
}

template <>
inline TensorProto ToTensor<uint64_t>(const uint64_t &value) {
  TensorProto tp;
  tp.set_data_type(static_cast<int>(TensorProto::DataType::UINT64));
  tp.add_uint64_data(value);
  return tp;
}

template <>
inline TensorProto ToTensor<bool>(const bool &value) {
  TensorProto tp;
  tp.set_data_type(static_cast<int>(TensorProto::DataType::BOOL));
  tp.add_int32_data(value ? 1 : 0);
  return tp;
}

template <>
inline TensorProto ToTensor<std::string>(const std::string &value) {
  TensorProto tp;
  tp.set_data_type(static_cast<int>(TensorProto::DataType::STRING));
  auto &s = tp.add_string_data();
  s = value;
  return tp;
}

// Vector specializations

template <>
inline TensorProto ToTensor<float>(const std::vector<float> &values) {
  TensorProto tp;
  tp.set_data_type(static_cast<int>(TensorProto::DataType::FLOAT));
  tp.add_dims(static_cast<int64_t>(values.size()));
  for (const auto &v : values)
    tp.add_float_data(v);
  return tp;
}

template <>
inline TensorProto ToTensor<double>(const std::vector<double> &values) {
  TensorProto tp;
  tp.set_data_type(static_cast<int>(TensorProto::DataType::DOUBLE));
  tp.add_dims(static_cast<int64_t>(values.size()));
  for (const auto &v : values)
    tp.add_double_data(v);
  return tp;
}

template <>
inline TensorProto ToTensor<int32_t>(const std::vector<int32_t> &values) {
  TensorProto tp;
  tp.set_data_type(static_cast<int>(TensorProto::DataType::INT32));
  tp.add_dims(static_cast<int64_t>(values.size()));
  for (const auto &v : values)
    tp.add_int32_data(v);
  return tp;
}

template <>
inline TensorProto ToTensor<int64_t>(const std::vector<int64_t> &values) {
  TensorProto tp;
  tp.set_data_type(static_cast<int>(TensorProto::DataType::INT64));
  tp.add_dims(static_cast<int64_t>(values.size()));
  for (const auto &v : values)
    tp.add_int64_data(v);
  return tp;
}

template <>
inline TensorProto ToTensor<uint64_t>(const std::vector<uint64_t> &values) {
  TensorProto tp;
  tp.set_data_type(static_cast<int>(TensorProto::DataType::UINT64));
  tp.add_dims(static_cast<int64_t>(values.size()));
  for (const auto &v : values)
    tp.add_uint64_data(v);
  return tp;
}

template <>
inline TensorProto ToTensor<bool>(const std::vector<bool> &values) {
  TensorProto tp;
  tp.set_data_type(static_cast<int>(TensorProto::DataType::BOOL));
  tp.add_dims(static_cast<int64_t>(values.size()));
  for (const auto &v : values)
    tp.add_int32_data(v ? 1 : 0);
  return tp;
}

// ---------------------------------------------------------------------------
// ParseData overloads for TensorProto* (used in shape inference).
// ---------------------------------------------------------------------------

template <typename T>
std::vector<T> ParseDataFromTensorProto(const TensorProto *tp);

template <>
inline std::vector<int64_t> ParseDataFromTensorProto<int64_t>(const TensorProto *tp) {
  std::vector<int64_t> result;
  if (tp == nullptr)
    return result;
  const auto &data = tp->ref_int64_data();
  result.insert(result.end(), data.begin(), data.end());
  return result;
}

template <>
inline std::vector<int32_t> ParseDataFromTensorProto<int32_t>(const TensorProto *tp) {
  std::vector<int32_t> result;
  if (tp == nullptr)
    return result;
  const auto &data = tp->ref_int32_data();
  result.insert(result.end(), data.begin(), data.end());
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE
