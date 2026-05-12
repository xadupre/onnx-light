// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "assertions.h"
#include "onnx_pb.h"

#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

/**
 * @brief Stores an in-memory tensor payload and metadata.
 *
 * Represents the ONNX tensor fields used by the lightweight parser, including
 * dimensions, element type, typed repeated values, raw bytes, optional segment
 * information, and external-data metadata.
 */
struct Tensor final {
private:
  bool is_segment_{false};
  int64_t segment_begin_{0};
  int64_t segment_end_{0};
  bool has_name_{false};
  std::string name_;
  int32_t elem_type_{static_cast<int32_t>(ONNX_LIGHT_NAMESPACE::TensorProto::DataType::UNDEFINED)};
  std::vector<int64_t> sizes_;
  std::vector<float> float_data_;
  std::vector<double> double_data_;
  std::vector<int32_t> int32_data_;
  std::vector<int64_t> int64_data_;
  std::vector<uint64_t> uint64_data_;
  std::vector<std::string> string_data_;
  bool is_raw_data_{false};
  std::string raw_data_;
  std::vector<std::pair<std::string, std::string>> external_data_;
  ONNX_LIGHT_NAMESPACE::TensorProto::DataLocation data_location_{
      ONNX_LIGHT_NAMESPACE::TensorProto::DEFAULT};

public:
  /**
   * @brief Returns the tensor dimensions.
   */
  const std::vector<int64_t> &sizes() const { return sizes_; }
  /**
   * @brief Returns mutable access to the tensor dimensions.
   */
  std::vector<int64_t> &sizes() { return sizes_; }

  /**
   * @brief Returns the total number of elements.
   */
  int64_t elem_num() const {
    return std::accumulate(sizes_.begin(), sizes_.end(), 1, std::multiplies<int64_t>{});
  }

  /**
   * @brief Returns the number of elements from a given dimension to the end.
   * @param dim Dimension index, supports negative indexing from the back.
   */
  int64_t size_from_dim(int dim) const {
    if (dim < 0) {
      dim += static_cast<int>(sizes_.size());
    }
    ONNX_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes_.size())
    return std::accumulate(sizes_.begin() + dim, sizes_.end(), 1, std::multiplies<int64_t>{});
  }

  /**
   * @brief Returns the ONNX element type.
   */
  int32_t elem_type() const { return elem_type_; }
  /**
   * @brief Returns mutable access to the ONNX element type.
   */
  int32_t &elem_type() { return elem_type_; }

  /**
   * @brief Returns mutable access to string tensor values.
   */
  std::vector<std::string> &strings() { return string_data_; }
  /**
   * @brief Returns string tensor values.
   */
  const std::vector<std::string> &strings() const { return string_data_; }
  /**
   * @brief Returns mutable access to float tensor values.
   */
  std::vector<float> &floats() { return float_data_; }
  /**
   * @brief Returns float tensor values.
   */
  const std::vector<float> &floats() const { return float_data_; }
  /**
   * @brief Returns mutable access to double tensor values.
   */
  std::vector<double> &doubles() { return double_data_; }
  /**
   * @brief Returns double tensor values.
   */
  const std::vector<double> &doubles() const { return double_data_; }
  /**
   * @brief Returns mutable access to int32 tensor values.
   */
  std::vector<int32_t> &int32s() { return int32_data_; }
  /**
   * @brief Returns int32 tensor values.
   */
  const std::vector<int32_t> &int32s() const { return int32_data_; }
  /**
   * @brief Returns mutable access to int64 tensor values.
   */
  std::vector<int64_t> &int64s() { return int64_data_; }
  /**
   * @brief Returns int64 tensor values.
   */
  const std::vector<int64_t> &int64s() const { return int64_data_; }
  /**
   * @brief Returns mutable access to uint64 tensor values.
   */
  std::vector<uint64_t> &uint64s() { return uint64_data_; }
  /**
   * @brief Returns uint64 tensor values.
   */
  const std::vector<uint64_t> &uint64s() const { return uint64_data_; }
  /**
   * @brief Returns raw tensor bytes.
   */
  const std::string &raw() const { return raw_data_; }

  /**
   * @brief Sets raw tensor bytes and marks the tensor as raw-backed.
   * @param raw_data Raw tensor payload.
   */
  void set_raw_data(std::string raw_data) {
    is_raw_data_ = true;
    raw_data_ = std::move(raw_data);
  }

  /**
   * @brief Returns a mutable pointer to typed tensor data.
   * @tparam T Element type to access.
   */
  template <typename T> T *data();

  /**
   * @brief Returns a const pointer to typed tensor data.
   * @tparam T Element type to access.
   */
  template <typename T> const T *data() const;

  /**
   * @brief Returns whether a segment is defined.
   */
  bool is_segment() const { return is_segment_; }
  /**
   * @brief Returns the inclusive segment begin index.
   */
  int64_t segment_begin() const { return segment_begin_; }
  /**
   * @brief Returns the exclusive segment end index.
   */
  int64_t segment_end() const { return segment_end_; }

  /**
   * @brief Sets segment bounds and marks the segment as present.
   * @param begin Inclusive begin index.
   * @param end Exclusive end index.
   */
  void set_segment_begin_and_end(int64_t begin, int64_t end) {
    is_segment_ = true;
    segment_begin_ = begin;
    segment_end_ = end;
  }

  /**
   * @brief Returns whether the tensor name is set.
   */
  bool hasName() const { return has_name_; }
  /**
   * @brief Returns the tensor name.
   */
  const std::string &name() const { return name_; }

  /**
   * @brief Sets the tensor name.
   * @param name Tensor name.
   */
  void setName(std::string name) {
    has_name_ = true;
    name_ = std::move(name);
  }

  /**
   * @brief Returns whether tensor values are stored as raw bytes.
   */
  bool is_raw_data() const { return is_raw_data_; }

  /**
   * @brief Returns external-data entries.
   */
  const std::vector<std::pair<std::string, std::string>> &external_data() const {
    return external_data_;
  }

  /**
   * @brief Returns mutable external-data entries.
   */
  std::vector<std::pair<std::string, std::string>> &external_data() { return external_data_; }

  /**
   * @brief Returns whether the data location differs from the default.
   */
  bool has_data_location() const {
    return data_location_ != ONNX_LIGHT_NAMESPACE::TensorProto::DEFAULT;
  }

  /**
   * @brief Returns the tensor data location.
   */
  const ONNX_LIGHT_NAMESPACE::TensorProto::DataLocation &data_location() const {
    return data_location_;
  }
  /**
   * @brief Returns mutable access to the tensor data location.
   */
  ONNX_LIGHT_NAMESPACE::TensorProto::DataLocation &data_location() { return data_location_; }
};

template <> inline std::string *Tensor::data<std::string>() {
  ONNX_ASSERTM(!is_raw_data(), "data type is string. string content is required to be stored in "
                               "repeated bytes string_data field."
                               "raw_data type cannot be string.")
  return string_data_.data();
}

template <> inline const std::string *Tensor::data<std::string>() const {
  ONNX_ASSERTM(!is_raw_data(), "data type is string. string content is required to be stored in "
                               "repeated bytes string_data field."
                               "raw_data type cannot be string.")
  return string_data_.data();
}

#define define_data(type, field)                                                                   \
  template <> inline type *Tensor::data<type>() {                                                  \
    if (is_raw_data_) {                                                                            \
      return reinterpret_cast<type *>(raw_data_.data());                                           \
    }                                                                                              \
    return field.data();                                                                           \
  }                                                                                                \
                                                                                                   \
  template <> inline const type *Tensor::data<type>() const {                                      \
    if (is_raw_data_) {                                                                            \
      return reinterpret_cast<const type *>(raw_data_.data());                                     \
    }                                                                                              \
    return field.data();                                                                           \
  }

define_data(float, float_data_) define_data(double, double_data_) define_data(int32_t, int32_data_)
    define_data(int64_t, int64_data_) define_data(uint64_t, uint64_data_)
#undef define_data

} // namespace ONNX_LIGHT_NAMESPACE
