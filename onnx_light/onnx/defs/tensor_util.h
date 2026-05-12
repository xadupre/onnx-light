// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "onnx/common/tensor.h"

namespace ONNX_LIGHT_NAMESPACE {

template <typename T> std::vector<T> ParseData(const Tensor *tensor);

// Creates a scalar TensorProto from a C++ value.
template <typename T> TensorProto ToTensor(const T &value);

// Creates a 1D TensorProto from a vector of values.
template <typename T> TensorProto ToTensor(const std::vector<T> &values);

// Identity overload for TensorProto.
inline TensorProto ToTensor(const TensorProto &tensor) { return tensor; }

} // namespace ONNX_LIGHT_NAMESPACE
