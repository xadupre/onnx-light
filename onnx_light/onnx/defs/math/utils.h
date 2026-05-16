// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file math/utils.h
 * @brief Declares reusable math-operator schema and inference helpers.
 *
 * This header defines utility APIs for math-domain operators, including TopK
 * schema generation, scalar extraction from TensorProto constants, and MatMul
 * shape inference helpers.
 */

#pragma once

#include <string>
#include <vector>

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/onnx_pb.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace defs {
namespace math {
namespace utils {

std::function<void(OpSchema &)> TopKOpGenerator(std::vector<std::string> allowed_types);

template <typename T> T GetScalarValueFromTensor(const ONNX_LIGHT_NAMESPACE::TensorProto *t) {
  if (t == nullptr) {
    return T{};
  }

  auto data_type = t->data_type();
  switch (data_type) {
  case ONNX_LIGHT_NAMESPACE::TensorProto::FLOAT:
    return static_cast<T>(ONNX_LIGHT_NAMESPACE::ParseData<float>(t).at(0));
  case ONNX_LIGHT_NAMESPACE::TensorProto::DOUBLE:
    return static_cast<T>(ONNX_LIGHT_NAMESPACE::ParseData<double>(t).at(0));
  case ONNX_LIGHT_NAMESPACE::TensorProto::INT32:
    return static_cast<T>(ONNX_LIGHT_NAMESPACE::ParseData<int32_t>(t).at(0));
  case ONNX_LIGHT_NAMESPACE::TensorProto::INT64:
    return static_cast<T>(ONNX_LIGHT_NAMESPACE::ParseData<int64_t>(t).at(0));
  default:
    fail_shape_inference("Unsupported input data type of ", data_type);
  }
}

void MatMulShapeInference(ONNX_LIGHT_NAMESPACE::InferenceContext &ctx, int input1Idx,
                          int input2Idx);

void QLinearMatMulShapeInference(ONNX_LIGHT_NAMESPACE::InferenceContext &ctx);

const char *QLinearMatMulDoc();

int MathOpTwoIntegers(const std::string &op_type, int a, int b);

} // namespace utils
} // namespace math
} // namespace defs
} // namespace ONNX_LIGHT_NAMESPACE
