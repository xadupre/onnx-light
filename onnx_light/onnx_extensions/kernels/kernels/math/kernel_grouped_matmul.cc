// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {
namespace {

constexpr const char *kName = "kernel::GroupedMatMul";

void ValidatePayload(const Tensor &tensor) {
  const int64_t count = tensor.shape.product(0, tensor.shape.size(), kName);
  const size_t bytes = PackedByteSize(tensor.data_type, count);
  EXT_ENFORCE_INVALID(tensor.size_bytes() == bytes && (bytes == 0 || tensor.bytes() != nullptr),
                      kName, " tensor payload does not match its shape.");
}

template <typename T, typename Stored, typename Decode, typename Encode>
void Compute(const Tensor &input, const Tensor &weights, const Tensor &indices, const Tensor *bias,
             Tensor &output, Decode decode, Encode encode) {
  const auto *x = reinterpret_cast<const Stored *>(input.bytes());
  const auto *w = reinterpret_cast<const Stored *>(weights.bytes());
  const auto *b = bias ? reinterpret_cast<const Stored *>(bias->bytes()) : nullptr;
  auto *y = reinterpret_cast<Stored *>(output.mutable_bytes());
  const int64_t k = input.shape[1], n = weights.shape[2], s = indices.shape[1];
  for (int64_t i = 0; i < indices.element_count(); ++i) {
    const int64_t group =
        indices.data_type == DataType::INT64 ? indices.As<int64_t>()[i] : indices.As<int32_t>()[i];
    for (int64_t j = 0; j < n; ++j) {
      T value = T{0};
      for (int64_t l = 0; l < k; ++l) {
        value += decode(x[(i / s) * k + l]) * decode(w[(group * k + l) * n + j]);
      }
      if (b != nullptr) {
        // Matches MatMul followed by Add, including the intermediate dtype rounding.
        value = decode(encode(value)) + decode(b[group * n + j]);
      }
      y[i * n + j] = encode(value);
    }
  }
}

} // namespace

Tensor GroupedMatMul::operator()(const Tensor &input, const Tensor &weights,
                                 const Tensor &group_indices, const Tensor *bias,
                                 RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(input.shape.size() == 2 && weights.shape.size() == 3 &&
                          group_indices.shape.size() == 2,
                      kName, " expects input [M,K], weights [G,K,N], and indices [M,S].");
  EXT_ENFORCE_INVALID(input.shape[1] == weights.shape[1] &&
                          input.shape[0] == group_indices.shape[0],
                      kName, " incompatible input dimensions.");
  EXT_ENFORCE_INVALID(input.data_type == DataType::FLOAT || input.data_type == DataType::DOUBLE ||
                          input.data_type == DataType::FLOAT16 ||
                          input.data_type == DataType::BFLOAT16,
                      kName, " expects floating-point inputs.");
  EXT_ENFORCE_INVALID(weights.data_type == input.data_type, kName,
                      " weights must match the input dtype.");
  EXT_ENFORCE_INVALID(group_indices.data_type == DataType::INT32 ||
                          group_indices.data_type == DataType::INT64,
                      kName, " indices must be INT32 or INT64.");
  ValidatePayload(input);
  ValidatePayload(weights);
  ValidatePayload(group_indices);
  if (bias != nullptr) {
    EXT_ENFORCE_INVALID(bias->data_type == input.data_type &&
                            bias->shape == Shape({weights.shape[0], weights.shape[2]}),
                        kName, " bias must have shape [G,N] and match the input dtype.");
    ValidatePayload(*bias);
  }
  for (int64_t i = 0; i < group_indices.element_count(); ++i) {
    const int64_t group = group_indices.data_type == DataType::INT64
                              ? group_indices.As<int64_t>()[i]
                              : group_indices.As<int32_t>()[i];
    EXT_ENFORCE_INVALID(group >= 0 && group < weights.shape[0], kName,
                        " group index must be in [0,G).");
  }
  const Shape shape{input.shape[0], group_indices.shape[1], weights.shape[2]};
  const size_t bytes = PackedByteSize(input.data_type, shape.product(0, shape.size(), kName));
  Tensor output = rt ? rt->MakeOutputTensor(0, input.data_type, shape, bytes)
                     : MakeOutputTensor(input.data_type, shape, bytes, nullptr);
  switch (static_cast<DataType>(input.data_type)) {
  case DataType::FLOAT:
    Compute<float, float>(
        input, weights, group_indices, bias, output, [](float v) { return v; },
        [](float v) { return v; });
    break;
  case DataType::DOUBLE:
    Compute<double, double>(
        input, weights, group_indices, bias, output, [](double v) { return v; },
        [](double v) { return v; });
    break;
  case DataType::FLOAT16:
    Compute<float, uint16_t>(input, weights, group_indices, bias, output, Float16BitsToFloat,
                             FloatToFloat16Bits);
    break;
  case DataType::BFLOAT16:
    Compute<float, uint16_t>(input, weights, group_indices, bias, output, Bfloat16BitsToFloat,
                             FloatToBfloat16Bits);
    break;
  default:
    break;
  }
  return output;
}

void GroupedMatMul::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 3);
  EXT_ENFORCE_INVALID(node.input_size() <= 4, kName, " expects at most four inputs.");
  RequireOutputCount(node, 1);
  SetOutput(node, 0,
            (*this)(GetInput(node, 0, rt), GetInput(node, 1, rt), GetInput(node, 2, rt),
                    GetOptionalInput(node, 3, rt), &rt),
            rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
