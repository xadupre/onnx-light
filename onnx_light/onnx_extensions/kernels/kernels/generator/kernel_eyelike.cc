// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/generator/include_generator_kernels.h"

#include "onnx_core/runtime/runtime_context.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

bool IsSupportedEyeLikeDtype(int32_t dtype) {
  switch (static_cast<DataType>(dtype)) {
  case DataType::FLOAT:
  case DataType::DOUBLE:
  case DataType::INT8:
  case DataType::UINT8:
  case DataType::INT16:
  case DataType::UINT16:
  case DataType::INT32:
  case DataType::UINT32:
  case DataType::INT64:
  case DataType::UINT64:
  case DataType::BOOL:
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
    return true;
  default:
    return false;
  }
}

} // namespace

void EyeLike::operator()(const Tensor &input, int64_t k, int32_t dtype, Tensor &output) const {
  EXT_ENFORCE_INVALID(input.shape.size() == 2, "kernel::EyeLike: input must be 2-dimensional.");
  const int64_t rows = input.shape[0];
  const int64_t cols = input.shape[1];
  EXT_ENFORCE_INVALID(rows >= 0 && cols >= 0,
                      "kernel::EyeLike: input dimensions must be non-negative.");

  EXT_ENFORCE_INVALID(IsSupportedEyeLikeDtype(dtype), "kernel::EyeLike: unsupported output dtype.");
  EXT_ENFORCE_INVALID(output.data_type == dtype,
                      "kernel::EyeLike preallocated output must have the expected dtype.");
  const bool shape_matches =
      output.shape.size() == 2 && output.shape[0] == rows && output.shape[1] == cols;
  EXT_ENFORCE_INVALID(shape_matches,
                      "kernel::EyeLike preallocated output shape must match the produced tensor "
                      "shape.");

  std::memset(output.mutable_bytes(), 0, output.size_bytes());

  // Write 1 at each diagonal position using typed accessors for direct assignment (no
  // intermediate buffer or extra copy). FLOAT16/BFLOAT16 have no typed accessor, so
  // their two-byte representations are written directly into the output buffer.
  auto write_diagonal = [&](auto *ptr, auto one_val) {
    for (int64_t i = 0; i < rows; ++i) {
      const int64_t j = i + k;
      if (j >= 0 && j < cols)
        ptr[static_cast<std::size_t>(i * cols + j)] = one_val;
    }
  };
  auto write_diagonal_u16 = [&](uint16_t one_val) {
    uint8_t *ptr = output.mutable_bytes();
    const std::array<uint8_t, sizeof(uint16_t)> one_bytes =
        std::bit_cast<std::array<uint8_t, sizeof(uint16_t)>>(one_val);
    for (int64_t i = 0; i < rows; ++i) {
      const int64_t j = i + k;
      if (j >= 0 && j < cols) {
        const std::size_t offset = static_cast<std::size_t>(i * cols + j) * sizeof(one_val);
        ptr[offset] = one_bytes[0];
        ptr[offset + 1] = one_bytes[1];
      }
    }
  };

  switch (static_cast<DataType>(dtype)) {
  case DataType::FLOAT:
    write_diagonal(output.AsFloat(), 1.0f);
    break;
  case DataType::DOUBLE:
    write_diagonal(output.AsDouble(), 1.0);
    break;
  case DataType::INT8:
    write_diagonal(output.AsInt8(), int8_t{1});
    break;
  case DataType::UINT8:
    write_diagonal(output.AsUint8(), uint8_t{1});
    break;
  case DataType::INT16:
    write_diagonal(output.AsInt16(), int16_t{1});
    break;
  case DataType::UINT16:
    write_diagonal(output.AsUint16(), uint16_t{1});
    break;
  case DataType::INT32:
    write_diagonal(output.AsInt32(), int32_t{1});
    break;
  case DataType::UINT32:
    write_diagonal(output.AsUint32(), uint32_t{1});
    break;
  case DataType::INT64:
    write_diagonal(output.AsInt64(), int64_t{1});
    break;
  case DataType::UINT64:
    write_diagonal(output.AsUint64(), uint64_t{1});
    break;
  case DataType::BOOL:
    write_diagonal(output.AsBool(), uint8_t{1});
    break;
  case DataType::FLOAT16:
    write_diagonal_u16(0x3C00);
    break; // 1.0 in float16
  case DataType::BFLOAT16:
    write_diagonal_u16(0x3F80);
    break; // 1.0 in bfloat16
  default:
    EXT_THROW_INVALID("kernel::EyeLike: unsupported output dtype ", dtype, ".");
  }
}

Tensor EyeLike::operator()(const Tensor &input, int64_t k, int32_t dtype,
                           RuntimeContext *rt) const {
  const int32_t out_dtype = (dtype != 0) ? dtype : input.data_type;
  const size_t n_bytes = PackedByteSize(out_dtype, input.element_count());
  Tensor output = MakeOutputTensor(out_dtype, input.shape, n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(input, k, out_dtype, output);
  return output;
}

void EyeLike::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const int64_t k = GetAttributeIntOrDefault(node, "k", 0);
  const int64_t dtype = GetAttributeIntOrDefault(node, "dtype", 0);
  onnx_kernels::kernel::EyeLike kernel(rt.kernel_ctx());
  SetOutput(node, 0, kernel(x, k, static_cast<int32_t>(dtype), &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
