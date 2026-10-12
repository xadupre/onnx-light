// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr uint32_t kTuningAbi = 1;
constexpr int64_t kPortableParallelMinimum = core::runtime::kParallelForGrainSize;
constexpr std::array<int32_t, 18> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT),          static_cast<int32_t>(DataType::DOUBLE),
    static_cast<int32_t>(DataType::INT32),          static_cast<int32_t>(DataType::INT64),
    static_cast<int32_t>(DataType::UINT8),          static_cast<int32_t>(DataType::INT8),
    static_cast<int32_t>(DataType::BOOL),           static_cast<int32_t>(DataType::FLOAT8E4M3FN),
    static_cast<int32_t>(DataType::FLOAT8E4M3FNUZ), static_cast<int32_t>(DataType::FLOAT8E5M2),
    static_cast<int32_t>(DataType::FLOAT8E5M2FNUZ), static_cast<int32_t>(DataType::FLOAT8E8M0),
    static_cast<int32_t>(DataType::UINT16),         static_cast<int32_t>(DataType::INT16),
    static_cast<int32_t>(DataType::FLOAT16),        static_cast<int32_t>(DataType::BFLOAT16),
    static_cast<int32_t>(DataType::UINT32),         static_cast<int32_t>(DataType::UINT64),
};

onnx_kernels::Shape ResolvePermOrDefault(const onnx_kernels::Shape &perm, std::size_t rank) {
  if (perm.empty()) {
    onnx_kernels::Shape reversed;
    reversed.assign(rank, 0);
    for (std::size_t i = 0; i < rank; ++i) {
      reversed[i] = static_cast<int64_t>(rank - 1 - i);
    }
    return reversed;
  }
  EXT_ENFORCE_INVALID(perm.size() == rank, "kernel::Transpose: perm length must match input rank.");
  onnx_kernels::Shape seen;
  seen.assign(rank, 0);
  for (int64_t p : perm) {
    EXT_ENFORCE_INVALID(p >= 0 && static_cast<std::size_t>(p) < rank,
                        "kernel::Transpose: perm has an out-of-range axis.");
    EXT_ENFORCE_INVALID(!seen[static_cast<std::size_t>(p)],
                        "kernel::Transpose: perm has duplicate axes.");
    seen[static_cast<std::size_t>(p)] = 1;
  }
  return perm;
}

onnx_kernels::Shape ComputeStrides(const onnx_kernels::Shape &shape) {
  if (shape.empty()) {
    return {};
  }
  onnx_kernels::Shape strides;
  strides.assign(shape.size(), 1);
  for (std::size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

int64_t BlockGrainSize(int64_t minimum_elements, int64_t block_elements) {
  return std::max<int64_t>(1, minimum_elements / block_elements +
                                  static_cast<int64_t>(minimum_elements % block_elements != 0));
}

} // namespace

Transpose::Transpose(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "Transpose", kSupportedElementTypes, kPortableParallelMinimum,
                            kTuningAbi) {}

ONNX_LIGHT_REGISTER_PARALLEL_TUNING_SCHEMA(Transpose)

Tensor Transpose::operator()(const Tensor &data, const onnx_kernels::Shape &perm,
                             RuntimeContext *rt) const {
  const onnx_kernels::Shape resolved_perm = ResolvePermOrDefault(perm, data.shape.size());
  onnx_kernels::Shape out_shape;
  out_shape.assign(resolved_perm.size(), 0);
  for (std::size_t i = 0; i < resolved_perm.size(); ++i) {
    out_shape[i] = data.shape[static_cast<std::size_t>(resolved_perm[i])];
  }

  const size_t output_n_bytes = PackedByteSize(data.data_type, data.element_count());
  Tensor output = (rt ? rt->MakeOutputTensor(0, data.data_type, out_shape, output_n_bytes)
                      : MakeOutputTensor(data.data_type, out_shape, output_n_bytes, nullptr));
  (*this)(data, perm, output);
  return output;
}

void Transpose::operator()(const Tensor &data, const onnx_kernels::Shape &perm,
                           Tensor &output) const {
  const onnx_kernels::Shape resolved_perm = ResolvePermOrDefault(perm, data.shape.size());
  onnx_kernels::Shape out_shape;
  out_shape.assign(resolved_perm.size(), 0);
  for (std::size_t i = 0; i < resolved_perm.size(); ++i) {
    out_shape[i] = data.shape[static_cast<std::size_t>(resolved_perm[i])];
  }

  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::Transpose: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Transpose: preallocated output shape mismatch.");

  const std::size_t elem_size = ElementSize(data.data_type);
  const onnx_kernels::Shape in_strides = ComputeStrides(data.shape);
  const onnx_kernels::Shape out_strides = ComputeStrides(out_shape);
  const int64_t total = output.element_count();
  if (total == 0) {
    return;
  }

  std::size_t contiguous_axis_begin = out_shape.size();
  int64_t expected_input_axis = static_cast<int64_t>(data.shape.size()) - 1;
  while (contiguous_axis_begin > 0 &&
         resolved_perm[contiguous_axis_begin - 1] == expected_input_axis) {
    --contiguous_axis_begin;
    --expected_input_axis;
  }

  int64_t block_elements = 1;
  for (std::size_t i = contiguous_axis_begin; i < out_shape.size(); ++i) {
    block_elements *= out_shape[i];
  }
  const std::size_t block_bytes = static_cast<std::size_t>(block_elements) * elem_size;

  // Identity permutations are one contiguous transfer. A single memcpy is faster
  // than splitting the same bandwidth-bound copy across the thread pool.
  if (contiguous_axis_begin == 0) {
    std::memcpy(output.mutable_bytes(), data.bytes(), block_bytes);
    return;
  }

  const int64_t block_count = total / block_elements;
  ParallelFor(
      block_count, BlockGrainSize(tuning().parallel_minimum_elements, block_elements),
      [&](int64_t begin, int64_t end) {
        for (int64_t out_block = begin; out_block < end; ++out_block) {
          int64_t remaining = out_block * block_elements;
          int64_t in_idx = 0;
          for (std::size_t i = 0; i < contiguous_axis_begin; ++i) {
            const int64_t coord = remaining / out_strides[i];
            remaining %= out_strides[i];
            in_idx += coord * in_strides[static_cast<std::size_t>(resolved_perm[i])];
          }
          std::memcpy(output.mutable_bytes() + static_cast<std::size_t>(out_block) * block_bytes,
                      data.bytes() + static_cast<std::size_t>(in_idx) * elem_size, block_bytes);
        }
      },
      "Transpose");
}

void Transpose::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &data = GetInput(node, 0, rt);
  const onnx_kernels::Shape perm = GetAttributeIntsOrDefault(node, "perm", {});
  SetOutput(node, 0, (*this)(data, perm, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
