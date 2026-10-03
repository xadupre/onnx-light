// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/kernels/parallel_for.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr uint32_t kTuningAbi = 1;
constexpr int64_t kPortableParallelMinimum = core::runtime::kParallelForGrainSize;
constexpr std::array<int32_t, 4> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT), static_cast<int32_t>(DataType::DOUBLE),
    static_cast<int32_t>(DataType::FLOAT16), static_cast<int32_t>(DataType::BFLOAT16)};

int64_t ResolveAxis(int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank, "kernel::Softmax: axis is out of range.");
  return resolved;
}

} // namespace

Softmax::Softmax(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "Softmax", kSupportedElementTypes, kPortableParallelMinimum,
                            kTuningAbi) {}

ONNX_LIGHT_REGISTER_PARALLEL_TUNING_SCHEMA(Softmax)

Tensor Softmax::operator()(const Tensor &x, int64_t axis, RuntimeContext *rt) const {
  // FLOAT16/BFLOAT16 inputs are computed in float32 and demoted back, mirroring
  // the half-precision handling in the other math kernels.
  if (IsHalfPrecision(x.data_type)) {
    const Tensor x_f = PromoteToFloat32(x, rt);
    Tensor y_f = (*this)(x_f, axis, rt);
    return DemoteFromFloat32(y_f, x.data_type, rt);
  }
  const int32_t out_dtype = (static_cast<DataType>(x.data_type) == DataType::DOUBLE)
                                ? static_cast<int32_t>(DataType::DOUBLE)
                                : static_cast<int32_t>(DataType::FLOAT);
  const size_t elem_size =
      (static_cast<DataType>(x.data_type) == DataType::DOUBLE) ? sizeof(double) : sizeof(float);
  const size_t y_n_bytes = static_cast<size_t>(x.element_count()) * elem_size;
  Tensor y = rt ? rt->MakeOutputTensor(0, out_dtype, x.shape, y_n_bytes)
                : MakeOutputTensor(out_dtype, x.shape, y_n_bytes, nullptr);
  (*this)(x, axis, y);
  return y;
}

void Softmax::operator()(const Tensor &x, int64_t axis, Tensor &output) const {
  const bool is_double = static_cast<DataType>(x.data_type) == DataType::DOUBLE;
  EXT_ENFORCE_INVALID(x.data_type == DataType::FLOAT || is_double,
                      "kernel::Softmax only supports FLOAT and DOUBLE tensors.");
  EXT_ENFORCE_INVALID(output.data_type == x.data_type,
                      "kernel::Softmax preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Softmax preallocated output shape must match input shape.");
  EXT_ENFORCE_INVALID(output.mutable_bytes() != x.bytes(),
                      "kernel::Softmax does not support aliasing input/output buffers.");

  const int64_t n = x.element_count();
  const size_t elem_size = is_double ? sizeof(double) : sizeof(float);
  const size_t expected_bytes = static_cast<size_t>(n) * elem_size;
  EXT_ENFORCE_INVALID(output.size_bytes() == expected_bytes,
                      "kernel::Softmax preallocated output buffer has unexpected size in bytes.");

  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank > 0, "kernel::Softmax: input rank must be >= 1.");
  const int64_t resolved_axis = ResolveAxis(axis, rank);

  int64_t outer = 1;
  for (int64_t d = 0; d < resolved_axis; ++d) {
    outer *= x.shape[static_cast<size_t>(d)];
  }
  const int64_t axis_dim = x.shape[static_cast<size_t>(resolved_axis)];
  int64_t inner = 1;
  for (int64_t d = resolved_axis + 1; d < rank; ++d) {
    inner *= x.shape[static_cast<size_t>(d)];
  }
  if (n == 0) {
    return;
  }
  const int64_t rows = outer * inner;
  const int64_t grain = std::max<int64_t>(
      1, tuning().parallel_minimum_elements / axis_dim +
             static_cast<int64_t>(tuning().parallel_minimum_elements % axis_dim != 0));

  if (is_double) {
    const double *px = x.AsDouble();
    double *py = output.AsDouble();
    ParallelFor(
        rows, grain,
        [&](int64_t begin, int64_t end) {
          for (int64_t row = begin; row < end; ++row) {
            const int64_t o = row / inner;
            const int64_t i = row % inner;
            double max_v = -std::numeric_limits<double>::infinity();
            for (int64_t a = 0; a < axis_dim; ++a) {
              const int64_t offset = (o * axis_dim + a) * inner + i;
              max_v = std::max(max_v, px[static_cast<size_t>(offset)]);
            }
            double sum = 0.0;
            for (int64_t a = 0; a < axis_dim; ++a) {
              const int64_t offset = (o * axis_dim + a) * inner + i;
              sum += std::exp(px[static_cast<size_t>(offset)] - max_v);
            }
            for (int64_t a = 0; a < axis_dim; ++a) {
              const int64_t offset = (o * axis_dim + a) * inner + i;
              py[static_cast<size_t>(offset)] =
                  std::exp(px[static_cast<size_t>(offset)] - max_v) / sum;
            }
          }
        },
        "Softmax");
    return;
  }

  const float *px = x.AsFloat();
  float *py = output.AsFloat();
  ParallelFor(
      rows, grain,
      [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
          const int64_t o = row / inner;
          const int64_t i = row % inner;
          float max_v = -std::numeric_limits<float>::infinity();
          for (int64_t a = 0; a < axis_dim; ++a) {
            const int64_t offset = (o * axis_dim + a) * inner + i;
            max_v = std::max(max_v, px[static_cast<size_t>(offset)]);
          }
          float sum = 0.0f;
          for (int64_t a = 0; a < axis_dim; ++a) {
            const int64_t offset = (o * axis_dim + a) * inner + i;
            sum += std::exp(px[static_cast<size_t>(offset)] - max_v);
          }
          for (int64_t a = 0; a < axis_dim; ++a) {
            const int64_t offset = (o * axis_dim + a) * inner + i;
            py[static_cast<size_t>(offset)] =
                std::exp(px[static_cast<size_t>(offset)] - max_v) / sum;
          }
        }
      },
      "Softmax");
}

void Softmax::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const int64_t axis = GetAttributeIntOrDefault(node, "axis", -1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, axis, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
