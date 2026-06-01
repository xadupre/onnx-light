// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

constexpr const char *kCumSumName = "kernel::CumSum";
constexpr const char *kCumProdName = "kernel::CumProd";

// Resolves a possibly-negative axis (ONNX semantics: ``axis`` in
// ``[-rank, rank - 1]``) to a non-negative axis.
int64_t ResolveAxis(const char *op_name, int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank,
                      std::string(op_name) + ": axis is out of range.");
  return resolved;
}

int64_t ReadAxisScalar(const char *op_name, const Tensor &axis) {
  EXT_ENFORCE_INVALID(axis.element_count() == 1,
                      std::string(op_name) + ": axis must be a 0-D tensor.");
  switch (axis.data_type) {
  case DataType::INT32:
    return static_cast<int64_t>(axis.AsInt32()[0]);
  case DataType::INT64:
    return axis.AsInt64()[0];
  default:
    throw std::invalid_argument(std::string(op_name) + ": axis must be INT32 or INT64.");
  }
}

// Computes strides such that ``outer = product of dims before axis``,
// ``inner = product of dims after axis`` and ``dim = shape[axis]``.
void SplitShape(const std::vector<int64_t> &shape, int64_t axis, int64_t &outer, int64_t &dim,
                int64_t &inner) {
  outer = 1;
  for (int64_t i = 0; i < axis; ++i) {
    outer *= shape[static_cast<size_t>(i)];
  }
  dim = shape[static_cast<size_t>(axis)];
  inner = 1;
  for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i) {
    inner *= shape[i];
  }
}

template <typename T, typename Op>
void CumulativeInPlace(const Tensor &x, int64_t axis, bool exclusive, bool reverse, Tensor &output,
                       T identity, Op op) {
  int64_t outer = 0, dim = 0, inner = 0;
  SplitShape(x.shape, axis, outer, dim, inner);
  const T *px = reinterpret_cast<const T *>(x.data.data());
  T *py = reinterpret_cast<T *>(output.data.data());

  // Iterate over the outer/inner cartesian product and accumulate along the
  // axis dimension. Iteration order along the axis is reversed when
  // ``reverse`` is true. In exclusive mode the first written element is
  // ``identity`` and each subsequent element accumulates the *previous*
  // input value.
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      const int64_t base = (o * dim) * inner + i;
      const int64_t step = inner;
      if (!reverse) {
        T acc = identity;
        if (exclusive) {
          // Iterate forward: write identity at position 0, then propagate
          // a running accumulator built from the previous input value. We
          // read px[base + (k-1)*step] *before* writing py[base + k*step],
          // which is safe even when px and py alias because no later read
          // depends on a previously written position.
          for (int64_t k = 0; k < dim; ++k) {
            const T cur = px[base + k * step];
            py[base + k * step] = acc;
            acc = op(acc, cur);
          }
        } else {
          for (int64_t k = 0; k < dim; ++k) {
            acc = op(acc, px[base + k * step]);
            py[base + k * step] = acc;
          }
        }
      } else {
        T acc = identity;
        if (exclusive) {
          for (int64_t k = dim - 1; k >= 0; --k) {
            const T cur = px[base + k * step];
            py[base + k * step] = acc;
            acc = op(acc, cur);
          }
        } else {
          for (int64_t k = dim - 1; k >= 0; --k) {
            acc = op(acc, px[base + k * step]);
            py[base + k * step] = acc;
          }
        }
      }
    }
  }
}

template <typename T> Tensor CumAlloc(const Tensor &x) {
  return Tensor("", x.data_type, x.shape,
                std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * sizeof(T)));
}

void ValidateOutput(const char *op_name, const Tensor &x, const Tensor &output) {
  EXT_ENFORCE_INVALID(output.data_type == x.data_type,
                      std::string(op_name) + ": output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      std::string(op_name) + ": output shape must match input shape.");
  EXT_ENFORCE_INVALID(output.data.size() == x.data.size(),
                      std::string(op_name) + ": output buffer size mismatch.");
}

template <typename Op>
void DispatchCumulative(const char *op_name, const Tensor &x, int64_t axis, bool exclusive,
                        bool reverse, Tensor &output, Op /*unused*/) {
  switch (x.data_type) {
  case DataType::FLOAT:
    CumulativeInPlace<float>(x, axis, exclusive, reverse, output, Op::template Identity<float>(),
                             Op{});
    return;
  case DataType::DOUBLE:
    CumulativeInPlace<double>(x, axis, exclusive, reverse, output, Op::template Identity<double>(),
                              Op{});
    return;
  case DataType::INT32:
    CumulativeInPlace<int32_t>(x, axis, exclusive, reverse, output,
                               Op::template Identity<int32_t>(), Op{});
    return;
  case DataType::INT64:
    CumulativeInPlace<int64_t>(x, axis, exclusive, reverse, output,
                               Op::template Identity<int64_t>(), Op{});
    return;
  default:
    throw std::invalid_argument(std::string(op_name) +
                                " only supports FLOAT, DOUBLE, INT32 and INT64 tensors.");
  }
}

struct SumOp {
  template <typename T> T operator()(T a, T b) const { return static_cast<T>(a + b); }
  template <typename T> static T Identity() { return static_cast<T>(0); }
};

struct ProdOp {
  template <typename T> T operator()(T a, T b) const { return static_cast<T>(a * b); }
  template <typename T> static T Identity() { return static_cast<T>(1); }
};

} // namespace

Tensor CumSum::operator()(const Tensor &x, const Tensor &axis, bool exclusive, bool reverse) const {
  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank >= 1, std::string(kCumSumName) + " requires a non-scalar input.");
  const int64_t a = ResolveAxis(kCumSumName, ReadAxisScalar(kCumSumName, axis), rank);
  Tensor out;
  switch (x.data_type) {
  case DataType::FLOAT:
    out = CumAlloc<float>(x);
    break;
  case DataType::DOUBLE:
    out = CumAlloc<double>(x);
    break;
  case DataType::INT32:
    out = CumAlloc<int32_t>(x);
    break;
  case DataType::INT64:
    out = CumAlloc<int64_t>(x);
    break;
  default:
    throw std::invalid_argument(std::string(kCumSumName) +
                                " only supports FLOAT, DOUBLE, INT32 and INT64 tensors.");
  }
  DispatchCumulative(kCumSumName, x, a, exclusive, reverse, out, SumOp{});
  return out;
}

void CumSum::operator()(const Tensor &x, const Tensor &axis, bool exclusive, bool reverse,
                        Tensor &output) const {
  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank >= 1, std::string(kCumSumName) + " requires a non-scalar input.");
  const int64_t a = ResolveAxis(kCumSumName, ReadAxisScalar(kCumSumName, axis), rank);
  ValidateOutput(kCumSumName, x, output);
  DispatchCumulative(kCumSumName, x, a, exclusive, reverse, output, SumOp{});
}

Tensor CumProd::operator()(const Tensor &x, const Tensor &axis, bool exclusive,
                           bool reverse) const {
  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank >= 1, std::string(kCumProdName) + " requires a non-scalar input.");
  const int64_t a = ResolveAxis(kCumProdName, ReadAxisScalar(kCumProdName, axis), rank);
  Tensor out;
  switch (x.data_type) {
  case DataType::FLOAT:
    out = CumAlloc<float>(x);
    break;
  case DataType::DOUBLE:
    out = CumAlloc<double>(x);
    break;
  case DataType::INT32:
    out = CumAlloc<int32_t>(x);
    break;
  case DataType::INT64:
    out = CumAlloc<int64_t>(x);
    break;
  default:
    throw std::invalid_argument(std::string(kCumProdName) +
                                " only supports FLOAT, DOUBLE, INT32 and INT64 tensors.");
  }
  DispatchCumulative(kCumProdName, x, a, exclusive, reverse, out, ProdOp{});
  return out;
}

void CumProd::operator()(const Tensor &x, const Tensor &axis, bool exclusive, bool reverse,
                         Tensor &output) const {
  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank >= 1, std::string(kCumProdName) + " requires a non-scalar input.");
  const int64_t a = ResolveAxis(kCumProdName, ReadAxisScalar(kCumProdName, axis), rank);
  ValidateOutput(kCumProdName, x, output);
  DispatchCumulative(kCumProdName, x, a, exclusive, reverse, output, ProdOp{});
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
