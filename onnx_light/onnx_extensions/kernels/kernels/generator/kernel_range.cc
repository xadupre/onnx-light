// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/kernels/kernels/generator/include_generator_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

template <typename T> T ReadScalar(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.element_count() == 1, "kernel::Range: '", name,
                      "' must be a scalar (single-element) tensor.");
  return t.As<T>()[0];
}

// Fills ``n`` range elements into ``out_ptr`` starting at ``s`` with step ``d``.
template <typename T> void FillRangeBuffer(T s, T d, int64_t n, T *out_ptr) {
  for (int64_t i = 0; i < n; ++i) {
    out_ptr[i] = s + static_cast<T>(i) * d;
  }
}

// Fills ``n`` half-precision range elements (float16 or bfloat16) into
// ``out_ptr``. Accumulation is done in 32-bit float (the v27 stash_type
// semantics) and each value is encoded via ``encode``.
void FillRangeHalfBuffer(float s, float d, int64_t n, uint16_t *out_ptr,
                         uint16_t (*encode)(float)) {
  for (int64_t i = 0; i < n; ++i) {
    out_ptr[i] = encode(s + static_cast<float>(i) * d);
  }
}

// number_of_elements = max(ceil((limit - start) / delta), 0)
// Computed in double to handle both integer and float types uniformly,
// matching the upstream schema's shape-inference formula.
template <typename T> int64_t ComputeRangeCount(T s, T l, T d) {
  int64_t n = static_cast<int64_t>(
      std::ceil((static_cast<double>(l) - static_cast<double>(s)) / static_cast<double>(d)));
  return std::max<int64_t>(n, 0);
}

template <typename T>
Tensor ComputeRange(const Tensor &start, const Tensor &limit, const Tensor &delta, int32_t dtype,
                    RawBufferAllocator *allocator) {
  const T s = ReadScalar<T>(start, "start");
  const T l = ReadScalar<T>(limit, "limit");
  const T d = ReadScalar<T>(delta, "delta");
  EXT_ENFORCE_INVALID(d != T(0), "kernel::Range: 'delta' must be non-zero.");

  const int64_t n = ComputeRangeCount(s, l, d);
  Tensor output = MakeOutputTensor(dtype, {n}, PackedByteSize(dtype, n), allocator);
  FillRangeBuffer(s, d, n, reinterpret_cast<T *>(output.mutable_bytes()));
  return output;
}

template <typename T>
void ComputeRangeInto(const Tensor &start, const Tensor &limit, const Tensor &delta,
                      int32_t expected_dtype, Tensor &output) {
  const T s = ReadScalar<T>(start, "start");
  const T l = ReadScalar<T>(limit, "limit");
  const T d = ReadScalar<T>(delta, "delta");
  EXT_ENFORCE_INVALID(d != T(0), "kernel::Range: 'delta' must be non-zero.");

  const int64_t n = ComputeRangeCount(s, l, d);
  EXT_ENFORCE_INVALID(output.data_type == expected_dtype,
                      "kernel::Range preallocated output must have the expected dtype.");
  EXT_ENFORCE_INVALID(output.shape.size() == 1 && output.shape[0] == n,
                      "kernel::Range preallocated output shape must match the produced tensor "
                      "shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == PackedByteSize(expected_dtype, n),
                      "kernel::Range preallocated output buffer has unexpected size in bytes.");
  if (n > 0) {
    FillRangeBuffer(s, d, n, reinterpret_cast<T *>(output.mutable_bytes()));
  }
}

// Reads a scalar tensor stored as the raw IEEE-754 binary16 ``float16``
// bit pattern and returns its value as a ``float``.
float ReadFloat16Scalar(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.element_count() == 1, "kernel::Range: '", name,
                      "' must be a scalar (single-element) tensor.");
  return Float16BitsToFloat(*reinterpret_cast<const uint16_t *>(t.bytes()));
}

// Reads a scalar tensor stored as the raw ``bfloat16`` bit pattern
// (the upper 16 bits of an IEEE-754 binary32 ``float``) and returns its
// value as a ``float``.
float ReadBfloat16Scalar(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.element_count() == 1, "kernel::Range: '", name,
                      "' must be a scalar (single-element) tensor.");
  return Bfloat16BitsToFloat(*reinterpret_cast<const uint16_t *>(t.bytes()));
}

// IEEE-754 binary16 / bfloat16 encoders (round-to-nearest-even) are provided
// by ``onnx_core/runtime/kernels/cast_helper.h`` as ``FloatToFloat16Bits``
// and ``FloatToBfloat16Bits``.

// Computes a Range output whose element type is float16 or bfloat16 by
// accumulating in 32-bit float (the v27 ``stash_type`` semantics). The
// encoder/decoder pair (``read``/``encode``) selects between the two
// half-precision layouts.
Tensor ComputeRangeHalf(const Tensor &start, const Tensor &limit, const Tensor &delta,
                        int32_t dtype, float (*read)(const Tensor &, const char *),
                        uint16_t (*encode)(float), RawBufferAllocator *allocator) {
  const float s = read(start, "start");
  const float l = read(limit, "limit");
  const float d = read(delta, "delta");
  EXT_ENFORCE_INVALID(d != 0.0f, "kernel::Range: 'delta' must be non-zero.");

  const int64_t n = ComputeRangeCount(s, l, d);
  Tensor output = MakeOutputTensor(dtype, {n}, PackedByteSize(dtype, n), allocator);
  FillRangeHalfBuffer(s, d, n, reinterpret_cast<uint16_t *>(output.mutable_bytes()), encode);
  return output;
}

void ComputeRangeHalfInto(const Tensor &start, const Tensor &limit, const Tensor &delta,
                          int32_t expected_dtype, Tensor &output,
                          float (*read)(const Tensor &, const char *), uint16_t (*encode)(float)) {
  const float s = read(start, "start");
  const float l = read(limit, "limit");
  const float d = read(delta, "delta");
  EXT_ENFORCE_INVALID(d != 0.0f, "kernel::Range: 'delta' must be non-zero.");

  const int64_t n = ComputeRangeCount(s, l, d);
  EXT_ENFORCE_INVALID(output.data_type == expected_dtype,
                      "kernel::Range preallocated output must have the expected dtype.");
  EXT_ENFORCE_INVALID(output.shape.size() == 1 && output.shape[0] == n,
                      "kernel::Range preallocated output shape must match the produced tensor "
                      "shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == PackedByteSize(expected_dtype, n),
                      "kernel::Range preallocated output buffer has unexpected size in bytes.");
  if (n > 0) {
    FillRangeHalfBuffer(s, d, n, reinterpret_cast<uint16_t *>(output.mutable_bytes()), encode);
  }
}

} // namespace

Tensor Range::operator()(const Tensor &start, const Tensor &limit, const Tensor &delta,
                         RuntimeContext *rt) const {
  if (rt != nullptr) {
    Tensor produced = (*this)(start, limit, delta, nullptr);
    Tensor output =
        rt->MakeOutputTensor(0, produced.data_type, produced.shape, produced.size_bytes());
    (*this)(start, limit, delta, output);
    return output;
  }
  EXT_ENFORCE_INVALID(start.data_type == limit.data_type && start.data_type == delta.data_type,
                      "kernel::Range: 'start', 'limit' and 'delta' must share the same dtype.");
  RawBufferAllocator *allocator = nullptr;
  switch (static_cast<DataType>(start.data_type)) {
  case DataType::FLOAT:
    return ComputeRange<float>(start, limit, delta, start.data_type, allocator);
  case DataType::DOUBLE:
    return ComputeRange<double>(start, limit, delta, start.data_type, allocator);
  case DataType::INT16:
    return ComputeRange<int16_t>(start, limit, delta, start.data_type, allocator);
  case DataType::INT32:
    return ComputeRange<int32_t>(start, limit, delta, start.data_type, allocator);
  case DataType::INT64:
    return ComputeRange<int64_t>(start, limit, delta, start.data_type, allocator);
  case DataType::FLOAT16:
    return ComputeRangeHalf(start, limit, delta, start.data_type, &ReadFloat16Scalar,
                            &FloatToFloat16Bits, allocator);
  case DataType::BFLOAT16:
    return ComputeRangeHalf(start, limit, delta, start.data_type, &ReadBfloat16Scalar,
                            &FloatToBfloat16Bits, allocator);
  default:
    EXT_THROW_INVALID("unsupported data type ", start.data_type, ", ",
                      "kernel::Range: unsupported input dtype.");
  }
}

void Range::operator()(const Tensor &start, const Tensor &limit, const Tensor &delta,
                       Tensor &output) const {
  EXT_ENFORCE_INVALID(start.data_type == limit.data_type && start.data_type == delta.data_type,
                      "kernel::Range: 'start', 'limit' and 'delta' must share the same dtype.");
  switch (static_cast<DataType>(start.data_type)) {
  case DataType::FLOAT:
    ComputeRangeInto<float>(start, limit, delta, start.data_type, output);
    break;
  case DataType::DOUBLE:
    ComputeRangeInto<double>(start, limit, delta, start.data_type, output);
    break;
  case DataType::INT16:
    ComputeRangeInto<int16_t>(start, limit, delta, start.data_type, output);
    break;
  case DataType::INT32:
    ComputeRangeInto<int32_t>(start, limit, delta, start.data_type, output);
    break;
  case DataType::INT64:
    ComputeRangeInto<int64_t>(start, limit, delta, start.data_type, output);
    break;
  case DataType::FLOAT16:
    ComputeRangeHalfInto(start, limit, delta, start.data_type, output, &ReadFloat16Scalar,
                         &FloatToFloat16Bits);
    break;
  case DataType::BFLOAT16:
    ComputeRangeHalfInto(start, limit, delta, start.data_type, output, &ReadBfloat16Scalar,
                         &FloatToBfloat16Bits);
    break;
  default:
    EXT_THROW_INVALID("unsupported data type ", start.data_type, ", ",
                      "kernel::Range: unsupported input dtype.");
  }
}

void Range::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 3);
  RequireOutputCount(node, 1);
  const Tensor &start = GetInput(node, 0, rt.tensors());
  const Tensor &limit = GetInput(node, 1, rt.tensors());
  const Tensor &delta = GetInput(node, 2, rt.tensors());
  onnx_kernels::kernel::Range k(rt.kernel_ctx());
  SetOutput(node, 0, k(start, limit, delta, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
