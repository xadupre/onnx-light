// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/elementwise_helpers.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/temporary_buffer.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {
constexpr const char *kPowName = "kernel::Pow";
constexpr std::array<int32_t, 5> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT), static_cast<int32_t>(DataType::FLOAT16),
    static_cast<int32_t>(DataType::BFLOAT16), static_cast<int32_t>(DataType::INT32),
    static_cast<int32_t>(DataType::INT64)};

constexpr const char *kSupportedBaseTypesMsg =
    " only supports FLOAT, FLOAT16, BFLOAT16, INT32 and INT64 base inputs.";
constexpr const char *kSupportedExponentTypesMsg =
    " only supports FLOAT, FLOAT16, BFLOAT16, INT32, INT64, UINT32 and UINT64 exponent inputs.";

// Evaluate ``base ^ exp`` honouring the output dtype semantics of ONNX Pow.
//   * Floating-point base: use ``std::pow`` directly (with the exponent cast
//     to the base type) which correctly handles fractional exponents, ``NaN``
//     and ``+/- inf``.
//   * Integer base: compute in ``double`` precision and cast back to the base
//     dtype. This matches the reference outputs produced by the upstream
//     ``onnx.backend.test.case.node.pow.Pow`` test cases (``test_pow_types_*``)
//     for all observed base/exponent combinations and mirrors NumPy's
//     ``numpy.power`` behaviour for non-negative integer exponents.
template <typename TBase, typename TExp> TBase PowOne(TBase base, TExp exp) {
  if constexpr (std::is_floating_point<TBase>::value) {
    return static_cast<TBase>(std::pow(base, static_cast<TBase>(exp)));
  } else {
    return static_cast<TBase>(std::pow(static_cast<double>(base), static_cast<double>(exp)));
  }
}

// Broadcasted iteration over ``x`` and ``y`` writing ``PowOne(x_i, y_i)`` into
// ``pz``. The dtype-specific buffer pointers are passed in pre-cast.
template <typename TBase, typename TExp>
void PowLoop(const detail::BroadcastInfo &bi, const TBase *px, const TExp *py, TBase *pz,
             int64_t grain) {
  // Fast paths: equal-shape and scalar broadcasting.
  if (bi.shape_x == bi.shape_y) {
    ParallelFor(bi.element_count, grain, [px, py, pz](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        pz[static_cast<size_t>(i)] = PowOne<TBase, TExp>(px[i], py[i]);
      }
    });
    return;
  }
  if (bi.nx == 1 || bi.ny == 1) {
    ParallelFor(bi.element_count, grain,
                [px, py, pz, nx = bi.nx, ny = bi.ny](int64_t begin, int64_t end) {
                  for (int64_t i = begin; i < end; ++i) {
                    const TBase a = nx == 1 ? px[0] : px[i];
                    const TExp b = ny == 1 ? py[0] : py[i];
                    pz[static_cast<size_t>(i)] = PowOne<TBase, TExp>(a, b);
                  }
                });
    return;
  }

  ParallelFor(bi.element_count, grain, [px, py, pz, &bi](int64_t begin, int64_t end) {
    const size_t rank = bi.shape.size();
    Shape idx;
    idx.assign(rank, 0);
    int64_t remaining = begin;
    for (size_t d = rank; d-- > 0;) {
      idx[d] = remaining % bi.shape[d];
      remaining /= bi.shape[d];
    }
    for (int64_t flat = begin; flat < end; ++flat) {
      int64_t ox = 0, oy = 0;
      for (size_t d = 0; d < rank; ++d) {
        ox += idx[d] * bi.strides_x[d];
        oy += idx[d] * bi.strides_y[d];
      }
      pz[static_cast<size_t>(flat)] = PowOne<TBase, TExp>(px[ox], py[oy]);
      for (size_t d = rank; d-- > 0;) {
        if (++idx[d] < bi.shape[d]) {
          break;
        }
        idx[d] = 0;
      }
    }
  });
}

// Compute broadcast shape/strides without enforcing that ``x`` and ``y`` share
// a dtype. ``Pow`` is the only element-wise binary kernel in the backend test
// library whose two inputs may have different dtypes, so the standard
// :cpp:func:`detail::CheckBinaryBroadcast` / :cpp:func:`detail::CheckBinaryBroadcastInOut`
// helpers (which require
// ``x.data_type == y.data_type``) cannot be used.
detail::BroadcastInfo BroadcastShape(const Tensor &x, const Tensor &y) {
  const size_t rank = x.shape.size() > y.shape.size() ? x.shape.size() : y.shape.size();
  Shape sx, sy, out;
  sx.assign(rank, 1);
  sy.assign(rank, 1);
  out.assign(rank, 1);
  for (size_t i = 0; i < x.shape.size(); ++i) {
    sx[rank - x.shape.size() + i] = x.shape[i];
  }
  for (size_t i = 0; i < y.shape.size(); ++i) {
    sy[rank - y.shape.size() + i] = y.shape[i];
  }
  for (size_t d = 0; d < rank; ++d) {
    if (sx[d] == sy[d] || sx[d] == 1 || sy[d] == 1) {
      out[d] = sx[d] >= sy[d] ? sx[d] : sy[d];
    } else {
      EXT_THROW_INVALID(kPowName, " input shapes are not multidirectional-broadcastable.");
    }
  }

  detail::BroadcastInfo bi;
  bi.shape = std::move(out);
  bi.shape_x = sx;
  bi.shape_y = sy;
  bi.nx = x.element_count();
  bi.ny = y.element_count();
  bi.element_count = 1;
  for (int64_t d : bi.shape) {
    bi.element_count *= d;
  }
  bi.strides_x.assign(rank, 0);
  bi.strides_y.assign(rank, 0);
  int64_t acc_x = 1, acc_y = 1;
  for (size_t i = rank; i-- > 0;) {
    bi.strides_x[i] = sx[i] == 1 ? 0 : acc_x;
    bi.strides_y[i] = sy[i] == 1 ? 0 : acc_y;
    acc_x *= sx[i];
    acc_y *= sy[i];
  }
  return bi;
}

template <typename TExp>
void PowHalfLoop(const detail::BroadcastInfo &bi, const uint16_t *px, const TExp *py, uint16_t *pz,
                 detail::HalfDecodeFunc decode, detail::HalfEncodeFunc encode, int64_t grain) {
  if (bi.shape_x == bi.shape_y) {
    ParallelFor(bi.element_count, grain, [px, py, pz, decode, encode](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        pz[static_cast<size_t>(i)] = encode(std::pow(decode(px[i]), static_cast<float>(py[i])));
      }
    });
    return;
  }
  if (bi.nx == 1 || bi.ny == 1) {
    ParallelFor(bi.element_count, grain,
                [px, py, pz, decode, encode, nx = bi.nx, ny = bi.ny](int64_t begin, int64_t end) {
                  for (int64_t i = begin; i < end; ++i) {
                    const float a = nx == 1 ? decode(px[0]) : decode(px[i]);
                    const float b = static_cast<float>(ny == 1 ? py[0] : py[i]);
                    pz[static_cast<size_t>(i)] = encode(std::pow(a, b));
                  }
                });
    return;
  }
  ParallelFor(bi.element_count, grain,
              [px, py, pz, decode, encode, &bi](int64_t begin, int64_t end) {
                const size_t rank = bi.shape.size();
                Shape idx;
                idx.assign(rank, 0);
                int64_t remaining = begin;
                for (size_t d = rank; d-- > 0;) {
                  idx[d] = remaining % bi.shape[d];
                  remaining /= bi.shape[d];
                }
                for (int64_t flat = begin; flat < end; ++flat) {
                  int64_t ox = 0, oy = 0;
                  for (size_t d = 0; d < rank; ++d) {
                    ox += idx[d] * bi.strides_x[d];
                    oy += idx[d] * bi.strides_y[d];
                  }
                  pz[static_cast<size_t>(flat)] =
                      encode(std::pow(decode(px[ox]), static_cast<float>(py[oy])));
                  for (size_t d = rank; d-- > 0;) {
                    if (++idx[d] < bi.shape[d]) {
                      break;
                    }
                    idx[d] = 0;
                  }
                }
              });
}

template <typename TBase, typename TExp>
void PowDispatchExp(const Tensor &x, const Tensor &y, Tensor &output,
                    const detail::BroadcastInfo &bi, int64_t grain) {
  const TBase *px = reinterpret_cast<const TBase *>(x.bytes());
  const TExp *py = reinterpret_cast<const TExp *>(y.bytes());
  TBase *pz = reinterpret_cast<TBase *>(output.mutable_bytes());
  PowLoop<TBase, TExp>(bi, px, py, pz, grain);
}

template <typename TBase>
void PowDispatchBase(const Tensor &x, const Tensor &y, Tensor &output,
                     const detail::BroadcastInfo &bi, int64_t grain) {
  switch (y.data_type) {
  case DataType::FLOAT:
    return PowDispatchExp<TBase, float>(x, y, output, bi, grain);
  case DataType::INT32:
    return PowDispatchExp<TBase, int32_t>(x, y, output, bi, grain);
  case DataType::INT64:
    return PowDispatchExp<TBase, int64_t>(x, y, output, bi, grain);
  case DataType::UINT32:
    return PowDispatchExp<TBase, uint32_t>(x, y, output, bi, grain);
  case DataType::UINT64:
    return PowDispatchExp<TBase, uint64_t>(x, y, output, bi, grain);
  default:
    EXT_THROW_INVALID(kPowName, ": unsupported data type ", y.data_type,
                      kSupportedExponentTypesMsg);
  }
}

template <typename TExp>
void PowDispatchHalfExp(const Tensor &x, const Tensor &y, Tensor &output,
                        const detail::BroadcastInfo &bi, detail::HalfDecodeFunc decode,
                        detail::HalfEncodeFunc encode, int64_t grain) {
  const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
  const TExp *py = reinterpret_cast<const TExp *>(y.bytes());
  uint16_t *pz = reinterpret_cast<uint16_t *>(output.mutable_bytes());
  PowHalfLoop<TExp>(bi, px, py, pz, decode, encode, grain);
}

void PowDispatchHalfBase(const Tensor &x, const Tensor &y, Tensor &output,
                         const detail::BroadcastInfo &bi, detail::HalfDecodeFunc decode,
                         detail::HalfEncodeFunc encode, RawBufferAllocator *allocator,
                         int64_t grain) {
  switch (y.data_type) {
  case DataType::FLOAT:
    return PowDispatchHalfExp<float>(x, y, output, bi, decode, encode, grain);
  case DataType::FLOAT16: {
    const int64_t ny = y.element_count();
    detail::TemporaryTypedBuffer<float> fy(static_cast<size_t>(ny), allocator, kPowName);
    const uint16_t *raw_py = reinterpret_cast<const uint16_t *>(y.bytes());
    for (int64_t i = 0; i < ny; ++i)
      fy.data()[static_cast<size_t>(i)] = Float16BitsToFloat(raw_py[i]);
    const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
    uint16_t *pz = reinterpret_cast<uint16_t *>(output.mutable_bytes());
    return PowHalfLoop<float>(bi, px, fy.data(), pz, decode, encode, grain);
  }
  case DataType::BFLOAT16: {
    const int64_t ny = y.element_count();
    detail::TemporaryTypedBuffer<float> fy(static_cast<size_t>(ny), allocator, kPowName);
    const uint16_t *raw_py = reinterpret_cast<const uint16_t *>(y.bytes());
    for (int64_t i = 0; i < ny; ++i)
      fy.data()[static_cast<size_t>(i)] = Bfloat16BitsToFloat(raw_py[i]);
    const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
    uint16_t *pz = reinterpret_cast<uint16_t *>(output.mutable_bytes());
    return PowHalfLoop<float>(bi, px, fy.data(), pz, decode, encode, grain);
  }
  case DataType::INT32:
    return PowDispatchHalfExp<int32_t>(x, y, output, bi, decode, encode, grain);
  case DataType::INT64:
    return PowDispatchHalfExp<int64_t>(x, y, output, bi, decode, encode, grain);
  case DataType::UINT32:
    return PowDispatchHalfExp<uint32_t>(x, y, output, bi, decode, encode, grain);
  case DataType::UINT64:
    return PowDispatchHalfExp<uint64_t>(x, y, output, bi, decode, encode, grain);
  default:
    EXT_THROW_INVALID(kPowName, ": unsupported data type ", y.data_type,
                      kSupportedExponentTypesMsg);
  }
}

size_t BaseDtypeSize(int32_t dtype) {
  switch (dtype) {
  case DataType::FLOAT:
    return sizeof(float);
  case DataType::FLOAT16:
    return sizeof(uint16_t);
  case DataType::BFLOAT16:
    return sizeof(uint16_t);
  case DataType::INT32:
    return sizeof(int32_t);
  case DataType::INT64:
    return sizeof(int64_t);
  default:
    EXT_THROW_INVALID(kPowName, ": unsupported data type ", dtype, kSupportedBaseTypesMsg);
  }
}

const char *BaseDtypeName(int32_t dtype) {
  switch (dtype) {
  case DataType::FLOAT:
    return "FLOAT";
  case DataType::FLOAT16:
    return "FLOAT16";
  case DataType::BFLOAT16:
    return "BFLOAT16";
  case DataType::INT32:
    return "INT32";
  case DataType::INT64:
    return "INT64";
  default:
    EXT_THROW_INVALID(kPowName, ": unsupported data type ", dtype, kSupportedBaseTypesMsg);
  }
}

void PowDispatch(const Tensor &x, const Tensor &y, Tensor &output, const detail::BroadcastInfo &bi,
                 RawBufferAllocator *allocator, int64_t grain) {
  switch (x.data_type) {
  case DataType::FLOAT:
    return PowDispatchBase<float>(x, y, output, bi, grain);
  case DataType::INT32:
    return PowDispatchBase<int32_t>(x, y, output, bi, grain);
  case DataType::INT64:
    return PowDispatchBase<int64_t>(x, y, output, bi, grain);
  case DataType::FLOAT16:
    return PowDispatchHalfBase(x, y, output, bi, Float16BitsToFloat, FloatToFloat16Bits, allocator,
                               grain);
  case DataType::BFLOAT16:
    return PowDispatchHalfBase(x, y, output, bi, Bfloat16BitsToFloat, FloatToBfloat16Bits,
                               allocator, grain);
  default:
    EXT_THROW_INVALID(kPowName, ": unsupported data type ", x.data_type, kSupportedBaseTypesMsg);
  }
}
} // namespace

Pow::Pow(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "Pow", kSupportedElementTypes, kParallelForGrainSize) {}

void Pow::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Pow", kSupportedElementTypes, kParallelForGrainSize);
}

Tensor Pow::operator()(const Tensor &x, const Tensor &y, RuntimeContext *rt) const {
  const detail::BroadcastInfo bi = BroadcastShape(x, y);
  const size_t elem_size = BaseDtypeSize(x.data_type);
  const size_t z_n_bytes = static_cast<size_t>(bi.element_count) * elem_size;
  RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;
  Tensor z = MakeOutputTensor(x.data_type, bi.shape, z_n_bytes, allocator);
  PowDispatch(x, y, z, bi, allocator, tuning().parallel_minimum_elements);
  return z;
}

void Pow::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  const detail::BroadcastInfo bi = BroadcastShape(x, y);
  const size_t elem_size = BaseDtypeSize(x.data_type);
  const size_t expected_bytes = static_cast<size_t>(bi.element_count) * elem_size;
  detail::CheckPreallocatedOutput(kPowName, BaseDtypeName(x.data_type), x.data_type, bi.shape,
                                  expected_bytes, output);
  RawBufferAllocator *allocator = output.has_allocation() ? output.allocation_owner() : nullptr;
  PowDispatch(x, y, output, bi, allocator, tuning().parallel_minimum_elements);
}

void Pow::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
