// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/elementwise_helpers.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {
constexpr const char *kPowName = "kernel::Pow";

constexpr const char *kSupportedBaseTypesMsg = " only supports FLOAT, INT32 and INT64 base inputs.";
constexpr const char *kSupportedExponentTypesMsg =
    " only supports FLOAT, INT32, INT64, UINT32 and UINT64 exponent inputs.";

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
void PowLoop(const detail::BroadcastInfo &bi, const TBase *px, const TExp *py, TBase *pz) {
  // Fast paths: equal-shape and scalar broadcasting.
  if (bi.shape_x == bi.shape_y) {
    for (int64_t i = 0; i < bi.element_count; ++i) {
      pz[static_cast<size_t>(i)] = PowOne<TBase, TExp>(px[i], py[i]);
    }
    return;
  }
  if (bi.nx == 1 || bi.ny == 1) {
    for (int64_t i = 0; i < bi.element_count; ++i) {
      const TBase a = bi.nx == 1 ? px[0] : px[i];
      const TExp b = bi.ny == 1 ? py[0] : py[i];
      pz[static_cast<size_t>(i)] = PowOne<TBase, TExp>(a, b);
    }
    return;
  }

  const size_t rank = bi.shape.size();
  std::vector<int64_t> idx(rank, 0);
  for (int64_t flat = 0; flat < bi.element_count; ++flat) {
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
}

// Compute broadcast shape/strides without enforcing that ``x`` and ``y`` share
// a dtype. ``Pow`` is the only element-wise binary kernel in the backend test
// library whose two inputs may have different dtypes, so the standard
// :cpp:func:`detail::CheckBinaryBroadcast` / :cpp:func:`detail::CheckBinaryBroadcastInOut`
// helpers (which require
// ``x.data_type == y.data_type``) cannot be used.
detail::BroadcastInfo BroadcastShape(const Tensor &x, const Tensor &y) {
  const size_t rank = x.shape.size() > y.shape.size() ? x.shape.size() : y.shape.size();
  std::vector<int64_t> sx(rank, 1), sy(rank, 1), out(rank, 1);
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
      throw std::invalid_argument(std::string(kPowName) +
                                  " input shapes are not multidirectional-broadcastable.");
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

template <typename TBase, typename TExp>
void PowDispatchExp(const Tensor &x, const Tensor &y, Tensor &output,
                    const detail::BroadcastInfo &bi) {
  const TBase *px = reinterpret_cast<const TBase *>(x.data.data());
  const TExp *py = reinterpret_cast<const TExp *>(y.data.data());
  TBase *pz = reinterpret_cast<TBase *>(output.data.data());
  PowLoop<TBase, TExp>(bi, px, py, pz);
}

template <typename TBase>
void PowDispatchBase(const Tensor &x, const Tensor &y, Tensor &output,
                     const detail::BroadcastInfo &bi) {
  switch (y.data_type) {
  case DataType::FLOAT:
    return PowDispatchExp<TBase, float>(x, y, output, bi);
  case DataType::INT32:
    return PowDispatchExp<TBase, int32_t>(x, y, output, bi);
  case DataType::INT64:
    return PowDispatchExp<TBase, int64_t>(x, y, output, bi);
  case DataType::UINT32:
    return PowDispatchExp<TBase, uint32_t>(x, y, output, bi);
  case DataType::UINT64:
    return PowDispatchExp<TBase, uint64_t>(x, y, output, bi);
  default:
    throw std::invalid_argument(std::string(kPowName) + kSupportedExponentTypesMsg);
  }
}

size_t BaseDtypeSize(int32_t dtype) {
  switch (dtype) {
  case DataType::FLOAT:
    return sizeof(float);
  case DataType::INT32:
    return sizeof(int32_t);
  case DataType::INT64:
    return sizeof(int64_t);
  default:
    throw std::invalid_argument(std::string(kPowName) + kSupportedBaseTypesMsg);
  }
}

const char *BaseDtypeName(int32_t dtype) {
  switch (dtype) {
  case DataType::FLOAT:
    return "FLOAT";
  case DataType::INT32:
    return "INT32";
  case DataType::INT64:
    return "INT64";
  default:
    throw std::invalid_argument(std::string(kPowName) + kSupportedBaseTypesMsg);
  }
}

void PowDispatch(const Tensor &x, const Tensor &y, Tensor &output,
                 const detail::BroadcastInfo &bi) {
  switch (x.data_type) {
  case DataType::FLOAT:
    return PowDispatchBase<float>(x, y, output, bi);
  case DataType::INT32:
    return PowDispatchBase<int32_t>(x, y, output, bi);
  case DataType::INT64:
    return PowDispatchBase<int64_t>(x, y, output, bi);
  default:
    throw std::invalid_argument(std::string(kPowName) + kSupportedBaseTypesMsg);
  }
}
} // namespace

Tensor Pow::operator()(const Tensor &x, const Tensor &y) const {
  const detail::BroadcastInfo bi = BroadcastShape(x, y);
  const size_t elem_size = BaseDtypeSize(x.data_type);
  Tensor z("", x.data_type, bi.shape,
           std::vector<uint8_t>(static_cast<size_t>(bi.element_count) * elem_size));
  PowDispatch(x, y, z, bi);
  return z;
}

void Pow::operator()(const Tensor &x, const Tensor &y, Tensor &output) const {
  const detail::BroadcastInfo bi = BroadcastShape(x, y);
  const size_t elem_size = BaseDtypeSize(x.data_type);
  const size_t expected_bytes = static_cast<size_t>(bi.element_count) * elem_size;
  detail::CheckPreallocatedOutput(kPowName, BaseDtypeName(x.data_type), x.data_type, bi.shape,
                                  expected_bytes, output);
  PowDispatch(x, y, output, bi);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
