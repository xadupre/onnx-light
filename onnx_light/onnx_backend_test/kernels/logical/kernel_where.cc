// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"
#include "onnx_light_helpers.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

constexpr const char *kWhereName = "kernel::Where";

struct TernaryBroadcastInfo {
  std::vector<int64_t> shape;
  std::vector<int64_t> strides_c;
  std::vector<int64_t> strides_x;
  std::vector<int64_t> strides_y;
  int64_t element_count = 0;
};

TernaryBroadcastInfo CheckWhereBroadcast(const Tensor &condition, const Tensor &x,
                                         const Tensor &y) {
  EXT_ENFORCE_INVALID(condition.data_type == static_cast<int32_t>(DataType::BOOL),
                      "kernel::Where only supports BOOL condition tensor.");
  EXT_ENFORCE_INVALID(x.data_type == y.data_type,
                      "kernel::Where inputs ``x`` and ``y`` must share the same dtype.");

  const size_t rank = std::max(condition.shape.size(), std::max(x.shape.size(), y.shape.size()));
  std::vector<int64_t> sc(rank, 1), sx(rank, 1), sy(rank, 1), out(rank, 1);
  for (size_t i = 0; i < condition.shape.size(); ++i) {
    sc[rank - condition.shape.size() + i] = condition.shape[i];
  }
  for (size_t i = 0; i < x.shape.size(); ++i) {
    sx[rank - x.shape.size() + i] = x.shape[i];
  }
  for (size_t i = 0; i < y.shape.size(); ++i) {
    sy[rank - y.shape.size() + i] = y.shape[i];
  }

  for (size_t d = 0; d < rank; ++d) {
    const int64_t dc = sc[d];
    const int64_t dx = sx[d];
    const int64_t dy = sy[d];
    const int64_t max_dim = std::max(dc, std::max(dx, dy));
    EXT_ENFORCE_INVALID((dc == 1 || dc == max_dim) && (dx == 1 || dx == max_dim) &&
                            (dy == 1 || dy == max_dim),
                        "kernel::Where input shapes are not multidirectional-broadcastable.");
    out[d] = max_dim;
  }

  TernaryBroadcastInfo bi;
  bi.shape = std::move(out);
  bi.element_count = 1;
  for (int64_t d : bi.shape) {
    bi.element_count *= d;
  }

  bi.strides_c.assign(rank, 0);
  bi.strides_x.assign(rank, 0);
  bi.strides_y.assign(rank, 0);
  int64_t acc_c = 1, acc_x = 1, acc_y = 1;
  for (size_t i = rank; i-- > 0;) {
    bi.strides_c[i] = sc[i] == 1 ? 0 : acc_c;
    bi.strides_x[i] = sx[i] == 1 ? 0 : acc_x;
    bi.strides_y[i] = sy[i] == 1 ? 0 : acc_y;
    acc_c *= sc[i];
    acc_x *= sx[i];
    acc_y *= sy[i];
  }

  return bi;
}

template <typename T>
Tensor WhereAllocTyped(const Tensor &condition, const Tensor &x, const Tensor &y) {
  const TernaryBroadcastInfo bi = CheckWhereBroadcast(condition, x, y);
  Tensor out("", x.data_type, bi.shape,
             std::vector<uint8_t>(static_cast<size_t>(bi.element_count) * sizeof(T)));
  const uint8_t *pc = condition.AsBool();
  const T *px = x.As<T>();
  const T *py = y.As<T>();
  T *po = out.As<T>();

  const size_t rank = bi.shape.size();
  std::vector<int64_t> idx(rank, 0);
  for (int64_t flat = 0; flat < bi.element_count; ++flat) {
    int64_t oc = 0, ox = 0, oy = 0;
    for (size_t d = 0; d < rank; ++d) {
      oc += idx[d] * bi.strides_c[d];
      ox += idx[d] * bi.strides_x[d];
      oy += idx[d] * bi.strides_y[d];
    }
    po[static_cast<size_t>(flat)] = pc[oc] != 0 ? px[ox] : py[oy];
    for (size_t d = rank; d-- > 0;) {
      if (++idx[d] < bi.shape[d]) {
        break;
      }
      idx[d] = 0;
    }
  }

  return out;
}

template <typename T>
void WhereInPlaceTyped(const Tensor &condition, const Tensor &x, const Tensor &y, Tensor &output) {
  const TernaryBroadcastInfo bi = CheckWhereBroadcast(condition, x, y);
  EXT_ENFORCE_INVALID(output.data_type == x.data_type,
                      "kernel::Where preallocated output dtype must match x/y dtype.");
  EXT_ENFORCE_INVALID(
      output.shape == bi.shape,
      "kernel::Where preallocated output shape must match the broadcasted input shape.");
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(bi.element_count) * sizeof(T),
                      "kernel::Where preallocated output buffer has unexpected size in bytes.");

  const uint8_t *pc = condition.AsBool();
  const T *px = x.As<T>();
  const T *py = y.As<T>();
  T *po = output.As<T>();

  const size_t rank = bi.shape.size();
  std::vector<int64_t> idx(rank, 0);
  for (int64_t flat = 0; flat < bi.element_count; ++flat) {
    int64_t oc = 0, ox = 0, oy = 0;
    for (size_t d = 0; d < rank; ++d) {
      oc += idx[d] * bi.strides_c[d];
      ox += idx[d] * bi.strides_x[d];
      oy += idx[d] * bi.strides_y[d];
    }
    po[static_cast<size_t>(flat)] = pc[oc] != 0 ? px[ox] : py[oy];
    for (size_t d = rank; d-- > 0;) {
      if (++idx[d] < bi.shape[d]) {
        break;
      }
      idx[d] = 0;
    }
  }
}

Tensor WhereAllocString(const Tensor &condition, const Tensor &x, const Tensor &y) {
  const TernaryBroadcastInfo bi = CheckWhereBroadcast(condition, x, y);
  EXT_ENFORCE_INVALID(static_cast<int64_t>(x.string_data.size()) == x.element_count(),
                      "kernel::Where input ``x`` string_data size does not match its shape.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(y.string_data.size()) == y.element_count(),
                      "kernel::Where input ``y`` string_data size does not match its shape.");

  Tensor out = Tensor::MakeString("", bi.shape,
                                  std::vector<std::string>(static_cast<size_t>(bi.element_count)));
  const uint8_t *pc = condition.AsBool();

  const size_t rank = bi.shape.size();
  std::vector<int64_t> idx(rank, 0);
  for (int64_t flat = 0; flat < bi.element_count; ++flat) {
    int64_t oc = 0, ox = 0, oy = 0;
    for (size_t d = 0; d < rank; ++d) {
      oc += idx[d] * bi.strides_c[d];
      ox += idx[d] * bi.strides_x[d];
      oy += idx[d] * bi.strides_y[d];
    }
    out.string_data[static_cast<size_t>(flat)] = pc[oc] != 0
                                                     ? x.string_data[static_cast<size_t>(ox)]
                                                     : y.string_data[static_cast<size_t>(oy)];
    for (size_t d = rank; d-- > 0;) {
      if (++idx[d] < bi.shape[d]) {
        break;
      }
      idx[d] = 0;
    }
  }

  return out;
}

void WhereInPlaceString(const Tensor &condition, const Tensor &x, const Tensor &y, Tensor &output) {
  const TernaryBroadcastInfo bi = CheckWhereBroadcast(condition, x, y);
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::STRING),
                      "kernel::Where preallocated output dtype must match x/y dtype.");
  EXT_ENFORCE_INVALID(
      output.shape == bi.shape,
      "kernel::Where preallocated output shape must match the broadcasted input shape.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(output.string_data.size()) == bi.element_count,
                      "kernel::Where preallocated output string_data has unexpected size.");

  const uint8_t *pc = condition.AsBool();

  const size_t rank = bi.shape.size();
  std::vector<int64_t> idx(rank, 0);
  for (int64_t flat = 0; flat < bi.element_count; ++flat) {
    int64_t oc = 0, ox = 0, oy = 0;
    for (size_t d = 0; d < rank; ++d) {
      oc += idx[d] * bi.strides_c[d];
      ox += idx[d] * bi.strides_x[d];
      oy += idx[d] * bi.strides_y[d];
    }
    output.string_data[static_cast<size_t>(flat)] = pc[oc] != 0
                                                        ? x.string_data[static_cast<size_t>(ox)]
                                                        : y.string_data[static_cast<size_t>(oy)];
    for (size_t d = rank; d-- > 0;) {
      if (++idx[d] < bi.shape[d]) {
        break;
      }
      idx[d] = 0;
    }
  }
}

} // namespace

Tensor Where::operator()(const Tensor &condition, const Tensor &x, const Tensor &y) const {
  switch (x.data_type) {
  case DataType::BOOL:
    return WhereAllocTyped<uint8_t>(condition, x, y);
  case DataType::FLOAT:
    return WhereAllocTyped<float>(condition, x, y);
  case DataType::DOUBLE:
    return WhereAllocTyped<double>(condition, x, y);
  case DataType::INT8:
    return WhereAllocTyped<int8_t>(condition, x, y);
  case DataType::INT16:
    return WhereAllocTyped<int16_t>(condition, x, y);
  case DataType::INT32:
    return WhereAllocTyped<int32_t>(condition, x, y);
  case DataType::INT64:
    return WhereAllocTyped<int64_t>(condition, x, y);
  case DataType::UINT8:
    return WhereAllocTyped<uint8_t>(condition, x, y);
  case DataType::UINT16:
    return WhereAllocTyped<uint16_t>(condition, x, y);
  case DataType::UINT32:
    return WhereAllocTyped<uint32_t>(condition, x, y);
  case DataType::UINT64:
    return WhereAllocTyped<uint64_t>(condition, x, y);
  case DataType::STRING:
    return WhereAllocString(condition, x, y);
  default:
    throw std::invalid_argument(std::string(kWhereName) +
                                " only supports BOOL, FLOAT, DOUBLE, INT8, INT16, INT32, "
                                "INT64, UINT8, UINT16, UINT32, UINT64 and STRING x/y inputs.");
  }
}

void Where::operator()(const Tensor &condition, const Tensor &x, const Tensor &y,
                       Tensor &output) const {
  switch (x.data_type) {
  case DataType::BOOL:
    return WhereInPlaceTyped<uint8_t>(condition, x, y, output);
  case DataType::FLOAT:
    return WhereInPlaceTyped<float>(condition, x, y, output);
  case DataType::DOUBLE:
    return WhereInPlaceTyped<double>(condition, x, y, output);
  case DataType::INT8:
    return WhereInPlaceTyped<int8_t>(condition, x, y, output);
  case DataType::INT16:
    return WhereInPlaceTyped<int16_t>(condition, x, y, output);
  case DataType::INT32:
    return WhereInPlaceTyped<int32_t>(condition, x, y, output);
  case DataType::INT64:
    return WhereInPlaceTyped<int64_t>(condition, x, y, output);
  case DataType::UINT8:
    return WhereInPlaceTyped<uint8_t>(condition, x, y, output);
  case DataType::UINT16:
    return WhereInPlaceTyped<uint16_t>(condition, x, y, output);
  case DataType::UINT32:
    return WhereInPlaceTyped<uint32_t>(condition, x, y, output);
  case DataType::UINT64:
    return WhereInPlaceTyped<uint64_t>(condition, x, y, output);
  case DataType::STRING:
    return WhereInPlaceString(condition, x, y, output);
  default:
    throw std::invalid_argument(std::string(kWhereName) +
                                " only supports BOOL, FLOAT, DOUBLE, INT8, INT16, INT32, "
                                "INT64, UINT8, UINT16, UINT32, UINT64 and STRING x/y inputs.");
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
