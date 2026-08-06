// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_light_helpers.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kWhereName = "kernel::Where";

// ``uint8_t`` aliases both ``UINT8`` and ``BOOL`` storage, but ``Tensor::As``
// validates the element type strictly. These helpers route ``BOOL`` tensors
// through ``AsBool`` so the shared ``uint8_t`` code path serves both dtypes.
template <typename T> const T *WhereTypedInput(const Tensor &t) {
  if constexpr (std::is_same_v<T, uint8_t>) {
    return t.data_type == static_cast<int32_t>(DataType::BOOL) ? t.AsBool() : t.As<T>();
  } else {
    return t.As<T>();
  }
}

template <typename T> T *WhereTypedOutput(Tensor &t) {
  if constexpr (std::is_same_v<T, uint8_t>) {
    return t.data_type == static_cast<int32_t>(DataType::BOOL) ? t.AsBool() : t.As<T>();
  } else {
    return t.As<T>();
  }
}

struct TernaryBroadcastInfo {
  onnx_kernels::Shape shape;
  onnx_kernels::Shape strides_c;
  onnx_kernels::Shape strides_x;
  onnx_kernels::Shape strides_y;
  int64_t element_count = 0;
};

TernaryBroadcastInfo CheckWhereBroadcast(const Tensor &condition, const Tensor &x,
                                         const Tensor &y) {
  EXT_ENFORCE_INVALID(condition.data_type == static_cast<int32_t>(DataType::BOOL),
                      "kernel::Where only supports BOOL condition tensor.");
  EXT_ENFORCE_INVALID(x.data_type == y.data_type,
                      "kernel::Where inputs ``x`` and ``y`` must share the same dtype.");

  const size_t rank = std::max(condition.shape.size(), std::max(x.shape.size(), y.shape.size()));
  Shape sc;
  sc.assign(rank, 1);
  Shape sx(sc), sy(sc), out(sc);
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
Tensor WhereAllocTyped(const Tensor &condition, const Tensor &x, const Tensor &y,
                       RawBufferAllocator *allocator) {
  const TernaryBroadcastInfo bi = CheckWhereBroadcast(condition, x, y);
  const size_t out_n_bytes = static_cast<size_t>(bi.element_count) * sizeof(T);
  Tensor out = MakeOutputTensor(x.data_type, bi.shape, out_n_bytes, allocator);
  const uint8_t *pc = condition.AsBool();
  const T *px = WhereTypedInput<T>(x);
  const T *py = WhereTypedInput<T>(y);
  T *po = WhereTypedOutput<T>(out);

  const size_t rank = bi.shape.size();
  onnx_kernels::Shape idx;
  idx.assign(rank, 0);
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
  EXT_ENFORCE_INVALID(output.size_bytes() == static_cast<size_t>(bi.element_count) * sizeof(T),
                      "kernel::Where preallocated output buffer has unexpected size in bytes.");

  const uint8_t *pc = condition.AsBool();
  const T *px = WhereTypedInput<T>(x);
  const T *py = WhereTypedInput<T>(y);
  T *po = WhereTypedOutput<T>(output);

  const size_t rank = bi.shape.size();
  onnx_kernels::Shape idx;
  idx.assign(rank, 0);
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
  onnx_kernels::Shape idx;
  idx.assign(rank, 0);
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
  onnx_kernels::Shape idx;
  idx.assign(rank, 0);
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

Tensor Where::operator()(const Tensor &condition, const Tensor &x, const Tensor &y,
                         RuntimeContext *rt) const {
  RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;
  switch (x.data_type) {
  case DataType::BOOL:
    return WhereAllocTyped<uint8_t>(condition, x, y, allocator);
  case DataType::FLOAT:
    return WhereAllocTyped<float>(condition, x, y, allocator);
  case DataType::DOUBLE:
    return WhereAllocTyped<double>(condition, x, y, allocator);
  case DataType::INT8:
    return WhereAllocTyped<int8_t>(condition, x, y, allocator);
  case DataType::INT16:
    return WhereAllocTyped<int16_t>(condition, x, y, allocator);
  case DataType::INT32:
    return WhereAllocTyped<int32_t>(condition, x, y, allocator);
  case DataType::INT64:
    return WhereAllocTyped<int64_t>(condition, x, y, allocator);
  case DataType::UINT8:
    return WhereAllocTyped<uint8_t>(condition, x, y, allocator);
  case DataType::UINT16:
    return WhereAllocTyped<uint16_t>(condition, x, y, allocator);
  case DataType::UINT32:
    return WhereAllocTyped<uint32_t>(condition, x, y, allocator);
  case DataType::UINT64:
    return WhereAllocTyped<uint64_t>(condition, x, y, allocator);
  case DataType::STRING:
    return WhereAllocString(condition, x, y);
  default:
    EXT_THROW_INVALID(kWhereName, ": unsupported data type ", x.data_type,
                      ", only supports BOOL, FLOAT, DOUBLE, INT8, INT16, INT32, "
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
    EXT_THROW_INVALID(kWhereName, ": unsupported data type ", x.data_type,
                      ", only supports BOOL, FLOAT, DOUBLE, INT8, INT16, INT32, "
                      "INT64, UINT8, UINT16, UINT32, UINT64 and STRING x/y inputs.");
  }
}

void Where::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 3);
  RequireOutputCount(node, 1);
  const Tensor &a = GetInput(node, 0, rt.tensors());
  const Tensor &b = GetInput(node, 1, rt.tensors());
  const Tensor &c = GetInput(node, 2, rt.tensors());
  SetOutput(node, 0, (*this)(a, b, c, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
