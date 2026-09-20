// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_dlpack.h"
#include <algorithm>
#include <bit>
#include <limits>
#include <memory>

namespace ONNX_LIGHT_NAMESPACE {

namespace {
// Share diagnostic construction instead of instantiating it at every validation site.
void CheckDLPack(bool condition, const char *message) {
  if (!condition) {
    EXT_THROW_INVALID(message);
  }
}
} // namespace

DLDataType DLPackDataTypeFromOnnx(int32_t data_type, const char *producer) {
  auto make = [](uint8_t code, uint8_t bits) -> DLDataType { return {code, bits, 1}; };
  switch (static_cast<TensorProto::DataType>(data_type)) {
  case TensorProto::FLOAT:
    return make(kDLFloat, 32);
  case TensorProto::DOUBLE:
    return make(kDLFloat, 64);
  case TensorProto::FLOAT16:
    return make(kDLFloat, 16);
  case TensorProto::BFLOAT16:
    return make(kDLBfloat, 16);
  case TensorProto::UINT8:
    return make(kDLUInt, 8);
  case TensorProto::INT8:
    return make(kDLInt, 8);
  case TensorProto::UINT16:
    return make(kDLUInt, 16);
  case TensorProto::INT16:
    return make(kDLInt, 16);
  case TensorProto::UINT32:
    return make(kDLUInt, 32);
  case TensorProto::INT32:
    return make(kDLInt, 32);
  case TensorProto::UINT64:
    return make(kDLUInt, 64);
  case TensorProto::INT64:
    return make(kDLInt, 64);
  case TensorProto::BOOL:
    return make(kDLBool, 8);
  case TensorProto::COMPLEX64:
    return make(kDLComplex, 64);
  case TensorProto::COMPLEX128:
    return make(kDLComplex, 128);
  case TensorProto::FLOAT8E4M3FN:
    return make(kDLFloat8_e4m3fn, 8);
  case TensorProto::FLOAT8E4M3FNUZ:
    return make(kDLFloat8_e4m3fnuz, 8);
  case TensorProto::FLOAT8E5M2:
    return make(kDLFloat8_e5m2, 8);
  case TensorProto::FLOAT8E5M2FNUZ:
    return make(kDLFloat8_e5m2fnuz, 8);
  case TensorProto::FLOAT8E8M0:
    return make(kDLFloat8_e8m0fnu, 8);
  default:
    EXT_THROW_INVALID(producer, " DLPack: data type '",
                      TensorProto::DataType_Name(static_cast<TensorProto::DataType>(data_type)),
                      "' cannot be exported through DLPack (STRING and sub-byte packed types are "
                      "not supported).");
  }
}

DLPackMetadata ValidateDLPack(const TensorProto &tensor) {
  DLPackMetadata metadata{DLPackDataTypeFromOnnx(tensor.data_type(), "TensorProto"), {}};
  CheckDLPack(!tensor.has_segment(), "TensorProto DLPack: segmented tensors are not supported.");
  CheckDLPack(tensor.has_raw_data(),
              "TensorProto DLPack: requires raw_data; typed-field-only payloads and "
              "unloaded external data are not supported.");
  const auto &raw = tensor.ref_raw_data();
  const size_t itemsize = metadata.dtype.bits / 8;
  if (itemsize > 1 && std::endian::native != std::endian::little) {
    throw DLPackBufferError("TensorProto DLPack: multi-byte raw_data requires a little-endian "
                            "host; byte swapping would require a copy.");
  }
  CheckDLPack(tensor.dims_size() <= 128, "TensorProto DLPack: rank must not exceed 128.");
  metadata.shape.reserve(tensor.dims_size());
  const uint64_t limit =
      std::min<uint64_t>(std::numeric_limits<size_t>::max(), std::numeric_limits<int64_t>::max());
  uint64_t elements = 1;
  bool empty = false;
  for (int64_t dim : tensor.ref_dims()) {
    CheckDLPack(dim >= 0, "TensorProto DLPack: dimensions must be non-negative.");
    const uint64_t extent = std::max<uint64_t>(static_cast<uint64_t>(dim), 1);
    CheckDLPack(extent <= limit / elements, "TensorProto DLPack: shape or strides overflow.");
    elements *= extent;
    empty |= dim == 0;
    metadata.shape.push_back(dim);
  }
  CheckDLPack(elements <= limit / itemsize, "TensorProto DLPack: payload byte count overflows.");
  const size_t expected = empty ? 0 : static_cast<size_t>(elements * itemsize);
  CheckDLPack(raw.size() == expected,
              "TensorProto DLPack: raw_data size does not match shape and data type.");
  const size_t alignment = metadata.dtype.code == kDLComplex ? itemsize / 2 : itemsize;
  if (expected && reinterpret_cast<uintptr_t>(raw.data()) % alignment != 0) {
    throw DLPackBufferError("TensorProto DLPack: raw_data is not aligned to its element "
                            "(or complex component) width; alignment would require a copy.");
  }
  return metadata;
}

namespace {
struct DLPackOwner {
  DLManagedTensor managed{};
  DLPackMetadata metadata;
  utils::ByteSpan storage;
};
} // namespace

DLManagedTensor *ReleaseDLPack(TensorProto &tensor) {
  auto metadata = ValidateDLPack(tensor);
  const auto &raw = static_cast<const TensorProto &>(tensor).ref_raw_data();
  CheckDLPack(raw.is_borrowed() || !raw.has_active_exports(),
              "TensorProto ReleaseDLPack: owned raw_data has active DLPack exports; "
              "release those consumers before transferring its storage.");
  CheckDLPack(!raw.is_borrowed() || raw.owner().use_count() != 0,
              "TensorProto ReleaseDLPack: borrowed raw_data requires a lifetime owner.");
  auto owner = std::make_unique<DLPackOwner>();
  owner->metadata = std::move(metadata);
  owner->managed.dl_tensor = {raw.empty() ? nullptr : const_cast<uint8_t *>(raw.data()),
                              {kDLCPU, 0},
                              static_cast<int32_t>(owner->metadata.shape.size()),
                              owner->metadata.dtype,
                              owner->metadata.shape.data(),
                              nullptr,
                              0};
  owner->managed.manager_ctx = owner.get();
  owner->managed.deleter = [](DLManagedTensor *managed) {
    delete static_cast<DLPackOwner *>(managed->manager_ctx);
  };
  // ByteSpan's noexcept move detaches pointer, size, presence, allocation and lifetime token.
  owner->storage = std::move(tensor.ref_raw_data());
  return &owner.release()->managed;
}

} // namespace ONNX_LIGHT_NAMESPACE
