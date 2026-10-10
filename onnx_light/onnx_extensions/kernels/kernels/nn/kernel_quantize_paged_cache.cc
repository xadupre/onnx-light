// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_extensions/kernels/kernels/quantization/include_quantization_kernels.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"
#include "onnx_proto/onnx_verify.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {
namespace {

const RuntimeValue &Field(const RuntimeValue &value, const char *name) {
  const auto it = value.fields.find(name);
  EXT_ENFORCE_INVALID(value.kind == RuntimeValue::Kind::kStruct && it != value.fields.end(),
                      "QuantizePagedCache: missing field ", name, ".");
  return it->second;
}

int64_t PageLength(const RuntimeValue &page) {
  const auto &length = Field(page, "length");
  EXT_ENFORCE_INVALID(
      length.kind == RuntimeValue::Kind::kTensor && length.tensor.data_type == DataType::INT64 &&
          length.tensor.shape.empty() && length.tensor.size_bytes() == sizeof(int64_t),
      "QuantizePagedCache: page length must be an INT64 scalar.");
  int64_t result;
  std::memcpy(&result, length.tensor.bytes(), sizeof(result));
  EXT_ENFORCE_INVALID(result > 0, "QuantizePagedCache: page length must be positive.");
  return result;
}

void CheckScale(const Tensor &scale, const char *name) {
  EXT_ENFORCE_INVALID(scale.data_type == DataType::FLOAT && scale.shape.empty() &&
                          scale.size_bytes() == sizeof(float) && scale.bytes() != nullptr &&
                          std::isfinite(scale.AsFloat()[0]) && scale.AsFloat()[0] > 0,
                      "QuantizePagedCache: ", name,
                      " must be a positive finite scalar FLOAT tensor.");
}

void CheckZeroPoint(const Tensor &zero_point, const char *name) {
  const auto type = static_cast<TensorProto::DataType>(zero_point.data_type);
  EXT_ENFORCE_INVALID(
      (IsAffineStorageType(type) || type == TensorProto::FLOAT || type == TensorProto::FLOAT16 ||
       type == TensorProto::BFLOAT16) &&
          zero_point.shape.empty() && zero_point.size_bytes() > 0 && zero_point.bytes() != nullptr,
      "QuantizePagedCache: ", name,
      " must be a scalar INT8, UINT8, INT4, UINT4, INT2, UINT2, FLOAT, FLOAT16 or BFLOAT16 "
      "tensor.");
}

void CopyScalar(TensorProto &target, const Tensor &source) {
  target.set_data_type(static_cast<TensorProto::DataType>(source.data_type));
  target.set_raw_data(source.bytes(), source.size_bytes());
}

RuntimeValue QuantizePayload(const RuntimeValue &source, int64_t length, const Tensor &scale,
                             const Tensor &zero_point, const KernelContext &context,
                             const StructTypeCatalogue &catalogue, RuntimeContext *rt) {
  Tensor decoded =
      DecodePagedCachePayload(source, length, catalogue, rt ? rt->allocator() : context.allocator);
  const int64_t capacity =
      source.kind == RuntimeValue::Kind::kTensor
          ? source.tensor.shape[2]
          : source.Encoded().logical_type().tensor_type().shape().dim(2).dim_value();
  if (capacity != length) {
    auto shape = decoded.shape;
    shape[2] = capacity;
    Tensor padded =
        MakeOutputTensor(DataType::FLOAT, shape, PackedByteSize(DataType::FLOAT, shape.product()),
                         rt ? rt->allocator() : context.allocator);
    // Unused rows need initialized storage, not reads from the source's invalid tail.
    std::memset(padded.mutable_bytes(), 0, padded.size_bytes());
    const int64_t outer = decoded.shape.product(0, 2, "QuantizePagedCache");
    const size_t valid_bytes = PackedByteSize(DataType::FLOAT, length * decoded.shape[3]);
    const size_t capacity_bytes = PackedByteSize(DataType::FLOAT, capacity * decoded.shape[3]);
    for (int64_t i = 0; i < outer; ++i)
      std::memcpy(padded.mutable_bytes() + static_cast<size_t>(i) * capacity_bytes,
                  decoded.bytes() + static_cast<size_t>(i) * valid_bytes, valid_bytes);
    decoded = std::move(padded);
  }
  if (!IsAffineStorageType(static_cast<TensorProto::DataType>(zero_point.data_type))) {
    Cast cast(context);
    return RuntimeValue(cast(decoded, zero_point.data_type, rt).RetainStorage());
  }
  QuantizeLinear quantize(context);
  Tensor codes = quantize(decoded, scale, zero_point, rt);
  const uint32_t width = FixedBitWidth(static_cast<TensorProto::DataType>(codes.data_type));
  const uint32_t used_bits = static_cast<uint32_t>(codes.element_count() % (8 / width)) * width;
  // EncodedValueProto requires zero padding in the last packed byte.
  if (used_bits != 0)
    codes.mutable_bytes()[codes.size_bytes() - 1] &= static_cast<uint8_t>((1U << used_bits) - 1);
  codes = std::move(codes).RetainStorage();

  EncodedValueProto encoded;
  auto *logical = encoded.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(DataType::FLOAT);
  for (int64_t dimension : decoded.shape)
    logical->mutable_shape()->add_dim()->set_dim_value(dimension);
  auto *affine = encoded.mutable_affine();
  affine->set_storage_type(static_cast<TensorProto::DataType>(zero_point.data_type));
  CopyScalar(*affine->mutable_scale(), scale);
  CopyScalar(*affine->mutable_zero_point(), zero_point);
  encoded.ref_raw_data().assign_borrowed(codes.bytes(), codes.size_bytes(), codes.borrowed_owner());
  return RuntimeValue(std::move(encoded)).Retain(catalogue);
}

} // namespace

RuntimeValue QuantizePagedCache::operator()(const RuntimeValue &cache, const Tensor &block_indices,
                                            const Tensor &key_scale, const Tensor &key_zero_point,
                                            const Tensor &value_scale,
                                            const Tensor &value_zero_point,
                                            const StructTypeCatalogue &catalogue,
                                            RuntimeContext *rt) const {
  const auto &blocks_value = Field(cache, "blocks");
  EXT_ENFORCE_INVALID(cache.fields.size() == 1 &&
                          blocks_value.kind == RuntimeValue::Kind::kSequence,
                      "QuantizePagedCache: cache must contain only a blocks sequence.");
  EXT_ENFORCE_INVALID(block_indices.data_type == DataType::INT64 &&
                          block_indices.shape.size() == 1 && block_indices.shape[0] >= 0,
                      "QuantizePagedCache: block_indices must be a rank-one INT64 tensor.");
  const int64_t index_count = block_indices.shape[0];
  EXT_ENFORCE_INVALID(
      static_cast<uint64_t>(index_count) <= std::numeric_limits<size_t>::max() / sizeof(int64_t) &&
          block_indices.size_bytes() == static_cast<size_t>(index_count) * sizeof(int64_t) &&
          (index_count == 0 || block_indices.bytes() != nullptr),
      "QuantizePagedCache: block_indices buffer extent does not match its shape.");
  CheckScale(key_scale, "key_scale");
  CheckScale(value_scale, "value_scale");
  CheckZeroPoint(key_zero_point, "key_zero_point");
  CheckZeroPoint(value_zero_point, "value_zero_point");

  RuntimeValue result = cache.BorrowView().Retain(catalogue);
  // Validates every page through the owner-preserving exporter without decoding payloads.
  result.ToPagedCache("", catalogue);

  std::vector<size_t> indices;
  indices.reserve(static_cast<size_t>(index_count));
  std::unordered_set<size_t> seen;
  for (int64_t i = 0; i < index_count; ++i) {
    const int64_t index = block_indices.AsInt64()[i];
    EXT_ENFORCE_INVALID(index >= 0 && static_cast<size_t>(index) < blocks_value.elements.size(),
                        "QuantizePagedCache: block index is out of range.");
    EXT_ENFORCE_INVALID(seen.insert(static_cast<size_t>(index)).second,
                        "QuantizePagedCache: block indices must be unique.");
    indices.push_back(static_cast<size_t>(index));
  }

  auto &blocks = result.fields.at("blocks").elements;
  for (size_t index : indices) {
    RuntimeValue page = blocks[index].BorrowView();
    EXT_ENFORCE_INVALID(page.fields.size() == 4, "QuantizePagedCache: unexpected page fields.");
    const int64_t length = PageLength(page);
    page.fields.at("key") =
        QuantizePayload(Field(page, "key"), length, key_scale, key_zero_point, ctx_, catalogue, rt);
    page.fields.at("value") = QuantizePayload(Field(page, "value"), length, value_scale,
                                              value_zero_point, ctx_, catalogue, rt);
    blocks.Set(index, std::move(page));
  }
  return std::move(result).Retain(catalogue);
}

void QuantizePagedCache::Run(RuntimeContext &rt) {
  EXT_ENFORCE_INVALID(node_ != nullptr, "QuantizePagedCache: missing node.");
  const NodeProto &node = *node_;
  RequireInputCount(node, 6);
  RequireOutputCount(node, 1);
  const auto cache = rt.values().find(node.input(0));
  EXT_ENFORCE_INVALID(cache != rt.values().end(), "QuantizePagedCache: missing paged cache input.");
  RuntimeValue result =
      (*this)(cache->second, GetInput(node, 1, rt), GetInput(node, 2, rt), GetInput(node, 3, rt),
              GetInput(node, 4, rt), GetInput(node, 5, rt), rt.struct_type_catalogue(), &rt);
  rt.PutValue(node.output(0), std::move(result));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
