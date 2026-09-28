// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"
#include "onnx_proto/onnx_verify.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {
namespace {

const TypeProto &InputType(const ShapesContext &ctx, const std::string &name) {
  EXT_ENFORCE_INVALID(ctx.HasType(name), "QuantizePagedCache: missing type for ", name, ".");
  return ctx.GetType(name);
}

void CheckTensor(const ShapesContext &ctx, const std::string &name, int32_t data_type, int rank) {
  const auto &type = InputType(ctx, name);
  EXT_ENFORCE_INVALID(type.has_tensor_type() && type.tensor_type().elem_type() == data_type,
                      "QuantizePagedCache: unexpected tensor type for ", name, ".");
  if (type.tensor_type().has_shape())
    EXT_ENFORCE_INVALID(type.tensor_type().shape().dim_size() == rank,
                        "QuantizePagedCache: unexpected rank for ", name, ".");
}

void CheckZeroPoint(const ShapesContext &ctx, const std::string &name) {
  const auto &type = InputType(ctx, name);
  EXT_ENFORCE_INVALID(type.has_tensor_type(), "QuantizePagedCache: zero points must be tensors.");
  const auto data_type = static_cast<TensorProto::DataType>(type.tensor_type().elem_type());
  EXT_ENFORCE_INVALID(
      IsAffineStorageType(data_type) || data_type == TensorProto::FLOAT ||
          data_type == TensorProto::FLOAT16 || data_type == TensorProto::BFLOAT16,
      "QuantizePagedCache: zero points must select an affine integer or supported dense floating "
      "storage type.");
  if (type.tensor_type().has_shape())
    EXT_ENFORCE_INVALID(type.tensor_type().shape().dim_size() == 0,
                        "QuantizePagedCache: zero points must be scalars.");
}

} // namespace

void ComputeShapeQuantizePagedCache(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "QuantizePagedCache", "ComputeShapeQuantizePagedCache");
  EXT_ENFORCE_INVALID(node.domain() == "onnx_light" && node.input_size() == 6 &&
                          node.output_size() == 1 && !node.output(0).empty(),
                      "QuantizePagedCache: expects six inputs and one output.");
  for (const auto &input : node.input())
    EXT_ENFORCE_INVALID(!input.empty(), "QuantizePagedCache: inputs must not be omitted.");
  const auto &cache = InputType(ctx, node.input(0));
  EXT_ENFORCE_INVALID(cache.has_struct_type(),
                      "QuantizePagedCache: cache must be a structured value.");
  CheckTensor(ctx, node.input(1), TensorProto::INT64, 1);
  CheckTensor(ctx, node.input(2), TensorProto::FLOAT, 0);
  CheckZeroPoint(ctx, node.input(3));
  CheckTensor(ctx, node.input(4), TensorProto::FLOAT, 0);
  CheckZeroPoint(ctx, node.input(5));
  ctx.SetType(node.output(0), cache);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
