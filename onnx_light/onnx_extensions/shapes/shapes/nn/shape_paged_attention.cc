// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <array>

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {
namespace {

using Dimension = TensorShapeProto::Dimension;

Dimension MergeDimension(const Dimension &left, const Dimension &right) {
  EXT_ENFORCE_INVALID(!left.has_dim_value() || !right.has_dim_value() ||
                          left.dim_value() == right.dim_value(),
                      "PagedAttention: incompatible dimensions.");
  if (left.has_dim_value() || (!right.has_dim_value() && left.has_dim_param()))
    return left;
  return right;
}

std::array<Dimension, 4> TensorDimensions(const TypeProto &type) {
  EXT_ENFORCE_INVALID(type.has_tensor_type(), "PagedAttention: expected a tensor type.");
  const auto &tensor = type.tensor_type();
  EXT_ENFORCE_INVALID(tensor.elem_type() == TensorProto::UNDEFINED ||
                          tensor.elem_type() == TensorProto::FLOAT,
                      "PagedAttention: tensors must be FLOAT.");
  std::array<Dimension, 4> dims;
  if (tensor.has_shape()) {
    EXT_ENFORCE_INVALID(tensor.shape().dim_size() == 4,
                        "PagedAttention: tensors must have rank 4.");
    for (size_t i = 0; i < dims.size(); ++i)
      dims[i] = tensor.shape().dim(static_cast<int>(i));
  }
  for (size_t i = 0; i < dims.size(); ++i)
    EXT_ENFORCE_INVALID(!dims[i].has_dim_value() ||
                            (i < 2 ? dims[i].dim_value() == 1
                                   : (i == 2 ? dims[i].dim_value() >= 0 : dims[i].dim_value() > 0)),
                        "PagedAttention: expected [1,1,L,D] with L >= 0 and D > 0.");
  return dims;
}

TypeProto InputType(const ShapesContext &ctx, const std::string &name) {
  if (ctx.HasType(name))
    return ctx.GetType(name);
  ValueInfoProto info;
  EXT_ENFORCE_INVALID(ctx.Has(name) && core::symbolic::SymTensorToValueInfo(ctx.Get(name), info),
                      "PagedAttention: missing tensor descriptor for ", name, ".");
  return info.type();
}

const StructTypeProto::Structure &Structure(const ShapesContext &ctx, const TypeProto &type,
                                            int fields) {
  EXT_ENFORCE_INVALID(type.has_struct_type(), "PagedAttention: expected a cache structure.");
  const auto &resolved = ctx.ResolveStructType(type.struct_type());
  EXT_ENFORCE_INVALID(resolved.has_structure() && resolved.structure().field_size() == fields,
                      "PagedAttention: invalid cache fields.");
  return resolved.structure();
}

const TypeProto &Field(const StructTypeProto::Structure &structure, const char *name) {
  for (const auto &field : structure.field())
    if (field.name() == name)
      return field.type();
  EXT_THROW_INVALID("PagedAttention: missing cache field ", name, ".");
}

} // namespace

void ComputeShapePagedAttention(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "PagedAttention", "ComputeShapePagedAttention");
  EXT_ENFORCE_INVALID(node.domain() == "onnx_light" && node.input_size() == 4 &&
                          node.output_size() == 2 && !node.output(0).empty() &&
                          !node.output(1).empty() && node.output(0) != node.output(1),
                      "PagedAttention: expects Q,K,V,past and Y,present in domain onnx_light.");
  for (const auto &input : node.input())
    EXT_ENFORCE_INVALID(!input.empty(), "PagedAttention: inputs must not be omitted.");
  const auto q = TensorDimensions(InputType(ctx, node.input(0)));
  const auto k = TensorDimensions(InputType(ctx, node.input(1)));
  const auto v = TensorDimensions(InputType(ctx, node.input(2)));
  const auto length = MergeDimension(MergeDimension(q[2], k[2]), v[2]);
  const auto key_width = MergeDimension(q[3], k[3]);
  auto value_width = v[3];

  EXT_ENFORCE_INVALID(ctx.HasType(node.input(3)), "PagedAttention: missing past cache type.");
  const TypeProto past = ctx.GetType(node.input(3));
  const auto &blocks = Field(Structure(ctx, past, 1), "blocks");
  EXT_ENFORCE_INVALID(blocks.has_sequence_type() && blocks.sequence_type().has_elem_type(),
                      "PagedAttention: blocks must be a typed sequence.");
  const auto &page = Structure(ctx, blocks.sequence_type().elem_type(), 4);
  for (const char *name : {"start", "length"}) {
    const auto &type = Field(page, name);
    EXT_ENFORCE_INVALID(
        type.has_tensor_type() && type.tensor_type().elem_type() == TensorProto::INT64 &&
            (!type.tensor_type().has_shape() || type.tensor_type().shape().dim_size() == 0),
        "PagedAttention: start/length must be INT64 scalars.");
  }
  const auto cached_key = TensorDimensions(Field(page, "key"));
  const auto cached_value = TensorDimensions(Field(page, "value"));
  MergeDimension(cached_key[2], cached_value[2]);
  MergeDimension(key_width, cached_key[3]);
  value_width = MergeDimension(value_width, cached_value[3]);

  TypeProto output;
  auto *tensor = output.mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  tensor->mutable_shape()->add_dim()->set_dim_value(1);
  tensor->mutable_shape()->add_dim()->set_dim_value(1);
  *tensor->mutable_shape()->add_dim() = length;
  *tensor->mutable_shape()->add_dim() = value_width;
  ctx.SetType(node.output(0), output);
  ctx.SetType(node.output(1), past);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
