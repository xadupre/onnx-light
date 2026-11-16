// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "onnx_core/symbolic/symbolic_helper.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {
namespace {

namespace expressions = ::ONNX_LIGHT_NAMESPACE::core::expressions;

struct PackingParameters {
  int64_t group_size;
  int64_t width;
};

SymDim UnknownCount(const std::string &input_name) {
  constexpr char digits[] = "0123456789abcdef";
  std::string name = "Unpack_";
  for (const unsigned char byte : input_name) {
    name.push_back(digits[byte >> 4]);
    name.push_back(digits[byte & 15]);
  }
  name += "_count";
  return SymDim(std::move(name));
}

const AttributeProto *IntegerAttribute(const NodeProto &node, const char *name) {
  const auto *attribute = FindAttribute(node, name);
  EXT_ENFORCE_INVALID(attribute == nullptr || attribute->type() == AttributeProto::INT,
                      node.op_type(), ": attribute '", name, "' must be an integer.");
  return attribute;
}

PackingParameters Parameters(const NodeProto &node) {
  const auto *bits = IntegerAttribute(node, "bits");
  const auto *radix = IntegerAttribute(node, "radix");
  const auto *group = IntegerAttribute(node, "group_size");
  if (bits != nullptr) {
    EXT_ENFORCE_INVALID(radix == nullptr && group == nullptr, node.op_type(),
                        ": specify either bits or radix and group_size.");
    EXT_ENFORCE_INVALID(bits->i() >= 1 && bits->i() <= 8, node.op_type(),
                        ": bits must be in [1, 8].");
    return {1, bits->i()};
  }
  EXT_ENFORCE_INVALID(radix != nullptr && group != nullptr, node.op_type(),
                      ": specify either bits or radix and group_size.");
  EXT_ENFORCE_INVALID(radix->i() >= 2 && radix->i() <= 256 && group->i() > 0, node.op_type(),
                      ": radix must be in [2, 256] and group_size positive.");
  int64_t limit = 1;
  for (int64_t i = 0; i < group->i(); ++i) {
    EXT_ENFORCE_INVALID(limit <= std::numeric_limits<int64_t>::max() / radix->i(), node.op_type(),
                        ": radix^group_size must fit in positive INT64.");
    limit *= radix->i();
  }
  int64_t width = 0;
  for (int64_t maximum = limit - 1; maximum != 0; maximum /= 2) {
    ++width;
  }
  return {group->i(), width};
}

SymDim WordCount(const SymDim &count, const PackingParameters &parameters, int64_t word_bits) {
  if (count.IsInt()) {
    const int64_t n = count.AsInt();
    EXT_ENFORCE_INVALID(n >= 0, "Pack/Unpack: count must be nonnegative.");
    const int64_t groups = n / parameters.group_size + (n % parameters.group_size != 0);
    EXT_ENFORCE_INVALID(groups / word_bits <=
                            std::numeric_limits<int64_t>::max() / parameters.width,
                        "Pack/Unpack: packed dimension exceeds INT64.");
    const int64_t whole = (groups / word_bits) * parameters.width;
    const int64_t tail = ((groups % word_bits) * parameters.width + word_bits - 1) / word_bits;
    EXT_ENFORCE_INVALID(whole <= std::numeric_limits<int64_t>::max() - tail,
                        "Pack/Unpack: packed dimension exceeds INT64.");
    return SymDim(whole + tail);
  }
  if (parameters.group_size == 1 && parameters.width == word_bits) {
    return count;
  }

  // Quotient/remainder decomposition avoids overflowing the intermediate
  // products, including when a symbolic count later resolves near INT64_MAX.
  const auto n = core::symbolic::ToDimType(count);
  const expressions::DimType group{parameters.group_size};
  const expressions::DimType word{word_bits};
  const expressions::DimType width{parameters.width};
  const auto groups = expressions::dim_add(
      expressions::dim_div(n, group),
      expressions::dim_div(expressions::dim_add(expressions::dim_mod(n, group),
                                                expressions::DimType{parameters.group_size - 1}),
                           group));
  const auto whole = expressions::dim_mul(expressions::dim_div(groups, word), width);
  const auto tail = expressions::dim_div(
      expressions::dim_add(expressions::dim_mul(expressions::dim_mod(groups, word), width),
                           expressions::DimType{word_bits - 1}),
      word);
  return core::symbolic::FromDimType(expressions::dim_add(whole, tail));
}

void ComputePackingShape(ShapesContext &ctx, const NodeProto &node, bool unpack) {
  CheckNodeOpAndOutput(node, unpack ? "Unpack" : "Pack", "ComputePackingShape");
  EXT_ENFORCE_INVALID(node.input_size() >= (unpack ? 2 : 1), node.op_type(),
                      ": missing required input.");
  const auto parameters = Parameters(node);
  const auto *axis_attribute = IntegerAttribute(node, "axis");
  const auto &input = ctx.Get(node.input(0));
  auto shape = input.Shape();
  const int64_t rank = static_cast<int64_t>(shape.Rank());
  EXT_ENFORCE_INVALID(rank > 0, node.op_type(), ": input must have rank at least one.");
  for (size_t i = 0; i < shape.Rank(); ++i) {
    EXT_ENFORCE_INVALID(!shape[i].IsInt() || shape[i].AsInt() >= 0, node.op_type(),
                        ": input dimensions must be nonnegative.");
  }
  int64_t axis = axis_attribute == nullptr ? -1 : axis_attribute->i();
  EXT_ENFORCE_INVALID(axis >= -rank && axis < rank, node.op_type(), ": axis is out of range.");
  if (axis < 0) {
    axis += rank;
  }

  TensorType storage_type = input.Dtype();
  if (unpack) {
    EXT_ENFORCE_INVALID(storage_type == TensorType::kUint8 || storage_type == TensorType::kUint32,
                        "Unpack: input must be UINT8 or UINT32.");
  } else {
    EXT_ENFORCE_INVALID(input.Dtype() == TensorType::kUint8, "Pack: input must be UINT8.");
    const auto *output_type = IntegerAttribute(node, "output_type");
    const int64_t type =
        output_type == nullptr ? static_cast<int64_t>(TensorProto::UINT8) : output_type->i();
    EXT_ENFORCE_INVALID(type == TensorProto::UINT8 || type == TensorProto::UINT32,
                        "Pack: output_type must be UINT8 or UINT32.");
    storage_type = type == TensorProto::UINT32 ? TensorType::kUint32 : TensorType::kUint8;
  }
  const int64_t word_bits = storage_type == TensorType::kUint32 ? 32 : 8;
  const size_t index = static_cast<size_t>(axis);
  if (unpack) {
    const auto &count_tensor = ctx.Get(node.input(1));
    EXT_ENFORCE_INVALID(count_tensor.Dtype() == TensorType::kInt64 &&
                            count_tensor.Shape().Rank() == 0,
                        "Unpack: count must be a scalar INT64 tensor.");
    SymDim count = UnknownCount(node.input(1));
    if (count_tensor.HasValueAsShape()) {
      EXT_ENFORCE_INVALID(count_tensor.ValueAsShape().Rank() == 1,
                          "Unpack: count must contain exactly one value.");
      count = count_tensor.ValueAsShape()[0];
    } else if (count_tensor.Data() != nullptr) {
      int64_t value;
      std::memcpy(&value, count_tensor.Data(), sizeof(value));
      count = SymDim(value);
    } else if (parameters.group_size == 1 && parameters.width == word_bits) {
      count = shape[index];
    }
    if (count.IsInt()) {
      const auto required = WordCount(count, parameters, word_bits);
      EXT_ENFORCE_INVALID(!shape[index].IsInt() || shape[index].AsInt() == required.AsInt(),
                          "Unpack: packed axis must equal the exact required word count.");
    }
    shape[index] = std::move(count);
  } else {
    shape[index] = WordCount(shape[index], parameters, word_bits);
  }
  ctx.Set(node.output(0),
          SymTensor(nullptr, unpack ? TensorType::kUint8 : storage_type, std::move(shape)));
}

} // namespace

void ComputeShapePack(ShapesContext &ctx, const NodeProto &node) {
  ComputePackingShape(ctx, node, false);
}

void ComputeShapeUnpack(ShapesContext &ctx, const NodeProto &node) {
  ComputePackingShape(ctx, node, true);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
