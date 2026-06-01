// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/traditionalml/shape_traditionalml.h"

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace traditionalml {

namespace {

int64_t VocabularyLength(const NodeProto &node) {
  const AttributeProto *str_vocab = FindAttribute(node, "string_vocabulary");
  const AttributeProto *int_vocab = FindAttribute(node, "int64_vocabulary");
  const bool has_str = str_vocab != nullptr && str_vocab->strings_size() > 0;
  const bool has_int = int_vocab != nullptr && int_vocab->ints_size() > 0;
  EXT_ENFORCE_INVALID(
      has_str != has_int,
      "ComputeShapeDictVectorizer: exactly one of 'string_vocabulary' or 'int64_vocabulary' "
      "must be specified and non-empty.");
  return has_str ? static_cast<int64_t>(str_vocab->strings_size())
                 : static_cast<int64_t>(int_vocab->ints_size());
}

} // namespace

void ComputeShapeDictVectorizer(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "DictVectorizer", "ComputeShapeDictVectorizer");

  const int64_t c = VocabularyLength(node);
  OptimShape output_shape;
  output_shape.PushBack(OptimDim(c));

  // The output element type is the value type of the input map. When the input
  // is recorded in the context as a typed tensor (e.g. coming from a producer
  // node that already encoded the map's value type as a tensor dtype), we
  // forward that dtype; otherwise we mark the dtype as undefined.
  TensorType output_dtype = TensorType::kUndefined;
  if (ctx.Has(x)) {
    const OptimTensor &input = ctx.Get(x);
    if (input.Dtype() != TensorType::kUndefined) {
      output_dtype = input.Dtype();
    }
  }
  (void)x;
  ctx.Set(node.output(0), OptimTensor(nullptr, output_dtype, std::move(output_shape)));
}

} // namespace traditionalml
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
