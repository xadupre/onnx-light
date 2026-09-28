// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

TypeProto TensorType(int32_t data_type, std::initializer_list<int64_t> dimensions) {
  TypeProto type;
  auto *tensor = type.mutable_tensor_type();
  tensor->set_elem_type(static_cast<TensorProto::DataType>(data_type));
  for (int64_t dimension : dimensions)
    tensor->mutable_shape()->add_dim()->set_dim_value(dimension);
  return type;
}

NodeProto Node() {
  NodeProto node;
  node.set_domain("onnx_light");
  node.set_op_type("QuantizePagedCache");
  for (const char *name :
       {"cache", "indices", "key_scale", "key_zero", "value_scale", "value_zero"})
    node.add_input(name);
  node.add_output("output");
  return node;
}

core::shapes::ShapesContext Context() {
  ::ONNX_LIGHT_NAMESPACE::onnx_shapes::RegisterShapeFunctions();
  core::shapes::ShapesContext context;
  context.SetOpsetVersion("onnx_light", 1);
  TypeProto cache;
  cache.mutable_struct_type();
  context.SetType("cache", cache);
  context.SetType("indices", TensorType(TensorProto::INT64, {2}));
  context.SetType("key_scale", TensorType(TensorProto::FLOAT, {}));
  context.SetType("key_zero", TensorType(TensorProto::INT2, {}));
  context.SetType("value_scale", TensorType(TensorProto::FLOAT, {}));
  context.SetType("value_zero", TensorType(TensorProto::UINT8, {}));
  return context;
}

} // namespace

TEST(QuantizePagedCacheShape, PreservesCacheAndAcceptsIndependentStorageTypes) {
  auto context = Context();
  context.ComputeShapeNode(Node());
  EXPECT_TRUE(context.GetType("output").Equals(context.GetType("cache")));

  context = Context();
  context.SetType("key_zero", TensorType(TensorProto::FLOAT16, {}));
  context.SetType("value_zero", TensorType(TensorProto::BFLOAT16, {}));
  context.ComputeShapeNode(Node());
  EXPECT_TRUE(context.GetType("output").Equals(context.GetType("cache")));
}

TEST(QuantizePagedCacheShape, RejectsInvalidIndicesScalesAndZeroPoints) {
  for (int failure = 0; failure < 3; ++failure) {
    auto context = Context();
    if (failure == 0)
      context.SetType("indices", TensorType(TensorProto::INT32, {2}));
    else if (failure == 1)
      context.SetType("key_scale", TensorType(TensorProto::FLOAT, {1}));
    else
      context.SetType("value_zero", TensorType(TensorProto::DOUBLE, {}));
    EXPECT_THROW(context.ComputeShapeNode(Node()), std::invalid_argument);
    EXPECT_FALSE(context.HasType("output"));
  }
}
