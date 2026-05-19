#include "onnx.h"
#include "onnx/shape_inference/implementation.h"
#include <gtest/gtest.h>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

template <class Type> void CreateDims(Type &proto, int num_dims) {
  auto &shape = proto.ref_shape();
  shape.ref_dim().clear();
  for (int i = 0; i < num_dims; ++i) {
    shape.add_dim();
  }
}

template <class Type> void SetDimValues(Type &proto, const std::vector<int64_t> &values) {
  auto &dims = proto.ref_shape().ref_dim();
  ASSERT_EQ(dims.size(), values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i] != -1) {
      dims[i].set_dim_value(values[i]);
    }
  }
}

template <class Type>
void SetDimParams(Type &proto, const std::vector<const std::string *> &values) {
  auto &dims = proto.ref_shape().ref_dim();
  ASSERT_EQ(dims.size(), values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i] != nullptr) {
      dims[i].set_dim_param(*values[i]);
    }
  }
}

void MergeDimensionInfo(const TensorShapeProto::Dimension &source,
                        TensorShapeProto::Dimension &target, size_t index) {
  if (source.has_dim_value()) {
    if (target.has_dim_value() && target.ref_dim_value() != source.ref_dim_value()) {
      throw std::runtime_error("Mismatch between dimension values at index " +
                               std::to_string(index));
    }
    target.set_dim_value(source.ref_dim_value());
    return;
  }
  if (!source.has_dim_param()) {
    return;
  }
  if (!target.has_dim_value() && !target.has_dim_param()) {
    target.set_dim_param(source.ref_dim_param());
  }
}

template <class Type> void MergeInShapeInfo(const Type &source, Type &target) {
  if (!source.has_shape()) {
    return;
  }
  if (!target.has_shape()) {
    target.ref_shape() = source.ref_shape();
    return;
  }
  const auto &source_dims = source.ref_shape().ref_dim();
  auto &target_dims = target.ref_shape().ref_dim();
  if (source_dims.size() != target_dims.size()) {
    throw std::runtime_error("Mismatch between number of inferred and declared dimensions");
  }
  for (size_t i = 0; i < source_dims.size(); ++i) {
    MergeDimensionInfo(source_dims[i], target_dims[i], i);
  }
}

} // namespace

TEST(onnx_shape_inference, mergeShapeInfo_HasShape) {
  TypeProto::Tensor source_tensor;
  TypeProto::Tensor target_tensor;
  CreateDims(source_tensor, 1);
  SetDimValues(source_tensor, {1});
  MergeInShapeInfo(source_tensor, target_tensor);
  ASSERT_EQ(target_tensor.ref_shape().ref_dim().size(), 1u);
  EXPECT_EQ(target_tensor.ref_shape().ref_dim()[0].ref_dim_value(), 1);

  TypeProto::SparseTensor source_sparse;
  TypeProto::SparseTensor target_sparse;
  CreateDims(source_sparse, 1);
  SetDimValues(source_sparse, {1});
  MergeInShapeInfo(source_sparse, target_sparse);
  ASSERT_EQ(target_sparse.ref_shape().ref_dim().size(), 1u);
  EXPECT_EQ(target_sparse.ref_shape().ref_dim()[0].ref_dim_value(), 1);
}

TEST(onnx_shape_inference, mergeShapeInfo_PreferValueOverParam) {
  const std::string param = "A";
  TypeProto::Tensor source;
  TypeProto::Tensor target;
  CreateDims(source, 1);
  SetDimValues(source, {1});
  CreateDims(target, 1);
  SetDimParams(target, {&param});
  MergeInShapeInfo(source, target);
  ASSERT_EQ(target.ref_shape().ref_dim().size(), 1u);
  EXPECT_TRUE(target.ref_shape().ref_dim()[0].has_dim_value());
  EXPECT_EQ(target.ref_shape().ref_dim()[0].ref_dim_value(), 1);
}

TEST(onnx_shape_inference, mergeShapeInfo_SourceParamCopiedWhenTargetUnknown) {
  const std::string param = "A";
  TypeProto::Tensor source;
  TypeProto::Tensor target;
  CreateDims(source, 1);
  SetDimParams(source, {&param});
  CreateDims(target, 1);
  MergeInShapeInfo(source, target);
  ASSERT_EQ(target.ref_shape().ref_dim().size(), 1u);
  EXPECT_EQ(target.ref_shape().ref_dim()[0].ref_dim_param(), "A");
}

TEST(onnx_shape_inference, mergeShapeInfo_CombineShapes) {
  TypeProto::Tensor source;
  TypeProto::Tensor target;
  CreateDims(source, 2);
  SetDimValues(source, {-1, 2});
  CreateDims(target, 2);
  SetDimValues(target, {1, -1});
  MergeInShapeInfo(source, target);
  ASSERT_EQ(target.ref_shape().ref_dim().size(), 2u);
  EXPECT_EQ(target.ref_shape().ref_dim()[0].ref_dim_value(), 1);
  EXPECT_EQ(target.ref_shape().ref_dim()[1].ref_dim_value(), 2);
}

TEST(onnx_shape_inference, mergeShapeInfo_Mismatches) {
  TypeProto::Tensor rank_source;
  TypeProto::Tensor rank_target;
  CreateDims(rank_source, 2);
  SetDimValues(rank_source, {-1, 2});
  CreateDims(rank_target, 3);
  SetDimValues(rank_target, {1, -1, 1});
  EXPECT_THROW(MergeInShapeInfo(rank_source, rank_target), std::runtime_error);

  TypeProto::Tensor source;
  TypeProto::Tensor target;
  CreateDims(source, 2);
  SetDimValues(source, {2, 2});
  CreateDims(target, 2);
  SetDimValues(target, {2, 1});
  EXPECT_THROW(MergeInShapeInfo(source, target), std::runtime_error);

  const std::string param_a = "A";
  const std::string param_b = "B";
  TypeProto::SparseTensor source_sparse;
  TypeProto::SparseTensor target_sparse;
  CreateDims(source_sparse, 1);
  SetDimParams(source_sparse, {&param_a});
  CreateDims(target_sparse, 1);
  SetDimParams(target_sparse, {&param_b});
  MergeInShapeInfo(source_sparse, target_sparse);
  ASSERT_EQ(target_sparse.ref_shape().ref_dim().size(), 1u);
  EXPECT_EQ(target_sparse.ref_shape().ref_dim()[0].ref_dim_param(), "B");
}

TEST(onnx_shape_inference, InferFunctionOutputTypes_Basic) {
  // Create a simple function: Y = Add(X, W)
  FunctionProto function;
  auto *opset = function.add_opset_import();
  opset->set_domain("");
  opset->set_version(14);
  function.set_name("test_add_function");
  function.set_domain("");
  *function.add_input() = "X";
  *function.add_input() = "W";
  *function.add_output() = "Y";

  // Add a single Add node
  NodeProto *add_node = function.add_node();
  add_node->set_op_type("Add");
  *add_node->add_input() = "X";
  *add_node->add_input() = "W";
  *add_node->add_output() = "Y";

  // Create input types: two float tensors with shape [3, 4]
  std::vector<TypeProto> input_types(2);
  for (int i = 0; i < 2; ++i) {
    TypeProto::Tensor *tensor = input_types[i].add_tensor_type();
    tensor->set_elem_type(1); // FLOAT
    TensorShapeProto *shape = tensor->add_shape();
    shape->add_dim()->set_dim_value(3);
    shape->add_dim()->set_dim_value(4);
  }

  // No attributes for this simple function
  std::vector<AttributeProto> attributes;

  // Infer output types
  std::vector<TypeProto> output_types =
      shape_inference::InferFunctionOutputTypes(function, input_types, attributes);

  // Verify results
  ASSERT_EQ(output_types.size(), 1);
  EXPECT_TRUE(output_types[0].has_tensor_type());
  EXPECT_EQ(output_types[0].ref_tensor_type().ref_elem_type(), 1); // FLOAT
  EXPECT_TRUE(output_types[0].ref_tensor_type().has_shape());
  EXPECT_EQ(output_types[0].ref_tensor_type().ref_shape().ref_dim().size(), 2);
  EXPECT_EQ(output_types[0].ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 3);
  EXPECT_EQ(output_types[0].ref_tensor_type().ref_shape().ref_dim()[1].ref_dim_value(), 4);
}

TEST(onnx_shape_inference, InferFunctionOutputTypes_MultipleOutputs) {
  // Create a function with multiple outputs: Y1, Y2 = Split(X)
  FunctionProto function;
  auto *opset = function.add_opset_import();
  opset->set_domain("");
  opset->set_version(14);
  function.set_name("test_split_function");
  function.set_domain("");
  *function.add_input() = "X";
  *function.add_output() = "Y1";
  *function.add_output() = "Y2";

  // Add a Split node
  NodeProto *split_node = function.add_node();
  split_node->set_op_type("Split");
  *split_node->add_input() = "X";
  *split_node->add_output() = "Y1";
  *split_node->add_output() = "Y2";

  // Add axis attribute
  AttributeProto *axis_attr = split_node->add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(AttributeProto::AttributeType::INT);
  axis_attr->set_i(0);

  // Create input type: float tensor with shape [4, 3]
  std::vector<TypeProto> input_types(1);
  TypeProto::Tensor *tensor = input_types[0].add_tensor_type();
  tensor->set_elem_type(1); // FLOAT
  TensorShapeProto *shape = tensor->add_shape();
  shape->add_dim()->set_dim_value(4);
  shape->add_dim()->set_dim_value(3);

  std::vector<AttributeProto> attributes;

  // Infer output types
  std::vector<TypeProto> output_types =
      shape_inference::InferFunctionOutputTypes(function, input_types, attributes);

  // Verify results - should have two outputs
  ASSERT_EQ(output_types.size(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_TRUE(output_types[i].has_tensor_type());
    EXPECT_EQ(output_types[i].ref_tensor_type().ref_elem_type(), 1); // FLOAT
  }
}

TEST(onnx_shape_inference, InferFunctionOutputTypes_WithAttributes) {
  // Create a function that uses an attribute: Y = Pad(X)
  FunctionProto function;
  auto *opset = function.add_opset_import();
  opset->set_domain("");
  opset->set_version(14);
  function.set_name("test_pad_function");
  function.set_domain("");
  *function.add_input() = "X";
  *function.add_output() = "Y";
  *function.add_attribute() = "pads";

  // Add a Pad node that references the function attribute
  NodeProto *pad_node = function.add_node();
  pad_node->set_op_type("Constant");
  *pad_node->add_output() = "pads_value";
  AttributeProto *value_attr = pad_node->add_attribute();
  value_attr->set_name("value");
  value_attr->set_ref_attr_name("pads");

  NodeProto *pad_node2 = function.add_node();
  pad_node2->set_op_type("Pad");
  *pad_node2->add_input() = "X";
  *pad_node2->add_input() = "pads_value";
  *pad_node2->add_output() = "Y";

  // Create input type
  std::vector<TypeProto> input_types(1);
  TypeProto::Tensor *tensor = input_types[0].add_tensor_type();
  tensor->set_elem_type(1); // FLOAT
  TensorShapeProto *shape = tensor->add_shape();
  shape->add_dim()->set_dim_value(2);
  shape->add_dim()->set_dim_value(3);

  // Create attribute value
  std::vector<AttributeProto> attributes(1);
  attributes[0].set_name("pads");
  attributes[0].set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto *pads_tensor = attributes[0].add_t();
  pads_tensor->set_data_type(TensorProto::DataType::INT64);
  pads_tensor->ref_dims().push_back(4);
  pads_tensor->add_int64_data(0);
  pads_tensor->add_int64_data(0);
  pads_tensor->add_int64_data(1);
  pads_tensor->add_int64_data(1);

  // Infer output types
  std::vector<TypeProto> output_types =
      shape_inference::InferFunctionOutputTypes(function, input_types, attributes);

  // Verify results
  ASSERT_EQ(output_types.size(), 1);
  EXPECT_TRUE(output_types[0].has_tensor_type());
  EXPECT_EQ(output_types[0].ref_tensor_type().ref_elem_type(), 1); // FLOAT
}

TEST(onnx_shape_inference, InferFunctionOutputTypes_MissingOptionalInput) {
  // Create a function with an optional input
  FunctionProto function;
  auto *opset = function.add_opset_import();
  opset->set_domain("");
  opset->set_version(14);
  function.set_name("test_optional_function");
  function.set_domain("");
  *function.add_input() = "X";
  *function.add_input() = "Y"; // optional
  *function.add_output() = "Z";

  // Add an Identity node
  NodeProto *identity_node = function.add_node();
  identity_node->set_op_type("Identity");
  *identity_node->add_input() = "X";
  *identity_node->add_output() = "Z";

  // Create input types with second input missing (VALUE_NOT_SET)
  std::vector<TypeProto> input_types(2);
  TypeProto::Tensor *tensor = input_types[0].add_tensor_type();
  tensor->set_elem_type(1); // FLOAT
  TensorShapeProto *shape = tensor->add_shape();
  shape->add_dim()->set_dim_value(5);
  // input_types[1] is left as VALUE_NOT_SET to indicate missing optional parameter

  std::vector<AttributeProto> attributes;

  // Infer output types
  std::vector<TypeProto> output_types =
      shape_inference::InferFunctionOutputTypes(function, input_types, attributes);

  // Verify results
  ASSERT_EQ(output_types.size(), 1);
  EXPECT_TRUE(output_types[0].has_tensor_type());
  EXPECT_EQ(output_types[0].ref_tensor_type().ref_elem_type(), 1); // FLOAT
  EXPECT_TRUE(output_types[0].ref_tensor_type().has_shape());
  EXPECT_EQ(output_types[0].ref_tensor_type().ref_shape().ref_dim().size(), 1);
  EXPECT_EQ(output_types[0].ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 5);
}

TEST(onnx_shape_inference, InferShapesImpl_ModelGraph) {
  ModelProto model;
  model.set_ir_version(IR_VERSION);
  auto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(14);

  GraphProto *graph = model.mutable_graph();
  graph->set_name("infer_shapes_impl_graph");

  ValueInfoProto *input = graph->add_input();
  input->set_name("X");
  TypeProto::Tensor *input_tensor = input->mutable_type()->mutable_tensor_type();
  input_tensor->set_elem_type(TensorProto::FLOAT);
  TensorShapeProto *input_shape = input_tensor->mutable_shape();
  input_shape->add_dim()->set_dim_value(2);
  input_shape->add_dim()->set_dim_value(3);

  ValueInfoProto *output = graph->add_output();
  output->set_name("Y");
  TypeProto::Tensor *output_tensor = output->mutable_type()->mutable_tensor_type();
  output_tensor->set_elem_type(TensorProto::FLOAT);

  NodeProto *add_node = graph->add_node();
  add_node->set_op_type("Add");
  *add_node->add_input() = "X";
  *add_node->add_input() = "X";
  *add_node->add_output() = "Y";

  shape_inference::InferShapes(model);

  const auto &inferred_output = model.ref_graph().ref_output()[0];
  ASSERT_TRUE(inferred_output.has_type());
  ASSERT_TRUE(inferred_output.ref_type().has_tensor_type());
  ASSERT_EQ(inferred_output.ref_type().ref_tensor_type().elem_type(), TensorProto::FLOAT);
  ASSERT_TRUE(inferred_output.ref_type().ref_tensor_type().has_shape());
  const auto &dims = inferred_output.ref_type().ref_tensor_type().ref_shape().ref_dim();
  constexpr size_t kExpectedDims = 2U;
  constexpr int64_t kExpectedDim0 = 2;
  constexpr int64_t kExpectedDim1 = 3;
  ASSERT_EQ(dims.size(), kExpectedDims);
  EXPECT_EQ(dims[0].ref_dim_value(), kExpectedDim0);
  EXPECT_EQ(dims[1].ref_dim_value(), kExpectedDim1);
}

TEST(onnx_shape_inference, GetValueCaseString_TensorType) {
  TypeProto type;
  type.add_tensor_type();
  EXPECT_EQ(shape_inference::GetValueCaseString(type), "tensor_type");
}

TEST(onnx_shape_inference, GetValueCaseString_SequenceType) {
  TypeProto type;
  type.add_sequence_type();
  EXPECT_EQ(shape_inference::GetValueCaseString(type), "sequence_type");
}

TEST(onnx_shape_inference, GetValueCaseString_MapType) {
  TypeProto type;
  type.add_map_type();
  EXPECT_EQ(shape_inference::GetValueCaseString(type), "map_type");
}

TEST(onnx_shape_inference, GetValueCaseString_OptionalType) {
  TypeProto type;
  type.add_optional_type();
  EXPECT_EQ(shape_inference::GetValueCaseString(type), "optional_type");
}

TEST(onnx_shape_inference, GetValueCaseString_SparseTensorType) {
  TypeProto type;
  type.add_sparse_tensor_type();
  EXPECT_EQ(shape_inference::GetValueCaseString(type), "sparse_tensor_type");
}

TEST(onnx_shape_inference, GetValueCaseString_NotSet) {
  TypeProto type;
  EXPECT_EQ(shape_inference::GetValueCaseString(type), "NOT_SET");
}

TEST(onnx_shape_inference, ArrayFeatureExtractorSymbolicDimConvertsToStdString) {
  const std::string symbolic_dim = "K";
  TensorShapeProto::Dimension dim;
  dim.set_dim_param(symbolic_dim);
  std::string converted = dim.dim_param().as_string();
  EXPECT_EQ(converted, symbolic_dim);
}

TEST(onnx_shape_inference, CastMapSupportsAllCastToValues) {
  struct CastCase {
    const char *cast_to;
  };
  const CastCase test_cases[] = {
      {"TO_FLOAT"},
      {"TO_INT64"},
      {"TO_STRING"},
  };

  for (const auto &test_case : test_cases) {
    SCOPED_TRACE(test_case.cast_to);
    AttributeProto cast_to_attr;
    cast_to_attr.set_name("cast_to");
    cast_to_attr.set_type(AttributeProto::AttributeType::STRING);
    cast_to_attr.set_s(test_case.cast_to);
    const auto &cast_to = cast_to_attr.ref_s();

    EXPECT_TRUE(cast_to == test_case.cast_to);
  }
}
