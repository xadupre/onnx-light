#include "onnx.h"
#include "onnx_lib/shape_inference/implementation.h"
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

TEST(onnx_shape_inference, InferShapesImpl_RNNBidirectionalLayout0) {
  ModelProto model;
  model.set_ir_version(IR_VERSION);
  auto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(22);

  GraphProto *graph = model.mutable_graph();
  graph->set_name("infer_shapes_impl_rnn_graph");

  ValueInfoProto *x = graph->add_input();
  x->set_name("X");
  TypeProto::Tensor *x_tensor = x->mutable_type()->mutable_tensor_type();
  x_tensor->set_elem_type(TensorProto::FLOAT);
  TensorShapeProto *x_shape = x_tensor->mutable_shape();
  x_shape->add_dim()->set_dim_value(4);
  x_shape->add_dim()->set_dim_value(2);
  x_shape->add_dim()->set_dim_value(3);

  ValueInfoProto *w = graph->add_input();
  w->set_name("W");
  TypeProto::Tensor *w_tensor = w->mutable_type()->mutable_tensor_type();
  w_tensor->set_elem_type(TensorProto::FLOAT);
  TensorShapeProto *w_shape = w_tensor->mutable_shape();
  w_shape->add_dim()->set_dim_value(2);
  w_shape->add_dim()->set_dim_value(5);
  w_shape->add_dim()->set_dim_value(3);

  ValueInfoProto *r = graph->add_input();
  r->set_name("R");
  TypeProto::Tensor *r_tensor = r->mutable_type()->mutable_tensor_type();
  r_tensor->set_elem_type(TensorProto::FLOAT);
  TensorShapeProto *r_shape = r_tensor->mutable_shape();
  r_shape->add_dim()->set_dim_value(2);
  r_shape->add_dim()->set_dim_value(5);
  r_shape->add_dim()->set_dim_value(5);

  ValueInfoProto *y = graph->add_output();
  y->set_name("Y");
  y->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);

  ValueInfoProto *y_h = graph->add_output();
  y_h->set_name("Y_h");
  y_h->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);

  NodeProto *rnn_node = graph->add_node();
  rnn_node->set_op_type("RNN");
  *rnn_node->add_input() = "X";
  *rnn_node->add_input() = "W";
  *rnn_node->add_input() = "R";
  *rnn_node->add_output() = "Y";
  *rnn_node->add_output() = "Y_h";

  AttributeProto *direction = rnn_node->add_attribute();
  direction->set_name("direction");
  direction->set_type(AttributeProto::STRING);
  direction->set_s("bidirectional");
  AttributeProto *hidden_size = rnn_node->add_attribute();
  hidden_size->set_name("hidden_size");
  hidden_size->set_type(AttributeProto::INT);
  hidden_size->set_i(5);

  shape_inference::InferShapes(model);

  const auto &y_dims =
      model.ref_graph().ref_output()[0].ref_type().ref_tensor_type().ref_shape().ref_dim();
  ASSERT_EQ(y_dims.size(), 4u);
  EXPECT_EQ(y_dims[0].ref_dim_value(), 4);
  EXPECT_EQ(y_dims[1].ref_dim_value(), 2);
  EXPECT_EQ(y_dims[2].ref_dim_value(), 2);
  EXPECT_EQ(y_dims[3].ref_dim_value(), 5);

  const auto &y_h_dims =
      model.ref_graph().ref_output()[1].ref_type().ref_tensor_type().ref_shape().ref_dim();
  ASSERT_EQ(y_h_dims.size(), 3u);
  EXPECT_EQ(y_h_dims[0].ref_dim_value(), 2);
  EXPECT_EQ(y_h_dims[1].ref_dim_value(), 2);
  EXPECT_EQ(y_h_dims[2].ref_dim_value(), 5);
}

TEST(onnx_shape_inference, InferShapesImpl_LSTMLayout1WithYc) {
  ModelProto model;
  model.set_ir_version(IR_VERSION);
  auto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(22);

  GraphProto *graph = model.mutable_graph();
  graph->set_name("infer_shapes_impl_lstm_graph");

  ValueInfoProto *x = graph->add_input();
  x->set_name("X");
  TypeProto::Tensor *x_tensor = x->mutable_type()->mutable_tensor_type();
  x_tensor->set_elem_type(TensorProto::FLOAT);
  TensorShapeProto *x_shape = x_tensor->mutable_shape();
  x_shape->add_dim()->set_dim_value(2);
  x_shape->add_dim()->set_dim_value(6);
  x_shape->add_dim()->set_dim_value(3);

  ValueInfoProto *w = graph->add_input();
  w->set_name("W");
  TypeProto::Tensor *w_tensor = w->mutable_type()->mutable_tensor_type();
  w_tensor->set_elem_type(TensorProto::FLOAT);
  TensorShapeProto *w_shape = w_tensor->mutable_shape();
  w_shape->add_dim()->set_dim_value(1);
  w_shape->add_dim()->set_dim_value(16);
  w_shape->add_dim()->set_dim_value(3);

  ValueInfoProto *r = graph->add_input();
  r->set_name("R");
  TypeProto::Tensor *r_tensor = r->mutable_type()->mutable_tensor_type();
  r_tensor->set_elem_type(TensorProto::FLOAT);
  TensorShapeProto *r_shape = r_tensor->mutable_shape();
  r_shape->add_dim()->set_dim_value(1);
  r_shape->add_dim()->set_dim_value(16);
  r_shape->add_dim()->set_dim_value(4);

  ValueInfoProto *y = graph->add_output();
  y->set_name("Y");
  y->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);

  ValueInfoProto *y_h = graph->add_output();
  y_h->set_name("Y_h");
  y_h->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);

  ValueInfoProto *y_c = graph->add_output();
  y_c->set_name("Y_c");
  y_c->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);

  NodeProto *lstm_node = graph->add_node();
  lstm_node->set_op_type("LSTM");
  *lstm_node->add_input() = "X";
  *lstm_node->add_input() = "W";
  *lstm_node->add_input() = "R";
  *lstm_node->add_output() = "Y";
  *lstm_node->add_output() = "Y_h";
  *lstm_node->add_output() = "Y_c";

  AttributeProto *hidden_size = lstm_node->add_attribute();
  hidden_size->set_name("hidden_size");
  hidden_size->set_type(AttributeProto::INT);
  hidden_size->set_i(4);
  AttributeProto *layout = lstm_node->add_attribute();
  layout->set_name("layout");
  layout->set_type(AttributeProto::INT);
  layout->set_i(1);

  shape_inference::InferShapes(model);

  const auto &y_dims =
      model.ref_graph().ref_output()[0].ref_type().ref_tensor_type().ref_shape().ref_dim();
  ASSERT_EQ(y_dims.size(), 4u);
  EXPECT_EQ(y_dims[0].ref_dim_value(), 2);
  EXPECT_EQ(y_dims[1].ref_dim_value(), 6);
  EXPECT_EQ(y_dims[2].ref_dim_value(), 1);
  EXPECT_EQ(y_dims[3].ref_dim_value(), 4);

  const auto &y_h_dims =
      model.ref_graph().ref_output()[1].ref_type().ref_tensor_type().ref_shape().ref_dim();
  ASSERT_EQ(y_h_dims.size(), 3u);
  EXPECT_EQ(y_h_dims[0].ref_dim_value(), 2);
  EXPECT_EQ(y_h_dims[1].ref_dim_value(), 1);
  EXPECT_EQ(y_h_dims[2].ref_dim_value(), 4);

  const auto &y_c_dims =
      model.ref_graph().ref_output()[2].ref_type().ref_tensor_type().ref_shape().ref_dim();
  ASSERT_EQ(y_c_dims.size(), 3u);
  EXPECT_EQ(y_c_dims[0].ref_dim_value(), 2);
  EXPECT_EQ(y_c_dims[1].ref_dim_value(), 1);
  EXPECT_EQ(y_c_dims[2].ref_dim_value(), 4);
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

TEST(onnx_shape_inference, GetAttributeProtoElemTypeAndLength) {
  AttributeProto ints_attr;
  ints_attr.set_name("ints_attr");
  ints_attr.set_type(AttributeProto::AttributeType::INTS);
  ints_attr.add_ints(1);
  ints_attr.add_ints(2);
  auto [ints_elem_type, ints_length] = getAttributeProtoElemTypeAndLength(&ints_attr);
  EXPECT_EQ(ints_elem_type, TensorProto::DataType::INT64);
  EXPECT_EQ(ints_length, 2);

  AttributeProto floats_attr;
  floats_attr.set_name("floats_attr");
  floats_attr.set_type(AttributeProto::AttributeType::FLOATS);
  floats_attr.add_floats(1.5f);
  auto [floats_elem_type, floats_length] = getAttributeProtoElemTypeAndLength(&floats_attr);
  EXPECT_EQ(floats_elem_type, TensorProto::DataType::FLOAT);
  EXPECT_EQ(floats_length, 1);

  AttributeProto strings_attr;
  strings_attr.set_name("strings_attr");
  strings_attr.set_type(AttributeProto::AttributeType::STRINGS);
  *strings_attr.add_strings() = "a";
  *strings_attr.add_strings() = "b";
  *strings_attr.add_strings() = "c";
  auto [strings_elem_type, strings_length] = getAttributeProtoElemTypeAndLength(&strings_attr);
  EXPECT_EQ(strings_elem_type, TensorProto::DataType::STRING);
  EXPECT_EQ(strings_length, 3);

  AttributeProto tensor_attr;
  tensor_attr.set_name("tensor_attr");
  tensor_attr.set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto *tensor = tensor_attr.mutable_t();
  tensor->set_data_type(TensorProto::DataType::INT32);
  tensor->add_dims(4);
  auto [tensor_elem_type, tensor_length] = getAttributeProtoElemTypeAndLength(&tensor_attr);
  EXPECT_EQ(tensor_elem_type, TensorProto::DataType::INT32);
  EXPECT_EQ(tensor_length, 4);

  AttributeProto empty_attr;
  auto [empty_elem_type, empty_length] = getAttributeProtoElemTypeAndLength(&empty_attr);
  EXPECT_EQ(empty_elem_type, TensorProto::DataType::UNDEFINED);
  EXPECT_EQ(empty_length, 0);
}

TEST(onnx_shape_inference, GetAttributeProtoElemTypeAndLength_RejectsNon1DTensor) {
  AttributeProto tensor_attr;
  tensor_attr.set_name("tensor_attr");
  tensor_attr.set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto *tensor = tensor_attr.mutable_t();
  tensor->set_data_type(TensorProto::DataType::INT64);
  tensor->add_dims(2);
  tensor->add_dims(3);
  EXPECT_THROW(getAttributeProtoElemTypeAndLength(&tensor_attr), InferenceError);
}

namespace {

TypeProto MakeTensorTypeProto(int32_t elem_type) {
  TypeProto tp;
  tp.add_tensor_type()->set_elem_type(elem_type);
  return tp;
}

TypeProto MakeTensorTypeProtoWithShape(int32_t elem_type, const std::vector<int64_t> &dims) {
  TypeProto tp;
  auto *tensor = tp.add_tensor_type();
  tensor->set_elem_type(elem_type);
  auto *shape = tensor->add_shape();
  for (int64_t d : dims) {
    auto *dim = shape->add_dim();
    if (d >= 0) {
      dim->set_dim_value(d);
    }
  }
  return tp;
}

TypeProto MakeSparseTensorTypeProto(int32_t elem_type) {
  TypeProto tp;
  tp.add_sparse_tensor_type()->set_elem_type(elem_type);
  return tp;
}

TypeProto MakeSparseTensorTypeProtoWithShape(int32_t elem_type, const std::vector<int64_t> &dims) {
  TypeProto tp;
  auto *sparse = tp.add_sparse_tensor_type();
  sparse->set_elem_type(elem_type);
  auto *shape = sparse->add_shape();
  for (int64_t d : dims) {
    auto *dim = shape->add_dim();
    if (d >= 0) {
      dim->set_dim_value(d);
    }
  }
  return tp;
}

} // namespace

// ---- CheckTensorShapesAndTypes tests (via checkShapesAndTypes) ----

// Tensor: matching elem type and shape passes without error.
TEST(CheckTensorShapesAndTypes, Tensor_NoError_MatchingElemTypeAndShape) {
  TypeProto inferred = MakeTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3, 4});
  TypeProto existing = MakeTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3, 4});
  EXPECT_NO_THROW(shape_inference::checkShapesAndTypes(inferred, existing));
}

// Tensor: inferred elem type UNDEFINED is compatible with any existing elem type.
TEST(CheckTensorShapesAndTypes, Tensor_NoError_InferredElemTypeUndefined) {
  TypeProto inferred = MakeTensorTypeProtoWithShape(TensorProto::DataType::UNDEFINED, {3});
  TypeProto existing = MakeTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3});
  EXPECT_NO_THROW(shape_inference::checkShapesAndTypes(inferred, existing));
}

// Tensor: existing elem type UNDEFINED is compatible with any inferred elem type.
TEST(CheckTensorShapesAndTypes, Tensor_NoError_ExistingElemTypeUndefined) {
  TypeProto inferred = MakeTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3});
  TypeProto existing = MakeTensorTypeProtoWithShape(TensorProto::DataType::UNDEFINED, {3});
  EXPECT_NO_THROW(shape_inference::checkShapesAndTypes(inferred, existing));
}

// Tensor: no shape on inferred side returns without error.
TEST(CheckTensorShapesAndTypes, Tensor_NoError_InferredHasNoShape) {
  TypeProto inferred = MakeTensorTypeProto(TensorProto::DataType::FLOAT);
  TypeProto existing = MakeTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3, 4});
  EXPECT_NO_THROW(shape_inference::checkShapesAndTypes(inferred, existing));
}

// Tensor: no shape on existing side returns without error.
TEST(CheckTensorShapesAndTypes, Tensor_NoError_ExistingHasNoShape) {
  TypeProto inferred = MakeTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3, 4});
  TypeProto existing = MakeTensorTypeProto(TensorProto::DataType::FLOAT);
  EXPECT_NO_THROW(shape_inference::checkShapesAndTypes(inferred, existing));
}

// Tensor: mismatched elem types both defined throws InferenceError.
TEST(CheckTensorShapesAndTypes, Tensor_ElemTypeMismatch) {
  TypeProto inferred = MakeTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3});
  TypeProto existing = MakeTensorTypeProtoWithShape(TensorProto::DataType::DOUBLE, {3});
  EXPECT_THROW(shape_inference::checkShapesAndTypes(inferred, existing),
               ONNX_LIGHT_NAMESPACE::InferenceError);
}

// Tensor: different ranks throw InferenceError.
TEST(CheckTensorShapesAndTypes, Tensor_RankMismatch) {
  TypeProto inferred = MakeTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3, 4});
  TypeProto existing = MakeTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3, 4, 5});
  EXPECT_THROW(shape_inference::checkShapesAndTypes(inferred, existing),
               ONNX_LIGHT_NAMESPACE::InferenceError);
}

// Tensor: conflicting dim values throw InferenceError.
TEST(CheckTensorShapesAndTypes, Tensor_DimValueMismatch) {
  TypeProto inferred = MakeTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3, 4});
  TypeProto existing = MakeTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3, 5});
  EXPECT_THROW(shape_inference::checkShapesAndTypes(inferred, existing),
               ONNX_LIGHT_NAMESPACE::InferenceError);
}

// Tensor: symbolic dim on either side does not trigger a dim value conflict.
TEST(CheckTensorShapesAndTypes, Tensor_NoError_SymbolicDimVsDimValue) {
  // inferred has symbolic dim, existing has concrete dim value: no conflict
  TypeProto inferred = MakeTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {-1, 4});
  TypeProto existing = MakeTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3, 4});
  EXPECT_NO_THROW(shape_inference::checkShapesAndTypes(inferred, existing));
}

// SparseTensor: matching elem type and shape passes without error.
TEST(CheckTensorShapesAndTypes, SparseTensor_NoError_MatchingElemTypeAndShape) {
  TypeProto inferred = MakeSparseTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {10});
  TypeProto existing = MakeSparseTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {10});
  EXPECT_NO_THROW(shape_inference::checkShapesAndTypes(inferred, existing));
}

// SparseTensor: inferred elem type UNDEFINED is compatible with any existing elem type.
TEST(CheckTensorShapesAndTypes, SparseTensor_NoError_InferredElemTypeUndefined) {
  TypeProto inferred = MakeSparseTensorTypeProto(TensorProto::DataType::UNDEFINED);
  TypeProto existing = MakeSparseTensorTypeProto(TensorProto::DataType::FLOAT);
  EXPECT_NO_THROW(shape_inference::checkShapesAndTypes(inferred, existing));
}

// SparseTensor: mismatched elem types both defined throws InferenceError.
TEST(CheckTensorShapesAndTypes, SparseTensor_ElemTypeMismatch) {
  TypeProto inferred = MakeSparseTensorTypeProto(TensorProto::DataType::FLOAT);
  TypeProto existing = MakeSparseTensorTypeProto(TensorProto::DataType::DOUBLE);
  EXPECT_THROW(shape_inference::checkShapesAndTypes(inferred, existing),
               ONNX_LIGHT_NAMESPACE::InferenceError);
}

// SparseTensor: different ranks throw InferenceError.
TEST(CheckTensorShapesAndTypes, SparseTensor_RankMismatch) {
  TypeProto inferred = MakeSparseTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3, 4});
  TypeProto existing = MakeSparseTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3});
  EXPECT_THROW(shape_inference::checkShapesAndTypes(inferred, existing),
               ONNX_LIGHT_NAMESPACE::InferenceError);
}

// SparseTensor: conflicting dim values throw InferenceError.
TEST(CheckTensorShapesAndTypes, SparseTensor_DimValueMismatch) {
  TypeProto inferred = MakeSparseTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3, 4});
  TypeProto existing = MakeSparseTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3, 9});
  EXPECT_THROW(shape_inference::checkShapesAndTypes(inferred, existing),
               ONNX_LIGHT_NAMESPACE::InferenceError);
}

// checkShapesAndTypes: VALUE_NOT_SET on either side returns without error.
TEST(CheckTensorShapesAndTypes, checkShapesAndTypes_NoError_InferredValueNotSet) {
  TypeProto inferred; // VALUE_NOT_SET
  TypeProto existing = MakeTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3});
  EXPECT_NO_THROW(shape_inference::checkShapesAndTypes(inferred, existing));
}

TEST(CheckTensorShapesAndTypes, checkShapesAndTypes_NoError_ExistingValueNotSet) {
  TypeProto inferred = MakeTensorTypeProtoWithShape(TensorProto::DataType::FLOAT, {3});
  TypeProto existing; // VALUE_NOT_SET
  EXPECT_NO_THROW(shape_inference::checkShapesAndTypes(inferred, existing));
}

// checkShapesAndTypes: mismatched type case (tensor vs sparse tensor) throws InferenceError.
TEST(CheckTensorShapesAndTypes, checkShapesAndTypes_TypeCaseMismatch) {
  TypeProto inferred = MakeTensorTypeProto(TensorProto::DataType::FLOAT);
  TypeProto existing = MakeSparseTensorTypeProto(TensorProto::DataType::FLOAT);
  EXPECT_THROW(shape_inference::checkShapesAndTypes(inferred, existing),
               ONNX_LIGHT_NAMESPACE::InferenceError);
}

namespace {

// Builds a model graph holding a single FlexAttention node with rank-4
// Q/K/V inputs of the given shapes (all FLOAT) and an empty-typed output
// ``Y`` so that ``InferShapes`` is responsible for populating it.
ModelProto MakeFlexAttentionModel(const std::vector<int64_t> &q_shape,
                                  const std::vector<int64_t> &k_shape,
                                  const std::vector<int64_t> &v_shape) {
  ModelProto model;
  model.set_ir_version(IR_VERSION);
  auto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(14);
  auto *preview_opset = model.add_opset_import();
  preview_opset->set_domain("ai.onnx.preview");
  preview_opset->set_version(1);

  GraphProto *graph = model.mutable_graph();
  graph->set_name("flex_attention_graph");

  auto add_input = [&](const char *name, const std::vector<int64_t> &shape) {
    ValueInfoProto *vi = graph->add_input();
    vi->set_name(name);
    TypeProto::Tensor *t = vi->mutable_type()->mutable_tensor_type();
    t->set_elem_type(TensorProto::FLOAT);
    TensorShapeProto *s = t->mutable_shape();
    for (int64_t d : shape) {
      s->add_dim()->set_dim_value(d);
    }
  };
  add_input("Q", q_shape);
  add_input("K", k_shape);
  add_input("V", v_shape);

  ValueInfoProto *output = graph->add_output();
  output->set_name("Y");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);

  NodeProto *node = graph->add_node();
  node->set_op_type("FlexAttention");
  node->set_domain("ai.onnx.preview");
  *node->add_input() = "Q";
  *node->add_input() = "K";
  *node->add_input() = "V";
  *node->add_output() = "Y";
  return model;
}

} // namespace

// FlexAttention: same number of Q and KV heads, output is (B, Hq, L, Dv).
TEST(onnx_shape_inference, InferShapesImpl_FlexAttention_BasicShape) {
  ModelProto model = MakeFlexAttentionModel(/*q=*/{2, 4, 8, 16},
                                            /*k=*/{2, 4, 12, 16},
                                            /*v=*/{2, 4, 12, 32});

  shape_inference::InferShapes(model);

  const auto &out = model.ref_graph().ref_output()[0];
  ASSERT_TRUE(out.ref_type().has_tensor_type());
  EXPECT_EQ(out.ref_type().ref_tensor_type().elem_type(), TensorProto::FLOAT);
  const auto &dims = out.ref_type().ref_tensor_type().ref_shape().ref_dim();
  ASSERT_EQ(dims.size(), 4u);
  EXPECT_EQ(dims[0].ref_dim_value(), 2);  // batch
  EXPECT_EQ(dims[1].ref_dim_value(), 4);  // q_num_heads
  EXPECT_EQ(dims[2].ref_dim_value(), 8);  // q_seq_len
  EXPECT_EQ(dims[3].ref_dim_value(), 32); // v_head_size
}

// FlexAttention with Grouped Query Attention: q_num_heads is a multiple of
// kv_num_heads. Output's head dim must reflect q_num_heads, not kv_num_heads.
TEST(onnx_shape_inference, InferShapesImpl_FlexAttention_GroupedQueryAttention) {
  ModelProto model = MakeFlexAttentionModel(/*q=*/{1, 8, 5, 6},
                                            /*k=*/{1, 2, 7, 6},
                                            /*v=*/{1, 2, 7, 6});

  shape_inference::InferShapes(model);

  const auto &dims =
      model.ref_graph().ref_output()[0].ref_type().ref_tensor_type().ref_shape().ref_dim();
  ASSERT_EQ(dims.size(), 4u);
  EXPECT_EQ(dims[0].ref_dim_value(), 1);
  EXPECT_EQ(dims[1].ref_dim_value(), 8);
  EXPECT_EQ(dims[2].ref_dim_value(), 5);
  EXPECT_EQ(dims[3].ref_dim_value(), 6);
}

// FlexAttention rejects mismatched K/V sequence lengths.
TEST(onnx_shape_inference, InferShapesImpl_FlexAttention_RejectsMismatchedKvSeqLen) {
  ModelProto model = MakeFlexAttentionModel(/*q=*/{1, 2, 3, 4},
                                            /*k=*/{1, 2, 5, 4},
                                            /*v=*/{1, 2, 7, 4});
  EXPECT_THROW(shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                                            ShapeInferenceOptions(false, 1, false)),
               ONNX_LIGHT_NAMESPACE::InferenceError);
}

// FlexAttention rejects q_num_heads that is not a multiple of kv_num_heads.
TEST(onnx_shape_inference, InferShapesImpl_FlexAttention_RejectsInvalidGqa) {
  ModelProto model = MakeFlexAttentionModel(/*q=*/{1, 7, 3, 4},
                                            /*k=*/{1, 2, 3, 4},
                                            /*v=*/{1, 2, 3, 4});
  EXPECT_THROW(shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                                            ShapeInferenceOptions(false, 1, false)),
               ONNX_LIGHT_NAMESPACE::InferenceError);
}

// FlexAttention rejects non-rank-4 inputs.
TEST(onnx_shape_inference, InferShapesImpl_FlexAttention_RejectsWrongRank) {
  ModelProto model = MakeFlexAttentionModel(/*q=*/{1, 2, 3},
                                            /*k=*/{1, 2, 3},
                                            /*v=*/{1, 2, 3});
  EXPECT_THROW(shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                                            ShapeInferenceOptions(false, 1, false)),
               ONNX_LIGHT_NAMESPACE::InferenceError);
}
