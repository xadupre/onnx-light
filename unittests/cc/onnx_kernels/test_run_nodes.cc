// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_kernels/kernels/training/include_training_kernels.h"
#include "onnx_kernels/run_nodes.h"
#include "onnx_kernels/simple_tensor.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_kernels::DataType;
using onnx_kernels::KernelDispatchTable;
using onnx_kernels::RunFunction;
using onnx_kernels::RunGraph;
using onnx_kernels::RunModel;
using onnx_kernels::RunNode;
using onnx_kernels::RunNodes;
using onnx_kernels::RuntimeContext;
using onnx_kernels::Tensor;
using onnx_kernels::TensorFromProto;
using onnx_kernels::TensorMap;
using onnx_kernels::kernel::KernelContext;

namespace Test {

namespace {

// Builds a single-node ``NodeProto`` of type ``op_type`` with the
// requested input and output names.
NodeProto MakeNode(const std::string &op_type, const std::vector<std::string> &inputs,
                   const std::vector<std::string> &outputs, const std::string &domain = "") {
  NodeProto node;
  node.set_op_type(op_type);
  if (!domain.empty()) {
    node.set_domain(domain);
  }
  for (const auto &name : inputs) {
    node.add_input(name);
  }
  for (const auto &name : outputs) {
    node.add_output(name);
  }
  return node;
}

} // namespace

TEST(RunNodes, DispatchTableContainsRegisteredOps) {
  const auto &table = KernelDispatchTable();
  // Spot-check the initial registered baseline of element-wise math ops.
  EXPECT_NE(table.find("ai.onnx:Add"), table.end());
  EXPECT_NE(table.find("ai.onnx:Sub"), table.end());
  EXPECT_NE(table.find("ai.onnx:Mul"), table.end());
  EXPECT_NE(table.find("ai.onnx:Div"), table.end());
  EXPECT_NE(table.find("ai.onnx:Neg"), table.end());
  EXPECT_NE(table.find("ai.onnx:Abs"), table.end());

  // Spot-check the extended registration set covering the rest of the
  // unary / binary / variadic / attribute-driven math + logical
  // kernels (see ``onnx_kernels/kernel_dispatch_table.cc``).
  // Unary math, no attributes.
  EXPECT_NE(table.find("ai.onnx:Cos"), table.end());
  EXPECT_NE(table.find("ai.onnx:Erf"), table.end());
  EXPECT_NE(table.find("ai.onnx:Sigmoid"), table.end());
  EXPECT_NE(table.find("ai.onnx:Tanh"), table.end());
  // Binary math (Gemm is attribute-driven; MatMul/Pow are not).
  EXPECT_NE(table.find("ai.onnx:Gemm"), table.end());
  EXPECT_NE(table.find("ai.onnx:MatMul"), table.end());
  EXPECT_NE(table.find("ai.onnx:Pow"), table.end());
  // Variadic reducers.
  EXPECT_NE(table.find("ai.onnx:Sum"), table.end());
  EXPECT_NE(table.find("ai.onnx:Max"), table.end());
  EXPECT_NE(table.find("ai.onnx:Min"), table.end());
  EXPECT_NE(table.find("ai.onnx:Mean"), table.end());
  // Reduction operators.
  EXPECT_NE(table.find("ai.onnx:ArgMax"), table.end());
  EXPECT_NE(table.find("ai.onnx:ArgMin"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceL1"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceL2"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceLogSum"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceLogSumExp"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceMax"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceMean"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceMin"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceProd"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceSum"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceSumSquare"), table.end());
  // Attribute-driven kernels.
  EXPECT_NE(table.find("ai.onnx:Softmax"), table.end());
  EXPECT_NE(table.find("ai.onnx:LeakyRelu"), table.end());
  EXPECT_NE(table.find("ai.onnx:HardSigmoid"), table.end());
  EXPECT_NE(table.find("ai.onnx:Selu"), table.end());
  EXPECT_NE(table.find("ai.onnx:Gelu"), table.end());
  EXPECT_NE(table.find("ai.onnx:Mod"), table.end());
  EXPECT_NE(table.find("ai.onnx:Clip"), table.end());
  EXPECT_NE(table.find("ai.onnx:Attention"), table.end());
  EXPECT_NE(table.find("ai.onnx:GRU"), table.end());
  EXPECT_NE(table.find("ai.onnx:NonMaxSuppression"), table.end());
  EXPECT_NE(table.find("ai.onnx:NonZero"), table.end());
  EXPECT_NE(table.find("ai.onnx:IsInf"), table.end());
  EXPECT_NE(table.find("ai.onnx:BitShift"), table.end());
  EXPECT_NE(table.find("ai.onnx:BitCast"), table.end());
  EXPECT_NE(table.find("ai.onnx:Einsum"), table.end());
  EXPECT_NE(table.find("ai.onnx:DFT"), table.end());
  EXPECT_NE(table.find("ai.onnx:TopK"), table.end());
  // Quantization kernels.
  EXPECT_NE(table.find("ai.onnx:QuantizeLinear"), table.end());
  EXPECT_NE(table.find("ai.onnx:DequantizeLinear"), table.end());
  EXPECT_NE(table.find("ai.onnx:DynamicQuantizeLinear"), table.end());
  // Generator kernels.
  EXPECT_NE(table.find("ai.onnx:EyeLike"), table.end());
  EXPECT_NE(table.find("ai.onnx:AffineGrid"), table.end());
  EXPECT_NE(table.find("ai.onnx:ImageDecoder"), table.end());
  // Tensor shape kernels.
  EXPECT_NE(table.find("ai.onnx:Cast"), table.end());
  EXPECT_NE(table.find("ai.onnx:Shape"), table.end());
  EXPECT_NE(table.find("ai.onnx:Size"), table.end());
  EXPECT_NE(table.find("ai.onnx:DepthToSpace"), table.end());
  EXPECT_NE(table.find("ai.onnx:Gather"), table.end());
  EXPECT_NE(table.find("ai.onnx:GatherND"), table.end());
  EXPECT_NE(table.find("ai.onnx:Pad"), table.end());
  // Logical / bitwise kernels.
  EXPECT_NE(table.find("ai.onnx:And"), table.end());
  EXPECT_NE(table.find("ai.onnx:Or"), table.end());
  EXPECT_NE(table.find("ai.onnx:Xor"), table.end());
  EXPECT_NE(table.find("ai.onnx:Not"), table.end());
  EXPECT_NE(table.find("ai.onnx:Equal"), table.end());
  EXPECT_NE(table.find("ai.onnx:Greater"), table.end());
  EXPECT_NE(table.find("ai.onnx:Less"), table.end());
  EXPECT_NE(table.find("ai.onnx:Where"), table.end());
  EXPECT_NE(table.find("ai.onnx:IsNaN"), table.end());
  EXPECT_NE(table.find("ai.onnx:BitwiseAnd"), table.end());
  EXPECT_NE(table.find("ai.onnx:BitwiseNot"), table.end());
  // ai.onnx.preview.training optimizers.
  EXPECT_NE(table.find("ai.onnx.preview.training:Adagrad"), table.end());
  EXPECT_NE(table.find("ai.onnx.preview.training:Adam"), table.end());
  EXPECT_NE(table.find("ai.onnx.preview.training:Momentum"), table.end());
  // ai.onnx.preview kernels.
  EXPECT_NE(table.find("ai.onnx.preview:FlexAttention"), table.end());
  // ai.onnx.ml kernels.
  EXPECT_NE(table.find("ai.onnx.ml:SVMRegressor"), table.end());
  EXPECT_NE(table.find("ai.onnx.ml:SVMClassifier"), table.end());
  EXPECT_NE(table.find("ai.onnx.ml:ArrayFeatureExtractor"), table.end());
  EXPECT_NE(table.find("ai.onnx.ml:Binarizer"), table.end());
  EXPECT_NE(table.find("ai.onnx.ml:Imputer"), table.end());
  EXPECT_NE(table.find("ai.onnx.ml:FeatureVectorizer"), table.end());
  EXPECT_NE(table.find("ai.onnx.ml:LabelEncoder"), table.end());
  // Linear attention (opset 27).
  EXPECT_NE(table.find("ai.onnx:LinearAttention"), table.end());
  // Sequence operators (opset 11+).
  EXPECT_NE(table.find("ai.onnx:SequenceConstruct"), table.end());
  EXPECT_NE(table.find("ai.onnx:SequenceEmpty"), table.end());
  EXPECT_NE(table.find("ai.onnx:SequenceAt"), table.end());
  EXPECT_NE(table.find("ai.onnx:SequenceErase"), table.end());
  EXPECT_NE(table.find("ai.onnx:SequenceInsert"), table.end());
  EXPECT_NE(table.find("ai.onnx:SequenceLength"), table.end());
  EXPECT_NE(table.find("ai.onnx:ConcatFromSequence"), table.end());
  EXPECT_NE(table.find("ai.onnx:SplitToSequence"), table.end());
  // Optional kernels (opset 15+).
  EXPECT_NE(table.find("ai.onnx:Optional"), table.end());
  EXPECT_NE(table.find("ai.onnx:OptionalGetElement"), table.end());
  EXPECT_NE(table.find("ai.onnx:OptionalHasElement"), table.end());
  // Text kernels (ai.onnx).
  EXPECT_NE(table.find("ai.onnx:RegexFullMatch"), table.end());
}

TEST(RunNodes, RunNodeSingleAdd) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f});
  rt.tensors()["y"] = Tensor::FromFloat("y", {3}, {10.0f, 20.0f, 30.0f});
  NodeProto node = MakeNode("Add", {"x", "y"}, {"z"});
  RunNode(node, rt);
  ASSERT_NE(rt.tensors().find("z"), rt.tensors().end());
  const Tensor &z = rt.tensors()["z"];
  EXPECT_EQ(z.name, "z");
  EXPECT_EQ(z.shape, std::vector<int64_t>({3}));
  ASSERT_EQ(z.element_count(), 3);
  const float *got = z.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 11.0f);
  EXPECT_FLOAT_EQ(got[1], 22.0f);
  EXPECT_FLOAT_EQ(got[2], 33.0f);
}

TEST(RunNodes, RunNodeGemmWithoutBiasUsesSchemaDefaults) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["a"] = Tensor::FromFloat("a", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  rt.tensors()["b"] = Tensor::FromFloat("b", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
  NodeProto node = MakeNode("Gemm", {"a", "b"}, {"y"});
  RunNode(node, rt);
  ASSERT_NE(rt.tensors().find("y"), rt.tensors().end());
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, std::vector<int64_t>({2, 2}));
  ASSERT_EQ(y.element_count(), 4);
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 19.0f);
  EXPECT_FLOAT_EQ(got[1], 22.0f);
  EXPECT_FLOAT_EQ(got[2], 43.0f);
  EXPECT_FLOAT_EQ(got[3], 50.0f);
}

TEST(RunNodes, RunNodeNormalisesDefaultDomain) {
  // The default ONNX domain is the empty string. The dispatcher must
  // normalise it to ``ai.onnx`` before looking up the kernel.
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2}, {-1.5f, 2.5f});
  NodeProto node = MakeNode("Abs", {"x"}, {"y"}); // empty domain
  EXPECT_TRUE(node.domain().as_string().empty());
  RunNode(node, rt);
  const float *got = rt.tensors()["y"].AsFloat();
  ASSERT_EQ(rt.tensors()["y"].element_count(), 2);
  EXPECT_FLOAT_EQ(got[0], 1.5f);
  EXPECT_FLOAT_EQ(got[1], 2.5f);
}

TEST(RunNodes, RunNodeCastFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3}, {-1.5f, 0.0f, 2.75f});
  NodeProto node = MakeNode("Cast", {"x"}, {"y"});
  AttributeProto *to = node.add_attribute();
  to->set_name("to");
  to->set_type(AttributeProto::INT);
  to->set_i(static_cast<int64_t>(DataType::INT32));

  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("y");
  EXPECT_EQ(y.data_type, static_cast<int32_t>(DataType::INT32));
  EXPECT_EQ(y.shape, std::vector<int64_t>({3}));
  const int32_t *got = y.AsInt32();
  EXPECT_EQ(got[0], -1);
  EXPECT_EQ(got[1], 0);
  EXPECT_EQ(got[2], 2);
}

TEST(RunNodes, RunNodeBitCastFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(26)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3}, {0.0f, 1.0f, -1.0f});
  NodeProto node = MakeNode("BitCast", {"x"}, {"y"});
  AttributeProto *to = node.add_attribute();
  to->set_name("to");
  to->set_type(AttributeProto::INT);
  to->set_i(static_cast<int64_t>(DataType::INT32));

  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("y");
  EXPECT_EQ(y.data_type, static_cast<int32_t>(DataType::INT32));
  EXPECT_EQ(y.shape, std::vector<int64_t>({3}));
  const int32_t *got = y.AsInt32();
  EXPECT_EQ(got[0], 0);
  EXPECT_EQ(got[1], 1065353216);
  EXPECT_EQ(got[2], -1082130432);
}

TEST(RunNodes, RunNodeNonMaxSuppressionFromDispatchTable) {
  // NonMaxSuppression was introduced in ONNX opset 10; use opset 11 to match
  // the other NonMaxSuppression kernel tests in this repository.
  RuntimeContext rt(KernelContext(DefaultOpset(11)));
  rt.tensors()["boxes"] =
      Tensor::FromFloat("boxes", {1, 2, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 10.0f, 1.0f, 11.0f});
  rt.tensors()["scores"] = Tensor::FromFloat("scores", {1, 1, 2}, {0.9f, 0.8f});
  rt.tensors()["max_output_boxes_per_class"] =
      Tensor::FromInt64("max_output_boxes_per_class", {1}, {10});
  rt.tensors()["iou_threshold"] = Tensor::FromFloat("iou_threshold", {1}, {0.5f});
  rt.tensors()["score_threshold"] = Tensor::FromFloat("score_threshold", {1}, {0.0f});

  NodeProto node = MakeNode(
      "NonMaxSuppression",
      {"boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"},
      {"selected_indices"});
  RunNode(node, rt);

  const Tensor &selected = rt.tensors().at("selected_indices");
  EXPECT_EQ(selected.shape, (std::vector<int64_t>{2, 3}));
  const int64_t *py = selected.AsInt64();
  const std::vector<int64_t> expected = {0, 0, 0, 0, 0, 1};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(py[i], expected[i]);
  }
}

TEST(RunNodes, RunNodeNonZeroFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2, 3}, {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f});

  NodeProto node = MakeNode("NonZero", {"x"}, {"y"});
  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("y");
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 3}));
  const int64_t *py = y.AsInt64();
  const std::vector<int64_t> expected = {0, 0, 1, 0, 2, 1};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(py[i], expected[i]);
  }
}

TEST(RunNodes, RunNodeQuantizeLinearFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {4}, {0.0f, 2.0f, 3.0f, 254.0f});
  rt.tensors()["y_scale"] = Tensor::FromFloat("y_scale", {}, {2.0f});

  NodeProto node = MakeNode("QuantizeLinear", {"x", "y_scale"}, {"y"});
  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("y");
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT8));
  EXPECT_EQ(y.shape, std::vector<int64_t>({4}));
  ASSERT_EQ(y.element_count(), 4);
  const std::uint8_t *py = reinterpret_cast<const std::uint8_t *>(y.bytes());
  EXPECT_EQ(py[0], 0);
  EXPECT_EQ(py[1], 1);
  EXPECT_EQ(py[2], 2);
  EXPECT_EQ(py[3], 127);
}

TEST(RunNodes, RunNodeDequantizeLinearFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromUint8("x", {3}, {0, 3, 128});
  rt.tensors()["x_scale"] = Tensor::FromFloat("x_scale", {}, {2.0f});
  rt.tensors()["x_zero_point"] = Tensor::FromUint8("x_zero_point", {}, {128});

  NodeProto node = MakeNode("DequantizeLinear", {"x", "x_scale", "x_zero_point"}, {"y"});
  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("y");
  EXPECT_EQ(y.shape, std::vector<int64_t>({3}));
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -256.0f);
  EXPECT_FLOAT_EQ(py[1], -250.0f);
  EXPECT_FLOAT_EQ(py[2], 0.0f);
}

TEST(RunNodes, RunNodeDynamicQuantizeLinearFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {4}, {0.0f, 2.0f, -2.0f, 4.0f});

  NodeProto node = MakeNode("DynamicQuantizeLinear", {"x"}, {"y", "y_scale", "y_zero_point"});
  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("y");
  const Tensor &y_scale = rt.tensors().at("y_scale");
  const Tensor &y_zp = rt.tensors().at("y_zero_point");
  EXPECT_EQ(y.shape, std::vector<int64_t>({4}));
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT8));
  EXPECT_EQ(y_scale.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  EXPECT_EQ(y_zp.data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT8));
  EXPECT_TRUE(y_scale.shape.empty());
  EXPECT_TRUE(y_zp.shape.empty());
}

TEST(RunNodes, RunNodesOnRepeatedProtoFieldChain) {
  // Builds the small graph:  t = Mul(x, y);  out = Sub(t, z)
  // and runs it through the iterator overload that drives a
  // RepeatedProtoField<NodeProto> directly (mirroring how a caller
  // would feed ``graph.node()``).
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2}, {1.0f, 2.0f});
  rt.tensors()["y"] = Tensor::FromFloat("y", {2}, {3.0f, 4.0f});
  rt.tensors()["z"] = Tensor::FromFloat("z", {2}, {0.5f, 0.25f});

  utils::RepeatedProtoField<NodeProto> nodes;
  *nodes.Add() = MakeNode("Mul", {"x", "y"}, {"t"});
  *nodes.Add() = MakeNode("Sub", {"t", "z"}, {"out"});

  RunNodes(nodes, rt);

  ASSERT_NE(rt.tensors().find("t"), rt.tensors().end());
  ASSERT_NE(rt.tensors().find("out"), rt.tensors().end());
  const float *out = rt.tensors()["out"].AsFloat();
  ASSERT_EQ(rt.tensors()["out"].element_count(), 2);
  EXPECT_FLOAT_EQ(out[0], 1.0f * 3.0f - 0.5f);
  EXPECT_FLOAT_EQ(out[1], 2.0f * 4.0f - 0.25f);
  EXPECT_EQ(rt.tensors()["out"].name, "out");
}

TEST(RunNodes, RunNodesOnIteratorRangeFromVector) {
  // Same graph, but driven through the generic iterator overload so
  // any container whose elements dereference to ``NodeProto`` works.
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["a"] = Tensor::FromFloat("a", {1}, {6.0f});
  rt.tensors()["b"] = Tensor::FromFloat("b", {1}, {2.0f});

  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Div", {"a", "b"}, {"q"})); // q = 3
  nodes.push_back(MakeNode("Neg", {"q"}, {"out"}));    // out = -3

  RunNodes(nodes.begin(), nodes.end(), rt);

  const float *out = rt.tensors()["out"].AsFloat();
  ASSERT_EQ(rt.tensors()["out"].element_count(), 1);
  EXPECT_FLOAT_EQ(out[0], -3.0f);
}

TEST(RunNodes, RunNodeEyeLikeUsesAttributes) {
  // Verify the EyeLike trampoline forwards both the ``k`` and ``dtype``
  // attributes to ``kernel::EyeLike``. Input shape is copied (2x3) and
  // the output uses dtype=INT64 (=7) with ones on the super-diagonal.
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2, 3}, {0, 0, 0, 0, 0, 0});
  NodeProto node = MakeNode("EyeLike", {"x"}, {"y"});
  AttributeProto *attr_k = node.add_attribute();
  attr_k->set_name("k");
  attr_k->set_type(AttributeProto::AttributeType::INT);
  attr_k->set_i(1);
  AttributeProto *attr_dtype = node.add_attribute();
  attr_dtype->set_name("dtype");
  attr_dtype->set_type(AttributeProto::AttributeType::INT);
  attr_dtype->set_i(7); // INT64
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, std::vector<int64_t>({2, 3}));
  EXPECT_EQ(y.data_type, 7);
  const int64_t *got = y.AsInt64();
  EXPECT_EQ(got[0], 0);
  EXPECT_EQ(got[1], 1);
  EXPECT_EQ(got[2], 0);
  EXPECT_EQ(got[3], 0);
  EXPECT_EQ(got[4], 0);
  EXPECT_EQ(got[5], 1);
}

TEST(RunNodes, RunNodeAffineGridUsesAttributes) {
  RuntimeContext rt(KernelContext(DefaultOpset(20)));
  rt.tensors()["theta"] =
      Tensor::FromFloat("theta", {1, 2, 3}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
  rt.tensors()["size"] = Tensor::FromInt64("size", {4}, {1, 1, 2, 2});

  NodeProto node = MakeNode("AffineGrid", {"theta", "size"}, {"grid"});
  AttributeProto *align_corners_attr = node.add_attribute();
  align_corners_attr->set_name("align_corners");
  align_corners_attr->set_type(AttributeProto::AttributeType::INT);
  align_corners_attr->set_i(1);

  RunNode(node, rt);

  const Tensor &grid = rt.tensors()["grid"];
  EXPECT_EQ(grid.shape, (std::vector<int64_t>{1, 2, 2, 2}));
  const float *got = grid.AsFloat();
  EXPECT_FLOAT_EQ(got[0], -1.0f);
  EXPECT_FLOAT_EQ(got[1], -1.0f);
  EXPECT_FLOAT_EQ(got[2], 1.0f);
  EXPECT_FLOAT_EQ(got[3], -1.0f);
  EXPECT_FLOAT_EQ(got[4], -1.0f);
  EXPECT_FLOAT_EQ(got[5], 1.0f);
  EXPECT_FLOAT_EQ(got[6], 1.0f);
  EXPECT_FLOAT_EQ(got[7], 1.0f);
}

TEST(RunNodes, RunNodeImageDecoderFromDispatchTable) {
  // ``ImageDecoder`` was introduced in ONNX opset 20. The reference kernel
  // does not link an image-decoding library and returns the empty
  // ``(0, 0, C)`` matrix mandated by the schema; ``pixel_format`` drives
  // the channel count.
  RuntimeContext rt(KernelContext(DefaultOpset(20)));
  rt.tensors()["encoded_stream"] =
      Tensor::FromUint8("encoded_stream", {4}, std::vector<uint8_t>{0x00, 0x01, 0x02, 0x03});

  NodeProto node = MakeNode("ImageDecoder", {"encoded_stream"}, {"image"});
  AttributeProto *pixel_format_attr = node.add_attribute();
  pixel_format_attr->set_name("pixel_format");
  pixel_format_attr->set_type(AttributeProto::AttributeType::STRING);
  pixel_format_attr->set_s("Grayscale");

  RunNode(node, rt);

  const Tensor &image = rt.tensors()["image"];
  EXPECT_EQ(image.data_type, static_cast<int32_t>(onnx_kernels::DataType::UINT8));
  EXPECT_EQ(image.shape, (std::vector<int64_t>{0, 0, 1}));
  EXPECT_EQ(image.element_count(), 0);
  EXPECT_TRUE(image.data.empty());
}

TEST(RunNodes, RunNodeImageDecoderDefaultsToRGB) {
  // Without ``pixel_format`` attribute, the default ``"RGB"`` produces a
  // 3-channel empty image.
  RuntimeContext rt(KernelContext(DefaultOpset(20)));
  rt.tensors()["encoded_stream"] =
      Tensor::FromUint8("encoded_stream", {2}, std::vector<uint8_t>{0xAA, 0xBB});

  NodeProto node = MakeNode("ImageDecoder", {"encoded_stream"}, {"image"});
  RunNode(node, rt);

  const Tensor &image = rt.tensors()["image"];
  EXPECT_EQ(image.shape, (std::vector<int64_t>{0, 0, 3}));
}

TEST(RunNodes, RunNodeExpandFromDispatchTable) {
  // Expand broadcasts a (3, 1) input to a (3, 4) shape.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3, 1}, {1, 2, 3});
  rt.tensors()["shape"] = Tensor::FromInt64("shape", {2}, {3, 4});
  NodeProto node = MakeNode("Expand", {"x", "shape"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 4}));
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 1.0f);
  EXPECT_FLOAT_EQ(got[3], 1.0f);
  EXPECT_FLOAT_EQ(got[4], 2.0f);
  EXPECT_FLOAT_EQ(got[11], 3.0f);
}

TEST(RunNodes, RunNodeTileFromDispatchTable) {
  // Tile repeats a (2, 2) input by (2, 2) to produce a (4, 4) output.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
  rt.tensors()["repeats"] = Tensor::FromInt64("repeats", {2}, {2, 2});
  NodeProto node = MakeNode("Tile", {"x", "repeats"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{4, 4}));
  const float *got = y.AsFloat();
  const std::vector<float> expected = {0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(got[i], expected[i]);
  }
}

TEST(RunNodes, RunNodeReshapeFromDispatchTable) {
  // Reshape with two inputs (data, shape) and the default ``allowzero`` (0).
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2, 3}, {1, 2, 3, 4, 5, 6});
  rt.tensors()["shape"] = Tensor::FromInt64("shape", {2}, {3, 2});
  NodeProto node = MakeNode("Reshape", {"x", "shape"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 2}));
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 1.0f);
  EXPECT_FLOAT_EQ(got[5], 6.0f);
}

TEST(RunNodes, RunNodeResizeScalesFromDispatchTable) {
  // Resize via the (X, roi="", scales) input form. Upsamples a 1x1x2x2
  // NCHW tensor by [1, 1, 2, 3] using nearest mode and the asymmetric
  // coordinate transformation.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["X"] = Tensor::FromFloat("X", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  rt.tensors()["scales"] = Tensor::FromFloat("scales", {4}, {1.0f, 1.0f, 2.0f, 3.0f});
  NodeProto node = MakeNode("Resize", {"X", "", "scales"}, {"Y"});
  AttributeProto *coord = node.add_attribute();
  coord->set_name("coordinate_transformation_mode");
  coord->set_type(AttributeProto::AttributeType::STRING);
  coord->set_s("asymmetric");

  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("Y");
  EXPECT_EQ(y.shape, (std::vector<int64_t>{1, 1, 4, 6}));
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));

  // Compare against the kernel's direct output to validate dispatch-time
  // wiring of inputs, attributes and outputs.
  onnx_kernels::kernel::Resize::Attributes attrs;
  attrs.coordinate_transformation_mode = "asymmetric";
  const onnx_kernels::kernel::Resize resize_kernel(rt.kernel_ctx());
  const Tensor y_ref = resize_kernel(rt.tensors().at("X"), rt.tensors().at("scales"), attrs);
  ASSERT_EQ(y.element_count(), y_ref.element_count());
  for (int64_t i = 0; i < y.element_count(); ++i) {
    EXPECT_FLOAT_EQ(y.AsFloat()[i], y_ref.AsFloat()[i]);
  }
}

TEST(RunNodes, RunNodeResizeSizesFromDispatchTable) {
  // Resize via the (X, roi="", scales="", sizes) input form.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["X"] = Tensor::FromFloat("X", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  rt.tensors()["sizes"] = Tensor::FromInt64("sizes", {4}, {1, 1, 4, 6});
  NodeProto node = MakeNode("Resize", {"X", "", "", "sizes"}, {"Y"});
  AttributeProto *coord = node.add_attribute();
  coord->set_name("coordinate_transformation_mode");
  coord->set_type(AttributeProto::AttributeType::STRING);
  coord->set_s("asymmetric");

  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("Y");
  EXPECT_EQ(y.shape, (std::vector<int64_t>{1, 1, 4, 6}));
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
}

TEST(RunNodes, RunNodeRegexFullMatchFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(20)));
  rt.tensors()["x"] = Tensor::FromStrings("x", {3}, {"abc", "abcdef", "xyz"});
  NodeProto node = MakeNode("RegexFullMatch", {"x"}, {"y"});
  AttributeProto *pattern = node.add_attribute();
  pattern->set_name("pattern");
  pattern->set_type(AttributeProto::AttributeType::STRING);
  pattern->set_s("abc");
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.data_type, DataType::BOOL);
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3}));
  const uint8_t *got = y.AsBool();
  ASSERT_NE(got, nullptr);
  EXPECT_EQ(got[0], 1u);
  EXPECT_EQ(got[1], 0u);
  EXPECT_EQ(got[2], 0u);
}

TEST(RunNodes, RunNodeGatherFromDispatchTable) {
  // Gather along ``axis=0`` selects whole rows from ``data``.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["data"] = Tensor::FromFloat("data", {3, 2}, {1, 2, 3, 4, 5, 6});
  rt.tensors()["indices"] = Tensor::FromInt64("indices", {2}, {2, 0});
  NodeProto node = MakeNode("Gather", {"data", "indices"}, {"y"});
  AttributeProto *axis = node.add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::AttributeType::INT);
  axis->set_i(0);
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 5.0f);
  EXPECT_FLOAT_EQ(got[1], 6.0f);
  EXPECT_FLOAT_EQ(got[2], 1.0f);
  EXPECT_FLOAT_EQ(got[3], 2.0f);
}

TEST(RunNodes, RunNodeGatherFromDispatchTableDefaultAxis) {
  // Default ``axis`` is 0; verify the dispatch entry handles a missing attribute.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["data"] = Tensor::FromFloat("data", {3}, {10, 20, 30});
  rt.tensors()["indices"] = Tensor::FromInt64("indices", {2}, {1, 2});
  NodeProto node = MakeNode("Gather", {"data", "indices"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2}));
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 20.0f);
  EXPECT_FLOAT_EQ(got[1], 30.0f);
}

TEST(RunNodes, RunNodeGatherNDFromDispatchTable) {
  // GatherND with ``batch_dims=0`` picks scalars from a 2-D ``data`` tensor.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["data"] = Tensor::FromFloat("data", {2, 2}, {1, 2, 3, 4});
  rt.tensors()["indices"] = Tensor::FromInt64("indices", {2, 2}, {0, 0, 1, 1});
  NodeProto node = MakeNode("GatherND", {"data", "indices"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2}));
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 1.0f);
  EXPECT_FLOAT_EQ(got[1], 4.0f);
}

TEST(RunNodes, RunNodeGatherNDWithBatchDimsFromDispatchTable) {
  // GatherND with ``batch_dims=1`` independently indexes each batch row.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["data"] =
      Tensor::FromFloat("data", {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  rt.tensors()["indices"] = Tensor::FromInt64("indices", {2, 1}, {1, 0});
  NodeProto node = MakeNode("GatherND", {"data", "indices"}, {"y"});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("batch_dims");
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(1);
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 3.0f);
  EXPECT_FLOAT_EQ(got[1], 4.0f);
  EXPECT_FLOAT_EQ(got[2], 7.0f);
  EXPECT_FLOAT_EQ(got[3], 8.0f);
}

TEST(RunNodes, RunNodeSqueezeAxesAsInput) {
  // Opset 13+: ``axes`` is provided as the optional second INT64 input.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1, 3, 1, 2}, {1, 2, 3, 4, 5, 6});
  rt.tensors()["axes"] = Tensor::FromInt64("axes", {2}, {0, 2});
  NodeProto node = MakeNode("Squeeze", {"x", "axes"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(y.element_count(), 6);
}

TEST(RunNodes, RunNodeSqueezeAxesAsAttribute) {
  // Opset <13: ``axes`` is an INTS attribute (also accepted by the trampoline
  // for backward compatibility).
  RuntimeContext rt(KernelContext(DefaultOpset(11)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1, 3, 1, 2}, {1, 2, 3, 4, 5, 6});
  NodeProto node = MakeNode("Squeeze", {"x"}, {"y"});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("axes");
  attr->set_type(AttributeProto::AttributeType::INTS);
  attr->add_ints(static_cast<int64_t>(0));
  attr->add_ints(static_cast<int64_t>(2));
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 2}));
}

TEST(RunNodes, RunNodeSqueezeAxesMissingInput) {
  // Opset 13+: ``axes`` is an optional input. When the node is declared with
  // only the ``data`` input (the optional ``axes`` slot is omitted entirely),
  // the kernel squeezes every dimension equal to 1, matching the ONNX spec.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1, 3, 1, 2}, {1, 2, 3, 4, 5, 6});
  NodeProto node = MakeNode("Squeeze", {"x"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(y.element_count(), 6);
}

TEST(RunNodes, RunNodeSqueezeAxesEmptyName) {
  // Opset 13+: a missing optional input is conventionally encoded as an empty
  // input name. The kernel must treat this the same as omitting the slot
  // entirely and squeeze all unit dimensions.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1, 3, 1, 2}, {1, 2, 3, 4, 5, 6});
  NodeProto node = MakeNode("Squeeze", {"x", ""}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(y.element_count(), 6);
}

TEST(RunNodes, RunNodeUnsqueezeAxesAsInput) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3, 2}, {1, 2, 3, 4, 5, 6});
  rt.tensors()["axes"] = Tensor::FromInt64("axes", {2}, {0, 2});
  NodeProto node = MakeNode("Unsqueeze", {"x", "axes"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{1, 3, 1, 2}));
}

TEST(RunNodes, RunNodeUnsqueezeScalarAxesInput) {
  // Some upstream function bodies (e.g. ai.onnx::AffineGrid) feed the
  // Unsqueeze ``axes`` input with a 0-D INT64 scalar instead of the 1-D
  // tensor required by the schema. For compatibility with those models
  // (which the upstream reference evaluator also accepts) the trampoline
  // tolerates a rank-0 axes tensor.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3, 2}, {1, 2, 3, 4, 5, 6});
  rt.tensors()["axes"] = Tensor::FromInt64("axes", {}, {-1});
  NodeProto node = MakeNode("Unsqueeze", {"x", "axes"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 2, 1}));
}

TEST(RunNodes, RunNodeShapeNoAttributes) {
  // Default attributes: returns the full shape as an INT64 1-D tensor.
  RuntimeContext rt(KernelContext(DefaultOpset(15)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2, 3, 4}, std::vector<float>(24, 0.0f));
  NodeProto node = MakeNode("Shape", {"x"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3}));
  EXPECT_EQ(y.data_type, 7); // INT64
  const int64_t *got = y.AsInt64();
  EXPECT_EQ(got[0], 2);
  EXPECT_EQ(got[1], 3);
  EXPECT_EQ(got[2], 4);
}

TEST(RunNodes, RunNodeShapeUsesStartAndEndAttributes) {
  // Verify the Shape trampoline forwards the ``start`` and ``end``
  // attributes to ``kernel::Shape``: ``shape[1:-1]`` of a 4-D input.
  RuntimeContext rt(KernelContext(DefaultOpset(15)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2, 3, 4, 5}, std::vector<float>(120, 0.0f));
  NodeProto node = MakeNode("Shape", {"x"}, {"y"});
  AttributeProto *attr_start = node.add_attribute();
  attr_start->set_name("start");
  attr_start->set_type(AttributeProto::AttributeType::INT);
  attr_start->set_i(1);
  AttributeProto *attr_end = node.add_attribute();
  attr_end->set_name("end");
  attr_end->set_type(AttributeProto::AttributeType::INT);
  attr_end->set_i(-1);
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2}));
  EXPECT_EQ(y.data_type, 7); // INT64
  const int64_t *got = y.AsInt64();
  EXPECT_EQ(got[0], 3);
  EXPECT_EQ(got[1], 4);
}

TEST(RunNodes, RunNodeEinsumUsesEquationAttribute) {
  // Verify the Einsum trampoline forwards the variadic inputs and the
  // ``equation`` STRING attribute to ``kernel::Einsum``. The equation
  // ``"ij,jk->ik"`` performs a 2x3 by 3x2 matrix product.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["a"] = Tensor::FromFloat("a", {2, 3}, {1, 2, 3, 4, 5, 6});
  rt.tensors()["b"] = Tensor::FromFloat("b", {3, 2}, {7, 8, 9, 10, 11, 12});
  NodeProto node = MakeNode("Einsum", {"a", "b"}, {"y"});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("equation");
  attr->set_type(AttributeProto::AttributeType::STRING);
  attr->set_s("ij,jk->ik");
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 58.0f);
  EXPECT_FLOAT_EQ(got[1], 64.0f);
  EXPECT_FLOAT_EQ(got[2], 139.0f);
  EXPECT_FLOAT_EQ(got[3], 154.0f);
}

TEST(RunNodes, RunNodeDFTOpset17UsesAxisAttribute) {
  // v17 DFT: ``axis`` / ``inverse`` / ``onesided`` are INT attributes.
  // Inputs are ``(input, dft_length?)``.
  RuntimeContext rt(KernelContext(DefaultOpset(17)));
  Tensor x = Tensor::FromFloat("x", {1, 4, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
  rt.tensors()["x"] = x;

  NodeProto node = MakeNode("DFT", {"x"}, {"y"});
  AttributeProto *axis = node.add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::AttributeType::INT);
  axis->set_i(1);

  RunNode(node, rt);

  onnx_kernels::kernel::DFT ref(rt.kernel_ctx());
  Tensor expected = ref(x, /*dft_length=*/nullptr, /*axis=*/1, /*onesided=*/false,
                        /*inverse=*/false);
  const Tensor &y = rt.tensors()["y"];
  ASSERT_EQ(y.shape, expected.shape);
  ASSERT_EQ(y.data.size(), expected.data.size());
  EXPECT_EQ(std::memcmp(y.data.data(), expected.data.data(), expected.data.size()), 0);
}

TEST(RunNodes, RunNodeDFTOpset20UsesAxisInput) {
  // v20 DFT: ``axis`` becomes the third (optional) input; only
  // ``inverse`` / ``onesided`` remain attributes.
  RuntimeContext rt(KernelContext(DefaultOpset(20)));
  Tensor x = Tensor::FromFloat("x", {1, 4, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor axis = Tensor::FromInt64("axis", {}, {1});
  rt.tensors()["x"] = x;
  rt.tensors()["axis"] = axis;

  // ``dft_length`` is omitted by passing an empty input name.
  NodeProto node = MakeNode("DFT", {"x", "", "axis"}, {"y"});

  RunNode(node, rt);

  onnx_kernels::kernel::DFT ref(rt.kernel_ctx());
  Tensor expected = ref(x, /*dft_length=*/nullptr, /*axis=*/1, /*onesided=*/false,
                        /*inverse=*/false);
  const Tensor &y = rt.tensors()["y"];
  ASSERT_EQ(y.shape, expected.shape);
  ASSERT_EQ(y.data.size(), expected.data.size());
  EXPECT_EQ(std::memcmp(y.data.data(), expected.data.data(), expected.data.size()), 0);
}

TEST(RunNodes, RunNodeDFTOpset17InverseOnesidedAttributes) {
  // v17 DFT with ``inverse`` and ``onesided`` attributes set.
  RuntimeContext rt(KernelContext(DefaultOpset(17)));
  Tensor x = Tensor::FromFloat("x", {1, 4, 2}, {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f});
  rt.tensors()["x"] = x;

  NodeProto node = MakeNode("DFT", {"x"}, {"y"});
  AttributeProto *axis = node.add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::AttributeType::INT);
  axis->set_i(1);
  AttributeProto *inverse = node.add_attribute();
  inverse->set_name("inverse");
  inverse->set_type(AttributeProto::AttributeType::INT);
  inverse->set_i(1);

  RunNode(node, rt);

  onnx_kernels::kernel::DFT ref(rt.kernel_ctx());
  Tensor expected = ref(x, /*dft_length=*/nullptr, /*axis=*/1, /*onesided=*/false,
                        /*inverse=*/true);
  const Tensor &y = rt.tensors()["y"];
  ASSERT_EQ(y.shape, expected.shape);
  ASSERT_EQ(y.data.size(), expected.data.size());
  EXPECT_EQ(std::memcmp(y.data.data(), expected.data.data(), expected.data.size()), 0);
}

TEST(RunNodes, RunNodeAdagradFromDispatchTable) {
  // Dispatches a single-tensor Adagrad node and checks the outputs
  // against ``kernel::Adagrad`` invoked directly.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["R"] = Tensor::FromFloat("R", {}, {0.1f});
  rt.tensors()["T"] = Tensor::FromInt64("T", {}, {0});
  rt.tensors()["X"] = Tensor::FromFloat("X", {2}, {1.0f, 2.0f});
  rt.tensors()["G"] = Tensor::FromFloat("G", {2}, {-0.5f, 0.25f});
  rt.tensors()["H"] = Tensor::FromFloat("H", {2}, {0.1f, 0.1f});

  NodeProto node = MakeNode("Adagrad", {"R", "T", "X", "G", "H"}, {"X_new", "H_new"},
                            "ai.onnx.preview.training");
  AttributeProto *eps = node.add_attribute();
  eps->set_name("epsilon");
  eps->set_type(AttributeProto::AttributeType::FLOAT);
  eps->set_f(1e-5f);
  AttributeProto *nc = node.add_attribute();
  nc->set_name("norm_coefficient");
  nc->set_type(AttributeProto::AttributeType::FLOAT);
  nc->set_f(0.001f);

  RunNode(node, rt);

  onnx_kernels::kernel::Adagrad ref(rt.kernel_ctx());
  std::vector<Tensor> expected = ref(rt.tensors()["R"], rt.tensors()["T"], {rt.tensors()["X"]},
                                     {rt.tensors()["G"]}, {rt.tensors()["H"]}, 1e-5f, 0.0f, 0.001f);

  const Tensor &x_new = rt.tensors()["X_new"];
  const Tensor &h_new = rt.tensors()["H_new"];
  ASSERT_EQ(x_new.shape, (std::vector<int64_t>{2}));
  ASSERT_EQ(h_new.shape, (std::vector<int64_t>{2}));
  EXPECT_FLOAT_EQ(x_new.AsFloat()[0], expected[0].AsFloat()[0]);
  EXPECT_FLOAT_EQ(x_new.AsFloat()[1], expected[0].AsFloat()[1]);
  EXPECT_FLOAT_EQ(h_new.AsFloat()[0], expected[1].AsFloat()[0]);
  EXPECT_FLOAT_EQ(h_new.AsFloat()[1], expected[1].AsFloat()[1]);
}

TEST(RunNodes, RunNodeMomentumNesterovFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["R"] = Tensor::FromFloat("R", {}, {0.1f});
  rt.tensors()["T"] = Tensor::FromInt64("T", {}, {0});
  rt.tensors()["X"] = Tensor::FromFloat("X", {2}, {1.2f, 2.8f});
  rt.tensors()["G"] = Tensor::FromFloat("G", {2}, {-0.94f, -2.5f});
  rt.tensors()["V"] = Tensor::FromFloat("V", {2}, {1.7f, 3.6f});

  NodeProto node = MakeNode("Momentum", {"R", "T", "X", "G", "V"}, {"X_new", "V_new"},
                            "ai.onnx.preview.training");
  AttributeProto *a = node.add_attribute();
  a->set_name("alpha");
  a->set_type(AttributeProto::AttributeType::FLOAT);
  a->set_f(0.95f);
  AttributeProto *b = node.add_attribute();
  b->set_name("beta");
  b->set_type(AttributeProto::AttributeType::FLOAT);
  b->set_f(1.0f);
  AttributeProto *nc = node.add_attribute();
  nc->set_name("norm_coefficient");
  nc->set_type(AttributeProto::AttributeType::FLOAT);
  nc->set_f(0.01f);
  AttributeProto *mode = node.add_attribute();
  mode->set_name("mode");
  mode->set_type(AttributeProto::AttributeType::STRING);
  mode->set_s("nesterov");

  RunNode(node, rt);

  onnx_kernels::kernel::Momentum ref(rt.kernel_ctx());
  std::vector<Tensor> expected =
      ref(rt.tensors()["R"], rt.tensors()["T"], {rt.tensors()["X"]}, {rt.tensors()["G"]},
          {rt.tensors()["V"]}, 0.95f, 1.0f, 0.01f, onnx_kernels::kernel::Momentum::Mode::kNesterov);

  const Tensor &x_new = rt.tensors()["X_new"];
  const Tensor &v_new = rt.tensors()["V_new"];
  EXPECT_FLOAT_EQ(x_new.AsFloat()[0], expected[0].AsFloat()[0]);
  EXPECT_FLOAT_EQ(x_new.AsFloat()[1], expected[0].AsFloat()[1]);
  EXPECT_FLOAT_EQ(v_new.AsFloat()[0], expected[1].AsFloat()[0]);
  EXPECT_FLOAT_EQ(v_new.AsFloat()[1], expected[1].AsFloat()[1]);
}

TEST(RunNodes, RunNodeAdamMultipleFromDispatchTable) {
  // Dispatches a two-tensor Adam node (N=2) and checks the 3*N=6
  // outputs against the direct kernel invocation.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["R"] = Tensor::FromFloat("R", {}, {0.05f});
  rt.tensors()["T"] = Tensor::FromInt64("T", {}, {5});
  rt.tensors()["X1"] = Tensor::FromFloat("X1", {2}, {0.5f, -0.5f});
  rt.tensors()["X2"] = Tensor::FromFloat("X2", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  rt.tensors()["G1"] = Tensor::FromFloat("G1", {2}, {0.1f, -0.2f});
  rt.tensors()["G2"] = Tensor::FromFloat("G2", {2, 2}, {-0.5f, 0.25f, 0.75f, -1.0f});
  rt.tensors()["V1"] = Tensor::FromFloat("V1", {2}, {0.01f, 0.02f});
  rt.tensors()["V2"] = Tensor::FromFloat("V2", {2, 2}, {0.05f, 0.05f, -0.05f, 0.0f});
  rt.tensors()["H1"] = Tensor::FromFloat("H1", {2}, {0.001f, 0.002f});
  rt.tensors()["H2"] = Tensor::FromFloat("H2", {2, 2}, {0.01f, 0.02f, 0.03f, 0.04f});

  NodeProto node = MakeNode("Adam", {"R", "T", "X1", "X2", "G1", "G2", "V1", "V2", "H1", "H2"},
                            {"X1n", "X2n", "V1n", "V2n", "H1n", "H2n"}, "ai.onnx.preview.training");
  for (const auto &kv :
       std::vector<std::pair<std::string, float>>{{"alpha", 0.9f},
                                                  {"beta", 0.999f},
                                                  {"epsilon", 1e-6f},
                                                  {"norm_coefficient", 0.0f},
                                                  {"norm_coefficient_post", 0.0f}}) {
    AttributeProto *a = node.add_attribute();
    a->set_name(kv.first.c_str());
    a->set_type(AttributeProto::AttributeType::FLOAT);
    a->set_f(kv.second);
  }

  RunNode(node, rt);

  onnx_kernels::kernel::Adam ref(rt.kernel_ctx());
  std::vector<Tensor> expected =
      ref(rt.tensors()["R"], rt.tensors()["T"], {rt.tensors()["X1"], rt.tensors()["X2"]},
          {rt.tensors()["G1"], rt.tensors()["G2"]}, {rt.tensors()["V1"], rt.tensors()["V2"]},
          {rt.tensors()["H1"], rt.tensors()["H2"]}, 0.9f, 0.999f, 1e-6f, 0.0f, 0.0f);

  const std::vector<std::string> out_names = {"X1n", "X2n", "V1n", "V2n", "H1n", "H2n"};
  for (size_t i = 0; i < out_names.size(); ++i) {
    const Tensor &got = rt.tensors()[out_names[i]];
    ASSERT_EQ(got.shape, expected[i].shape);
    ASSERT_EQ(got.element_count(), expected[i].element_count());
    for (int64_t j = 0; j < got.element_count(); ++j) {
      EXPECT_FLOAT_EQ(got.AsFloat()[j], expected[i].AsFloat()[j])
          << " at out=" << out_names[i] << " idx=" << j;
    }
  }
}

TEST(RunNodes, RunNodeMomentumRejectsBadInputCount) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["R"] = Tensor::FromFloat("R", {}, {0.1f});
  rt.tensors()["T"] = Tensor::FromInt64("T", {}, {0});
  rt.tensors()["X"] = Tensor::FromFloat("X", {1}, {1.0f});
  rt.tensors()["G"] = Tensor::FromFloat("G", {1}, {1.0f});
  // 2 + 2 = 4 inputs, not 2 + 3*N: should throw.
  NodeProto node =
      MakeNode("Momentum", {"R", "T", "X", "G"}, {"X_new"}, "ai.onnx.preview.training");
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

TEST(RunNodes, RunNodeUnsupportedOpTypeThrows) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1}, {1.0f});
  NodeProto node = MakeNode("ThisOpDoesNotExist", {"x"}, {"y"});
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

TEST(RunNodes, RunNodeMissingInputThrows) {
  RuntimeContext rt(KernelContext(DefaultOpset(18))); // empty: "x" is not present
  NodeProto node = MakeNode("Abs", {"x"}, {"y"});
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

TEST(RunNodes, RunNodeWrongInputCountThrows) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1}, {1.0f});
  // Add expects two inputs but we only declare one.
  NodeProto node = MakeNode("Add", {"x"}, {"y"});
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

TEST(RunNodes, RuntimeContextSetGetHas) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  EXPECT_FALSE(rt.Has("x"));
  rt.Set("x", Tensor::FromFloat("x", {1}, {7.0f}));
  EXPECT_TRUE(rt.Has("x"));
  EXPECT_FLOAT_EQ(rt.Get("x").AsFloat()[0], 7.0f);
  EXPECT_THROW(rt.Get("missing"), std::out_of_range);
}

TEST(RunNodes, RuntimeContextRemove) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {1}, {7.0f}));
  EXPECT_TRUE(rt.Remove("x"));
  EXPECT_FALSE(rt.Has("x"));
  EXPECT_THROW(rt.Get("x"), std::out_of_range);
  EXPECT_FALSE(rt.Remove("x"));
}

TEST(RunNodes, RuntimeContextEventLogSetReplaceRemove) {
  using onnx_kernels::TensorEventAction;
  using onnx_kernels::TensorEventKind;
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.set_events_enabled(true);
  EXPECT_TRUE(rt.events().empty());

  // Set -> add event with values populated (element_count <= 8). Default
  // kind for Set is "input".
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, -2.0f, 3.5f}));
  ASSERT_EQ(rt.events().size(), 1u);
  const auto &add_ev = rt.events()[0];
  EXPECT_EQ(add_ev.action, TensorEventAction::kAdd);
  EXPECT_EQ(add_ev.kind, TensorEventKind::kInput);
  EXPECT_EQ(add_ev.name, "x");
  EXPECT_EQ(add_ev.data_type, static_cast<int32_t>(DataType::FLOAT));
  EXPECT_EQ(add_ev.shape, (std::vector<int64_t>{3}));
  EXPECT_EQ(add_ev.value_count, 3);
  EXPECT_FLOAT_EQ(static_cast<float>(add_ev.values[0]), 1.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(add_ev.values[1]), -2.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(add_ev.values[2]), 3.5f);
  // Unused slots of the fixed-size buffer stay zero-initialised.
  EXPECT_DOUBLE_EQ(add_ev.values[3], 0.0);
  EXPECT_DOUBLE_EQ(add_ev.values[7], 0.0);
  EXPECT_GT(add_ev.timestamp_ns, 0);

  // Put on the same name -> replace event. Default kind for Put is
  // "intermediate".
  rt.Put("x", Tensor::FromInt32("x", {2}, {7, 8}));
  ASSERT_EQ(rt.events().size(), 2u);
  EXPECT_EQ(rt.events()[1].action, TensorEventAction::kReplace);
  EXPECT_EQ(rt.events()[1].kind, TensorEventKind::kIntermediate);
  EXPECT_EQ(rt.events()[1].data_type, static_cast<int32_t>(DataType::INT32));
  EXPECT_EQ(rt.events()[1].value_count, 2);
  EXPECT_DOUBLE_EQ(rt.events()[1].values[0], 7.0);
  EXPECT_DOUBLE_EQ(rt.events()[1].values[1], 8.0);

  // Remove -> remove event with empty shape and value_count = 0.
  EXPECT_TRUE(rt.Remove("x"));
  ASSERT_EQ(rt.events().size(), 3u);
  EXPECT_EQ(rt.events()[2].action, TensorEventAction::kRemove);
  EXPECT_EQ(rt.events()[2].name, "x");
  EXPECT_TRUE(rt.events()[2].shape.empty());
  EXPECT_EQ(rt.events()[2].value_count, 0);

  // No-op remove -> no extra event.
  EXPECT_FALSE(rt.Remove("x"));
  EXPECT_EQ(rt.events().size(), 3u);

  // Large tensor (> kTensorEventValueLimit) -> data_type = -1, empty shape,
  // values truncated to the first kTensorEventValueLimit entries.
  rt.Put("big", Tensor::FromInt32("big", {9}, {0, 1, 2, 3, 4, 5, 6, 7, 8}));
  ASSERT_EQ(rt.events().size(), 4u);
  EXPECT_EQ(rt.events()[3].action, TensorEventAction::kAdd);
  EXPECT_EQ(rt.events()[3].data_type, -1);
  EXPECT_TRUE(rt.events()[3].shape.empty());
  EXPECT_EQ(rt.events()[3].value_count, 8);
  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_DOUBLE_EQ(rt.events()[3].values[i], static_cast<double>(i));
  }

  // String tensor values land in string_values (numeric values buffer stays
  // zero-initialised).
  rt.Put("s", Tensor::FromStrings("s", {2}, {"a", "bc"}));
  ASSERT_EQ(rt.events().size(), 5u);
  EXPECT_EQ(rt.events()[4].data_type, static_cast<int32_t>(DataType::STRING));
  EXPECT_EQ(rt.events()[4].value_count, 2);
  EXPECT_EQ(rt.events()[4].string_values[0], "a");
  EXPECT_EQ(rt.events()[4].string_values[1], "bc");
  EXPECT_DOUBLE_EQ(rt.events()[4].values[0], 0.0);

  rt.ClearEvents();
  EXPECT_TRUE(rt.events().empty());
}

TEST(RunNodes, RuntimeContextEventLogCapturesRunGraphMutations) {
  // Smoke test: running a small chain of nodes through the dispatcher
  // populates the event log via SetOutput / Put on every produced tensor
  // and also appends one ``kRunNode`` event per dispatched node.
  using onnx_kernels::TensorEventAction;
  using onnx_kernels::TensorEventKind;
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.set_events_enabled(true);
  rt.Set("x", Tensor::FromFloat("x", {2}, {-1.0f, 2.0f}));
  rt.Set("z", Tensor::FromFloat("z", {2}, {10.0f, 20.0f}));
  rt.ClearEvents();

  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Abs", {"x"}, {"t"}));
  nodes.push_back(MakeNode("Add", {"t", "z"}, {"y"}));
  RunNodes(nodes.begin(), nodes.end(), rt);

  // Each node produces one tensor ``add`` event tagged as an
  // intermediate plus one ``run_node`` event summarising the dispatch.
  ASSERT_EQ(rt.events().size(), 4u);
  EXPECT_EQ(rt.events()[0].name, "t");
  EXPECT_EQ(rt.events()[0].action, TensorEventAction::kAdd);
  EXPECT_EQ(rt.events()[0].kind, TensorEventKind::kIntermediate);
  EXPECT_EQ(rt.events()[1].action, TensorEventAction::kRunNode);
  EXPECT_EQ(rt.events()[1].op_domain, "ai.onnx");
  EXPECT_EQ(rt.events()[1].op_type, "Abs");
  EXPECT_EQ(rt.events()[1].inputs, (std::vector<std::string>{"x"}));
  EXPECT_GE(rt.events()[1].duration_ns, 0);
  EXPECT_EQ(rt.events()[2].name, "y");
  EXPECT_EQ(rt.events()[2].action, TensorEventAction::kAdd);
  EXPECT_EQ(rt.events()[2].kind, TensorEventKind::kIntermediate);
  EXPECT_EQ(rt.events()[3].action, TensorEventAction::kRunNode);
  EXPECT_EQ(rt.events()[3].op_domain, "ai.onnx");
  EXPECT_EQ(rt.events()[3].op_type, "Add");
  EXPECT_EQ(rt.events()[3].inputs, (std::vector<std::string>{"t", "z"}));
  EXPECT_GE(rt.events()[3].duration_ns, 0);
}

// ---------------------------------------------------------------------------
// TensorFromProto tests
// ---------------------------------------------------------------------------

TEST(TensorFromProto, FloatTypedField) {
  TensorProto tp;
  tp.set_name("w");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::FLOAT);
  tp.add_float_data(1.0f);
  tp.add_float_data(2.0f);

  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.name, "w");
  EXPECT_EQ(t.shape, (std::vector<int64_t>{2}));
  EXPECT_EQ(t.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  ASSERT_EQ(t.element_count(), 2);
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(t.AsFloat()[1], 2.0f);
}

TEST(TensorFromProto, FloatRawData) {
  TensorProto tp;
  tp.set_name("r");
  tp.ref_dims().push_back(3);
  tp.set_data_type(TensorProto::DataType::FLOAT);
  const float vals[] = {3.0f, 4.0f, 5.0f};
  const auto *raw_ptr = reinterpret_cast<const uint8_t *>(vals);
  tp.ref_raw_data() = std::vector<uint8_t>(raw_ptr, raw_ptr + sizeof(vals));

  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.name, "r");
  ASSERT_EQ(t.element_count(), 3);
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 3.0f);
  EXPECT_FLOAT_EQ(t.AsFloat()[1], 4.0f);
  EXPECT_FLOAT_EQ(t.AsFloat()[2], 5.0f);
}

TEST(TensorFromProto, Int64TypedField) {
  TensorProto tp;
  tp.set_name("i64");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::INT64);
  tp.add_int64_data(100);
  tp.add_int64_data(-200);

  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
  ASSERT_EQ(t.element_count(), 2);
  EXPECT_EQ(t.AsInt64()[0], 100);
  EXPECT_EQ(t.AsInt64()[1], -200);
}

TEST(TensorFromProto, Int32TypedField) {
  TensorProto tp;
  tp.set_name("i32");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::INT32);
  tp.add_int32_data(7);
  tp.add_int32_data(-3);

  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.data_type, static_cast<int32_t>(TensorProto::DataType::INT32));
  ASSERT_EQ(t.element_count(), 2);
  EXPECT_EQ(t.AsInt32()[0], 7);
  EXPECT_EQ(t.AsInt32()[1], -3);
}

TEST(TensorFromProto, ScalarNoShape) {
  // Scalar TensorProto has no dims entry.
  TensorProto tp;
  tp.set_name("s");
  tp.set_data_type(TensorProto::DataType::FLOAT);
  tp.add_float_data(42.0f);

  Tensor t = TensorFromProto(tp);
  EXPECT_TRUE(t.shape.empty());
  EXPECT_EQ(t.element_count(), 1);
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 42.0f);
}

// ---------------------------------------------------------------------------
// RunGraph tests
// ---------------------------------------------------------------------------

TEST(RunGraph, InitializersLoadedAndNodesRun) {
  // Graph:  out = Add(x_input, w_init)
  //   x_input is provided by the caller; w_init comes from the initializer.
  TensorProto init_tp;
  init_tp.set_name("w_init");
  init_tp.ref_dims().push_back(2);
  init_tp.set_data_type(TensorProto::DataType::FLOAT);
  init_tp.add_float_data(10.0f);
  init_tp.add_float_data(20.0f);

  GraphProto graph;
  *graph.add_initializer() = init_tp;
  NodeProto *node = graph.add_node();
  node->set_op_type("Add");
  node->add_input("x_input");
  node->add_input("w_init");
  node->add_output("out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x_input", Tensor::FromFloat("x_input", {2}, {1.0f, 2.0f}));

  RunGraph(graph, rt);

  ASSERT_TRUE(rt.Has("out"));
  const float *res = rt.Get("out").AsFloat();
  EXPECT_FLOAT_EQ(res[0], 11.0f);
  EXPECT_FLOAT_EQ(res[1], 22.0f);
}

TEST(RunGraph, CallerInputOverridesInitializer) {
  // When the caller has already seeded a name that is also an initializer,
  // the caller's value must win.
  TensorProto init_tp;
  init_tp.set_name("w");
  init_tp.ref_dims().push_back(1);
  init_tp.set_data_type(TensorProto::DataType::FLOAT);
  init_tp.add_float_data(999.0f); // should be ignored

  GraphProto graph;
  *graph.add_initializer() = init_tp;
  NodeProto *node = graph.add_node();
  node->set_op_type("Abs");
  node->add_input("w");
  node->add_output("out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  // Override the initializer with the caller's value.
  rt.Set("w", Tensor::FromFloat("w", {1}, {-5.0f}));

  RunGraph(graph, rt);

  ASSERT_TRUE(rt.Has("out"));
  EXPECT_FLOAT_EQ(rt.Get("out").AsFloat()[0], 5.0f);
}

// ---------------------------------------------------------------------------
// RunFunction tests
// ---------------------------------------------------------------------------

TEST(RunFunction, NodesRun) {
  FunctionProto func;
  func.set_name("f");
  func.add_input("a");
  func.add_input("b");
  func.add_output("result");
  NodeProto *node = func.add_node();
  node->set_op_type("Mul");
  node->add_input("a");
  node->add_input("b");
  node->add_output("result");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("a", Tensor::FromFloat("a", {2}, {3.0f, 4.0f}));
  rt.Set("b", Tensor::FromFloat("b", {2}, {2.0f, 5.0f}));

  RunFunction(func, rt);

  ASSERT_TRUE(rt.Has("result"));
  const float *res = rt.Get("result").AsFloat();
  EXPECT_FLOAT_EQ(res[0], 6.0f);
  EXPECT_FLOAT_EQ(res[1], 20.0f);
}

// ---------------------------------------------------------------------------
// RunModel tests
// ---------------------------------------------------------------------------

TEST(RunModel, GraphRun) {
  // Build a minimal ModelProto with a single Add node.
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("Add");
  node->add_input("p");
  node->add_input("q");
  node->add_output("r");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("p", Tensor::FromFloat("p", {3}, {1.0f, 2.0f, 3.0f}));
  rt.Set("q", Tensor::FromFloat("q", {3}, {4.0f, 5.0f, 6.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("r"));
  const float *res = rt.Get("r").AsFloat();
  EXPECT_FLOAT_EQ(res[0], 5.0f);
  EXPECT_FLOAT_EQ(res[1], 7.0f);
  EXPECT_FLOAT_EQ(res[2], 9.0f);
}

TEST(RunModel, NoGraphThrows) {
  ModelProto model;
  model.set_ir_version(10);
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  EXPECT_THROW(RunModel(model, rt), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Model-local function dispatch tests
// ---------------------------------------------------------------------------

TEST(RunModel, NodeDispatchedToModelLocalFunction) {
  // Define a model-local function "MyAddMul" in domain "custom":
  //   inputs:  a, b, c
  //   output:  out  = (a + b) * c
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("MyAddMul");
  func->set_domain("custom");
  func->add_input("a");
  func->add_input("b");
  func->add_input("c");
  func->add_output("out");
  {
    NodeProto *n = func->add_node();
    n->set_op_type("Add");
    n->add_input("a");
    n->add_input("b");
    n->add_output("tmp");
  }
  {
    NodeProto *n = func->add_node();
    n->set_op_type("Mul");
    n->add_input("tmp");
    n->add_input("c");
    n->add_output("out");
  }

  // Main graph: y = MyAddMul(x, w, k) where (x, w, k) are graph inputs.
  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("MyAddMul");
  node->set_domain("custom");
  node->add_input("x");
  node->add_input("w");
  node->add_input("k");
  node->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {2}, {1.0f, 2.0f}));
  rt.Set("w", Tensor::FromFloat("w", {2}, {3.0f, 4.0f}));
  rt.Set("k", Tensor::FromFloat("k", {2}, {10.0f, 100.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  const float *res = rt.Get("y").AsFloat();
  ASSERT_EQ(rt.Get("y").element_count(), 2);
  EXPECT_FLOAT_EQ(res[0], (1.0f + 3.0f) * 10.0f);
  EXPECT_FLOAT_EQ(res[1], (2.0f + 4.0f) * 100.0f);
  // The function's internal "tmp" value must NOT leak into the caller's
  // tensor map: the child context is isolated.
  EXPECT_FALSE(rt.Has("tmp"));
}

TEST(RunModel, ModelLocalFunctionCanCallAnotherFunction) {
  // Define two model-local functions in "custom":
  //   AddOne(x) -> Add(x, one_init_passed_as_input)  -- here we use x+x
  //   Quad(x)   -> AddOne(AddOne(x))                 -- so result = x*4? No: just chained calls
  // To keep it simple and only exercise nesting, use:
  //   Twice(x)  -> Add(x, x)
  //   Quad(x)   -> Twice(Twice(x))   => 4*x
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *twice = model.add_functions();
  twice->set_name("Twice");
  twice->set_domain("custom");
  twice->add_input("x");
  twice->add_output("y");
  {
    NodeProto *n = twice->add_node();
    n->set_op_type("Add");
    n->add_input("x");
    n->add_input("x");
    n->add_output("y");
  }

  FunctionProto *quad = model.add_functions();
  quad->set_name("Quad");
  quad->set_domain("custom");
  quad->add_input("x");
  quad->add_output("y");
  {
    NodeProto *n = quad->add_node();
    n->set_op_type("Twice");
    n->set_domain("custom");
    n->add_input("x");
    n->add_output("t");
  }
  {
    NodeProto *n = quad->add_node();
    n->set_op_type("Twice");
    n->set_domain("custom");
    n->add_input("t");
    n->add_output("y");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("Quad");
  node->set_domain("custom");
  node->add_input("x");
  node->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.5f, -3.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  const float *res = rt.Get("y").AsFloat();
  ASSERT_EQ(rt.Get("y").element_count(), 3);
  EXPECT_FLOAT_EQ(res[0], 4.0f);
  EXPECT_FLOAT_EQ(res[1], 10.0f);
  EXPECT_FLOAT_EQ(res[2], -12.0f);
}

TEST(RunModel, ModelLocalFunctionCallsAnotherFunctionAcrossDomains) {
  // Define two model-local functions in different domains; the function
  // in domain "outer" calls into the function in domain "inner". This
  // exercises that the function registry propagated to the child
  // RuntimeContext is keyed by (domain, name, overload) and that
  // cross-domain function-to-function dispatch works.
  //
  //   inner::Square(x) -> Mul(x, x)
  //   outer::SquareThenAdd(x, y) -> Add(inner::Square(x), y)
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *square = model.add_functions();
  square->set_name("Square");
  square->set_domain("inner");
  square->add_input("x");
  square->add_output("y");
  {
    NodeProto *n = square->add_node();
    n->set_op_type("Mul");
    n->add_input("x");
    n->add_input("x");
    n->add_output("y");
  }

  FunctionProto *sqadd = model.add_functions();
  sqadd->set_name("SquareThenAdd");
  sqadd->set_domain("outer");
  sqadd->add_input("a");
  sqadd->add_input("b");
  sqadd->add_output("z");
  {
    NodeProto *n = sqadd->add_node();
    n->set_op_type("Square");
    n->set_domain("inner");
    n->add_input("a");
    n->add_output("a2");
  }
  {
    NodeProto *n = sqadd->add_node();
    n->set_op_type("Add");
    n->add_input("a2");
    n->add_input("b");
    n->add_output("z");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("SquareThenAdd");
  node->set_domain("outer");
  node->add_input("x");
  node->add_input("y");
  node->add_output("z");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));
  rt.Set("y", Tensor::FromFloat("y", {3}, {10.0f, 20.0f, 30.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("z"));
  const float *res = rt.Get("z").AsFloat();
  ASSERT_EQ(rt.Get("z").element_count(), 3);
  EXPECT_FLOAT_EQ(res[0], 1.0f * 1.0f + 10.0f);
  EXPECT_FLOAT_EQ(res[1], 2.0f * 2.0f + 20.0f);
  EXPECT_FLOAT_EQ(res[2], 3.0f * 3.0f + 30.0f);
  // The intermediate name produced inside the SquareThenAdd function
  // body must not leak into the caller's tensor map.
  EXPECT_FALSE(rt.Has("a2"));
}

TEST(RunModel, ModelLocalFunctionThreeLevelNestedCalls) {
  // Demonstrates that the function registry is propagated through
  // arbitrary nesting depth: the top-level graph calls Outer, Outer
  // calls Middle, and Middle calls Inner. Each level is a separate
  // FunctionProto in the model.
  //
  //   Inner(x)  -> Add(x, x)        => 2*x
  //   Middle(x) -> Inner(Inner(x))  => 4*x
  //   Outer(x)  -> Middle(Inner(x)) => 8*x
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *inner = model.add_functions();
  inner->set_name("Inner");
  inner->set_domain("custom");
  inner->add_input("x");
  inner->add_output("y");
  {
    NodeProto *n = inner->add_node();
    n->set_op_type("Add");
    n->add_input("x");
    n->add_input("x");
    n->add_output("y");
  }

  FunctionProto *middle = model.add_functions();
  middle->set_name("Middle");
  middle->set_domain("custom");
  middle->add_input("x");
  middle->add_output("y");
  {
    NodeProto *n = middle->add_node();
    n->set_op_type("Inner");
    n->set_domain("custom");
    n->add_input("x");
    n->add_output("t");
  }
  {
    NodeProto *n = middle->add_node();
    n->set_op_type("Inner");
    n->set_domain("custom");
    n->add_input("t");
    n->add_output("y");
  }

  FunctionProto *outer = model.add_functions();
  outer->set_name("Outer");
  outer->set_domain("custom");
  outer->add_input("x");
  outer->add_output("y");
  {
    NodeProto *n = outer->add_node();
    n->set_op_type("Inner");
    n->set_domain("custom");
    n->add_input("x");
    n->add_output("t");
  }
  {
    NodeProto *n = outer->add_node();
    n->set_op_type("Middle");
    n->set_domain("custom");
    n->add_input("t");
    n->add_output("y");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("Outer");
  node->set_domain("custom");
  node->add_input("x");
  node->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.5f, -3.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  const float *res = rt.Get("y").AsFloat();
  ASSERT_EQ(rt.Get("y").element_count(), 3);
  EXPECT_FLOAT_EQ(res[0], 8.0f);
  EXPECT_FLOAT_EQ(res[1], 20.0f);
  EXPECT_FLOAT_EQ(res[2], -24.0f);
}

TEST(RunModel, ModelLocalFunctionOverloadDisambiguation) {
  // Two functions share the same (domain, name) but differ by overload.
  // The dispatcher must pick the one matching node.overload.
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *f_sum = model.add_functions();
  f_sum->set_name("Combine");
  f_sum->set_domain("custom");
  f_sum->set_overload("sum");
  f_sum->add_input("a");
  f_sum->add_input("b");
  f_sum->add_output("out");
  {
    NodeProto *n = f_sum->add_node();
    n->set_op_type("Add");
    n->add_input("a");
    n->add_input("b");
    n->add_output("out");
  }

  FunctionProto *f_diff = model.add_functions();
  f_diff->set_name("Combine");
  f_diff->set_domain("custom");
  f_diff->set_overload("diff");
  f_diff->add_input("a");
  f_diff->add_input("b");
  f_diff->add_output("out");
  {
    NodeProto *n = f_diff->add_node();
    n->set_op_type("Sub");
    n->add_input("a");
    n->add_input("b");
    n->add_output("out");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *n1 = g->add_node();
  n1->set_op_type("Combine");
  n1->set_domain("custom");
  n1->set_overload("sum");
  n1->add_input("x");
  n1->add_input("y");
  n1->add_output("s");
  NodeProto *n2 = g->add_node();
  n2->set_op_type("Combine");
  n2->set_domain("custom");
  n2->set_overload("diff");
  n2->add_input("x");
  n2->add_input("y");
  n2->add_output("d");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {1}, {10.0f}));
  rt.Set("y", Tensor::FromFloat("y", {1}, {3.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("s"));
  ASSERT_TRUE(rt.Has("d"));
  EXPECT_FLOAT_EQ(rt.Get("s").AsFloat()[0], 13.0f);
  EXPECT_FLOAT_EQ(rt.Get("d").AsFloat()[0], 7.0f);
}

TEST(RunModel, ModelLocalFunctionWrongInputCountThrows) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *func = model.add_functions();
  func->set_name("F");
  func->set_domain("custom");
  func->add_input("a");
  func->add_input("b");
  func->add_output("out");
  NodeProto *fn = func->add_node();
  fn->set_op_type("Add");
  fn->add_input("a");
  fn->add_input("b");
  fn->add_output("out");

  GraphProto *g = model.add_graph();
  NodeProto *node = g->add_node();
  node->set_op_type("F");
  node->set_domain("custom");
  node->add_input("x"); // only 1, function expects 2
  node->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {1}, {1.0f}));

  EXPECT_THROW(RunModel(model, rt), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Model-local function with linked (ref_attr_name) attributes
// ---------------------------------------------------------------------------

namespace {

// Builds a constant-only sub-graph that emits a single FLOAT scalar
// named ``out_name`` set to ``value``. Used as a GRAPH attribute below.
void FillConstantBranch(GraphProto &g, const std::string &branch_name, const std::string &init_name,
                        const std::string &out_name, float value) {
  g.set_name(branch_name);
  TensorProto *init = g.add_initializer();
  init->set_name(init_name);
  init->set_data_type(TensorProto::DataType::FLOAT);
  init->add_float_data(value);
  // The If implementation expects each branch sub-graph to produce its
  // output via at least one node. Use Add(init, init) so the value is
  // doubled, mirroring the existing ``IfNodeWithBranchSubgraphs`` test.
  NodeProto *add = g.add_node();
  add->set_op_type("Add");
  add->add_input(init_name);
  add->add_input(init_name);
  add->add_output(out_name);
  g.add_output()->set_name(out_name);
}

} // namespace

TEST(RunModel, ModelLocalFunctionLinkedAttributeFromCallSite) {
  // Define a model-local function "Pick(cond)" whose body delegates to
  // an ``If`` node where both ``then_branch`` and ``else_branch`` are
  // attribute references (``ref_attr_name``) to call-site attributes of
  // the same name. Two distinct call-sites supply different branch
  // sub-graphs, proving that the attribute is resolved per call rather
  // than baked in at function-definition time.
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("Pick");
  func->set_domain("custom");
  func->add_input("cond");
  func->add_output("out");
  func->add_attribute("then_branch");
  func->add_attribute("else_branch");
  {
    NodeProto *n = func->add_node();
    n->set_op_type("If");
    n->add_input("cond");
    n->add_output("out");
    AttributeProto *tref = n->add_attribute();
    tref->set_name("then_branch");
    tref->set_ref_attr_name("then_branch");
    tref->set_type(AttributeProto::AttributeType::GRAPH);
    AttributeProto *eref = n->add_attribute();
    eref->set_name("else_branch");
    eref->set_ref_attr_name("else_branch");
    eref->set_type(AttributeProto::AttributeType::GRAPH);
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *call = g->add_node();
  call->set_op_type("Pick");
  call->set_domain("custom");
  call->add_input("cond");
  call->add_output("out");
  AttributeProto *tattr = call->add_attribute();
  tattr->set_name("then_branch");
  tattr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*tattr->add_g(), "then_g", "t", "z", 10.0f);
  AttributeProto *eattr = call->add_attribute();
  eattr->set_name("else_branch");
  eattr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*eattr->add_g(), "else_g", "e", "z", 1.0f);

  RuntimeContext rt_true(KernelContext(DefaultOpset(18)));
  rt_true.Set("cond", Tensor::FromBool("cond", {}, {1}));
  RunModel(model, rt_true);
  ASSERT_TRUE(rt_true.Has("out"));
  EXPECT_FLOAT_EQ(rt_true.Get("out").AsFloat()[0], 20.0f);

  RuntimeContext rt_false(KernelContext(DefaultOpset(18)));
  rt_false.Set("cond", Tensor::FromBool("cond", {}, {0}));
  RunModel(model, rt_false);
  ASSERT_TRUE(rt_false.Has("out"));
  EXPECT_FLOAT_EQ(rt_false.Get("out").AsFloat()[0], 2.0f);
}

TEST(RunModel, ModelLocalFunctionLinkedAttributeUsesDefault) {
  // The function declares typed defaults for ``then_branch`` and
  // ``else_branch`` via ``attribute_proto``. The call-site omits both
  // and the defaults must be used instead.
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("Pick");
  func->set_domain("custom");
  func->add_input("cond");
  func->add_output("out");
  AttributeProto *tdef = func->add_attribute_proto();
  tdef->set_name("then_branch");
  tdef->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*tdef->add_g(), "then_default_g", "t_def", "z", 5.0f);
  AttributeProto *edef = func->add_attribute_proto();
  edef->set_name("else_branch");
  edef->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*edef->add_g(), "else_default_g", "e_def", "z", 7.0f);
  {
    NodeProto *n = func->add_node();
    n->set_op_type("If");
    n->add_input("cond");
    n->add_output("out");
    AttributeProto *tref = n->add_attribute();
    tref->set_name("then_branch");
    tref->set_ref_attr_name("then_branch");
    tref->set_type(AttributeProto::AttributeType::GRAPH);
    AttributeProto *eref = n->add_attribute();
    eref->set_name("else_branch");
    eref->set_ref_attr_name("else_branch");
    eref->set_type(AttributeProto::AttributeType::GRAPH);
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *call = g->add_node();
  call->set_op_type("Pick");
  call->set_domain("custom");
  call->add_input("cond");
  call->add_output("out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  RunModel(model, rt);
  ASSERT_TRUE(rt.Has("out"));
  EXPECT_FLOAT_EQ(rt.Get("out").AsFloat()[0], 10.0f);
}

TEST(RunModel, ModelLocalFunctionDoesNotMutateModel) {
  // Verify that resolving ``ref_attr_name`` references operates on a
  // copy of the FunctionProto so the source ModelProto's function body
  // is unchanged after RunModel.
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("Pick");
  func->set_domain("custom");
  func->add_input("cond");
  func->add_output("out");
  func->add_attribute("then_branch");
  func->add_attribute("else_branch");
  {
    NodeProto *n = func->add_node();
    n->set_op_type("If");
    n->add_input("cond");
    n->add_output("out");
    AttributeProto *tref = n->add_attribute();
    tref->set_name("then_branch");
    tref->set_ref_attr_name("then_branch");
    tref->set_type(AttributeProto::AttributeType::GRAPH);
    AttributeProto *eref = n->add_attribute();
    eref->set_name("else_branch");
    eref->set_ref_attr_name("else_branch");
    eref->set_type(AttributeProto::AttributeType::GRAPH);
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *call = g->add_node();
  call->set_op_type("Pick");
  call->set_domain("custom");
  call->add_input("cond");
  call->add_output("out");
  AttributeProto *tattr = call->add_attribute();
  tattr->set_name("then_branch");
  tattr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*tattr->add_g(), "then_g", "t", "z", 10.0f);
  AttributeProto *eattr = call->add_attribute();
  eattr->set_name("else_branch");
  eattr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*eattr->add_g(), "else_g", "e", "z", 1.0f);

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  RunModel(model, rt);

  // The function body's attribute must still be a reference (no graph
  // baked in) so the same model can be executed again with a different
  // call-site attribute.
  const FunctionProto &saved = model.functions()[0];
  const AttributeProto &a = saved.node()[0].attribute()[0];
  EXPECT_EQ(a.ref_attr_name().as_string(), "then_branch");
  EXPECT_FALSE(a.has_g());
}

TEST(RunModel, IfNodeWithBranchSubgraphs) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *if_node = g->add_node();
  if_node->set_op_type("If");
  if_node->add_input("cond");
  if_node->add_output("out");

  AttributeProto *then_attr = if_node->add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *then_g = then_attr->add_g();
  then_g->set_name("then_graph");
  TensorProto *then_init = then_g->add_initializer();
  then_init->set_name("t");
  then_init->set_data_type(TensorProto::DataType::FLOAT);
  then_init->add_float_data(10.0f);
  NodeProto *then_add = then_g->add_node();
  then_add->set_op_type("Add");
  then_add->add_input("t");
  then_add->add_input("t");
  then_add->add_output("z");
  then_g->add_output()->set_name("z");

  AttributeProto *else_attr = if_node->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *else_g = else_attr->add_g();
  else_g->set_name("else_graph");
  TensorProto *else_init = else_g->add_initializer();
  else_init->set_name("e");
  else_init->set_data_type(TensorProto::DataType::FLOAT);
  else_init->add_float_data(1.0f);
  NodeProto *else_add = else_g->add_node();
  else_add->set_op_type("Add");
  else_add->add_input("e");
  else_add->add_input("e");
  else_add->add_output("z");
  else_g->add_output()->set_name("z");

  RuntimeContext rt_true(KernelContext(DefaultOpset(18)));
  rt_true.Set("cond", Tensor::FromBool("cond", {}, {1}));
  RunModel(model, rt_true);
  ASSERT_TRUE(rt_true.Has("out"));
  EXPECT_FLOAT_EQ(rt_true.Get("out").AsFloat()[0], 20.0f);

  RuntimeContext rt_false(KernelContext(DefaultOpset(18)));
  rt_false.Set("cond", Tensor::FromBool("cond", {}, {0}));
  RunModel(model, rt_false);
  ASSERT_TRUE(rt_false.Has("out"));
  EXPECT_FLOAT_EQ(rt_false.Get("out").AsFloat()[0], 2.0f);
}

TEST(RunModel, LoopNodeRunsBodySubgraph) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *loop = g->add_node();
  loop->set_op_type("Loop");
  loop->add_input("M");
  loop->add_input("cond");
  loop->add_input("s_init");
  loop->add_output("s_final");
  loop->add_output("scan");

  AttributeProto *body_attr = loop->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *body = body_attr->add_g();
  body->set_name("loop_body");
  body->add_input()->set_name("iter");
  body->add_input()->set_name("cond_in");
  body->add_input()->set_name("s_in");
  TensorProto *one = body->add_initializer();
  one->set_name("one");
  one->set_data_type(TensorProto::DataType::FLOAT);
  one->add_float_data(1.0f);
  NodeProto *add = body->add_node();
  add->set_op_type("Add");
  add->add_input("s_in");
  add->add_input("one");
  add->add_output("s_out");
  body->add_output()->set_name("cond_in");
  body->add_output()->set_name("s_out");
  body->add_output()->set_name("s_out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("M", Tensor::FromInt64("M", {}, {3}));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  rt.Set("s_init", Tensor::FromFloat("s_init", {}, {0.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("s_final"));
  ASSERT_TRUE(rt.Has("scan"));
  EXPECT_FLOAT_EQ(rt.Get("s_final").AsFloat()[0], 3.0f);
  ASSERT_EQ(rt.Get("scan").shape, (std::vector<int64_t>{3}));
  const float *scan = rt.Get("scan").AsFloat();
  EXPECT_FLOAT_EQ(scan[0], 1.0f);
  EXPECT_FLOAT_EQ(scan[1], 2.0f);
  EXPECT_FLOAT_EQ(scan[2], 3.0f);
}

TEST(RunModel, ScanNodeRunsBodySubgraph) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *scan = g->add_node();
  scan->set_op_type("Scan");
  scan->add_input("state0");
  scan->add_input("x");
  scan->add_output("state_final");
  scan->add_output("y");
  AttributeProto *num_attr = scan->add_attribute();
  num_attr->set_name("num_scan_inputs");
  num_attr->set_type(AttributeProto::AttributeType::INT);
  num_attr->set_i(1);

  AttributeProto *body_attr = scan->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *body = body_attr->add_g();
  body->set_name("scan_body");
  body->add_input()->set_name("state_in");
  body->add_input()->set_name("x_in");
  NodeProto *add = body->add_node();
  add->set_op_type("Add");
  add->add_input("state_in");
  add->add_input("x_in");
  add->add_output("state_out");
  body->add_output()->set_name("state_out");
  body->add_output()->set_name("state_out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("state0", Tensor::FromFloat("state0", {}, {0.0f}));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("state_final"));
  ASSERT_TRUE(rt.Has("y"));
  EXPECT_FLOAT_EQ(rt.Get("state_final").AsFloat()[0], 6.0f);
  ASSERT_EQ(rt.Get("y").shape, (std::vector<int64_t>{3}));
  const float *y = rt.Get("y").AsFloat();
  EXPECT_FLOAT_EQ(y[0], 1.0f);
  EXPECT_FLOAT_EQ(y[1], 3.0f);
  EXPECT_FLOAT_EQ(y[2], 6.0f);
}

// Mirrors backend test ``test_scan_sum``: opset 8 batched form of Scan
// where every state and scan input/output carries an outer batch dim of
// size 1 and the (always-empty here) ``sequence_lens`` placeholder
// occupies node.input(0).
//
//   sum_in = Add(sum_in, next), scan_out = Identity(sum_out)
//   initial=[[0,0]] [1,2], x=[[[1,2],[3,4],[5,6]]] [1,3,2]
//   y=[[9,12]] [1,2], z=[[[1,2],[4,6],[9,12]]] [1,3,2]
TEST(RunModel, ScanOpset8NodeRunsBodySubgraphWithBatchDim) {
  ModelProto model;
  model.set_ir_version(3);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(8);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *scan = g->add_node();
  scan->set_op_type("Scan");
  // Empty placeholder for sequence_lens, then initial state, then scan input.
  scan->add_input("");
  scan->add_input("initial");
  scan->add_input("x");
  scan->add_output("y");
  scan->add_output("z");
  AttributeProto *num_attr = scan->add_attribute();
  num_attr->set_name("num_scan_inputs");
  num_attr->set_type(AttributeProto::AttributeType::INT);
  num_attr->set_i(1);

  AttributeProto *body_attr = scan->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *body = body_attr->add_g();
  body->set_name("scan_body");
  body->add_input()->set_name("sum_in");
  body->add_input()->set_name("next");
  NodeProto *add = body->add_node();
  add->set_op_type("Add");
  add->add_input("sum_in");
  add->add_input("next");
  add->add_output("sum_out");
  NodeProto *id = body->add_node();
  id->set_op_type("Identity");
  id->add_input("sum_out");
  id->add_output("scan_out");
  body->add_output()->set_name("sum_out");
  body->add_output()->set_name("scan_out");

  RuntimeContext rt(KernelContext(DefaultOpset(8)));
  rt.Set("initial", Tensor::FromFloat("initial", {1, 2}, {0.0f, 0.0f}));
  rt.Set("x", Tensor::FromFloat("x", {1, 3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  ASSERT_TRUE(rt.Has("z"));

  const Tensor &y = rt.Get("y");
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 2}));
  const float *y_ptr = y.AsFloat();
  EXPECT_FLOAT_EQ(y_ptr[0], 9.0f);
  EXPECT_FLOAT_EQ(y_ptr[1], 12.0f);

  const Tensor &z = rt.Get("z");
  ASSERT_EQ(z.shape, (std::vector<int64_t>{1, 3, 2}));
  const float *z_ptr = z.AsFloat();
  EXPECT_FLOAT_EQ(z_ptr[0], 1.0f);
  EXPECT_FLOAT_EQ(z_ptr[1], 2.0f);
  EXPECT_FLOAT_EQ(z_ptr[2], 4.0f);
  EXPECT_FLOAT_EQ(z_ptr[3], 6.0f);
  EXPECT_FLOAT_EQ(z_ptr[4], 9.0f);
  EXPECT_FLOAT_EQ(z_ptr[5], 12.0f);
}

TEST(RunNodes, RunNodeLinearAttentionFromDispatchTable) {
  // Minimal test: B=1, T=1, q_num_heads=1, kv_num_heads=1, d_k=d_v=2.
  // query/key/value each have shape (1, 1, 2).
  RuntimeContext rt(KernelContext(DefaultOpset(27)));
  rt.tensors()["query"] = Tensor::FromFloat("query", {1, 1, 2}, {1.0f, 0.0f});
  rt.tensors()["key"] = Tensor::FromFloat("key", {1, 1, 2}, {1.0f, 0.0f});
  rt.tensors()["value"] = Tensor::FromFloat("value", {1, 1, 2}, {0.5f, 0.5f});

  NodeProto node =
      MakeNode("LinearAttention", {"query", "key", "value"}, {"output", "present_state"});
  AttributeProto *rule_attr = node.add_attribute();
  rule_attr->set_name("update_rule");
  rule_attr->set_type(AttributeProto::AttributeType::STRING);
  rule_attr->set_s("linear");
  AttributeProto *qh_attr = node.add_attribute();
  qh_attr->set_name("q_num_heads");
  qh_attr->set_type(AttributeProto::AttributeType::INT);
  qh_attr->set_i(1);
  AttributeProto *kvh_attr = node.add_attribute();
  kvh_attr->set_name("kv_num_heads");
  kvh_attr->set_type(AttributeProto::AttributeType::INT);
  kvh_attr->set_i(1);
  RunNode(node, rt);

  const Tensor &output = rt.tensors().at("output");
  EXPECT_EQ(output.shape, (std::vector<int64_t>{1, 1, 2}));
  const Tensor &present = rt.tensors().at("present_state");
  EXPECT_EQ(present.shape, (std::vector<int64_t>{1, 1, 2, 2}));
}

TEST(RunNodes, RunNodeFlexAttentionFromDispatchTable) {
  // Minimal test: B=1, q_num_heads=kv_num_heads=1, q_seq=kv_seq=2, head_size=2.
  // Q/K/V each have shape (1, 1, 2, 2).
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["Q"] = Tensor::FromFloat("Q", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  rt.tensors()["K"] = Tensor::FromFloat("K", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  rt.tensors()["V"] = Tensor::FromFloat("V", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  NodeProto node = MakeNode("FlexAttention", {"Q", "K", "V"}, {"Y"}, "ai.onnx.preview");
  RunNode(node, rt);

  const Tensor &Y = rt.tensors().at("Y");
  EXPECT_EQ(Y.shape, (std::vector<int64_t>{1, 1, 2, 2}));
  EXPECT_EQ(Y.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
}

TEST(RunNodes, RunNodeGRUFromDispatchTable) {
  // Single-step (seq_length=1) GRU with X/W/R only: requests Y_h as the
  // only output via an empty Y output name, mirroring the ``gru_defaults``
  // backend test case (batch=3, input=2, hidden=5).
  RuntimeContext rt(KernelContext(DefaultOpset(14)));

  constexpr int64_t kSeqLength = 1;
  constexpr int64_t kBatch = 3;
  constexpr int64_t kInput = 2;
  constexpr int64_t kHidden = 5;
  constexpr int64_t kNumGates = 3;
  constexpr float kWeightScale = 0.1f;

  rt.tensors()["X"] = Tensor::FromFloat("X", {kSeqLength, kBatch, kInput}, {1, 2, 3, 4, 5, 6});
  std::vector<float> w_data(static_cast<size_t>(kNumGates * kHidden * kInput), kWeightScale);
  std::vector<float> r_data(static_cast<size_t>(kNumGates * kHidden * kHidden), kWeightScale);
  rt.tensors()["W"] = Tensor::FromFloat("W", {1, kNumGates * kHidden, kInput}, w_data);
  rt.tensors()["R"] = Tensor::FromFloat("R", {1, kNumGates * kHidden, kHidden}, r_data);

  NodeProto node = MakeNode("GRU", {"X", "W", "R"}, {"", "Y_h"});
  AttributeProto *hs = node.add_attribute();
  hs->set_name("hidden_size");
  hs->set_type(AttributeProto::AttributeType::INT);
  hs->set_i(kHidden);

  RunNode(node, rt);

  // Y is suppressed (empty output name) so it must not appear in the tensors map.
  EXPECT_EQ(rt.tensors().find("Y"), rt.tensors().end());

  const Tensor &y_h = rt.tensors().at("Y_h");
  EXPECT_EQ(y_h.shape, (std::vector<int64_t>{1, kBatch, kHidden}));
  EXPECT_EQ(y_h.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));

  // Compare against the kernel's direct output to validate dispatch-time
  // wiring of inputs, attributes and outputs.
  const onnx_kernels::kernel::GRU gru_kernel(rt.kernel_ctx());
  auto [y_ref, y_h_ref] =
      gru_kernel(rt.tensors().at("X"), rt.tensors().at("W"), rt.tensors().at("R"));
  (void)y_ref;
  ASSERT_EQ(y_h.element_count(), y_h_ref.element_count());
  for (int64_t i = 0; i < y_h.element_count(); ++i) {
    EXPECT_FLOAT_EQ(y_h.AsFloat()[i], y_h_ref.AsFloat()[i]);
  }
}

TEST(RunNodes, RunNodeLSTMFromDispatchTableUniformSequenceLens) {
  // ``sequence_lens`` is accepted when every entry equals ``seq_length``
  // (no-op masking); the dispatch must produce the same outputs as the
  // 3-input form above. This mirrors the ``test_cc_lstm_with_peepholes``
  // backend case (which passes a uniform ``sequence_lens``).
  RuntimeContext rt(KernelContext(DefaultOpset(14)));

  constexpr int64_t kSeqLength = 1;
  constexpr int64_t kBatch = 2;
  constexpr int64_t kInput = 4;
  constexpr int64_t kHidden = 3;
  constexpr int64_t kNumGates = 4;
  constexpr int64_t kNumPeepholes = 3;
  constexpr float kWeightScale = 0.1f;

  rt.tensors()["X"] =
      Tensor::FromFloat("X", {kSeqLength, kBatch, kInput}, {1, 2, 3, 4, 5, 6, 7, 8});
  std::vector<float> w_data(static_cast<size_t>(kNumGates * kHidden * kInput), kWeightScale);
  std::vector<float> r_data(static_cast<size_t>(kNumGates * kHidden * kHidden), kWeightScale);
  std::vector<float> b_data(static_cast<size_t>(2 * kNumGates * kHidden), 0.0f);
  std::vector<float> h0_data(static_cast<size_t>(kBatch * kHidden), 0.0f);
  std::vector<float> c0_data(static_cast<size_t>(kBatch * kHidden), 0.0f);
  std::vector<float> p_data(static_cast<size_t>(kNumPeepholes * kHidden), kWeightScale);
  rt.tensors()["W"] = Tensor::FromFloat("W", {1, kNumGates * kHidden, kInput}, w_data);
  rt.tensors()["R"] = Tensor::FromFloat("R", {1, kNumGates * kHidden, kHidden}, r_data);
  rt.tensors()["B"] = Tensor::FromFloat("B", {1, 2 * kNumGates * kHidden}, b_data);
  rt.tensors()["sequence_lens"] =
      Tensor::FromInt32("sequence_lens", {kBatch},
                        {static_cast<int32_t>(kSeqLength), static_cast<int32_t>(kSeqLength)});
  rt.tensors()["initial_h"] = Tensor::FromFloat("initial_h", {1, kBatch, kHidden}, h0_data);
  rt.tensors()["initial_c"] = Tensor::FromFloat("initial_c", {1, kBatch, kHidden}, c0_data);
  rt.tensors()["P"] = Tensor::FromFloat("P", {1, kNumPeepholes * kHidden}, p_data);

  NodeProto node = MakeNode(
      "LSTM", {"X", "W", "R", "B", "sequence_lens", "initial_h", "initial_c", "P"}, {"", "Y_h"});
  AttributeProto *hs = node.add_attribute();
  hs->set_name("hidden_size");
  hs->set_type(AttributeProto::AttributeType::INT);
  hs->set_i(kHidden);

  RunNode(node, rt);

  const Tensor &y_h = rt.tensors().at("Y_h");
  EXPECT_EQ(y_h.shape, (std::vector<int64_t>{1, kBatch, kHidden}));
  EXPECT_EQ(y_h.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));

  // Non-uniform ``sequence_lens`` is still rejected.
  rt.tensors()["sequence_lens"] =
      Tensor::FromInt32("sequence_lens", {kBatch}, {static_cast<int32_t>(kSeqLength), 0});
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

TEST(RunNodes, RunNodeLSTMFromDispatchTable) {
  // Single-step (seq_length=1) LSTM with X/W/R only: requests Y_h as the
  // only output via an empty Y output name, mirroring the ``lstm_defaults``
  // backend test case (batch=3, input=2, hidden=3).
  RuntimeContext rt(KernelContext(DefaultOpset(14)));

  constexpr int64_t kSeqLength = 1;
  constexpr int64_t kBatch = 3;
  constexpr int64_t kInput = 2;
  constexpr int64_t kHidden = 3;
  constexpr int64_t kNumGates = 4;
  constexpr float kWeightScale = 0.1f;

  rt.tensors()["X"] = Tensor::FromFloat("X", {kSeqLength, kBatch, kInput}, {1, 2, 3, 4, 5, 6});
  std::vector<float> w_data(static_cast<size_t>(kNumGates * kHidden * kInput), kWeightScale);
  std::vector<float> r_data(static_cast<size_t>(kNumGates * kHidden * kHidden), kWeightScale);
  rt.tensors()["W"] = Tensor::FromFloat("W", {1, kNumGates * kHidden, kInput}, w_data);
  rt.tensors()["R"] = Tensor::FromFloat("R", {1, kNumGates * kHidden, kHidden}, r_data);

  NodeProto node = MakeNode("LSTM", {"X", "W", "R"}, {"", "Y_h"});
  AttributeProto *hs = node.add_attribute();
  hs->set_name("hidden_size");
  hs->set_type(AttributeProto::AttributeType::INT);
  hs->set_i(kHidden);

  RunNode(node, rt);

  // Y is suppressed (empty output name) so it must not appear in the tensors map.
  EXPECT_EQ(rt.tensors().find("Y"), rt.tensors().end());

  const Tensor &y_h = rt.tensors().at("Y_h");
  EXPECT_EQ(y_h.shape, (std::vector<int64_t>{1, kBatch, kHidden}));
  EXPECT_EQ(y_h.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));

  // Compare against the kernel's direct output to validate dispatch-time
  // wiring of inputs, attributes and outputs.
  const onnx_kernels::kernel::LSTM lstm_kernel(rt.kernel_ctx());
  auto [y_ref, y_h_ref] =
      lstm_kernel(rt.tensors().at("X"), rt.tensors().at("W"), rt.tensors().at("R"));
  (void)y_ref;
  ASSERT_EQ(y_h.element_count(), y_h_ref.element_count());
  for (int64_t i = 0; i < y_h.element_count(); ++i) {
    EXPECT_FLOAT_EQ(y_h.AsFloat()[i], y_h_ref.AsFloat()[i]);
  }
}

TEST(RunNodes, RunNodeSequenceConstructAndQueriesFromDispatchTable) {
  // Build a sequence of three tensors with SequenceConstruct, then
  // dispatch SequenceLength, SequenceAt and ConcatFromSequence and
  // check that the sequence-typed edge flows through the runtime
  // context's sequence map.
  RuntimeContext rt(KernelContext(DefaultOpset(11)));
  rt.tensors()["a"] = Tensor::FromFloat("a", {2}, {1.0f, 2.0f});
  rt.tensors()["b"] = Tensor::FromFloat("b", {2}, {3.0f, 4.0f});
  rt.tensors()["c"] = Tensor::FromFloat("c", {2}, {5.0f, 6.0f});
  rt.tensors()["pos1"] = Tensor::FromInt64("pos1", {}, {1});

  RunNode(MakeNode("SequenceConstruct", {"a", "b", "c"}, {"seq"}), rt);
  ASSERT_TRUE(rt.HasSequence("seq"));
  EXPECT_EQ(rt.GetSequence("seq").size(), 3u);
  EXPECT_EQ(rt.GetSequence("seq").elem_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));

  RunNode(MakeNode("SequenceLength", {"seq"}, {"n"}), rt);
  const Tensor &n = rt.tensors().at("n");
  EXPECT_EQ(n.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT64));
  ASSERT_EQ(n.element_count(), 1);
  EXPECT_EQ(n.AsInt64()[0], 3);

  RunNode(MakeNode("SequenceAt", {"seq", "pos1"}, {"middle"}), rt);
  const Tensor &middle = rt.tensors().at("middle");
  EXPECT_EQ(middle.shape, std::vector<int64_t>({2}));
  EXPECT_FLOAT_EQ(middle.AsFloat()[0], 3.0f);
  EXPECT_FLOAT_EQ(middle.AsFloat()[1], 4.0f);

  NodeProto concat = MakeNode("ConcatFromSequence", {"seq"}, {"flat"});
  AttributeProto *axis = concat.add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::INT);
  axis->set_i(0);
  RunNode(concat, rt);
  const Tensor &flat = rt.tensors().at("flat");
  EXPECT_EQ(flat.shape, std::vector<int64_t>({6}));
  const float *p = flat.AsFloat();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(p[i], static_cast<float>(i + 1));
  }
}

TEST(RunNodes, RunNodeSequenceEmptyInsertEraseFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(11)));
  rt.tensors()["a"] = Tensor::FromFloat("a", {1}, {7.0f});
  rt.tensors()["b"] = Tensor::FromFloat("b", {1}, {8.0f});

  NodeProto empty = MakeNode("SequenceEmpty", {}, {"seq0"});
  AttributeProto *dtype = empty.add_attribute();
  dtype->set_name("dtype");
  dtype->set_type(AttributeProto::INT);
  dtype->set_i(static_cast<int64_t>(onnx_kernels::DataType::FLOAT));
  RunNode(empty, rt);
  ASSERT_TRUE(rt.HasSequence("seq0"));
  EXPECT_TRUE(rt.GetSequence("seq0").empty());

  RunNode(MakeNode("SequenceInsert", {"seq0", "a"}, {"seq1"}), rt);
  EXPECT_EQ(rt.GetSequence("seq1").size(), 1u);
  RunNode(MakeNode("SequenceInsert", {"seq1", "b"}, {"seq2"}), rt);
  EXPECT_EQ(rt.GetSequence("seq2").size(), 2u);

  RunNode(MakeNode("SequenceErase", {"seq2"}, {"seq3"}), rt);
  EXPECT_EQ(rt.GetSequence("seq3").size(), 1u);
  EXPECT_FLOAT_EQ(rt.GetSequence("seq3").at(0).AsFloat()[0], 7.0f);
}

TEST(RunNodes, RunNodeSplitToSequenceFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(11)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  NodeProto split = MakeNode("SplitToSequence", {"x"}, {"out"});
  AttributeProto *keepdims = split.add_attribute();
  keepdims->set_name("keepdims");
  keepdims->set_type(AttributeProto::INT);
  keepdims->set_i(0);
  RunNode(split, rt);

  ASSERT_TRUE(rt.HasSequence("out"));
  const auto &seq = rt.GetSequence("out");
  EXPECT_EQ(seq.size(), 3u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(seq.at(i).shape, std::vector<int64_t>({2}));
    EXPECT_FLOAT_EQ(seq.at(i).AsFloat()[0], static_cast<float>(2 * i + 1));
    EXPECT_FLOAT_EQ(seq.at(i).AsFloat()[1], static_cast<float>(2 * i + 2));
  }
}

TEST(RuntimeContextCollectExternalInputs, FlatNodes) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Mul", {"x", "y"}, {"t"}));
  nodes.push_back(MakeNode("Sub", {"t", "z"}, {"out"}));
  nodes.push_back(MakeNode("Add", {"out", "x"}, {"final"}));

  auto inputs = RuntimeContext::CollectExternalInputs(nodes);
  EXPECT_EQ(inputs, std::vector<std::string>({"x", "y", "z"}));
}

TEST(RuntimeContextCollectExternalInputs, EmptyNodes) {
  std::vector<NodeProto> nodes;
  EXPECT_TRUE(RuntimeContext::CollectExternalInputs(nodes).empty());
}

TEST(RuntimeContextCollectExternalInputs, SkipsEmptyAndProducedNames) {
  std::vector<NodeProto> nodes;
  // Optional input "" must be ignored; output "t" produced internally
  // must not be reported as external.
  nodes.push_back(MakeNode("Resize", {"X", "", "scales"}, {"t"}));
  nodes.push_back(MakeNode("Abs", {"t"}, {"Y"}));

  auto inputs = RuntimeContext::CollectExternalInputs(nodes);
  EXPECT_EQ(inputs, std::vector<std::string>({"X", "scales"}));
}

TEST(RuntimeContextCollectExternalInputs, SubgraphCapturesOuterValues) {
  // Build an If node whose then_branch reads "cap1" from the outer
  // scope and whose else_branch reads "cap2"; "cond" feeds the If
  // input and "produced" is produced by an earlier node in the set.
  GraphProto then_branch;
  *then_branch.add_node() = MakeNode("Add", {"cap1", "produced"}, {"then_out"});
  then_branch.add_output()->set_name("then_out");

  GraphProto else_branch;
  *else_branch.add_node() = MakeNode("Add", {"cap2", "produced"}, {"else_out"});
  else_branch.add_output()->set_name("else_out");

  NodeProto if_node = MakeNode("If", {"cond"}, {"y"});
  AttributeProto *attr_then = if_node.add_attribute();
  attr_then->set_name("then_branch");
  attr_then->set_type(AttributeProto::AttributeType::GRAPH);
  *attr_then->mutable_g() = then_branch;
  AttributeProto *attr_else = if_node.add_attribute();
  attr_else->set_name("else_branch");
  attr_else->set_type(AttributeProto::AttributeType::GRAPH);
  *attr_else->mutable_g() = else_branch;

  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Identity", {"x"}, {"produced"}));
  nodes.push_back(if_node);

  auto inputs = RuntimeContext::CollectExternalInputs(nodes);
  // "produced" is produced by the outer set and must not be reported.
  // Order is first-seen.
  EXPECT_EQ(inputs, std::vector<std::string>({"x", "cond", "cap1", "cap2"}));
}

TEST(RuntimeContextCollectExternalInputs, SubgraphLocalNamesShadowOuter) {
  // Subgraph defines its own formal input "x", an initializer "k",
  // and produces "tmp" internally — none of these should be reported.
  // It additionally reads "outer_only" from the outer scope.
  GraphProto body;
  body.add_input()->set_name("x");
  TensorProto *init = body.add_initializer();
  init->set_name("k");
  init->set_data_type(static_cast<int32_t>(DataType::FLOAT));
  *body.add_node() = MakeNode("Add", {"x", "k"}, {"tmp"});
  *body.add_node() = MakeNode("Mul", {"tmp", "outer_only"}, {"body_out"});
  body.add_output()->set_name("body_out");

  NodeProto loop = MakeNode("Loop", {"M", "cond"}, {"y"});
  AttributeProto *attr = loop.add_attribute();
  attr->set_name("body");
  attr->set_type(AttributeProto::AttributeType::GRAPH);
  *attr->mutable_g() = body;

  std::vector<NodeProto> nodes = {loop};
  auto inputs = RuntimeContext::CollectExternalInputs(nodes);
  EXPECT_EQ(inputs, std::vector<std::string>({"M", "cond", "outer_only"}));
}

TEST(RuntimeContextCollectExternalInputs, NestedSubgraphCaptures) {
  // Outer If holds an inner If whose then_branch reads "deep".
  GraphProto inner_then;
  *inner_then.add_node() = MakeNode("Identity", {"deep"}, {"inner_out"});
  inner_then.add_output()->set_name("inner_out");
  GraphProto inner_else;
  *inner_else.add_node() = MakeNode("Identity", {"deep"}, {"inner_out"});
  inner_else.add_output()->set_name("inner_out");

  NodeProto inner_if = MakeNode("If", {"inner_cond"}, {"middle"});
  AttributeProto *a1 = inner_if.add_attribute();
  a1->set_name("then_branch");
  a1->set_type(AttributeProto::AttributeType::GRAPH);
  *a1->mutable_g() = inner_then;
  AttributeProto *a2 = inner_if.add_attribute();
  a2->set_name("else_branch");
  a2->set_type(AttributeProto::AttributeType::GRAPH);
  *a2->mutable_g() = inner_else;

  GraphProto outer_then;
  *outer_then.add_node() = inner_if;
  *outer_then.add_node() = MakeNode("Identity", {"middle"}, {"outer_out"});
  outer_then.add_output()->set_name("outer_out");
  GraphProto outer_else;
  *outer_else.add_node() = MakeNode("Identity", {"middle"}, {"outer_out"});
  outer_else.add_output()->set_name("outer_out");

  NodeProto outer_if = MakeNode("If", {"outer_cond"}, {"y"});
  AttributeProto *b1 = outer_if.add_attribute();
  b1->set_name("then_branch");
  b1->set_type(AttributeProto::AttributeType::GRAPH);
  *b1->mutable_g() = outer_then;
  AttributeProto *b2 = outer_if.add_attribute();
  b2->set_name("else_branch");
  b2->set_type(AttributeProto::AttributeType::GRAPH);
  *b2->mutable_g() = outer_else;

  std::vector<NodeProto> nodes = {outer_if};
  auto inputs = RuntimeContext::CollectExternalInputs(nodes);
  // outer_cond is read by the outer If node itself.
  // Inside outer_then: inner_if introduces inner_cond, and its branches
  // capture "deep" from above.
  // Inside outer_else: the lone Identity reads "middle", which is not
  // produced anywhere in outer_else (only in outer_then via inner_if),
  // so it is captured from the outer scope.
  EXPECT_EQ(inputs, std::vector<std::string>({"outer_cond", "inner_cond", "deep", "middle"}));
}

TEST(RuntimeContextCollectExternalInputs, DeduplicatesOrdering) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Add", {"a", "b"}, {"u"}));
  nodes.push_back(MakeNode("Mul", {"a", "u"}, {"v"})); // re-references "a"
  nodes.push_back(MakeNode("Sub", {"b", "v"}, {"w"})); // re-references "b"

  auto inputs = RuntimeContext::CollectExternalInputs(nodes);
  EXPECT_EQ(inputs, std::vector<std::string>({"a", "b"}));
}

// ---------------------------------------------------------------------------
// RuntimeContext isolation invariants when running a local function or a
// subgraph. See issue #2157.
// ---------------------------------------------------------------------------

// A model-local function must be invoked with an isolated tensor map: only
// its formal inputs (bound to the caller's actuals) are visible inside the
// function body. Names that exist in the caller's tensor map but are not
// passed as a function input must NOT be visible from inside the function.
// The construction-time ``kernel_ctx()`` is still shared so the function's
// nodes are dispatched against the same opset as the caller.
TEST(RunModel, LocalFunctionStartsWithEmptyTensorMap) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  // Function "F" has a single formal input "x" and a node that references
  // a name ("leak") which exists in the caller's tensor map but is NOT a
  // declared function input. Because the function runs in an isolated
  // child RuntimeContext that starts empty (then gets only "x" bound),
  // dispatching the body must fail: "leak" cannot be resolved.
  FunctionProto *func = model.add_functions();
  func->set_name("F");
  func->set_domain("custom");
  func->add_input("x");
  func->add_output("out");
  NodeProto *fn = func->add_node();
  fn->set_op_type("Add");
  fn->add_input("x");
  fn->add_input("leak"); // not a function input -> must be invisible.
  fn->add_output("out");

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *call = g->add_node();
  call->set_op_type("F");
  call->set_domain("custom");
  call->add_input("x");
  call->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {2}, {1.0f, 2.0f}));
  rt.Set("leak", Tensor::FromFloat("leak", {2}, {100.0f, 200.0f}));

  // The function body references "leak" which is not bound as a formal
  // input. With proper isolation the lookup throws.
  EXPECT_THROW(RunModel(model, rt), std::invalid_argument);

  // The caller's tensor map is untouched: "leak" is still there and the
  // function's formal output "out" did not leak in either.
  EXPECT_TRUE(rt.Has("leak"));
  EXPECT_FALSE(rt.Has("out"));
}

// Companion to the test above: the same model with "leak" passed as the
// function's formal input "x" succeeds, demonstrating that the function
// body sees only what was explicitly bound and that ``kernel_ctx()`` is
// shared (the Add kernel is dispatched against the caller's opset).
TEST(RunModel, LocalFunctionSharesKernelContextOnlyWithEmptyTensorMap) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("Twice");
  func->set_domain("custom");
  func->add_input("v");
  func->add_output("out");
  NodeProto *fn = func->add_node();
  fn->set_op_type("Add");
  fn->add_input("v");
  fn->add_input("v");
  fn->add_output("internal_tmp"); // an intermediate inside the function.
  NodeProto *fn2 = func->add_node();
  fn2->set_op_type("Identity");
  fn2->add_input("internal_tmp");
  fn2->add_output("out");

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *call = g->add_node();
  call->set_op_type("Twice");
  call->set_domain("custom");
  call->add_input("a");
  call->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("a", Tensor::FromFloat("a", {2}, {3.0f, 4.0f}));
  // Pre-populate the caller's tensor map with a name that collides with
  // a function-local intermediate. The function must not see or
  // overwrite this caller-side value; after the call the caller's
  // value must still be there.
  rt.Set("internal_tmp", Tensor::FromFloat("internal_tmp", {1}, {-1.0f}));

  RunModel(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  EXPECT_FLOAT_EQ(rt.Get("y").AsFloat()[0], 6.0f);
  EXPECT_FLOAT_EQ(rt.Get("y").AsFloat()[1], 8.0f);
  // The function intermediate must not leak back to the caller: the
  // caller's pre-existing "internal_tmp" is preserved unchanged.
  ASSERT_TRUE(rt.Has("internal_tmp"));
  ASSERT_EQ(rt.Get("internal_tmp").shape, (std::vector<int64_t>{1}));
  EXPECT_FLOAT_EQ(rt.Get("internal_tmp").AsFloat()[0], -1.0f);
}

// A local subgraph (e.g. the body of an If/Loop/Scan node) must start
// with a *copy* of the caller's tensor map so it can reference outer-scope
// names. Intermediates produced inside the subgraph must NOT be propagated
// back to the caller's tensor map: only the values declared as subgraph
// outputs are visible to the caller (under the node's output names).
TEST(RunModel, LocalSubgraphCopiesCallerTensorsButHidesIntermediates) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *if_node = g->add_node();
  if_node->set_op_type("If");
  if_node->add_input("cond");
  if_node->add_output("out");

  // ``then_branch`` references an outer-scope tensor ``outer`` (only
  // present in the caller's tensor map) via an Add node, and produces a
  // subgraph-local intermediate ``branch_tmp`` that must not leak back.
  AttributeProto *then_attr = if_node->add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *then_g = then_attr->add_g();
  then_g->set_name("then_graph");
  NodeProto *then_add = then_g->add_node();
  then_add->set_op_type("Add");
  then_add->add_input("outer");
  then_add->add_input("outer");
  then_add->add_output("branch_tmp");
  NodeProto *then_id = then_g->add_node();
  then_id->set_op_type("Identity");
  then_id->add_input("branch_tmp");
  then_id->add_output("z");
  then_g->add_output()->set_name("z");

  // Minimal else_branch, never taken in this test.
  AttributeProto *else_attr = if_node->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *else_g = else_attr->add_g();
  else_g->set_name("else_graph");
  NodeProto *else_id = else_g->add_node();
  else_id->set_op_type("Identity");
  else_id->add_input("outer");
  else_id->add_output("z");
  else_g->add_output()->set_name("z");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  rt.Set("outer", Tensor::FromFloat("outer", {2}, {5.0f, 7.0f}));

  RunModel(model, rt);

  // The subgraph can see ``outer`` (caller's tensor map was copied into
  // the child) and produced the declared output.
  ASSERT_TRUE(rt.Has("out"));
  ASSERT_EQ(rt.Get("out").shape, (std::vector<int64_t>{2}));
  EXPECT_FLOAT_EQ(rt.Get("out").AsFloat()[0], 10.0f);
  EXPECT_FLOAT_EQ(rt.Get("out").AsFloat()[1], 14.0f);

  // The caller's pre-existing tensors are preserved.
  EXPECT_TRUE(rt.Has("cond"));
  EXPECT_TRUE(rt.Has("outer"));

  // Subgraph intermediates do NOT leak into the caller's tensor map.
  EXPECT_FALSE(rt.Has("branch_tmp"));
  // The subgraph's formal output name ``z`` (distinct from the node's
  // output name ``out``) likewise stays inside the child context.
  EXPECT_FALSE(rt.Has("z"));
}

// ---------------------------------------------------------------------------
// Custom kernel registration through RuntimeContext::RegisterCustomKernel.
// ---------------------------------------------------------------------------

// A custom kernel registered for an op outside the built-in dispatch table
// is invoked by RunNode; its output is written back to the RuntimeContext.
TEST(RunNodes, RunNodeDispatchesCustomKernelForUnknownOp) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));

  int call_count = 0;
  rt.RegisterCustomKernel("my.domain", "Scale",
                          [&call_count](const NodeProto &node, RuntimeContext &ctx) {
                            ++call_count;
                            // Read the "factor" attribute (default 1.0f).
                            float factor = 1.0f;
                            for (int i = 0; i < node.attribute_size(); ++i) {
                              const AttributeProto &a = node.attribute(i);
                              if (a.name() == "factor") {
                                factor = a.f();
                              }
                            }
                            const Tensor &in = ctx.Get(node.input(0).as_string());
                            std::vector<float> out(static_cast<size_t>(in.element_count()));
                            const float *src = in.AsFloat();
                            for (size_t i = 0; i < out.size(); ++i) {
                              out[i] = src[i] * factor;
                            }
                            ctx.Put(node.output(0).as_string(),
                                    Tensor::FromFloat(node.output(0).as_string(), in.shape, out));
                          });

  NodeProto node = MakeNode("Scale", {"x"}, {"y"}, "my.domain");
  AttributeProto *attr = node.add_attribute();
  attr->set_name("factor");
  attr->set_type(AttributeProto::AttributeType::FLOAT);
  attr->set_f(3.0f);

  RunNode(node, rt);
  EXPECT_EQ(call_count, 1);
  const Tensor &y = rt.tensors().at("y");
  ASSERT_EQ(y.element_count(), 3);
  const float *yp = y.AsFloat();
  EXPECT_FLOAT_EQ(yp[0], 3.0f);
  EXPECT_FLOAT_EQ(yp[1], 6.0f);
  EXPECT_FLOAT_EQ(yp[2], 9.0f);
}

// A custom kernel registered under the default ONNX domain overrides the
// built-in dispatch-table entry with the same key.
TEST(RunNodes, RunNodeCustomKernelOverridesBuiltin) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {-1.0f, -2.0f, -3.0f}));

  // Replace Abs with negation: a custom override must take precedence over
  // the entry that KernelDispatchTable() would otherwise resolve.
  rt.RegisterCustomKernel("", "Abs", [](const NodeProto &node, RuntimeContext &ctx) {
    const Tensor &in = ctx.Get(node.input(0).as_string());
    std::vector<float> out(static_cast<size_t>(in.element_count()));
    const float *src = in.AsFloat();
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = -src[i];
    }
    ctx.Put(node.output(0).as_string(),
            Tensor::FromFloat(node.output(0).as_string(), in.shape, out));
  });

  NodeProto node = MakeNode("Abs", {"x"}, {"y"});
  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("y");
  ASSERT_EQ(y.element_count(), 3);
  const float *yp = y.AsFloat();
  EXPECT_FLOAT_EQ(yp[0], 1.0f);
  EXPECT_FLOAT_EQ(yp[1], 2.0f);
  EXPECT_FLOAT_EQ(yp[2], 3.0f);
}

// RunModel chains a built-in kernel and a custom kernel together; the
// CustomKernelMap survives across nodes within the same context.
TEST(RunModel, CustomKernelChainsWithBuiltinKernels) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("my.domain");
  custom_os->set_version(1);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *n1 = g->add_node();
  n1->set_op_type("Abs");
  n1->add_input("x");
  n1->add_output("a");
  NodeProto *n2 = g->add_node();
  n2->set_op_type("Scale");
  n2->set_domain("my.domain");
  n2->add_input("a");
  n2->add_output("y");
  AttributeProto *attr = n2->add_attribute();
  attr->set_name("factor");
  attr->set_type(AttributeProto::AttributeType::FLOAT);
  attr->set_f(2.0f);

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {-1.0f, -2.0f, -3.0f}));
  rt.RegisterCustomKernel("my.domain", "Scale", [](const NodeProto &node, RuntimeContext &ctx) {
    float factor = 1.0f;
    for (int i = 0; i < node.attribute_size(); ++i) {
      if (node.attribute(i).name() == "factor") {
        factor = node.attribute(i).f();
      }
    }
    const Tensor &in = ctx.Get(node.input(0).as_string());
    std::vector<float> out(static_cast<size_t>(in.element_count()));
    const float *src = in.AsFloat();
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = src[i] * factor;
    }
    ctx.Put(node.output(0).as_string(),
            Tensor::FromFloat(node.output(0).as_string(), in.shape, out));
  });

  RunModel(model, rt);
  const Tensor &y = rt.tensors().at("y");
  ASSERT_EQ(y.element_count(), 3);
  const float *yp = y.AsFloat();
  EXPECT_FLOAT_EQ(yp[0], 2.0f);
  EXPECT_FLOAT_EQ(yp[1], 4.0f);
  EXPECT_FLOAT_EQ(yp[2], 6.0f);
}

// Without a registered custom kernel, an unknown op fails as before.
TEST(RunNodes, RunNodeUnknownOpWithoutCustomKernelThrows) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));
  NodeProto node = MakeNode("Scale", {"x"}, {"y"}, "my.domain");
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Release-unused-intermediates tests
// ---------------------------------------------------------------------------

TEST(RunNodes, CollectNodeInputsPlainNode) {
  NodeProto node = MakeNode("Add", {"x", "y"}, {"z"});
  auto inputs = RuntimeContext::CollectNodeInputs(node);
  EXPECT_EQ(inputs, (std::vector<std::string>{"x", "y"}));
}

TEST(RunNodes, CollectNodeInputsSkipsEmptyAndDedups) {
  NodeProto node = MakeNode("Add", {"x", "", "x"}, {"z"});
  auto inputs = RuntimeContext::CollectNodeInputs(node);
  EXPECT_EQ(inputs, (std::vector<std::string>{"x"}));
}

TEST(RunNodes, CollectNodeInputsIncludesSubgraphCaptures) {
  // ``If`` node with then/else subgraphs each capturing an outer name.
  NodeProto node;
  node.set_op_type("If");
  node.add_input("cond");

  auto add_subgraph = [](NodeProto &n, const std::string &attr_name,
                         const std::string &captured_name, const std::string &out_name) {
    AttributeProto attr;
    attr.set_name(attr_name);
    attr.set_type(AttributeProto::AttributeType::GRAPH);
    GraphProto &g = attr.ref_g();
    // Body: out_name = Identity(captured_name)
    NodeProto inner;
    inner.set_op_type("Identity");
    inner.add_input(captured_name);
    inner.add_output(out_name);
    g.ref_node().push_back(inner);
    ValueInfoProto vi;
    vi.set_name(out_name);
    g.ref_output().push_back(vi);
    n.ref_attribute().push_back(attr);
  };
  add_subgraph(node, "then_branch", "a", "y");
  add_subgraph(node, "else_branch", "b", "y");

  auto inputs = RuntimeContext::CollectNodeInputs(node);
  EXPECT_EQ(inputs, (std::vector<std::string>{"cond", "a", "b"}));
}

TEST(RunNodes, ComputeReleasableInputsLastUse) {
  // x -> Abs -> t ; (t, z) -> Add -> y
  // "x" last-used at node 0, "t" last-used at node 1, "z" last-used at node 1.
  // With keep = {"y"}, after node 0 we can release "x"; after node 1 we can
  // release "t" and "z".
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Abs", {"x"}, {"t"}));
  nodes.push_back(MakeNode("Add", {"t", "z"}, {"y"}));

  std::unordered_set<std::string> keep{"y"};
  auto rel = RuntimeContext::ComputeReleasableInputs(nodes, keep);
  ASSERT_EQ(rel.size(), 2u);
  EXPECT_EQ(rel[0], (std::vector<std::string>{"x"}));
  EXPECT_EQ(rel[1], (std::vector<std::string>{"t", "z"}));
}

TEST(RunNodes, ComputeReleasableInputsKeepIsPreserved) {
  // Reuse the same intermediate: pretend "x" is also a declared graph output
  // (i.e. caller wants to keep "x" after the run). It must NOT be released.
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Abs", {"x"}, {"t"}));
  std::unordered_set<std::string> keep{"x", "t"};
  auto rel = RuntimeContext::ComputeReleasableInputs(nodes, keep);
  ASSERT_EQ(rel.size(), 1u);
  EXPECT_TRUE(rel[0].empty());
}

TEST(RunNodes, RunGraphReleaseIntermediatesRemovesUnusedAndEmitsEvent) {
  // y = Add(Abs(x), z) — after running, "t" (the intermediate) must be gone
  // from the context, "y" (declared output) must remain, and "x" / "z"
  // (graph inputs already in the context) must also remain.
  using onnx_kernels::TensorEventAction;

  GraphProto graph;
  ValueInfoProto vi_x;
  vi_x.set_name("x");
  ValueInfoProto vi_z;
  vi_z.set_name("z");
  ValueInfoProto vi_y;
  vi_y.set_name("y");
  graph.ref_input().push_back(vi_x);
  graph.ref_input().push_back(vi_z);
  graph.ref_output().push_back(vi_y);
  graph.ref_node().push_back(MakeNode("Abs", {"x"}, {"t"}));
  graph.ref_node().push_back(MakeNode("Add", {"t", "z"}, {"y"}));

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.set_events_enabled(true);
  rt.set_release_intermediates(true);
  rt.Set("x", Tensor::FromFloat("x", {2}, {-1.0f, 2.0f}));
  rt.Set("z", Tensor::FromFloat("z", {2}, {10.0f, 20.0f}));

  RunGraph(graph, rt);

  // "t" was released, "y" / "x" / "z" survived.
  EXPECT_FALSE(rt.Has("t"));
  EXPECT_TRUE(rt.Has("y"));
  EXPECT_TRUE(rt.Has("x"));
  EXPECT_TRUE(rt.Has("z"));

  // At least one kRemove event was emitted for "t".
  bool saw_remove_t = false;
  for (const auto &ev : rt.events()) {
    if (ev.action == TensorEventAction::kRemove && ev.name == "t") {
      saw_remove_t = true;
      break;
    }
  }
  EXPECT_TRUE(saw_remove_t);

  // Default behaviour (release disabled) keeps the intermediate around so
  // callers can still fetch it after run.
  RuntimeContext rt2(KernelContext(DefaultOpset(18)));
  rt2.Set("x", Tensor::FromFloat("x", {2}, {-1.0f, 2.0f}));
  rt2.Set("z", Tensor::FromFloat("z", {2}, {10.0f, 20.0f}));
  RunGraph(graph, rt2);
  EXPECT_TRUE(rt2.Has("t"));
  EXPECT_TRUE(rt2.Has("y"));
}

TEST(RunNodes, ExecutionPlanIsCachedAcrossRunGraphInvocations) {
  // GetExecutionPlan returns the same instance on subsequent calls for
  // the same GraphProto, so the release analysis is paid only once.
  GraphProto graph;
  ValueInfoProto vi_x;
  vi_x.set_name("x");
  ValueInfoProto vi_y;
  vi_y.set_name("y");
  graph.ref_input().push_back(vi_x);
  graph.ref_output().push_back(vi_y);
  graph.ref_node().push_back(MakeNode("Abs", {"x"}, {"t"}));
  graph.ref_node().push_back(MakeNode("Neg", {"t"}, {"y"}));

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.set_release_intermediates(true);
  const onnx_kernels::ExecutionPlan &plan1 = rt.GetExecutionPlan(graph);
  const onnx_kernels::ExecutionPlan &plan2 = rt.GetExecutionPlan(graph);
  EXPECT_EQ(&plan1, &plan2);
  EXPECT_EQ(plan1.num_nodes(), 2u);
  // "t" is releasable after node 1, "x" / "y" are in keep (input/output).
  EXPECT_TRUE(plan1.releasable()[0].empty());
  ASSERT_EQ(plan1.releasable()[1].size(), 1u);
  EXPECT_EQ(plan1.releasable()[1][0], "t");
  EXPECT_TRUE(plan1.keep().count("x"));
  EXPECT_TRUE(plan1.keep().count("y"));

  // Two successive RunGraph calls both reuse the cached plan.
  rt.Set("x", Tensor::FromFloat("x", {2}, {-1.0f, 2.0f}));
  RunGraph(graph, rt);
  EXPECT_FALSE(rt.Has("t"));
  EXPECT_TRUE(rt.Has("y"));
  rt.Remove("y");
  rt.Put("x", Tensor::FromFloat("x", {2}, {-3.0f, 4.0f}));
  RunGraph(graph, rt);
  EXPECT_FALSE(rt.Has("t"));
  EXPECT_TRUE(rt.Has("y"));
  // Cached plan still the same instance after both runs.
  EXPECT_EQ(&rt.GetExecutionPlan(graph), &plan1);
}

} // namespace Test