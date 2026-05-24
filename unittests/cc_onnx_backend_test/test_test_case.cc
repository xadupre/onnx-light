// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/controlflow/include_controlflow_cases.h"
#include "onnx_backend_test/cases/generator/include_generator_cases.h"
#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/cases/object_detection/include_object_detection_cases.h"
#include "onnx_backend_test/cases/optional/include_optional_cases.h"
#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"
#include "onnx_backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/cases/text/include_text_cases.h"
#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTestCases;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Expect;
using onnx_backend_test::OpsetId;
using onnx_backend_test::Tensor;
using onnx_backend_test::TestCase;

namespace Test {

// ---------------------------------------------------------------------------
// Framework-level tests for the backend test_case harness itself. Tests that
// inspect or exercise per-operator cases live in cases/<subfolder>/*.cc,
// mirroring the layout of onnx_backend_test/cases/.
// ---------------------------------------------------------------------------

TEST(BackendTestCase, TensorFromFloatRoundTrip) {
  Tensor t = Tensor::FromFloat("a", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  EXPECT_EQ(t.name, "a");
  EXPECT_EQ(t.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  EXPECT_EQ(t.shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(t.element_count(), 6);
  EXPECT_EQ(t.element_size(), sizeof(float));
  const float *p = t.AsFloat();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(p[i], static_cast<float>(i + 1));
  }
}

TEST(BackendTestCase, TensorScalarHasSingleElement) {
  Tensor t = Tensor::FromFloat("s", {}, {3.5f});
  EXPECT_EQ(t.element_count(), 1);
  EXPECT_EQ(t.shape.size(), 0u);
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 3.5f);
}

TEST(BackendTestCase, TensorRejectsShapeValueMismatch) {
  EXPECT_THROW(Tensor::FromFloat("x", {2, 3}, {1.0f}), std::invalid_argument);
}

TEST(BackendTestCase, TensorAsRejectsWrongDtype) {
  Tensor t = Tensor::FromFloat("x", {2}, {1.0f, 2.0f});
  EXPECT_THROW((void)t.AsInt64(), std::invalid_argument);
}

TEST(BackendTestCase, TensorTemplatedFromAndAs) {
  // Float
  Tensor tf = Tensor::From<float>("f", {2}, {1.5f, 2.5f});
  EXPECT_EQ(tf.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  EXPECT_FLOAT_EQ(tf.As<float>()[0], 1.5f);
  EXPECT_FLOAT_EQ(tf.As<float>()[1], 2.5f);
  const Tensor &ctf = tf;
  EXPECT_FLOAT_EQ(ctf.As<float>()[0], 1.5f);

  // Double
  Tensor td = Tensor::From<double>("d", {3}, {1.0, 2.0, 3.0});
  EXPECT_EQ(td.data_type, static_cast<int32_t>(TensorProto::DataType::DOUBLE));
  EXPECT_DOUBLE_EQ(td.As<double>()[2], 3.0);

  // Int32
  Tensor ti = Tensor::From<int32_t>("i", {2}, {-7, 8});
  EXPECT_EQ(ti.data_type, static_cast<int32_t>(TensorProto::DataType::INT32));
  EXPECT_EQ(ti.As<int32_t>()[0], -7);

  // Int64
  Tensor tl = Tensor::From<int64_t>("l", {1}, {1234567890123LL});
  EXPECT_EQ(tl.data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
  EXPECT_EQ(tl.As<int64_t>()[0], 1234567890123LL);

  // Wrong-type access throws via templated As<T>
  EXPECT_THROW((void)tf.As<int64_t>(), std::invalid_argument);
  EXPECT_THROW((void)ctf.As<double>(), std::invalid_argument);

  // Shape/value mismatch throws via templated From<T>
  EXPECT_THROW((void)Tensor::From<float>("x", {2, 3}, {1.0f}), std::invalid_argument);
}

TEST(BackendTestCase, ExpectBuildsSingleNodeModel) {
  NodeProto node;
  node.set_op_type("Add");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");

  OpsetId osid("", 14);

  std::vector<TestCase> registry;
  Expect(node,
         {Tensor::FromFloat("x", {2}, {1.0f, 2.0f}), Tensor::FromFloat("y", {2}, {3.0f, 4.0f})},
         {Tensor::FromFloat("z", {2}, {4.0f, 6.0f})}, "test_dummy_add", {osid}, "backend-test",
         registry);

  ASSERT_EQ(registry.size(), 1u);
  const TestCase &tc = registry[0];
  EXPECT_EQ(tc.name, "test_dummy_add");
  EXPECT_EQ(tc.kind, "node");
  EXPECT_EQ(tc.data_sets.size(), 1u);
  EXPECT_EQ(tc.data_sets[0].inputs.size(), 2u);
  EXPECT_EQ(tc.data_sets[0].outputs.size(), 1u);

  const GraphProto &graph = tc.model.ref_graph();
  EXPECT_EQ(graph.ref_node().size(), 1u);
  EXPECT_EQ(graph.ref_input().size(), 2u);
  EXPECT_EQ(graph.ref_output().size(), 1u);
  EXPECT_EQ(std::string(graph.ref_node()[0].ref_op_type().data(),
                        graph.ref_node()[0].ref_op_type().size()),
            "Add");
  ASSERT_EQ(tc.model.ref_opset_import().size(), 1u);
  EXPECT_EQ(tc.model.ref_opset_import()[0].version(), 14);
}

TEST(BackendTestCase, DefaultOpsetUsesEmptyDomain) {
  OpsetId osid = DefaultOpset(17);
  EXPECT_EQ(osid.domain, "");
  EXPECT_EQ(osid.version, 17);
}

TEST(BackendTestCase, OpsetIdConstructsWithCustomDomain) {
  OpsetId osid("ai.onnx.ml", 3);
  EXPECT_EQ(osid.domain, "ai.onnx.ml");
  EXPECT_EQ(osid.version, 3);
}

TEST(BackendTestCase, ExpectRejectsArityMismatch) {
  NodeProto node;
  node.set_op_type("Abs");
  node.add_input("x");
  node.add_output("y");

  std::vector<TestCase> registry;
  EXPECT_THROW(Expect(node, /*inputs=*/{}, /*outputs=*/{Tensor::FromFloat("y", {1}, {1.0f})}, "bad",
                      {}, "backend-test", registry),
               std::invalid_argument);
}

TEST(BackendTestCase, CollectReturnsExpectedNames) {
  auto cases = CollectTestCases();
  ASSERT_GE(cases.size(), 3u);
  bool has_abs = false, has_add = false, has_add_bcast = false;
  for (const auto &c : cases) {
    if (c.name == "test_cc_abs")
      has_abs = true;
    if (c.name == "test_cc_add")
      has_add = true;
    if (c.name == "test_cc_add_bcast")
      has_add_bcast = true;
  }
  EXPECT_TRUE(has_abs);
  EXPECT_TRUE(has_add);
  EXPECT_TRUE(has_add_bcast);
}

TEST(BackendTestCase, PerSubfolderCollectorsAggregateIntoMain) {
  std::vector<TestCase> math_only;
  onnx_backend_test::CollectMathTestCases(math_only);
  std::vector<TestCase> object_detection_only;
  onnx_backend_test::CollectObjectDetectionTestCases(object_detection_only);
  std::vector<TestCase> logical_only;
  onnx_backend_test::CollectLogicalTestCases(logical_only);
  std::vector<TestCase> tensor_only;
  onnx_backend_test::CollectTensorTestCases(tensor_only);
  std::vector<TestCase> controlflow_only;
  onnx_backend_test::CollectControlflowTestCases(controlflow_only);
  std::vector<TestCase> generator_only;
  onnx_backend_test::CollectGeneratorTestCases(generator_only);
  std::vector<TestCase> optional_only;
  onnx_backend_test::CollectOptionalTestCases(optional_only);
  std::vector<TestCase> quantization_only;
  onnx_backend_test::CollectQuantizationTestCases(quantization_only);
  std::vector<TestCase> reduction_only;
  onnx_backend_test::CollectReductionTestCases(reduction_only);
  std::vector<TestCase> sequence_only;
  onnx_backend_test::CollectSequenceTestCases(sequence_only);
  std::vector<TestCase> text_only;
  onnx_backend_test::CollectTextTestCases(text_only);
  std::vector<TestCase> traditionalml_only;
  onnx_backend_test::CollectTraditionalMLTestCases(traditionalml_only);

  EXPECT_FALSE(math_only.empty());
  EXPECT_FALSE(object_detection_only.empty());
  EXPECT_FALSE(logical_only.empty());
  EXPECT_FALSE(tensor_only.empty());
  EXPECT_FALSE(controlflow_only.empty());
  EXPECT_FALSE(generator_only.empty());
  EXPECT_FALSE(optional_only.empty());
  EXPECT_FALSE(quantization_only.empty());
  EXPECT_FALSE(reduction_only.empty());
  EXPECT_FALSE(sequence_only.empty());
  EXPECT_FALSE(text_only.empty());
  EXPECT_FALSE(traditionalml_only.empty());

  const auto all = CollectTestCases();
  EXPECT_EQ(all.size(), math_only.size() + logical_only.size() + tensor_only.size() +
                            controlflow_only.size() + generator_only.size() +
                            object_detection_only.size() + optional_only.size() +
                            quantization_only.size() + reduction_only.size() +
                            sequence_only.size() + text_only.size() + traditionalml_only.size());
}

} // namespace Test
