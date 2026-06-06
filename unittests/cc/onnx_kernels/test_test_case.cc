// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/controlflow/include_controlflow_cases.h"
#include "onnx_backend_test/cases/generator/include_generator_cases.h"
#include "onnx_backend_test/cases/image/include_image_cases.h"
#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/cases/object_detection/include_object_detection_cases.h"
#include "onnx_backend_test/cases/optional/include_optional_cases.h"
#include "onnx_backend_test/cases/preview/include_preview_cases.h"
#include "onnx_backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"
#include "onnx_backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/cases/text/include_text_cases.h"
#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_backend_test/cases/training/include_training_cases.h"
#include "onnx_backend_test/cases_for_shapes/empty_shape/include_empty_shape_cases.h"
#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_backend_test/cases_numerical/nan_inf/include_nan_inf_cases.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTestCases;
using onnx_backend_test::CollectTestCasesByName;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Expect;
using onnx_backend_test::OpsetId;
using onnx_kernels::Tensor;
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
  EXPECT_EQ(t.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
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
  EXPECT_EQ(tf.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  EXPECT_FLOAT_EQ(tf.As<float>()[0], 1.5f);
  EXPECT_FLOAT_EQ(tf.As<float>()[1], 2.5f);
  const Tensor &ctf = tf;
  EXPECT_FLOAT_EQ(ctf.As<float>()[0], 1.5f);

  // Double
  Tensor td = Tensor::From<double>("d", {3}, {1.0, 2.0, 3.0});
  EXPECT_EQ(td.data_type, static_cast<int32_t>(onnx_kernels::DataType::DOUBLE));
  EXPECT_DOUBLE_EQ(td.As<double>()[2], 3.0);

  // Int32
  Tensor ti = Tensor::From<int32_t>("i", {2}, {-7, 8});
  EXPECT_EQ(ti.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT32));
  EXPECT_EQ(ti.As<int32_t>()[0], -7);

  // Int64
  Tensor tl = Tensor::From<int64_t>("l", {1}, {1234567890123LL});
  EXPECT_EQ(tl.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT64));
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
  EXPECT_EQ(tc.tag, "");
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

TEST(BackendTestCase, TagDefaultsToEmptyForOrdinaryCases) {
  std::vector<TestCase> registry;
  onnx_backend_test::CollectMathTestCases(registry);
  ASSERT_FALSE(registry.empty());
  for (const auto &tc : registry) {
    EXPECT_EQ(tc.tag, "") << "case: " << tc.name;
  }
}

TEST(BackendTestCase, TagIsEmptyShapeForEmptyShapeCases) {
  std::vector<TestCase> registry;
  onnx_backend_test::CollectEmptyShapeTestCases(registry);
  ASSERT_FALSE(registry.empty());
  for (const auto &tc : registry) {
    EXPECT_EQ(tc.tag, "empty_shape") << "case: " << tc.name;
  }
}

TEST(BackendTestCase, TagIsNanInfForNanInfCases) {
  std::vector<TestCase> registry;
  onnx_backend_test::CollectNanInfTestCases(registry);
  ASSERT_FALSE(registry.empty());
  for (const auto &tc : registry) {
    EXPECT_EQ(tc.tag, "nan_inf") << "case: " << tc.name;
  }
}

TEST(BackendTestCase, TagIsInferenceForShapeInferenceCases) {
  std::vector<TestCase> registry;
  onnx_backend_test::CollectShapeInferenceTestCases(registry);
  ASSERT_FALSE(registry.empty());
  for (const auto &tc : registry) {
    EXPECT_EQ(tc.tag, "inference") << "case: " << tc.name;
  }
}

TEST(BackendTestCase, TagDefaultsToDomainForNonDefaultDomainNode) {
  // When the node belongs to a non-default operator domain (e.g.
  // ``ai.onnx.ml``) and no explicit tag is supplied, ``Expect`` should
  // default the tag to the node's domain so the test case can be grouped
  // by domain. The empty domain and ``"ai.onnx"`` both refer to the
  // standard ONNX domain and keep the empty tag.
  NodeProto node;
  node.set_op_type("Binarizer");
  node.set_domain("ai.onnx.ml");
  node.add_input("x");
  node.add_output("y");

  std::vector<TestCase> registry;
  Expect(node, {Tensor::FromFloat("x", {2}, {1.0f, 2.0f})},
         {Tensor::FromFloat("y", {2}, {0.0f, 1.0f})}, "test_cc_domain_tag",
         {OpsetId("ai.onnx.ml", 1)}, "backend-test", registry);

  ASSERT_EQ(registry.size(), 1u);
  EXPECT_EQ(registry[0].tag, "ai.onnx.ml");
}

TEST(BackendTestCase, TagStaysEmptyForDefaultDomainNode) {
  NodeProto node;
  node.set_op_type("Add");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");

  std::vector<TestCase> registry;
  Expect(node,
         {Tensor::FromFloat("x", {2}, {1.0f, 2.0f}), Tensor::FromFloat("y", {2}, {3.0f, 4.0f})},
         {Tensor::FromFloat("z", {2}, {4.0f, 6.0f})}, "test_cc_default_tag", {DefaultOpset(14)},
         "backend-test", registry);

  ASSERT_EQ(registry.size(), 1u);
  EXPECT_EQ(registry[0].tag, "");
}

TEST(BackendTestCase, ExplicitTagOverridesDomainDefault) {
  // An explicitly-supplied tag must take precedence over the domain-derived
  // default, including for non-default-domain nodes.
  NodeProto node;
  node.set_op_type("Binarizer");
  node.set_domain("ai.onnx.ml");
  node.add_input("x");
  node.add_output("y");

  std::vector<TestCase> registry;
  Expect(node, {Tensor::FromFloat("x", {2}, {1.0f, 2.0f})},
         {Tensor::FromFloat("y", {2}, {0.0f, 1.0f})}, "test_cc_explicit_tag",
         {OpsetId("ai.onnx.ml", 1)}, "backend-test", registry, "inference");

  ASSERT_EQ(registry.size(), 1u);
  EXPECT_EQ(registry[0].tag, "inference");
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

TEST(BackendTestCase, CollectByNameSubstringMatchesAbs) {
  auto cases = CollectTestCasesByName("abs");
  ASSERT_FALSE(cases.empty());
  for (const auto &c : cases) {
    EXPECT_NE(c.name.find("abs"), std::string::npos) << "case: " << c.name;
  }
  bool has_abs = false;
  for (const auto &c : cases) {
    if (c.name == "test_cc_abs") {
      has_abs = true;
      break;
    }
  }
  EXPECT_TRUE(has_abs);
}

TEST(BackendTestCase, CollectByNameAnchoredRegex) {
  auto cases = CollectTestCasesByName("^test_cc_add(_|$)");
  ASSERT_FALSE(cases.empty());
  bool has_add = false, has_add_bcast = false;
  for (const auto &c : cases) {
    EXPECT_EQ(c.name.rfind("test_cc_add", 0), 0u) << "case: " << c.name;
    if (c.name == "test_cc_add")
      has_add = true;
    if (c.name == "test_cc_add_bcast")
      has_add_bcast = true;
  }
  EXPECT_TRUE(has_add);
  EXPECT_TRUE(has_add_bcast);
}

TEST(BackendTestCase, CollectByNameEmptyPatternReturnsAll) {
  auto all_cases = CollectTestCases();
  auto by_empty = CollectTestCasesByName("");
  EXPECT_EQ(all_cases.size(), by_empty.size());
}

TEST(BackendTestCase, CollectByNameNoMatchReturnsEmpty) {
  auto cases = CollectTestCasesByName("__definitely_no_such_case__");
  EXPECT_TRUE(cases.empty());
}

TEST(BackendTestCase, CollectByNameInvalidRegexThrows) {
  EXPECT_THROW(CollectTestCasesByName("("), std::regex_error);
}

TEST(BackendTestCase, PerSubfolderCollectorsAggregateIntoMain) {
  std::vector<TestCase> controlflow_only;
  onnx_backend_test::CollectControlflowTestCases(controlflow_only);
  EXPECT_FALSE(controlflow_only.empty());

  std::vector<TestCase> generator_only;
  onnx_backend_test::CollectGeneratorTestCases(generator_only);
  EXPECT_FALSE(generator_only.empty());

  std::vector<TestCase> image_only;
  onnx_backend_test::CollectImageTestCases(image_only);
  EXPECT_FALSE(image_only.empty());

  std::vector<TestCase> logical_only;
  onnx_backend_test::CollectLogicalTestCases(logical_only);
  EXPECT_FALSE(logical_only.empty());

  std::vector<TestCase> math_only;
  onnx_backend_test::CollectMathTestCases(math_only);
  EXPECT_FALSE(math_only.empty());

  std::vector<TestCase> nn_only;
  onnx_backend_test::CollectNNTestCases(nn_only);
  EXPECT_FALSE(nn_only.empty());

  std::vector<TestCase> object_detection_only;
  onnx_backend_test::CollectObjectDetectionTestCases(object_detection_only);
  EXPECT_FALSE(object_detection_only.empty());

  std::vector<TestCase> optional_only;
  onnx_backend_test::CollectOptionalTestCases(optional_only);
  EXPECT_FALSE(optional_only.empty());

  std::vector<TestCase> preview_only;
  onnx_backend_test::CollectPreviewTestCases(preview_only);
  EXPECT_FALSE(preview_only.empty());

  std::vector<TestCase> quantization_only;
  onnx_backend_test::CollectQuantizationTestCases(quantization_only);
  EXPECT_FALSE(quantization_only.empty());

  std::vector<TestCase> reduction_only;
  onnx_backend_test::CollectReductionTestCases(reduction_only);
  EXPECT_FALSE(reduction_only.empty());

  std::vector<TestCase> sequence_only;
  onnx_backend_test::CollectSequenceTestCases(sequence_only);
  EXPECT_FALSE(sequence_only.empty());

  std::vector<TestCase> tensor_only;
  onnx_backend_test::CollectTensorTestCases(tensor_only);
  EXPECT_FALSE(tensor_only.empty());

  std::vector<TestCase> text_only;
  onnx_backend_test::CollectTextTestCases(text_only);
  EXPECT_FALSE(text_only.empty());

  std::vector<TestCase> traditionalml_only;
  onnx_backend_test::CollectTraditionalMLTestCases(traditionalml_only);
  EXPECT_FALSE(traditionalml_only.empty());

  std::vector<TestCase> training_only;
  onnx_backend_test::CollectTrainingTestCases(training_only);
  EXPECT_FALSE(training_only.empty());

  std::vector<TestCase> empty_shape_only;
  onnx_backend_test::CollectEmptyShapeTestCases(empty_shape_only);
  EXPECT_FALSE(empty_shape_only.empty());

  std::vector<TestCase> shape_inference_only;
  onnx_backend_test::CollectShapeInferenceTestCases(shape_inference_only);
  EXPECT_FALSE(shape_inference_only.empty());

  std::vector<TestCase> nan_inf_only;
  onnx_backend_test::CollectNanInfTestCases(nan_inf_only);
  EXPECT_FALSE(nan_inf_only.empty());

  const auto all = CollectTestCases();
  EXPECT_EQ(all.size(), math_only.size() + logical_only.size() + tensor_only.size() +
                            controlflow_only.size() + generator_only.size() + image_only.size() +
                            object_detection_only.size() + optional_only.size() +
                            preview_only.size() + quantization_only.size() + reduction_only.size() +
                            sequence_only.size() + text_only.size() + traditionalml_only.size() +
                            training_only.size() + nn_only.size() + empty_shape_only.size() +
                            shape_inference_only.size() + nan_inf_only.size());
}

TEST(BackendTestCase, CollectTestCasesFilterByOpTypeKeepsOnlyMatchingOps) {
  // The unfiltered registry contains many ops; passing "Abs" must keep only
  // ``Abs`` cases (registered by RegisterAbsCases in the math category).
  const auto all = CollectTestCases();
  const auto abs_only = CollectTestCases("Abs");
  ASSERT_FALSE(all.empty());
  ASSERT_FALSE(abs_only.empty());
  EXPECT_LT(abs_only.size(), all.size());
  for (const auto &tc : abs_only) {
    ASSERT_FALSE(tc.model.ref_graph().ref_node().empty());
    const auto &op = tc.model.ref_graph().ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op.data(), op.size()), "Abs");
  }
}

TEST(BackendTestCase, CollectCategoryFilterByOpTypeReturnsOnlyMatchingCases) {
  // Per-category collectors honour the op_type filter too.
  std::vector<TestCase> add_only;
  onnx_backend_test::CollectMathTestCases(add_only, "Add");
  ASSERT_FALSE(add_only.empty());
  for (const auto &tc : add_only) {
    ASSERT_FALSE(tc.model.ref_graph().ref_node().empty());
    const auto &op = tc.model.ref_graph().ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op.data(), op.size()), "Add");
  }

  // Asking for an op that lives in a different category yields no cases.
  std::vector<TestCase> none_for_math;
  onnx_backend_test::CollectMathTestCases(none_for_math, "If");
  EXPECT_TRUE(none_for_math.empty());

  // Empty op_type (the default) is a no-op and returns every case.
  std::vector<TestCase> all_math;
  onnx_backend_test::CollectMathTestCases(all_math);
  EXPECT_GT(all_math.size(), add_only.size());
}

TEST(BackendTestCase, CollectTraditionalMLFilterFindsArrayFeatureExtractorCases) {
  std::vector<TestCase> array_feature_extractor_only;
  onnx_backend_test::CollectTraditionalMLTestCases(array_feature_extractor_only,
                                              "ArrayFeatureExtractor");
  ASSERT_FALSE(array_feature_extractor_only.empty());
  for (const auto &tc : array_feature_extractor_only) {
    ASSERT_FALSE(tc.model.ref_graph().ref_node().empty());
    const auto &op = tc.model.ref_graph().ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op.data(), op.size()), "ArrayFeatureExtractor");
  }
}

TEST(BackendTestCase, CollectTraditionalMLFilterFindsOneHotEncoderCases) {
  std::vector<TestCase> one_hot_only;
  onnx_backend_test::CollectTraditionalMLTestCases(one_hot_only, "OneHotEncoder");
  ASSERT_FALSE(one_hot_only.empty());
  for (const auto &tc : one_hot_only) {
    ASSERT_FALSE(tc.model.ref_graph().ref_node().empty());
    const auto &op = tc.model.ref_graph().ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op.data(), op.size()), "OneHotEncoder");
  }
}

TEST(BackendTestCase, CollectTraditionalMLFilterFindsZipMapCases) {
  std::vector<TestCase> zipmap_only;
  onnx_backend_test::CollectTraditionalMLTestCases(zipmap_only, "ZipMap");
  ASSERT_FALSE(zipmap_only.empty());
  for (const auto &tc : zipmap_only) {
    ASSERT_FALSE(tc.model.ref_graph().ref_node().empty());
    const auto &op = tc.model.ref_graph().ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op.data(), op.size()), "ZipMap");
    ASSERT_FALSE(tc.model.ref_graph().ref_output().empty());
    const TypeProto &out_type = tc.model.ref_graph().ref_output()[0].ref_type();
    ASSERT_TRUE(out_type.has_sequence_type());
    ASSERT_TRUE(out_type.ref_sequence_type().ref_elem_type().has_map_type());
  }
}

TEST(BackendTestCase, CollectTraditionalMLFilterFindsSVMClassifierCases) {
  std::vector<TestCase> svm_only;
  onnx_backend_test::CollectTraditionalMLTestCases(svm_only, "SVMClassifier");
  ASSERT_FALSE(svm_only.empty());
  for (const auto &tc : svm_only) {
    ASSERT_FALSE(tc.model.ref_graph().ref_node().empty());
    const auto &op = tc.model.ref_graph().ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op.data(), op.size()), "SVMClassifier");
  }
}

TEST(BackendTestCase, CollectTraditionalMLFilterFindsSVMRegressorCases) {
  std::vector<TestCase> svm_only;
  onnx_backend_test::CollectTraditionalMLTestCases(svm_only, "SVMRegressor");
  ASSERT_FALSE(svm_only.empty());
  for (const auto &tc : svm_only) {
    ASSERT_FALSE(tc.model.ref_graph().ref_node().empty());
    const auto &op = tc.model.ref_graph().ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op.data(), op.size()), "SVMRegressor");
  }
}

TEST(BackendTestCase, CollectTraditionalMLFilterFindsLinearClassifierCases) {
  std::vector<TestCase> only;
  onnx_backend_test::CollectTraditionalMLTestCases(only, "LinearClassifier");
  ASSERT_FALSE(only.empty());
  for (const auto &tc : only) {
    ASSERT_FALSE(tc.model.ref_graph().ref_node().empty());
    const auto &op = tc.model.ref_graph().ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op.data(), op.size()), "LinearClassifier");
  }
}

TEST(BackendTestCase, CollectTraditionalMLFilterFindsLinearRegressorCases) {
  std::vector<TestCase> only;
  onnx_backend_test::CollectTraditionalMLTestCases(only, "LinearRegressor");
  ASSERT_FALSE(only.empty());
  for (const auto &tc : only) {
    ASSERT_FALSE(tc.model.ref_graph().ref_node().empty());
    const auto &op = tc.model.ref_graph().ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op.data(), op.size()), "LinearRegressor");
  }
}

TEST(BackendTestCase, CollectTraditionalMLFilterFindsTreeEnsembleRegressorCases) {
  std::vector<TestCase> only;
  onnx_backend_test::CollectTraditionalMLTestCases(only, "TreeEnsembleRegressor");
  ASSERT_FALSE(only.empty());
  for (const auto &tc : only) {
    ASSERT_FALSE(tc.model.ref_graph().ref_node().empty());
    const auto &op = tc.model.ref_graph().ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op.data(), op.size()), "TreeEnsembleRegressor");
  }
}

TEST(BackendTestCase, CollectTraditionalMLFilterFindsTreeEnsembleClassifierCases) {
  std::vector<TestCase> only;
  onnx_backend_test::CollectTraditionalMLTestCases(only, "TreeEnsembleClassifier");
  ASSERT_FALSE(only.empty());
  for (const auto &tc : only) {
    ASSERT_FALSE(tc.model.ref_graph().ref_node().empty());
    const auto &op = tc.model.ref_graph().ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op.data(), op.size()), "TreeEnsembleClassifier");
  }
}

TEST(BackendTestCase, CollectTraditionalMLFilterFindsTreeEnsembleCases) {
  std::vector<TestCase> only;
  onnx_backend_test::CollectTraditionalMLTestCases(only, "TreeEnsemble");
  ASSERT_FALSE(only.empty());
  for (const auto &tc : only) {
    ASSERT_FALSE(tc.model.ref_graph().ref_node().empty());
    const auto &op = tc.model.ref_graph().ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op.data(), op.size()), "TreeEnsemble");
  }
}

TEST(BackendTestCase, CollectPreservesPreExistingEntries) {
  // Build a registry with a single dummy ``Add`` case, then run a per-category
  // collector that also filters by a different op: the pre-existing ``Add``
  // entry must survive even though it does not match the filter.
  std::vector<TestCase> registry;
  NodeProto node;
  node.set_op_type("Add");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");
  Expect(node,
         {Tensor::FromFloat("x", {2}, {1.0f, 2.0f}), Tensor::FromFloat("y", {2}, {3.0f, 4.0f})},
         {Tensor::FromFloat("z", {2}, {4.0f, 6.0f})}, "pre_existing_add", {DefaultOpset(14)},
         "backend-test", registry);
  ASSERT_EQ(registry.size(), 1u);

  // Collect ``If`` cases only — the existing ``Add`` entry must be untouched.
  onnx_backend_test::CollectControlflowTestCases(registry, "If");
  ASSERT_GE(registry.size(), 2u);
  EXPECT_EQ(registry[0].name, "pre_existing_add");
  for (size_t i = 1; i < registry.size(); ++i) {
    const auto &op = registry[i].model.ref_graph().ref_node()[0].ref_op_type();
    EXPECT_EQ(std::string(op.data(), op.size()), "If");
  }
}

TEST(BackendTestCase, CollectEmptyFilterReturnsAllCases) {
  std::vector<TestCase> all_math;
  onnx_backend_test::CollectMathTestCases(all_math);
  std::vector<TestCase> all_math_default;
  onnx_backend_test::CollectMathTestCases(all_math_default, "");
  EXPECT_EQ(all_math.size(), all_math_default.size());
}

} // namespace Test
