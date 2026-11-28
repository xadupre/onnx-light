// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/shapes/shape_inference.h"
#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_extensions/patterns/algebra/common_pattern.h"
#include "onnx_extensions/patterns/collections/sequence_pattern.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/defs/function.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"
#include "test_case_utils.h"

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::CollectTestCases;
using core::backend_test::DataSet;
using core::backend_test::DefaultOpset;
using core::backend_test::ExpectBackendCaseTensorEqual;
using core::backend_test::ExpectTensorBitEqual;
using core::backend_test::TestCase;
using core::backend_test::TestCaseUnloadGuard;
using core::runtime::ExecutionPlan;
using core::runtime::Map;
using core::runtime::RegisterModelFunctions;
using core::runtime::RuntimeContext;
using core::runtime::RuntimeSession;
using core::runtime::TensorFromProto;
using onnx_kernels::kernel::KernelContext;

namespace Test {

using core::runtime::DataType;
using core::runtime::Tensor;

namespace {

constexpr int64_t kFallbackDefaultOpsetVersion = 18;

int64_t GetDefaultOpsetVersion(const ModelProto &model) {
  for (const auto &opset : model.ref_opset_import()) {
    if (opset.ref_domain().empty()) {
      return opset.version();
    }
  }
  return kFallbackDefaultOpsetVersion;
}

// Registers `model`'s local functions in `rt`, seeds `model.graph()`'s
// initializers into `rt`, and runs the graph by building its ExecutionPlan
// and driving it through a fresh RuntimeSession. This is what the removed
// `RunModel` used to do internally; every call site now builds the
// ExecutionPlan/RuntimeSession itself.
void RunModelViaSession(const ModelProto &model, RuntimeContext &rt) {
  RegisterModelFunctions(model, rt);
  const GraphProto &graph = model.ref_graph();
  const auto &inits = graph.initializer();
  for (size_t i = 0; i < inits.size(); ++i) {
    const TensorProto &tp = inits[i];
    const std::string init_name = tp.name();
    if (!rt.Has(init_name)) {
      rt.Set(init_name, TensorFromProto(tp), core::runtime::RuntimeEventKind::kInitializer);
    }
  }
  RuntimeSession session(model);
  session.Run(rt);
}

TEST(BackendRunModel, TopKRejectsNonPositiveK) {
  auto cases = CollectTestCases("TopK");
  size_t checked = 0;
  for (TestCase &tc : cases) {
    if (tc.name.find("test_cc_top_k_positive_k_opset_") != 0) {
      continue;
    }
    SCOPED_TRACE(tc.name);
    TestCaseUnloadGuard unload_guard(tc);
    const auto &model = tc.model();
    const auto &inputs = tc.data_sets()[0].inputs;
    for (const int64_t k : {0, -1}) {
      SCOPED_TRACE(k);
      RuntimeContext rt;
      rt.tensors()[inputs[0].name] = inputs[0];
      rt.tensors()[inputs[1].name] = Tensor::FromInt64(inputs[1].name, {1}, {k});
      EXPECT_THROW(RunModelViaSession(model, rt), std::invalid_argument);
    }
    ++checked;
  }
  EXPECT_EQ(checked, 3u);
}

TEST(BackendRunModel, ShapeBasedIdentityPreservesDynamicSlice) {
  for (bool computed_end : {false, true}) {
    SCOPED_TRACE(computed_end);
    core::builder::GraphBuilder source("dynamic_slice", [](const std::string &op_type) {
      return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
    });
    source.SetOpsetVersion("", 21);
    source.MakeInput("X", core::symbolic::TensorType::kFloat, {core::symbolic::SymDim(10)});
    source.MakeInitializer(MakeInitializer<int64_t>("zero", {1}, {0}));
    if (computed_end) {
      source.MakeInput("mask", core::symbolic::TensorType::kInt64, {core::symbolic::SymDim(20)});
      source.MakeNode("NonZero", {"mask"}, {"indices"});
      source.MakeNode("Size", {"indices"}, {"count"});
      source.MakeNode("Unsqueeze", {"count", "zero"}, {"end"});
    } else {
      source.MakeInput("end", core::symbolic::TensorType::kInt64, {core::symbolic::SymDim(1)});
    }
    source.MakeNode("Slice", {"X", "zero", "end", "zero"}, {"Y"});
    source.MakeOutput("Y");
    const ModelProto original = source.ToModel();
    core::builder::GraphBuilder builder(original, [](const std::string &op_type) {
      return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
    });
    std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
    patterns.push_back(std::make_unique<onnx_patterns::ShapeBasedIdentityPattern>());
    core::builder::GraphGraph graph(builder, std::move(patterns));
    graph.Optimize();
    const ModelProto optimized = builder.ToModel();
    ASSERT_EQ(optimized.graph().node().size(), original.graph().node().size());
    const NodeProto &slice = optimized.graph().node(optimized.graph().node_size() - 1);
    EXPECT_EQ(slice.op_type().value(), "Slice");
    EXPECT_EQ(slice.output(0), "Y");
    ASSERT_EQ(optimized.graph().output(0).type().tensor_type().shape().dim_size(), 1);
    EXPECT_FALSE(optimized.graph().output(0).type().tensor_type().shape().dim(0).has_dim_value());

    for (int64_t end : {0, 3, 10, 20}) {
      SCOPED_TRACE(end);
      for (const ModelProto *model : {&original, &optimized}) {
        RuntimeContext rt(KernelContext(DefaultOpset(21)));
        rt.tensors()["X"] = Tensor::FromFloat("X", {10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        if (computed_end) {
          std::vector<int64_t> mask(20, 0);
          for (int64_t i = 0; i < end; ++i) {
            mask[static_cast<size_t>(i)] = 1;
          }
          rt.tensors()["mask"] = Tensor::FromInt64("mask", {20}, mask);
        } else {
          rt.tensors()["end"] = Tensor::FromInt64("end", {1}, {end});
        }
        RunModelViaSession(*model, rt);
        const Tensor &output = rt.tensors().at("Y");
        const int64_t expected_size = std::min(end, int64_t{10});
        ASSERT_EQ(output.shape, std::vector<int64_t>{expected_size});
        ASSERT_EQ(output.element_count(), expected_size);
        for (int64_t i = 0; i < expected_size; ++i) {
          EXPECT_FLOAT_EQ(output.AsFloat()[i], static_cast<float>(i));
        }
      }
    }
  }
}

} // namespace

TEST(BackendRunModel, SupportedDtypeBatchIsLazyAndReloadable) {
  // Selects the one-per-type batch, not additional reference cases sharing its comparison policy.
  auto cases = core::backend_test::CollectTestCasesByName("^test_cc_supported_dtype_.*_[A-Z0-9]+$");
  ASSERT_EQ(cases.size(), 97u);
  std::set<std::pair<std::string, int32_t>> memberships;
  std::set<std::string> operators;
  for (TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);
    EXPECT_FALSE(tc.materialized());
    const auto &model = tc.model();
    const std::string op = model.graph().node(0).op_type().value();
    const auto &ds = tc.data_sets().at(0);
    ASSERT_FALSE(ds.inputs.empty());
    ASSERT_NO_THROW(checker::check_model(model, true));
    if (op == "MeanVarianceNormalization" && ds.inputs[0].data_type == DataType::DOUBLE) {
      ModelProto native_shape_model = model;
      native_shape_model.mutable_graph()->mutable_output(0)->clear_type();
      ASSERT_NO_THROW(core::shapes::InferShapesModel(native_shape_model));
      EXPECT_TRUE(
          native_shape_model.graph().output(0).type().Equals(model.graph().output(0).type()));
    }
    ASSERT_FALSE(ds.outputs.empty());
    EXPECT_TRUE(memberships.emplace(op, ds.inputs[0].data_type).second);
    operators.insert(op);
    for (const Tensor &input : ds.inputs) {
      EXPECT_GT(input.element_count(), 0);
    }
    for (const Tensor &output : ds.outputs) {
      EXPECT_GT(output.element_count(), 0);
    }
    const Tensor expected = ds.outputs[0];
    tc.unload();
    EXPECT_FALSE(tc.materialized());
    ExpectTensorBitEqual(tc.data_sets().at(0).outputs[0], expected);
    tc.unload();
  }
  EXPECT_EQ(operators.size(), 41u);
  auto benchmarks = core::backend_test::CollectTestCasesByName(
      "^test_cc_supported_dtype_", false, core::backend_test::TestMode::BENCHMARK);
  EXPECT_TRUE(benchmarks.empty());
}

TEST(BackendRunModel, SupportedDtypeComparisonRetainsExactReferences) {
  const Tensor positive_zero = Tensor::FromDouble("", {1}, {0.0});
  const Tensor negative_zero = Tensor::FromDouble("", {1}, {-0.0});
  for (const std::string name : {"test_cc_existing", "test_cc_supported_dtype_Transpose_DOUBLE"}) {
    TestCase tc(name);
    tc.rtol = 1.0;
    tc.atol = 1.0;
    EXPECT_NONFATAL_FAILURE(ExpectBackendCaseTensorEqual(positive_zero, negative_zero, tc),
                            "Tensor mismatch at byte");
  }
  TestCase integer_case("test_cc_supported_dtype_MatMul_INT64");
  integer_case.rtol = 1.0;
  integer_case.atol = 1.0;
  const Tensor one = Tensor::FromInt64("", {1}, {1});
  const Tensor two = Tensor::FromInt64("", {1}, {2});
  EXPECT_NONFATAL_FAILURE(ExpectBackendCaseTensorEqual(one, two, integer_case),
                          "Tensor mismatch at byte");
  TestCase float_case("test_cc_supported_dtype_MeanVarianceNormalization_DOUBLE");
  float_case.rtol = 1e-12;
  float_case.atol = 1e-14;
  ExpectBackendCaseTensorEqual(Tensor::FromDouble("", {1}, {1.0 - 1e-13}),
                               Tensor::FromDouble("", {1}, {1.0}), float_case);
  EXPECT_NONFATAL_FAILURE(ExpectBackendCaseTensorEqual(Tensor::FromDouble("", {1}, {0.99}),
                                                       Tensor::FromDouble("", {1}, {1.0}),
                                                       float_case),
                          "exceeds");
}

TEST(BackendRunModel, SoftmaxReferencesRejectCollapsedSmallResults) {
  auto cases = core::backend_test::CollectTestCasesByName(
      "^test_cc_reference_(logsoftmax_small_log_probability|sce_small_loss)_");
  ASSERT_FALSE(cases.empty());
  for (const TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);
    EXPECT_EQ(tc.atol, 0.0);
    EXPECT_EQ(tc.rtol, 1e-3);
    const Tensor expected = Tensor::FromDouble("", {1}, {3e-8});
    ExpectBackendCaseTensorEqual(Tensor::FromDouble("", {1}, {3.001e-8}), expected, tc);
    EXPECT_NONFATAL_FAILURE(
        ExpectBackendCaseTensorEqual(Tensor::FromDouble("", {1}, {0.0}), expected, tc), "");
  }
}

TEST(BackendRunModel, GroupedMatMulFunctionDecomposition) {
  RegisterAllOnnxOperatorSchemas();
  const auto *schema = OpSchemaRegistry::Schema("GroupedMatMul", 29, "");
  ASSERT_NE(schema, nullptr);
  auto cases = CollectTestCases("GroupedMatMul");
  ASSERT_FALSE(cases.empty());
  for (auto &tc : cases) {
    SCOPED_TRACE(tc.name);
    TestCaseUnloadGuard guard(tc);
    ModelProto model = tc.model();
    NodeProto node = model.graph().node(0);
    FunctionBodyBuildContextImpl context(node);
    FunctionProto function;
    ASSERT_TRUE(schema->BuildContextDependentFunction(context, function));
    model.mutable_graph()->clear_node();
    FunctionExpandHelper(node, function, *model.mutable_graph());
    RuntimeContext rt;
    for (const auto &input : tc.data_sets()[0].inputs) {
      rt.Set(input.name, input);
    }
    ASSERT_NO_THROW(RunModelViaSession(model, rt));
    const auto &expected = tc.data_sets()[0].outputs[0];
    ExpectTensorBitEqual(rt.Get(expected.name), expected);
  }
}

// Covers onnx/onnx#8489 and independent boolean/float causal-mask backend cases.
TEST(BackendRunModel, AttentionCausalMaskExpanded) {
  std::vector<std::string> names = {
      "test_cc_attention_4d_attn_mask_4d_causal", "test_cc_attention_4d_attn_mask_3d_causal",
      "test_cc_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal",
      "test_cc_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal"};
  for (int mask_rank : {2, 3, 4}) {
    for (const char *mask_type : {"bool", "float"}) {
      for (const char *suffix : {"", "_with_past"}) {
        names.push_back("test_cc_attention_causal_mask_composition_" + std::to_string(mask_rank) +
                        "d_" + mask_type + suffix);
      }
    }
  }
  auto cases = CollectTestCases("Attention");
  for (const auto &name : names) {
    SCOPED_TRACE(name);
    auto it = std::find_if(cases.begin(), cases.end(),
                           [&](const TestCase &test_case) { return test_case.name == name; });
    ASSERT_NE(it, cases.end());
    TestCaseUnloadGuard unload_guard(*it);
    const ModelProto &model = it->model();
    const NodeProto &node = model.graph().node(0);
    std::vector<TypeProto> input_types;
    for (const auto &input : model.graph().input()) {
      input_types.push_back(input.type());
    }
    ASSERT_EQ(input_types.size(), static_cast<size_t>(node.input_size()));
    ASSERT_FALSE(it->data_sets().empty());
    for (int version : {23, 24}) {
      SCOPED_TRACE(version);
      const auto *schema = OpSchemaRegistry::Schema("Attention", version, "");
      ASSERT_NE(schema, nullptr);
      FunctionBodyBuildContextImpl ctx(node, input_types);
      FunctionProto function;
      schema->BuildContextDependentFunction(ctx, function);
      ASSERT_GT(function.node_size(), 0);

      ModelProto expanded = model;
      expanded.mutable_graph()->clear_node();
      FunctionExpandHelper(node, function, *expanded.mutable_graph());
      expanded.clear_opset_import();
      for (const auto &opset : function.opset_import()) {
        *expanded.add_opset_import() = opset;
      }
      for (const auto &expanded_node : expanded.graph().node()) {
        ASSERT_NE(expanded_node.op_type().value(), "Attention");
      }
      for (const DataSet &ds : it->data_sets()) {
        RuntimeContext rt;
        for (const Tensor &input : ds.inputs) {
          rt.tensors()[input.name] = input;
        }
        RunModelViaSession(expanded, rt);
        for (const Tensor &expected : ds.outputs) {
          SCOPED_TRACE(expected.name);
          ASSERT_TRUE(rt.Has(expected.name));
          const Tensor &actual = rt.Get(expected.name);
          ASSERT_EQ(actual.data_type, DataType::FLOAT);
          ASSERT_EQ(actual.shape, expected.shape);
          for (int64_t i = 0; i < expected.element_count(); ++i) {
            const float value = expected.AsFloat()[i];
            if (std::isinf(value)) {
              EXPECT_EQ(actual.AsFloat()[i], value);
            } else {
              EXPECT_NEAR(actual.AsFloat()[i], value, 1e-5f + 1e-5f * std::abs(value));
            }
          }
        }
      }
    }
  }
}

TEST(BackendRunModel, Float4CastCasesExcludeNaN) {
  auto cases = core::backend_test::CollectTestCasesByName(
      "^test_cc_cast(like)?_(FLOAT(16)?_to_FLOAT4E2M1|FLOAT4E2M1_to_FLOAT(16)?)$");
  ASSERT_EQ(cases.size(), 8u);
  const KernelContext ctx{DefaultOpset(23)};
  const onnx_kernels::kernel::Cast cast{ctx};
  for (TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);
    TestCaseUnloadGuard unload_guard(tc);
    for (const DataSet &ds : tc.data_sets()) {
      ASSERT_FALSE(ds.inputs.empty());
      ASSERT_EQ(ds.outputs.size(), 1u);
      Tensor input = cast(ds.inputs[0], static_cast<int32_t>(DataType::FLOAT));
      Tensor output = cast(ds.outputs[0], static_cast<int32_t>(DataType::FLOAT));
      ASSERT_EQ(input.element_count(), 15);
      ASSERT_EQ(output.element_count(), 15);
      for (int64_t i = 0; i < input.element_count(); ++i) {
        EXPECT_FALSE(std::isnan(input.AsFloat()[i]));
      }
      EXPECT_FLOAT_EQ(output.AsFloat()[8], -6.0f);
    }
  }
}

TEST(BackendRunModel, WindowSmallTypesRemainFinite) {
  for (const std::string op : {"HannWindow", "HammingWindow"}) {
    size_t checked = 0;
    for (auto &tc : CollectTestCases(op)) {
      if (tc.name.find("_symmetric") != std::string::npos) {
        continue;
      }
      const auto &ds = tc.data_sets()[0];
      const auto dtype = ds.outputs[0].data_type;
      if (dtype == DataType::FLOAT) {
        continue;
      }
      SCOPED_TRACE(tc.name);
      RuntimeContext rt;
      rt.Set(ds.inputs[0].name, ds.inputs[0]);
      ASSERT_NO_THROW(RunModelViaSession(tc.model(), rt));
      const Tensor &result = rt.Get(ds.outputs[0].name);
      ASSERT_EQ(result.data_type, dtype);
      const int64_t size = dtype == DataType::INT8 ? 64 : (op == "HannWindow" ? 32768 : 16384);
      ASSERT_EQ(result.shape, std::vector<int64_t>{size});
      if (dtype == DataType::INT8) {
        EXPECT_EQ(result.AsInt8()[size / 2], 1);
      } else {
        const auto decode = dtype == DataType::FLOAT16 ? core::runtime::Float16BitsToFloat
                                                       : core::runtime::Bfloat16BitsToFloat;
        const auto encode = dtype == DataType::FLOAT16 ? core::runtime::FloatToFloat16Bits
                                                       : core::runtime::FloatToBfloat16Bits;
        for (int64_t i = 0; i < size; ++i) {
          uint16_t bits;
          std::memcpy(&bits, result.bytes() + static_cast<size_t>(i) * sizeof(bits), sizeof(bits));
          ASSERT_TRUE(std::isfinite(decode(bits))) << "index=" << i;
          if (i == size * 3 / 4) {
            const float expected = op == "HannWindow" ? 0.5f : 25.0f / 46.0f;
            EXPECT_EQ(bits, encode(expected));
          }
        }
      }
      ++checked;
    }
    EXPECT_EQ(checked, 3u);
  }
}

TEST(BackendRunModel, WindowSmallTypesRetainValidation) {
  for (const std::string op : {"HannWindow", "HammingWindow"}) {
    for (auto &tc : CollectTestCases(op)) {
      if (tc.name !=
          (op == "HannWindow" ? "test_cc_hannwindow_float16" : "test_cc_hammingwindow_float16")) {
        continue;
      }
      SCOPED_TRACE(tc.name);
      const auto &input = tc.data_sets()[0].inputs[0];
      for (const Tensor &invalid :
           {Tensor::FromInt32(input.name, {}, {-1}), Tensor::FromInt32(input.name, {1}, {8})}) {
        RuntimeContext rt;
        rt.Set(input.name, invalid);
        EXPECT_THROW(RunModelViaSession(tc.model(), rt), std::invalid_argument);
      }
      ModelProto invalid_model = tc.model();
      invalid_model.mutable_graph()->mutable_node(0)->mutable_attribute(0)->set_i(
          (int64_t{1} << 32) + static_cast<int64_t>(DataType::FLOAT));
      RuntimeContext rt;
      rt.Set(input.name, input);
      EXPECT_THROW(RunModelViaSession(invalid_model, rt), std::invalid_argument);
    }
  }
}

TEST(BackendRunModel, DynamicQuantizeLinearZeroInputExpanded) {
  auto cases = CollectTestCases("DynamicQuantizeLinear");
  auto it = std::find_if(cases.begin(), cases.end(), [](const TestCase &test_case) {
    return test_case.name == "test_dynamicquantizelinear_zero_input";
  });
  ASSERT_NE(it, cases.end());
  TestCaseUnloadGuard unload_guard(*it);
  const ModelProto &model = it->model();
  const NodeProto &node = model.graph().node(0);
  const auto *schema = OpSchemaRegistry::Schema("DynamicQuantizeLinear", 11, "");
  ASSERT_NE(schema, nullptr);
  const FunctionProto *function = schema->GetFunction(11);
  ASSERT_NE(function, nullptr);

  ModelProto expanded = model;
  expanded.mutable_graph()->clear_node();
  FunctionExpandHelper(node, *function, *expanded.mutable_graph());
  expanded.clear_opset_import();
  for (const auto &opset : function->opset_import()) {
    *expanded.add_opset_import() = opset;
  }
  for (const auto &input : {0.0f, -0.0f}) {
    SCOPED_TRACE(std::signbit(input));
    const DataSet &ds = it->data_sets()[0];
    RuntimeContext rt;
    rt.tensors()[ds.inputs[0].name] =
        Tensor::FromFloat(ds.inputs[0].name, {2, 3}, std::vector<float>(6, input));
    RunModelViaSession(expanded, rt);
    for (const Tensor &expected : ds.outputs) {
      SCOPED_TRACE(expected.name);
      ASSERT_TRUE(rt.Has(expected.name));
      ExpectTensorBitEqual(rt.Get(expected.name), expected);
    }
  }
}

// SequenceMap is a sequence-typed control-flow op: its outputs are
// :cpp:struct:`Sequence` values held in ``RuntimeContext::sequences``,
// not :cpp:struct:`Tensor` values in ``RuntimeContext::tensors``, and
// every backend test case wraps the SequenceMap node in a 2-node
// graph (``SequenceConstruct`` + ``SequenceMap``). The
// per-case expected stacked tensor (registered in
// ``cases_sequence_map.cc``) is compared against the actual output
// sequence by re-stacking the latter through
// :cpp:class:`kernel::SequenceConstruct`.
TEST(BackendRunModel, SequenceMap) {
  std::vector<TestCase> cases = CollectTestCases("SequenceMap");
  ASSERT_FALSE(cases.empty()) << "No backend test cases found for SequenceMap";

  std::size_t executed = 0;
  for (TestCase &tc : cases) {
    TestCaseUnloadGuard unload_guard(tc);
    SCOPED_TRACE(tc.name);
    const onnx_kernels::kernel::KernelContext kctx(
        DefaultOpset(GetDefaultOpsetVersion(tc.model())));

    for (const DataSet &ds : tc.data_sets()) {
      RuntimeContext rt;
      for (const Tensor &t : ds.inputs) {
        rt.Set(t.name, t);
      }
      ASSERT_NO_THROW(RunModelViaSession(tc.model(), rt))
          << "Running the model threw for case " << tc.name;

      // Each expected output is the stacked-tensor materialisation of
      // the corresponding output sequence (see
      // ``cases_sequence_map.cc``); re-stack the actual sequence and
      // compare bit-exactly.
      for (const Tensor &expected : ds.outputs) {
        ASSERT_TRUE(rt.HasSequence(expected.name))
            << "Missing output sequence '" << expected.name << "' for case " << tc.name;
        const auto &out_seq = rt.GetSequence(expected.name);
        const std::vector<Tensor> values(out_seq.values.begin(), out_seq.values.end());
        Tensor actual = onnx_kernels::kernel::SequenceConstruct(kctx)(values);
        actual.name = expected.name;
        ExpectTensorBitEqual(actual, expected);
      }
      ++executed;
    }
  }
  EXPECT_GT(executed, 0u) << "No SequenceMap test cases exercised.";
}

// SplitToSequence is a sequence-typed op: its output is a
// :cpp:struct:`Sequence` value held in ``RuntimeContext::sequences``
// rather than a :cpp:struct:`Tensor` in ``RuntimeContext::tensors``, so
// it needs a dedicated check. Each backend case (see
// ``cases_split_to_sequence.cc``) registers its expected output as the
// stacked-tensor materialisation of the output sequence, so the actual
// output sequence is re-stacked through
// :cpp:class:`kernel::SequenceConstruct` and compared bit-exactly. Running
// through :cpp:func:`RunModelViaSession` drives the kernel with a real
// :cpp:class:`RuntimeContext`, exercising the runtime-allocator path for the
// kernel's split-size buffer.
TEST(BackendRunModel, SplitToSequence) {
  std::vector<TestCase> cases = CollectTestCases("SplitToSequence");
  ASSERT_FALSE(cases.empty()) << "No backend test cases found for SplitToSequence";

  std::size_t executed = 0;
  for (TestCase &tc : cases) {
    TestCaseUnloadGuard unload_guard(tc);
    const auto &graph = tc.model().ref_graph();
    if (graph.ref_node().size() != 1u || graph.ref_node()[0].ref_op_type() != "SplitToSequence") {
      continue;
    }
    SCOPED_TRACE(tc.name);
    const onnx_kernels::kernel::KernelContext kctx(
        DefaultOpset(GetDefaultOpsetVersion(tc.model())));

    for (const DataSet &ds : tc.data_sets()) {
      RuntimeContext rt;
      for (const Tensor &t : ds.inputs) {
        rt.Set(t.name, t);
      }
      ASSERT_NO_THROW(RunModelViaSession(tc.model(), rt))
          << "Running the model threw for case " << tc.name;

      // Each expected output is the stacked-tensor materialisation of the
      // corresponding output sequence (see ``cases_split_to_sequence.cc``);
      // re-stack the actual sequence and compare bit-exactly.
      for (const Tensor &expected : ds.outputs) {
        ASSERT_TRUE(rt.HasSequence(expected.name))
            << "Missing output sequence '" << expected.name << "' for case " << tc.name;
        const auto &out_seq = rt.GetSequence(expected.name);
        const std::vector<Tensor> values(out_seq.values.begin(), out_seq.values.end());
        Tensor actual = onnx_kernels::kernel::SequenceConstruct(kctx)(values);
        actual.name = expected.name;
        ExpectTensorBitEqual(actual, expected);
      }
      ++executed;
    }
  }
  EXPECT_GT(executed, 0u) << "No SplitToSequence test cases exercised.";
}

TEST(BackendRunModel, SplitToSequenceScalarRemainderSequenceAt) {
  std::vector<TestCase> cases = CollectTestCases("SplitToSequence");
  std::size_t checked = 0;
  for (TestCase &tc : cases) {
    if (tc.name != "test_cc_split_to_sequence_at_scalar_remainder" &&
        tc.name != "test_cc_split_to_sequence_at_scalar_remainder_keepdims0") {
      continue;
    }
    SCOPED_TRACE(tc.name);
    TestCaseUnloadGuard unload_guard(tc);
    const ModelProto &original = tc.model();
    const auto &original_nodes = original.ref_graph().ref_node();
    ASSERT_EQ(original_nodes.size(), 3u);
    EXPECT_EQ(original_nodes[0].ref_op_type(), "SplitToSequence");
    EXPECT_EQ(original_nodes[1].ref_op_type(), "SequenceAt");
    EXPECT_EQ(original_nodes[2].ref_op_type(), "SequenceAt");

    core::builder::GraphBuilder builder(original, [](const std::string &op_type) {
      return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
    });
    std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
    patterns.push_back(std::make_unique<onnx_patterns::SplitToSequenceSequenceAtPattern>());
    core::builder::GraphGraph graph(builder, std::move(patterns));
    ASSERT_FALSE(graph.Optimize().rewrites().empty());
    const ModelProto optimized = builder.ToModel();
    ASSERT_EQ(optimized.ref_graph().ref_node().size(), 1u);
    EXPECT_EQ(optimized.ref_graph().ref_node()[0].ref_op_type(), "Split");

    for (const ModelProto *model : {&original, &optimized}) {
      for (const DataSet &ds : tc.data_sets()) {
        RuntimeContext rt;
        for (const Tensor &input : ds.inputs) {
          rt.Set(input.name, input);
        }
        RunModelViaSession(*model, rt);
        ASSERT_EQ(ds.outputs.size(), 2u);
        for (const Tensor &expected : ds.outputs) {
          ASSERT_TRUE(rt.Has(expected.name));
          ExpectTensorBitEqual(rt.Get(expected.name), expected);
        }
      }
    }
    ++checked;
  }
  EXPECT_EQ(checked, 2u);
}

} // namespace Test
