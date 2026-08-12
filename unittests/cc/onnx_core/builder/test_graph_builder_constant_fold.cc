// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_builder.h"

#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_helper.h"
#include "onnx_op/operator_sets.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

using core::runtime::RuntimeContext;
using core::runtime::Tensor;

// Schema provider backed by the built-in ONNX operator schemas.
core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, /*init_doc=*/false);
  };
}

core::symbolic::SymShape MakeShape(std::initializer_list<int64_t> dims) {
  core::symbolic::SymShape shape;
  for (int64_t d : dims) {
    shape.PushBack(core::symbolic::SymDim(d));
  }
  return shape;
}

// RAII helper that installs a couple of tiny process-wide custom kernels
// (float elementwise ``Add`` / ``Neg`` and an ``INT64`` ``Shape``) for the
// duration of a test and removes them afterwards so global state never leaks
// across tests. The kernels assume equally shaped float operands, which is all
// the constant-folding tests below exercise.
class ScopedFoldingKernels {
public:
  ScopedFoldingKernels() {
    core::runtime::RegisterGlobalCustomKernel(
        "", "Add", [](const NodeProto &node, RuntimeContext &rt) {
          const Tensor &a = rt.tensors().at(std::string(node.input(0)));
          const Tensor &b = rt.tensors().at(std::string(node.input(1)));
          const int64_t n = a.element_count();
          std::vector<float> out(static_cast<std::size_t>(n));
          const float *pa = a.AsFloat();
          const float *pb = b.AsFloat();
          for (int64_t i = 0; i < n; ++i) {
            out[static_cast<std::size_t>(i)] =
                pa[static_cast<std::size_t>(i)] + pb[static_cast<std::size_t>(i)];
          }
          rt.tensors()[std::string(node.output(0))] =
              Tensor::FromFloat(std::string(node.output(0)), a.shape, out);
        });
    core::runtime::RegisterGlobalCustomKernel(
        "", "Neg", [](const NodeProto &node, RuntimeContext &rt) {
          const Tensor &a = rt.tensors().at(std::string(node.input(0)));
          const int64_t n = a.element_count();
          std::vector<float> out(static_cast<std::size_t>(n));
          const float *pa = a.AsFloat();
          for (int64_t i = 0; i < n; ++i) {
            out[static_cast<std::size_t>(i)] = -pa[static_cast<std::size_t>(i)];
          }
          rt.tensors()[std::string(node.output(0))] =
              Tensor::FromFloat(std::string(node.output(0)), a.shape, out);
        });
    core::runtime::RegisterGlobalCustomKernel(
        "", "Shape", [](const NodeProto &node, RuntimeContext &rt) {
          const Tensor &a = rt.tensors().at(std::string(node.input(0)));
          std::vector<int64_t> dims(a.shape.begin(), a.shape.end());
          rt.tensors()[std::string(node.output(0))] = Tensor::FromInt64(
              std::string(node.output(0)), {static_cast<int64_t>(dims.size())}, dims);
        });
  }

  ~ScopedFoldingKernels() {
    core::runtime::UnregisterGlobalCustomKernel("", "Add");
    core::runtime::UnregisterGlobalCustomKernel("", "Neg");
    core::runtime::UnregisterGlobalCustomKernel("", "Shape");
  }

  ScopedFoldingKernels(const ScopedFoldingKernels &) = delete;
  ScopedFoldingKernels &operator=(const ScopedFoldingKernels &) = delete;
};

// Returns the initializer named ``name`` in ``builder`` or nullptr.
const TensorProto *FindInitializer(const core::builder::GraphBuilder &builder,
                                   const std::string &name) {
  for (const TensorProto &init : builder.Initializers()) {
    if (init.name().value() == name) {
      return &init;
    }
  }
  return nullptr;
}

// Returns whether the default-domain operator ``op_type`` already has a runtime
// kernel registered process-wide. A kernels-ON build links the real kernels, so
// the "missing kernel" scenario exercised below cannot be reproduced there; the
// affected tests skip themselves in that case and only run in a kernels-OFF
// build (or before any real kernel is registered).
bool HasRuntimeKernel(const std::string &op_type) {
  const std::string key = kDefaultOnnxDomain + ":" + op_type;
  return core::runtime::KernelDispatchTable().count(key) != 0 ||
         core::runtime::GlobalCustomKernels().count(key) != 0;
}

} // namespace

TEST(GraphBuilderConstantFold, FoldsConstantWeightNodeIntoInitializer) {
  ScopedFoldingKernels kernels;
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(MakeInitializer<float>("a", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f}));
  const std::vector<std::string> sum = builder.MakeNode("Add", {"a", "b"}, {"s"});
  builder.MakeOutput(sum[0]);

  EXPECT_EQ(builder.ConstantFold(), 1u);
  EXPECT_EQ(builder.Nodes().size(), 0u);
  const TensorProto *folded = FindInitializer(builder, "s");
  ASSERT_NE(folded, nullptr);
  const core::runtime::Tensor tensor = core::runtime::TensorFromProto(*folded);
  ASSERT_EQ(tensor.element_count(), 4);
  const float *values = tensor.AsFloat();
  EXPECT_FLOAT_EQ(values[0], 11.0f);
  EXPECT_FLOAT_EQ(values[1], 22.0f);
  EXPECT_FLOAT_EQ(values[2], 33.0f);
  EXPECT_FLOAT_EQ(values[3], 44.0f);
}

TEST(GraphBuilderConstantFold, FoldsChainOfConstantNodesInOnePass) {
  ScopedFoldingKernels kernels;
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(MakeInitializer<float>("a", {2}, {1.0f, 2.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {2}, {3.0f, 4.0f}));
  const std::vector<std::string> sum = builder.MakeNode("Add", {"a", "b"}, {"s"});
  const std::vector<std::string> neg = builder.MakeNode("Neg", {sum[0]}, {"n"});
  builder.MakeOutput(neg[0]);

  EXPECT_EQ(builder.ConstantFold(), 2u);
  EXPECT_EQ(builder.Nodes().size(), 0u);
  const TensorProto *folded = FindInitializer(builder, "n");
  ASSERT_NE(folded, nullptr);
  const core::runtime::Tensor tensor = core::runtime::TensorFromProto(*folded);
  ASSERT_EQ(tensor.element_count(), 2);
  const float *values = tensor.AsFloat();
  EXPECT_FLOAT_EQ(values[0], -4.0f);
  EXPECT_FLOAT_EQ(values[1], -6.0f);
}

TEST(GraphBuilderConstantFold, DoesNotFoldNodeDependingOnGraphInput) {
  ScopedFoldingKernels kernels;
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  builder.MakeInitializer(MakeInitializer<float>("b", {2}, {3.0f, 4.0f}));
  const std::vector<std::string> sum = builder.MakeNode("Add", {"x", "b"}, {"s"});
  builder.MakeOutput(sum[0]);

  EXPECT_EQ(builder.ConstantFold(), 0u);
  EXPECT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(FindInitializer(builder, "s"), nullptr);
}

TEST(GraphBuilderConstantFold, FoldsNodeWhoseOutputIsGraphOutput) {
  // A constant node whose output is *only* a declared graph output (with no
  // other consumer) is folded into an initializer carrying that name; the value
  // remains a valid graph output.
  ScopedFoldingKernels kernels;
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(MakeInitializer<float>("a", {2}, {1.0f, 2.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {2}, {3.0f, 4.0f}));
  const std::vector<std::string> sum = builder.MakeNode("Add", {"a", "b"}, {"s"});
  builder.MakeOutput(sum[0]);

  EXPECT_EQ(builder.ConstantFold(), 1u);
  EXPECT_EQ(builder.Nodes().size(), 0u);
  const TensorProto *folded = FindInitializer(builder, "s");
  ASSERT_NE(folded, nullptr);
  const core::runtime::Tensor tensor = core::runtime::TensorFromProto(*folded);
  ASSERT_EQ(tensor.element_count(), 2);
  const float *values = tensor.AsFloat();
  EXPECT_FLOAT_EQ(values[0], 4.0f);
  EXPECT_FLOAT_EQ(values[1], 6.0f);
}

TEST(GraphBuilderConstantFold, DisabledOptionIsNoOp) {
  ScopedFoldingKernels kernels;
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(MakeInitializer<float>("a", {2}, {1.0f, 2.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {2}, {3.0f, 4.0f}));
  const std::vector<std::string> sum = builder.MakeNode("Add", {"a", "b"}, {"s"});
  builder.MakeOutput(sum[0]);

  core::builder::ConstantFoldingOptions options;
  options.enabled = false;
  EXPECT_EQ(builder.ConstantFold(options), 0u);
  EXPECT_EQ(builder.Nodes().size(), 1u);
}

TEST(GraphBuilderConstantFold, SkipsResultsOverElementThreshold) {
  ScopedFoldingKernels kernels;
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(MakeInitializer<float>("a", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
  const std::vector<std::string> sum = builder.MakeNode("Add", {"a", "b"}, {"s"});
  builder.MakeOutput(sum[0]);

  core::builder::ConstantFoldingOptions options;
  options.max_element_count = 3; // output has 4 elements -> skipped.
  EXPECT_EQ(builder.ConstantFold(options), 0u);
  EXPECT_EQ(builder.Nodes().size(), 1u);
}

TEST(GraphBuilderConstantFold, ExcludedOperatorIsNotFolded) {
  ScopedFoldingKernels kernels;
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(MakeInitializer<float>("a", {2}, {1.0f, 2.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {2}, {3.0f, 4.0f}));
  const std::vector<std::string> sum = builder.MakeNode("Add", {"a", "b"}, {"s"});
  builder.MakeOutput(sum[0]);

  core::builder::ConstantFoldingOptions options;
  options.excluded_ops = {{"", "Add"}};
  EXPECT_EQ(builder.ConstantFold(options), 0u);
  EXPECT_EQ(builder.Nodes().size(), 1u);
}

TEST(GraphBuilderConstantFold, WeightResultIsNotFoldedWhenFoldWeightsDisabled) {
  ScopedFoldingKernels kernels;
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(MakeInitializer<float>("a", {2}, {1.0f, 2.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {2}, {3.0f, 4.0f}));
  const std::vector<std::string> sum = builder.MakeNode("Add", {"a", "b"}, {"s"});
  builder.MakeOutput(sum[0]);

  core::builder::ConstantFoldingOptions options;
  options.fold_weights = false;
  EXPECT_EQ(builder.ConstantFold(options), 0u);
  EXPECT_EQ(builder.Nodes().size(), 1u);
}

TEST(GraphBuilderConstantFold, ShapeResultIsFoldedWhenFoldWeightsDisabled) {
  ScopedFoldingKernels kernels;
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(
      MakeInitializer<float>("a", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  const std::vector<std::string> shape = builder.MakeNode("Shape", {"a"}, {"sh"});
  builder.MakeOutput(shape[0]);

  core::builder::ConstantFoldingOptions options;
  options.fold_weights = false;
  EXPECT_EQ(builder.ConstantFold(options), 1u);
  EXPECT_EQ(builder.Nodes().size(), 0u);
  ASSERT_NE(FindInitializer(builder, "sh"), nullptr);
}

TEST(GraphBuilderConstantFold, MissingWeightKernelIsSkippedByDefault) {
  // No custom kernels installed: the weight-tagged Add has no runtime kernel.
  if (HasRuntimeKernel("Add")) {
    GTEST_SKIP() << "Add kernel is registered in this build variant.";
  }
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(MakeInitializer<float>("a", {2}, {1.0f, 2.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {2}, {3.0f, 4.0f}));
  const std::vector<std::string> sum = builder.MakeNode("Add", {"a", "b"}, {"s"});
  builder.MakeOutput(sum[0]);

  EXPECT_EQ(builder.ConstantFold(), 0u);
  EXPECT_EQ(builder.Nodes().size(), 1u);
}

TEST(GraphBuilderConstantFold, MissingWeightKernelRaisesWhenRequested) {
  if (HasRuntimeKernel("Add")) {
    GTEST_SKIP() << "Add kernel is registered in this build variant.";
  }
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(MakeInitializer<float>("a", {2}, {1.0f, 2.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {2}, {3.0f, 4.0f}));
  const std::vector<std::string> sum = builder.MakeNode("Add", {"a", "b"}, {"s"});
  builder.MakeOutput(sum[0]);

  core::builder::ConstantFoldingOptions options;
  options.raise_on_missing_weight_kernel = true;
  EXPECT_THROW(builder.ConstantFold(options), core::builder::BuilderError);
}

TEST(GraphBuilderConstantFold, MissingShapeKernelAlwaysRaises) {
  // No custom kernels installed: the shape-tagged Shape node has no kernel and
  // must raise regardless of the weight-kernel flag.
  if (HasRuntimeKernel("Shape")) {
    GTEST_SKIP() << "Shape kernel is registered in this build variant.";
  }
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(
      MakeInitializer<float>("a", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  const std::vector<std::string> shape = builder.MakeNode("Shape", {"a"}, {"sh"});
  builder.MakeOutput(shape[0]);

  EXPECT_THROW(builder.ConstantFold(), core::builder::BuilderError);
}

TEST(GraphBuilderConstantFold, FoldsShapeTaggedNodeWhenKernelAvailable) {
  ScopedFoldingKernels kernels;
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(
      MakeInitializer<float>("a", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  const std::vector<std::string> shape = builder.MakeNode("Shape", {"a"}, {"sh"});
  builder.MakeOutput(shape[0]);

  EXPECT_EQ(builder.ConstantFold(), 1u);
  EXPECT_EQ(builder.Nodes().size(), 0u);
  const TensorProto *folded = FindInitializer(builder, "sh");
  ASSERT_NE(folded, nullptr);
  const core::runtime::Tensor tensor = core::runtime::TensorFromProto(*folded);
  ASSERT_EQ(tensor.element_count(), 2);
  const int64_t *values = tensor.AsInt64();
  EXPECT_EQ(values[0], 2);
  EXPECT_EQ(values[1], 3);
}

TEST(GraphBuilderConstantFold, RecursesIntoSubgraphs) {
  ScopedFoldingKernels kernels;
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));

  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeInitializer(MakeInitializer<float>("c1", {2}, {1.0f, 2.0f}));
  body.MakeInitializer(MakeInitializer<float>("c2", {2}, {3.0f, 4.0f}));
  const std::vector<std::string> sum = body.MakeNode("Add", {"c1", "c2"}, {"s"});
  const std::vector<std::string> neg = body.MakeNode("Neg", {sum[0]}, {"n"});
  body.MakeOutput(neg[0]);

  EXPECT_EQ(builder.ConstantFold(), 2u);
  ASSERT_EQ(builder.Subgraph("body").Nodes().size(), 0u);
  const TensorProto *folded = FindInitializer(builder.Subgraph("body"), "n");
  ASSERT_NE(folded, nullptr);
  const core::runtime::Tensor tensor = core::runtime::TensorFromProto(*folded);
  ASSERT_EQ(tensor.element_count(), 2);
  const float *values = tensor.AsFloat();
  EXPECT_FLOAT_EQ(values[0], -4.0f);
  EXPECT_FLOAT_EQ(values[1], -6.0f);
}

} // namespace Test
