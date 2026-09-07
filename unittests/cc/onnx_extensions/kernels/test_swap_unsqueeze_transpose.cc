// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_extensions/patterns/layout/layout_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::runtime::Tensor RunLayoutModel(const ModelProto &model, const TensorProto &input) {
  core::runtime::RuntimeContext context(
      core::runtime::KernelContext(core::backend_test::DefaultOpset(18)));
  context.Set("x", core::runtime::TensorFromProto(input));
  for (const TensorProto &initializer : model.ref_graph().initializer()) {
    context.Set(initializer.name().value(), core::runtime::TensorFromProto(initializer),
                core::runtime::RuntimeEventKind::kInitializer);
  }
  const auto &plan = context.GetExecutionPlan(model.ref_graph());
  core::runtime::RuntimeSession session(plan);
  session.Run(context);
  return context.Get("out");
}

TEST(SwapUnsqueezeTransposePattern, NumericallyPreservesEverySmallPermutation) {
  for (int64_t rank = 1; rank <= 4; ++rank) {
    for (int64_t mask = 1; mask < (int64_t{1} << rank); ++mask) {
      std::vector<int64_t> axes;
      std::vector<int64_t> dims;
      for (int64_t axis = 0; axis < rank; ++axis) {
        if ((mask & (int64_t{1} << axis)) != 0) {
          axes.push_back(axis);
        } else {
          dims.push_back(static_cast<int64_t>(dims.size()) + 2);
        }
      }
      if (axes.size() > 2) {
        continue;
      }
      std::vector<int64_t> perm(static_cast<std::size_t>(rank));
      std::iota(perm.begin(), perm.end(), 0);
      do {
        for (bool negative : {false, true}) {
          SCOPED_TRACE(::testing::PrintToString(axes));
          SCOPED_TRACE(::testing::PrintToString(perm));
          SCOPED_TRACE(negative);
          core::builder::GraphBuilder builder("layout_numerical", [](const std::string &op) {
            return onnx_op::GetAllOnnxOpSchemasWithHistory(op, false);
          });
          builder.SetOpsetVersion("", 18);
          core::symbolic::SymShape shape;
          TensorProto input;
          input.set_name("x");
          input.set_data_type(static_cast<int>(TensorProto::DataType::FLOAT));
          int64_t size = 1;
          for (int64_t dim : dims) {
            shape.PushBack(core::symbolic::SymDim(dim));
            input.dims().push_back(dim);
            size *= dim;
          }
          for (int64_t i = 0; i < size; ++i) {
            input.float_data().push_back(static_cast<float>(i) + 0.25f);
          }
          builder.MakeInput("x", core::symbolic::TensorType::kFloat, shape);
          std::vector<int64_t> signed_axes = axes;
          if (negative) {
            for (int64_t &axis : signed_axes) {
              axis -= rank;
            }
            std::reverse(signed_axes.begin(), signed_axes.end());
          }
          builder.MakeInitializer(MakeInitializerShape("axes", signed_axes));
          builder.MakeNode("Unsqueeze", {"x", "axes"}, {"u"});
          NodeProto transpose = MakeNode("Transpose", {"u"}, {"out"});
          AddAttribute(transpose, "perm", perm);
          builder.MakeNode("Transpose", {"u"}, {"out"}, "", "", transpose.attribute());
          builder.MakeOutput("out");
          const ModelProto original = builder.ToModel();
          std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
          patterns.push_back(std::make_unique<onnx_patterns::SwapUnsqueezeTransposePattern>());
          core::builder::GraphGraph graph(builder, std::move(patterns));
          const auto rewrites = graph.Optimize();
          ASSERT_EQ(std::count_if(rewrites.begin(), rewrites.end(),
                                  [](const core::builder::LocalRewriting &rewrite) {
                                    return rewrite.pattern != nullptr &&
                                           rewrite.pattern->Name() == "SwapUnsqueezeTranspose";
                                  }),
                    1);
          const ModelProto optimized = builder.ToModel();
          const auto expected = RunLayoutModel(original, input);
          const auto actual = RunLayoutModel(optimized, input);
          ASSERT_EQ(expected.shape, actual.shape);
          ASSERT_EQ(expected.size_bytes(), actual.size_bytes());
          EXPECT_EQ(
              std::vector<uint8_t>(expected.bytes(), expected.bytes() + expected.size_bytes()),
              std::vector<uint8_t>(actual.bytes(), actual.bytes() + actual.size_bytes()));
        }
      } while (std::next_permutation(perm.begin(), perm.end()));
    }
  }
}

} // namespace
} // namespace Test
