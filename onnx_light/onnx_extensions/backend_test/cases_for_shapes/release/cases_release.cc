// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/inplace_reuse.h"
#include "onnx_extensions/backend_test/cases_for_shapes/release/include_release_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

constexpr int64_t kDefaultIrVersion = 10;

} // namespace

// ---------------------------------------------------------------------------
// ``Shape → Reshape`` — exercises the ``kRelease`` event path.
//
// ``Shape(X)`` produces an INT64 tensor ``S`` whose last (and only) consumer
// is ``Reshape``.  ``ComputeInPlaceReuseGraph`` must therefore emit a
// ``kRelease`` event for ``S`` at node index 1 (the ``Reshape`` node) and
// write ``onnx_light.release_after = "S"`` on that node.
//
// The expected metadata is pre-embedded into the model so tests can verify
// that ``ComputeContext::ComputeInPlaceReuseGraph`` reproduces it exactly:
//
//   * node 0 (Shape):   no release metadata.
//   * node 1 (Reshape): ``onnx_light.release_after = "S"``,
//                       ``onnx_light.not_used_after = "X"``.
// ---------------------------------------------------------------------------
// ``Add(X, W) → Relu`` — exercises ``kNotUsedAfterMetadataKey`` for both a
// declared graph input and a graph initializer.
//
// ``W`` is an initializer (not a graph input) consumed only by ``Add``.
// ``X`` is a declared graph input also consumed only by ``Add``.
// Both reach their last use at node 0, so ``not_used_after`` for node 0 must
// list both ``X`` and ``W`` (in node-input order).  The intermediate ``T``
// produced by ``Add`` is the sole input of ``Relu`` and is therefore released
// after node 1.
//
// The expected metadata is pre-embedded into the model so tests can verify
// that ``ComputeContext::ComputeInPlaceReuseGraph`` reproduces it exactly:
//
//   * node 0 (Add):  ``onnx_light.not_used_after = "X;W"``.
//   * node 1 (Relu): ``onnx_light.release_after = "T"``.
// ---------------------------------------------------------------------------
void RegisterReleaseCases(std::vector<TestCase> &registry, TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(18);

  // ---- case 1: Shape → Reshape -----------------------------------------------
  {
    const std::string name = "test_cc_release_shape_reshape";
    TestCase lazy_case(name, name, TestCaseKind::MODEL, TestCaseTag::RELEASE);
    lazy_case.build = [=](bool) -> BuiltCase {
      TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::RELEASE);
      tc.rtol = 1e-3;
      tc.atol = 1e-7;

      ModelProto &model = tc.emplace_model();
      InitModel(model, kDefaultIrVersion, {opset});

      GraphProto *graph = model.add_graph();
      graph->set_name(name);

      AddNode(*graph, "Shape", {"X"}, {"S"});
      AddNode(*graph, "Reshape", {"X", "S"}, {"Y"});

      // Concrete input shape [2, 3] so the model is executable end-to-end.
      AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {DimSpec(2), DimSpec(3)});
      AppendValueInfo(*graph->add_value_info(), "S", DataType::INT64, {DimSpec(2)});
      AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT, {DimSpec(2), DimSpec(3)});

      // Pre-embed the expected release metadata.
      // The graph has exactly two nodes: node 0 = Shape, node 1 = Reshape.
      // S is produced by Shape and consumed only by Reshape, so the release point
      // is node 1. X is a declared graph input and also reaches its last use on
      // node 1, so it is reported as "not used after". No shape-tag metadata is
      // involved here.
      // NOLINTNEXTLINE: nodes has exactly 2 elements (Shape + Reshape added above).
      (*graph->mutable_node())[1].add_metadata(core::compute::kReleaseAfterMetadataKey, "S");
      (*graph->mutable_node())[1].add_metadata(core::compute::kNotUsedAfterMetadataKey, "X");

      // Build the reference DataSet so the case is executable end-to-end.
      const Tensor x = Tensor::FromFloat("X", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
      Tensor s = MakeReferenceKernel<onnx_kernels::kernel::Shape>(opset).Invoke(
          [&](const auto &kernel) { return kernel(x, onnx_kernels::kernel::Shape::Attributes{}); });
      s.name = "S";
      Tensor y = MakeReferenceKernel<onnx_kernels::kernel::Reshape>(opset).Invoke(
          [&](const auto &kernel) { return kernel(x, s); });
      y.name = "Y";

      AppendDataSet(tc, {x}, {y});

      return tc.take_materialized();
    };
    registry.emplace_back(std::move(lazy_case));
  }

  // ---- case 2: Add(input, initializer) → Relu --------------------------------
  {
    const std::string name = "test_cc_release_initializer_add";
    TestCase lazy_case(name, name, TestCaseKind::MODEL, TestCaseTag::RELEASE);
    lazy_case.build = [=](bool) -> BuiltCase {
      TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::RELEASE);
      tc.rtol = 1e-3;
      tc.atol = 1e-7;

      ModelProto &model = tc.emplace_model();
      InitModel(model, kDefaultIrVersion, {opset});

      GraphProto *graph = model.add_graph();
      graph->set_name(name);

      AddNode(*graph, "Add", {"X", "W"}, {"T"});
      AddNode(*graph, "Relu", {"T"}, {"Y"});

      AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {DimSpec(2), DimSpec(3)});
      AppendValueInfo(*graph->add_value_info(), "T", DataType::FLOAT, {DimSpec(2), DimSpec(3)});
      AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT, {DimSpec(2), DimSpec(3)});
      AddInitializer<float>(*graph, "W", {2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

      // Pre-embed the expected release metadata.
      // Node 0 (Add): X (graph input) and W (initializer) both reach their last
      // use here, so they are listed under kNotUsedAfterMetadataKey in node-input
      // order.
      // Node 1 (Relu): T (the intermediate produced by Add) is released here.
      // NOLINTNEXTLINE: nodes has exactly 2 elements (Add + Relu added above).
      (*graph->mutable_node())[0].add_metadata(core::compute::kNotUsedAfterMetadataKey, "X;W");
      (*graph->mutable_node())[1].add_metadata(core::compute::kReleaseAfterMetadataKey, "T");

      // Build the reference DataSet so the case is executable end-to-end.
      const Tensor x = Tensor::FromFloat("X", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, -5.0f, -6.0f});
      const Tensor w = Tensor::FromFloat("W", {2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
      Tensor t = MakeReferenceKernel<onnx_kernels::kernel::Add>(opset).Invoke(
          [&](const auto &kernel) { return kernel(x, w); });
      t.name = "T";
      Tensor y = MakeReferenceKernel<onnx_kernels::kernel::Relu>(opset).Invoke(
          [&](const auto &kernel) { return kernel(t); });
      y.name = "Y";

      AppendDataSet(tc, {x}, {y});

      return tc.take_materialized();
    };
    registry.emplace_back(std::move(lazy_case));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
