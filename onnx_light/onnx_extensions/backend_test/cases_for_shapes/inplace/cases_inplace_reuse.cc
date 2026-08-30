// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inplace/include_inplace_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

constexpr int64_t kDefaultIrVersion = 10;
constexpr const char *kInPlaceReuseMetadataKey = "onnx_light.inplace_reuse";
constexpr const char *kReleaseAfterMetadataKey = "onnx_light.release_after";
constexpr const char *kNotUsedAfterMetadataKey = "onnx_light.not_used_after";

} // namespace

void RegisterInPlaceReuseCases(std::vector<TestCase> &registry, TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(14);
  const auto abs_kernel = MakeReferenceKernel<onnx_kernels::kernel::Abs>(opset);

  const std::string name = "test_cc_shape_inference_inplace_reuse";

  TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INPLACE);
  tc.rtol = 1e-3;
  tc.atol = 1e-7;
  tc.build = [=](bool) -> BuiltCase {
    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INPLACE);
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion, {opset});

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    AddNode(*graph, "Abs", {"X"}, {"A"});
    AddNode(*graph, "Abs", {"A"}, {"B"});
    AddNode(*graph, "Abs", {"B"}, {"Y"});

    const int32_t kFloat = static_cast<int32_t>(DataType::FLOAT);
    const std::vector<DimSpec> shape = {"batch", "features"};
    AppendValueInfo(*graph->add_input(), "X", kFloat, shape);
    AppendValueInfo(*graph->add_value_info(), "A", kFloat, shape);
    AppendValueInfo(*graph->add_value_info(), "B", kFloat, shape);
    AppendValueInfo(*graph->add_output(), "Y", kFloat, shape);

    (*graph->mutable_node())[0].add_metadata(kNotUsedAfterMetadataKey, "X");
    (*graph->mutable_node())[1].add_metadata(kInPlaceReuseMetadataKey, "0:0:equal");
    (*graph->mutable_node())[1].add_metadata(kReleaseAfterMetadataKey, "A");
    (*graph->mutable_node())[2].add_metadata(kInPlaceReuseMetadataKey, "0:0:equal");
    (*graph->mutable_node())[2].add_metadata(kReleaseAfterMetadataKey, "B");

    const Tensor x = Tensor::FromFloat(
        "X", {3, 4}, {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, -5.0f, -6.0f, 7.0f, 8.0f});
    Tensor y = abs_kernel.Invoke([&](const auto &kernel) {
      return kernel(abs_kernel.Invoke([&](const auto &kernel) { return kernel(abs_kernel(x)); }));
    });
    y.name = "Y";

    AppendDataSet(tc, {x}, {y});

    return tc.take_materialized();
  };
  registry.emplace_back(std::move(tc));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
