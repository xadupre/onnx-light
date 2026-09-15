// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

constexpr int64_t kDefaultIrVersion = 10;

} // namespace

void RegisterSliceSymbolicEndShapeInferenceCases(std::vector<TestCase> &registry,
                                                 TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(13);
  const std::string name("test_cc_shape_inference_slice_symbolic_end");
  TestCase lazy_case(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE, 1e-7, 1e-3);
  lazy_case.build = [name](bool) -> BuiltCase {
    const OpsetId opset = DefaultOpset(13);

    const KernelContext ctx_1{opset};
    const onnx_kernels::kernel::Slice kernel_1{ctx_1};
    const KernelContext ctx_2{opset};
    const onnx_kernels::kernel::Abs kernel_2{ctx_2};

    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE, 1e-7, 1e-3);

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion, {opset});
    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    AddInitializer<int64_t>(*graph, "starts", {1}, {0});
    AddInitializer<int64_t>(*graph, "ends", {1}, {-1});
    AddInitializer<int64_t>(*graph, "axes", {1}, {2});

    AddNode(*graph, "Slice", {"X", "starts", "ends", "axes"}, {"sliced"});
    AddNode(*graph, "Abs", {"sliced"}, {"Y"});

    AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {"a", "b", "c"});
    AppendValueInfo(*graph->add_value_info(), "sliced", DataType::FLOAT, {"a", "b", "c-1"});
    AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT, {"a", "b", "c-1"});

    const Tensor x = Tensor::FromFloat("X", {2, 3, 4},
                                       {
                                           1.0f,  2.0f,  3.0f,  4.0f,  //
                                           5.0f,  6.0f,  7.0f,  8.0f,  //
                                           9.0f,  10.0f, 11.0f, 12.0f, //
                                           13.0f, 14.0f, 15.0f, 16.0f, //
                                           17.0f, 18.0f, 19.0f, 20.0f, //
                                           21.0f, 22.0f, 23.0f, 24.0f, //
                                       });
    const Tensor starts = Tensor::FromInt64("", {1}, {0});
    const Tensor ends = Tensor::FromInt64("", {1}, {-1});
    const Tensor axes = Tensor::FromInt64("", {1}, {2});
    Tensor sliced = kernel_1(x, starts, ends, &axes, nullptr);
    Tensor y = kernel_2(sliced);
    y.name = "Y";
    AppendDataSet(tc, {x}, {std::move(y)});

    return tc.take_materialized();
  };
  registry.emplace_back(std::move(lazy_case));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
