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

// IR version used by the manually-built models below.
constexpr int64_t kDefaultIrVersion = 10;

} // namespace

// ---------------------------------------------------------------------------
// ``Shape → Gather → Expand → Abs`` — exercises *value-as-shape* propagation
// through the ``Gather`` operator. The model is:
//
//   x : float[N, D], y : float[1], initializer idx : int64[1] = [0]
//   shape_x  = Shape(x)                         # int64[2] = [N, D]
//   n_vec    = Gather(shape_x, idx, axis=0)      # int64[1] = [N]
//   expanded = Expand(y, n_vec)                  # float[N]
//   z        = Abs(expanded)                     # float[N]
//
// ``Shape`` lifts the symbolic dims of ``x`` into an INT64 tensor with a
// *value-as-shape* annotation ``[N, D]``.  ``Gather`` with the constant index
// ``[0]`` then slices that annotation to produce ``n_vec`` with VAS ``[N]``.
// ``Expand`` consumes ``n_vec`` as its target shape, so shape inference must
// recover the precise symbolic output shape ``float[N]`` from the VAS.
//
// The reference DataSet uses concrete sizes (``N=3, D=4``) so the case is
// executable end-to-end by ``BackendTestCaseRunModel``.
// ---------------------------------------------------------------------------
void RegisterGatherValueAsShapeShapeInferenceCases(std::vector<TestCase> &registry,
                                                   TestMode /*mode*/) {
  const OpsetId opset = DefaultOpset(20);

  const std::string name = "test_cc_shape_inference_gather_value_as_shape";
  TestCase lazy_case(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE, 1e-7, 1e-3);
  lazy_case.build = [name](bool) -> BuiltCase {
    const OpsetId opset = DefaultOpset(20);

    const KernelContext ctx_1{opset};
    const onnx_kernels::kernel::Shape kernel_1{ctx_1};
    const KernelContext ctx_2{opset};
    const onnx_kernels::kernel::Gather kernel_2{ctx_2};
    const KernelContext ctx_3{opset};
    const onnx_kernels::kernel::Expand kernel_3{ctx_3};
    const KernelContext ctx_4{opset};
    const onnx_kernels::kernel::Abs kernel_4{ctx_4};

    TestCase tc(name, name, TestCaseKind::MODEL, TestCaseTag::INFERENCE, 1e-7, 1e-3);

    ModelProto &model = tc.emplace_model();
    InitModel(model, kDefaultIrVersion, {opset});

    GraphProto *graph = model.add_graph();
    graph->set_name(name);

    // shape_x = Shape(x)
    AddNode(*graph, "Shape", {"x"}, {"shape_x"});

    // n_vec = Gather(shape_x, idx, axis=0)
    NodeProto &gather_node = AddNode(*graph, "Gather", {"shape_x", "idx"}, {"n_vec"});
    AddAxisAttribute(gather_node, 0);

    // expanded = Expand(y, n_vec)
    AddNode(*graph, "Expand", {"y", "n_vec"}, {"expanded"});

    // z = Abs(expanded)
    AddNode(*graph, "Abs", {"expanded"}, {"z"});

    // Initializer ``idx`` : int64[1] = [0] — index that selects the first dim (N).
    AddInitializerShape(*graph, "idx", {0});

    // Graph inputs: x with symbolic shape [N, D], y as float[1].
    AppendValueInfo(*graph->add_input(), "x", DataType::FLOAT, {DimSpec("N"), DimSpec("D")});
    AppendValueInfo(*graph->add_input(), "y", DataType::FLOAT, {DimSpec(int64_t{1})});

    // Intermediate value_info entries stripped by SnapshotAndStripValueInfo and
    // used as ground truth for the shape-inference assertions.
    AppendValueInfo(*graph->add_value_info(), "shape_x", DataType::INT64, {DimSpec(int64_t{2})});
    AppendValueInfo(*graph->add_value_info(), "n_vec", DataType::INT64, {DimSpec(int64_t{1})});
    AppendValueInfo(*graph->add_value_info(), "expanded", DataType::FLOAT, {DimSpec("N")});

    // Graph output: z : float[N].
    AppendValueInfo(*graph->add_output(), "z", DataType::FLOAT, {DimSpec("N")});

    // Build the reference DataSet — concrete N=3, D=4 tensors.
    constexpr int64_t kN = 3;
    constexpr int64_t kD = 4;

    std::vector<float> x_values(static_cast<size_t>(kN * kD));
    for (size_t i = 0; i < x_values.size(); ++i) {
      x_values[i] = static_cast<float>(i) * 0.1f + 1.0f;
    }
    Tensor x = Tensor::FromFloat("x", {kN, kD}, x_values);

    Tensor y = Tensor::FromFloat("y", {1}, {2.0f});

    // idx = int64[1] = [0]
    const Tensor idx = Tensor::FromInt64("idx", {1}, {0});

    // shape_x = Shape(x) = [3, 4]
    Tensor shape_x = kernel_1(x, onnx_kernels::kernel::Shape::Attributes{});
    shape_x.name = "shape_x";

    // n_vec = Gather(shape_x, idx, axis=0) = int64[1] = [3]
    Tensor n_vec = kernel_2(shape_x, idx, 0);
    n_vec.name = "n_vec";

    // expanded = Expand(y, n_vec) = float[3] = [2.0, 2.0, 2.0]
    Tensor expanded = kernel_3(y, n_vec);
    expanded.name = "expanded";

    // z = Abs(expanded) = float[3] = [2.0, 2.0, 2.0]
    Tensor z = kernel_4(expanded);
    z.name = "z";

    AppendDataSet(tc, {std::move(x), std::move(y)}, {std::move(z)});

    return tc.take_materialized();
  };
  registry.emplace_back(std::move(lazy_case));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
