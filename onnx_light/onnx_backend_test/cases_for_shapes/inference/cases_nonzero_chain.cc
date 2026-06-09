// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// IR version used by the manually-built models below.
constexpr int64_t kDefaultIrVersion = 10;

// Describes how the graph outputs (``nz`` and ``nz_float``) should be
// annotated in the model's ValueInfo. ``kNamedDims`` reuses the symbolic
// names from the ``plot_computed_shapes`` page (``nnz`` on ``nz`` and
// ``do1`` on ``nz_float`` / ``transposed_nz``); ``kAnonymousDims`` leaves
// the data-dependent dimensions without any name (default-constructed
// ``DimSpec()``) so that shape inference must derive them on its own.
enum class NonZeroOutputAnnotation { kAnonymousDims, kNamedDims };

// Builds the shared 7-node ``Abs → Relu → Add → Mul → NonZero → Transpose
// → Cast`` chain and registers the resulting TestCase.
//
// The model is executable: the reference kernels compute every intermediate
// tensor from a positive input (so ``Relu`` is identity on ``abs_out``) and
// ``NonZero`` produces a deterministic ``(rank, nnz)`` index tensor.
void RegisterNonZeroChainCase(const std::string &name, std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};

  // Input contains a deterministic mix of zero and positive entries so that
  // ``NonZero`` returns a non-trivial number of indices.
  const std::vector<int64_t> input_shape = {3, 4};
  const std::vector<float> x_values = {1.0f, 0.0f, 2.0f, 0.0f, //
                                       0.0f, 3.0f, 0.0f, 4.0f, //
                                       5.0f, 0.0f, 6.0f, 0.0f};
  Tensor x = Tensor::FromFloat("X", input_shape, x_values);

  // Reference computation:
  //   abs_out       = |X|
  //   relu_out      = Relu(abs_out) = abs_out (no negative entries)
  //   double_out    = abs_out + abs_out
  //   mul_out       = double_out * abs_out
  //   nz            = NonZero(mul_out)   shape (2, nnz)
  //   transposed_nz = Transpose(nz)  shape (nnz, 2)
  //   nz_float      = Cast(transposed_nz, FLOAT)
  Tensor abs_out = kernel::Abs(ctx)(x);
  Tensor relu_out = abs_out;
  relu_out.name = "";
  Tensor double_out = kernel::Add(ctx)(relu_out, relu_out);
  Tensor mul_out = kernel::Mul(ctx)(double_out, relu_out);
  Tensor nz = kernel::NonZero(ctx)(mul_out);
  nz.name = "nz";
  Tensor transposed_nz = kernel::Transpose(ctx)(nz, /*perm=*/{});
  Tensor nz_float = kernel::Cast(ctx)(transposed_nz, static_cast<int32_t>(DataType::FLOAT));
  nz_float.name = "nz_float";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  AddNode(*graph, "Abs", {"X"}, {"abs_out"});
  AddNode(*graph, "Relu", {"abs_out"}, {"relu_out"});
  AddNode(*graph, "Add", {"relu_out", "relu_out"}, {"double_out"});
  AddNode(*graph, "Mul", {"double_out", "relu_out"}, {"mul_out"});
  AddNode(*graph, "NonZero", {"mul_out"}, {"nz"});
  AddNode(*graph, "Transpose", {"nz"}, {"transposed_nz"});
  NodeProto &cast_node = AddNode(*graph, "Cast", {"transposed_nz"}, {"nz_float"});
  AddAttribute<int64_t>(cast_node, "to", static_cast<int64_t>(DataType::FLOAT));

  // Graph input X uses symbolic ``batch``/``seq`` dims; concrete sizes
  // ``[3, 4]`` are only used in the DataSet below.
  const int32_t kInt64 = static_cast<int32_t>(DataType::INT64);
  const int32_t kFloat = static_cast<int32_t>(DataType::FLOAT);
  const std::vector<DimSpec> symbolic_input_shape = {"batch", "seq"};
  AppendValueInfo(*graph->add_input(), "X", kFloat, symbolic_input_shape);

  // Intermediate ValueInfo entries. Tensors before ``NonZero`` keep the
  // input's symbolic ``[batch, seq]`` shape; ``transposed_nz`` has the same
  // data-dependent ``nnz`` dimension as ``nz``. The annotation style mirrors
  // the graph outputs (anonymous vs named ``nnz``/``do1``).
  AppendValueInfo(*graph->add_value_info(), "abs_out", kFloat, symbolic_input_shape);
  AppendValueInfo(*graph->add_value_info(), "relu_out", kFloat, symbolic_input_shape);
  AppendValueInfo(*graph->add_value_info(), "double_out", kFloat, symbolic_input_shape);
  AppendValueInfo(*graph->add_value_info(), "mul_out", kFloat, symbolic_input_shape);

  // Graph outputs: nz and nz_float. The rank dimension is always known
  // (equal to the input rank, 2), so it is declared with ``dim_value=2``.
  // The data-dependent ``nnz`` dimension stays symbolic — either named
  // (``"nnz"``/``"do1"``) or anonymous (default-constructed ``DimSpec()``).
  //
  // Intermediate value_info for ``nz`` would collide with the graph output of
  // the same name, so it is omitted. ``transposed_nz`` is a pure intermediate
  // and its shape mirrors ``nz_float`` (same dim layout, INT64 dtype).
  AppendValueInfo(*graph->add_value_info(), "transposed_nz", kInt64,
                  {DimSpec("do1"), DimSpec(2)});
  AppendValueInfo(*graph->add_output(), "nz", kInt64, {DimSpec(2), DimSpec("do1")});
  AppendValueInfo(*graph->add_output(), "nz_float", kFloat, {DimSpec("do1"), DimSpec(2)});

  // Provide a concrete DataSet so the case is executable end-to-end.
  Tensor nz_out = nz;
  nz_out.name = "nz";
  Tensor nz_float_out = nz_float;
  nz_float_out.name = "nz_float";
  AppendDataSet(tc, {x}, {std::move(nz_out), std::move(nz_float_out)});

  registry.emplace_back(std::move(tc));
}

} // namespace

void RegisterNonZeroChainNamedShapeInferenceCases(std::vector<TestCase> &registry) {
  RegisterNonZeroChainCase("test_cc_shape_inference_nonzero_chain_named", registry);
}

void RegisterDimensionExpressionShapeInferenceCase(std::vector<TestCase> &registry) {
  const std::string name("test_cc_shape_inference_nonzero_plus_expression");
  TestCase tc(name, name, "model", "inference", 1e-3, 1e-7);
  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {DefaultOpset(18)});
  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  TensorProto *m1 = graph->add_initializer();
  m1->set_name("m1");
  m1->set_data_type(TensorProto::DataType::INT64);
  m1->add_int64_data(-1);

  AddNode(*graph, "Abs", {"X"}, {"abs_out"});
  AddNode(*graph, "NonZero", {"abs_out"}, {"nz"});
  AddNode(*graph, "Reshape", {"nz", "m1"}, {"flat_nz"});
  AddNode(*graph, "Neg", {"flat_nz"}, {"Y"});

  AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {"batch", "seq"});
  AppendValueInfo(*graph->add_value_info(), "abs_out", DataType::FLOAT, {"batch", "seq"});
  AppendValueInfo(*graph->add_value_info(), "nz", DataType::INT64, {"dnz", "2"});
  AppendValueInfo(*graph->add_value_info(), "flat_nz", DataType::INT64, {"2*dnz"});
  AppendValueInfo(*graph->add_output(), "Y", DataType::INT64, {"2*dnz"});

  // Provide a concrete DataSet so the case is executable end-to-end.

  Tensor x = Tensor::FromFloat("X", {3, 4}, {1.0f, 0.0f, 2.0f, 0.0f,//
                                       0.0f, 3.0f, 0.0f, 4.0f, //
                                       5.0f, 0.0f, 6.0f, 0.0f});
  Tensor y = Tensor::FromInt64("Y", {12}, {0, 1, 0, 3, 1, 4, 1, 6, 3, 1, 3, 7});
  AppendDataSet(tc, {std::move(x)}, {std::move(y)});
  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
