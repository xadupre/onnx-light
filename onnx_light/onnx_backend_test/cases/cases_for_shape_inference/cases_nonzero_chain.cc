// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/cases_for_shape_inference/include_shape_inference_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"

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
// names from the ``plot_computed_shapes`` page (rank/nnz on ``nz`` and
// do1/do2 on ``nz_float``); ``kAnonymousDims`` leaves both dimensions
// without a name so that shape inference must derive them.
enum class NonZeroOutputAnnotation { kAnonymousDims, kNamedDims };

// Builds the shared 7-node ``Abs → Relu → Add → Mul → NonZero → Transpose
// → Cast`` chain and registers the resulting TestCase.
//
// The model is executable: the reference kernels compute every intermediate
// tensor from a positive input (so ``Relu`` is identity on ``abs_out``) and
// ``NonZero`` produces a deterministic ``(rank, nnz)`` index tensor.
void RegisterNonZeroChainCase(const std::string &name, NonZeroOutputAnnotation annotation,
                              std::vector<TestCase> &registry) {
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
  //   abs_out   = |X|
  //   relu_out  = Relu(abs_out) = abs_out (no negative entries)
  //   double_out = abs_out + abs_out
  //   mul_out   = double_out * abs_out
  //   nz        = NonZero(mul_out)   shape (2, nnz)
  //   transposed_nz = Transpose(nz)  shape (nnz, 2)
  //   nz_float  = Cast(transposed_nz, FLOAT)
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

  TestCase tc;
  tc.name = name;
  tc.model_name = name;
  tc.kind = "node";
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  model.set_ir_version(kDefaultIrVersion);
  model.set_producer_name("backend-test");
  OperatorSetIdProto proto;
  proto.set_domain(opset.domain);
  proto.set_version(opset.version);
  model.add_opset_import(proto);

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  const auto add_unary = [&](const char *op_type, const char *in_name, const char *out_name) {
    NodeProto *node = graph->add_node();
    node->set_op_type(op_type);
    node->add_input(in_name);
    node->add_output(out_name);
    return node;
  };
  const auto add_binary = [&](const char *op_type, const char *in_a, const char *in_b,
                              const char *out_name) {
    NodeProto *node = graph->add_node();
    node->set_op_type(op_type);
    node->add_input(in_a);
    node->add_input(in_b);
    node->add_output(out_name);
    return node;
  };

  add_unary("Abs", "X", "abs_out");
  add_unary("Relu", "abs_out", "relu_out");
  add_binary("Add", "relu_out", "relu_out", "double_out");
  add_binary("Mul", "double_out", "relu_out", "mul_out");
  add_unary("NonZero", "mul_out", "nz");
  add_unary("Transpose", "nz", "transposed_nz");
  NodeProto *cast_node = add_unary("Cast", "transposed_nz", "nz_float");
  AttributeProto *to_attr = cast_node->add_attribute();
  to_attr->set_name("to");
  to_attr->set_type(AttributeProto::AttributeType::INT);
  to_attr->set_i(static_cast<int64_t>(DataType::FLOAT));

  // Graph input: X with concrete dims.
  FillValueInfo(x, *graph->add_input());

  // Intermediate ValueInfo entries. Tensors before ``NonZero`` keep the
  // input's concrete ``[3, 4]`` shape; ``transposed_nz`` has the same
  // data-dependent ``nnz`` dimension as ``nz``. The annotation style mirrors
  // the graph outputs (anonymous vs named ``nnz``/``do1``).
  Tensor abs_vi = abs_out;
  abs_vi.name = "abs_out";
  Tensor relu_vi = abs_out;
  relu_vi.name = "relu_out";
  Tensor double_vi = double_out;
  double_vi.name = "double_out";
  Tensor mul_vi = mul_out;
  mul_vi.name = "mul_out";
  FillValueInfo(abs_vi, *graph->add_value_info());
  FillValueInfo(relu_vi, *graph->add_value_info());
  FillValueInfo(double_vi, *graph->add_value_info());
  FillValueInfo(mul_vi, *graph->add_value_info());

  // Graph outputs: nz and nz_float. The rank dimension is always known
  // (equal to the input rank, 2), so it is declared with ``dim_value=2``.
  // The data-dependent ``nnz`` dimension stays symbolic — either named
  // (``"nnz"``/``"do1"``) or anonymous (no dim_param/dim_value).
  const auto add_value_info_with_dims =
      [&](ValueInfoProto *vi, const std::string &out_name, int32_t elem_type,
          const std::vector<std::pair<int64_t, std::string>> &dims) {
        vi->set_name(out_name);
        TypeProto *tp = vi->add_type();
        TypeProto::Tensor *tt = tp->add_tensor_type();
        tt->set_elem_type(elem_type);
        TensorShapeProto *shape = tt->add_shape();
        for (const auto &d : dims) {
          auto *dim = shape->add_dim();
          if (d.first >= 0) {
            dim->set_dim_value(d.first);
          } else if (!d.second.empty()) {
            dim->set_dim_param(d.second);
          }
          // else: leave the dim unannotated (no dim_value, no dim_param).
        }
      };
  const auto add_output = [&](const std::string &out_name, int32_t elem_type,
                              const std::vector<std::pair<int64_t, std::string>> &dims) {
    add_value_info_with_dims(graph->add_output(), out_name, elem_type, dims);
  };

  // Intermediate value_info for ``nz`` would collide with the graph output of
  // the same name, so it is omitted. ``transposed_nz`` is a pure intermediate
  // and its shape mirrors ``nz_float`` (same dim layout, INT64 dtype).
  if (annotation == NonZeroOutputAnnotation::kNamedDims) {
    add_value_info_with_dims(graph->add_value_info(), "transposed_nz",
                             static_cast<int32_t>(DataType::INT64), {{-1, "do1"}, {2, ""}});
    add_output("nz", static_cast<int32_t>(DataType::INT64), {{2, ""}, {-1, "nnz"}});
    add_output("nz_float", static_cast<int32_t>(DataType::FLOAT), {{-1, "do1"}, {2, ""}});
  } else {
    add_value_info_with_dims(graph->add_value_info(), "transposed_nz",
                             static_cast<int32_t>(DataType::INT64), {{-1, ""}, {2, ""}});
    add_output("nz", static_cast<int32_t>(DataType::INT64), {{2, ""}, {-1, ""}});
    add_output("nz_float", static_cast<int32_t>(DataType::FLOAT), {{-1, ""}, {2, ""}});
  }

  // Provide a concrete DataSet so the case is executable end-to-end.
  DataSet ds;
  ds.inputs.push_back(x);
  Tensor nz_out = nz;
  nz_out.name = "nz";
  Tensor nz_float_out = nz_float;
  nz_float_out.name = "nz_float";
  ds.outputs.push_back(std::move(nz_out));
  ds.outputs.push_back(std::move(nz_float_out));
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));
}

} // namespace

void RegisterNonZeroChainAnonShapeInferenceCases(std::vector<TestCase> &registry) {
  RegisterNonZeroChainCase("test_cc_shape_inference_nonzero_chain_anon",
                           NonZeroOutputAnnotation::kAnonymousDims, registry);
}

void RegisterNonZeroChainNamedShapeInferenceCases(std::vector<TestCase> &registry) {
  RegisterNonZeroChainCase("test_cc_shape_inference_nonzero_chain_named",
                           NonZeroOutputAnnotation::kNamedDims, registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
