// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/optional/include_optional_cases.h"
#include "onnx_extensions/kernels/kernels/optional/include_optional_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// Optional — wraps a tensor input into an optional-of-tensor value (since
// opset 15 in the ai.onnx domain). The operator requires a ``type``
// attribute (TypeProto) describing the wrapped element type/shape. Because
// the project's runtime :ref:`Tensor` type does not model optional/sequence
// values, this case exercises only the "tensor element, present" path: the
// expected output's element type, shape and bytes are an exact copy of the
// single input tensor.
// ---------------------------------------------------------------------------
void RegisterOptionalCases(std::vector<TestCase> &registry, TestMode mode) {
  NodeProto node;
  node.set_op_type("Optional");
  node.add_input("input");
  node.add_output("output");

  const bool benchmark = (mode == TestMode::BENCHMARK);
  const std::vector<int64_t> shape =
      benchmark ? std::vector<int64_t>{512, 512} : std::vector<int64_t>{2, 3};
  const std::string case_name = benchmark ? "test_cc_optional_benchmark" : "test_cc_optional";

  // ``type`` attribute: TypeProto wrapping an Optional<Tensor<FLOAT, [2, 3]>>.
  AttributeProto *attr = node.add_attribute();
  attr->set_name("type");
  attr->set_type(AttributeProto::AttributeType::TYPE_PROTO);
  TypeProto *tp = attr->add_tp();
  TypeProto::Optional *opt_type = tp->add_optional_type();
  TypeProto *elem_type = opt_type->add_elem_type();
  TypeProto::Tensor *tensor_type = elem_type->add_tensor_type();
  tensor_type->set_elem_type(static_cast<int>(DataType::FLOAT));
  TensorShapeProto *tp_shape = tensor_type->add_shape();
  for (int64_t d : shape) {
    TensorShapeProto::Dimension *dim = tp_shape->add_dim();
    dim->set_dim_value(d);
  }

  const OpsetId opset = DefaultOpset(15);

  Expect(registry, std::move(node), case_name, {opset},
         [=]() -> IoData {
           Tensor input =
               benchmark ? RandnTensor(DataType::FLOAT, shape, 4001)
                         : Tensor::FromFloat("", shape, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
           Tensor output = MakeReferenceKernel<onnx_kernels::kernel::Optional>(opset).Invoke(
               [&](const auto &kernel) { return kernel(input); });
           return IoData{{std::move(input)}, {std::move(output)}};
         },
         "backend-test", TestCaseTag::NONE,
         {OptionalTypeSpec(TensorTypeSpec(static_cast<int32_t>(DataType::FLOAT), shape))});
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
