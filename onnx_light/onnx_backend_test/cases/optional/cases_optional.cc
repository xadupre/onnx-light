// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/optional/include_optional_cases.h"
#include "onnx_backend_test/kernels/optional/include_optional_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Optional — wraps a tensor input into an optional-of-tensor value (since
// opset 15 in the ai.onnx domain). The operator requires a ``type``
// attribute (TypeProto) describing the wrapped element type/shape. Because
// the project's runtime :ref:`Tensor` type does not model optional/sequence
// values, this case exercises only the "tensor element, present" path: the
// expected output's element type, shape and bytes are an exact copy of the
// single input tensor.
// ---------------------------------------------------------------------------
void RegisterOptionalCases(std::vector<TestCase> &registry) {
  NodeProto node;
  node.set_op_type("Optional");
  node.add_input("input");
  node.add_output("output");

  const std::vector<int64_t> shape = {2, 3};
  Tensor input = Tensor::FromFloat("", shape, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});

  // ``type`` attribute: TypeProto wrapping an Optional<Tensor<FLOAT, [2, 3]>>.
  AttributeProto *attr = node.add_attribute();
  attr->set_name("type");
  attr->set_type(AttributeProto::AttributeType::TYPE_PROTO);
  TypeProto *tp = attr->add_tp();
  TypeProto::Optional *opt_type = tp->add_optional_type();
  TypeProto *elem_type = opt_type->add_elem_type();
  TypeProto::Tensor *tensor_type = elem_type->add_tensor_type();
  tensor_type->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
  TensorShapeProto *tp_shape = tensor_type->add_shape();
  for (int64_t d : shape) {
    TensorShapeProto::Dimension *dim = tp_shape->add_dim();
    dim->set_dim_value(d);
  }

  const OpsetId opset = DefaultOpset(15);
  Tensor output = kernel::Optional(kernel::KernelContext(opset))(input);

  Expect(node, {input}, {output}, "test_cc_optional", {opset}, "backend-test", registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
