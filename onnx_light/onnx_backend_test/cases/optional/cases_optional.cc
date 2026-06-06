// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/optional/include_optional_cases.h"
#include "onnx_kernels/kernels/optional/include_optional_kernels.h"
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
  tensor_type->set_elem_type(static_cast<int>(DataType::FLOAT));
  TensorShapeProto *tp_shape = tensor_type->add_shape();
  for (int64_t d : shape) {
    TensorShapeProto::Dimension *dim = tp_shape->add_dim();
    dim->set_dim_value(d);
  }

  const OpsetId opset = DefaultOpset(15);
  const kernel::KernelContext ctx{opset};
  Tensor output = kernel::Optional(ctx)(input);

  Expect(node, {input}, {output}, "test_cc_optional", {opset}, "backend-test", registry);

  // ``Expect()`` populates the output value-info as a TensorTypeProto via the
  // generic ``FillValueInfo`` helper. The ONNX ``Optional`` operator however
  // requires its output type to match the ``type`` attribute (here an
  // ``Optional<Tensor<FLOAT, [2, 3]>>``) — runtimes such as ONNX Runtime
  // perform this type check and would reject the model otherwise. Promote
  // the just-emitted output to an ``OptionalTypeProto`` wrapping the
  // existing tensor type.
  GraphProto &graph = registry.back().model.ref_graph();
  ValueInfoProto &out_vi = *graph.mutable_output(0);
  TypeProto &out_tp = out_vi.ref_type();
  TypeProto::Optional *out_opt = out_tp.add_optional_type();
  TypeProto *out_elem = out_opt->add_elem_type();
  TypeProto::Tensor *out_tensor = out_elem->add_tensor_type();
  out_tensor->set_elem_type(static_cast<int>(DataType::FLOAT));
  TensorShapeProto *out_shape = out_tensor->add_shape();
  for (int64_t d : shape) {
    out_shape->add_dim()->set_dim_value(d);
  }
  // Drop the now-redundant tensor_type oneof field so only optional_type is
  // serialized for this value-info.
  out_tp.reset_tensor_type();
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
