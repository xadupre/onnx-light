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
  const KernelContext ctx{opset};

  Expect(registry, std::move(node), case_name, {opset}, [=]() -> IoData {
    Tensor input = benchmark
                       ? RandnTensor(DataType::FLOAT, shape, 4001)
                       : Tensor::FromFloat("", shape, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
    Tensor output = onnx_kernels::kernel::Optional(ctx)(input);
    return IoData{{std::move(input)}, {std::move(output)}};
  });

  // ``Expect()`` populates the output value-info as a TensorTypeProto via the
  // generic ``FillValueInfo`` helper. The ONNX ``Optional`` operator however
  // requires its output type to match the ``type`` attribute (here an
  // ``Optional<Tensor<FLOAT, [2, 3]>>``) — runtimes such as ONNX Runtime
  // perform this type check and would reject the model otherwise. Promote
  // the just-emitted output to an ``OptionalTypeProto`` wrapping the
  // existing tensor type.
  GraphProto &graph = registry.back().model().ref_graph();
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

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
