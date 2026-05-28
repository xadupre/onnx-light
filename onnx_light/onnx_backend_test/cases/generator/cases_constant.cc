// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/generator/include_generator_cases.h"
#include "onnx_backend_test/kernels/generator/include_generator_kernels.h"
#include "onnx_backend_test/random.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Constant — produces a tensor output whose data type, shape and bytes are
// taken from the ``value`` attribute (since opset 1; the ``value`` TENSOR
// attribute is the form used here and matches the original opset-1 schema).
// Uses a small, fully deterministic value so this library does not depend
// on a PRNG.
// ---------------------------------------------------------------------------
void RegisterConstantCases(std::vector<TestCase> &registry) {
  NodeProto node;
  node.set_op_type("Constant");
  node.add_output("y");

  const Tensor value = Tensor::FromFloat("", {2, 3}, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});

  AttributeProto *attr = node.add_attribute();
  attr->set_name("value");
  attr->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto *t = attr->add_t();
  t->set_data_type(static_cast<TensorProto::DataType>(value.data_type));
  for (int64_t d : value.shape) {
    t->add_dims(static_cast<uint64_t>(d));
  }
  // Pack the value bytes as raw_data (row-major little-endian, matching the
  // layout used by ``Tensor::data``).
  t->set_raw_data(utils::ByteSpan(value.data));

  const OpsetId opset = DefaultOpset(13);
  Tensor y = kernel::Constant(kernel::KernelContext(opset))(value);

  Expect(node, /*inputs=*/{}, {y}, "test_cc_constant", {opset}, "backend-test", registry);

  // Upstream ONNX backend test case for the ``Constant`` operator (mirrors the
  // ``onnx.backend.test.case.node.constant.Constant`` Python class). The
  // upstream case uses ``np.random.randn(5, 5).astype(np.float32)`` as the
  // ``value`` attribute; we use the deterministic ``Randn`` helper here so the
  // registry remains reproducible without depending on NumPy.
  //
  // From Constant.export():
  {
    const std::vector<int64_t> values_shape = {5, 5};
    const Tensor values =
        Tensor::FromFloat("", values_shape, Randn<float>(values_shape, /*seed=*/5));

    NodeProto upstream_node;
    upstream_node.set_op_type("Constant");
    upstream_node.add_output("values");

    AttributeProto *upstream_attr = upstream_node.add_attribute();
    upstream_attr->set_name("value");
    upstream_attr->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *ut = upstream_attr->add_t();
    ut->set_name("const_tensor");
    ut->set_data_type(static_cast<TensorProto::DataType>(values.data_type));
    for (int64_t d : values.shape) {
      ut->add_dims(static_cast<uint64_t>(d));
    }
    ut->set_raw_data(utils::ByteSpan(values.data));

    Tensor y_upstream = kernel::Constant(kernel::KernelContext(opset))(values);
    Expect(upstream_node, /*inputs=*/{}, {y_upstream}, "test_constant", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
