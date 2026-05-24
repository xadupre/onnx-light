// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/generator/include_generator_cases.h"
#include "onnx_backend_test/kernels/generator/include_generator_kernels.h"
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
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
