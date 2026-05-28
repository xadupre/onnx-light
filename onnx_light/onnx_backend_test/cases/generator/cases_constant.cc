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
  const kernel::KernelContext ctx{opset};
  Tensor y = kernel::Constant(ctx)(value);

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

    Tensor y_upstream = kernel::Constant(ctx)(values);
    Expect(upstream_node, /*inputs=*/{}, {y_upstream}, "test_constant", {opset}, "backend-test",
           registry);
  }

  // Attribute-variant cases. Constant (since opset 12) accepts exclusive
  // ``value_float``/``value_floats``/``value_int``/``value_ints``/
  // ``value_string``/``value_strings`` attributes as alternatives to the
  // ``value`` TENSOR attribute. Each variant produces a tensor of the
  // corresponding scalar/vector shape and element type, per the operator
  // schema. The ``sparse_value`` attribute is intentionally not covered here
  // because it produces a sparse tensor output (type ``sparse_tensor(T)``),
  // which the dense-tensor backend test harness cannot represent.
  const OpsetId v12 = DefaultOpset(12);

  // value_float: scalar FLOAT output.
  {
    NodeProto n;
    n.set_op_type("Constant");
    n.add_output("y");
    AttributeProto *a = n.add_attribute();
    a->set_name("value_float");
    a->set_type(AttributeProto::AttributeType::FLOAT);
    a->set_f(3.5f);
    Tensor y = Tensor::FromFloat("", /*shape=*/{}, {3.5f});
    Expect(n, /*inputs=*/{}, {y}, "test_cc_constant_value_float", {v12}, "backend-test", registry);
  }

  // value_floats: 1-D FLOAT output.
  {
    NodeProto n;
    n.set_op_type("Constant");
    n.add_output("y");
    AttributeProto *a = n.add_attribute();
    a->set_name("value_floats");
    a->set_type(AttributeProto::AttributeType::FLOATS);
    const std::vector<float> vals = {1.0f, 2.5f, -3.25f, 4.75f};
    for (float v : vals) {
      a->floats().push_back(v);
    }
    Tensor y = Tensor::FromFloat("", {static_cast<int64_t>(vals.size())}, vals);
    Expect(n, /*inputs=*/{}, {y}, "test_cc_constant_value_floats", {v12}, "backend-test", registry);
  }

  // value_int: scalar INT64 output.
  {
    NodeProto n;
    n.set_op_type("Constant");
    n.add_output("y");
    AttributeProto *a = n.add_attribute();
    a->set_name("value_int");
    a->set_type(AttributeProto::AttributeType::INT);
    a->set_i(42);
    Tensor y = Tensor::FromInt64("", /*shape=*/{}, {static_cast<int64_t>(42)});
    Expect(n, /*inputs=*/{}, {y}, "test_cc_constant_value_int", {v12}, "backend-test", registry);
  }

  // value_ints: 1-D INT64 output.
  {
    NodeProto n;
    n.set_op_type("Constant");
    n.add_output("y");
    AttributeProto *a = n.add_attribute();
    a->set_name("value_ints");
    a->set_type(AttributeProto::AttributeType::INTS);
    const std::vector<int64_t> vals = {-1, 0, 1, 2, 3};
    for (int64_t v : vals) {
      a->ints().push_back(v);
    }
    Tensor y = Tensor::FromInt64("", {static_cast<int64_t>(vals.size())}, vals);
    Expect(n, /*inputs=*/{}, {y}, "test_cc_constant_value_ints", {v12}, "backend-test", registry);
  }

  // value_string: scalar STRING output.
  {
    NodeProto n;
    n.set_op_type("Constant");
    n.add_output("y");
    AttributeProto *a = n.add_attribute();
    a->set_name("value_string");
    a->set_type(AttributeProto::AttributeType::STRING);
    a->set_s(utils::String("hello"));
    Tensor y = Tensor::FromStrings("", /*shape=*/{}, {std::string("hello")});
    Expect(n, /*inputs=*/{}, {y}, "test_cc_constant_value_string", {v12}, "backend-test", registry);
  }

  // value_strings: 1-D STRING output.
  {
    NodeProto n;
    n.set_op_type("Constant");
    n.add_output("y");
    AttributeProto *a = n.add_attribute();
    a->set_name("value_strings");
    a->set_type(AttributeProto::AttributeType::STRINGS);
    const std::vector<std::string> vals = {"a", "bc", "def"};
    for (const std::string &v : vals) {
      *a->add_strings() = utils::String(v);
    }
    Tensor y = Tensor::FromStrings("", {static_cast<int64_t>(vals.size())}, vals);
    Expect(n, /*inputs=*/{}, {y}, "test_cc_constant_value_strings", {v12}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
