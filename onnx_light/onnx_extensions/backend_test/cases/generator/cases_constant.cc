// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/generator/include_generator_cases.h"
#include "onnx_extensions/kernels/kernels/generator/include_generator_kernels.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// Constant — produces a tensor output whose data type, shape and bytes are
// taken from the ``value`` attribute (since opset 1; the ``value`` TENSOR
// attribute is the form used here and matches the original opset-1 schema).
// Uses a small, fully deterministic value so this library does not depend
// on a PRNG.
// ---------------------------------------------------------------------------
void RegisterConstantCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Constant");
    node.add_output("y");

    Tensor value = RandnTensor(DataType::FLOAT, {kBenchmarkElementwiseSize}, 987654321ULL);

    AttributeProto *attr = node.add_attribute();
    attr->set_name("value");
    attr->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *t = attr->add_t();
    t->set_data_type(static_cast<DataType>(value.data_type));
    for (int64_t d : value.shape) {
      t->add_dims(static_cast<uint64_t>(d));
    }
    t->set_raw_data(utils::ByteSpan(value.data));

    const OpsetId opset = DefaultOpset(13);

    Expect(registry, std::move(node), "test_cc_constant_benchmark", {opset},
           /*in_counts=*/{}, {kBenchmarkElementwiseSize}, []() mutable -> IoData {
             Tensor value = RandnTensor(DataType::FLOAT, {kBenchmarkElementwiseSize}, 987654321ULL);

             const OpsetId opset = DefaultOpset(13);

             const KernelContext constant_kernel_ctx{opset};
             const onnx_kernels::kernel::Constant constant_kernel{constant_kernel_ctx};

             Tensor y = constant_kernel(std::move(value));
             return IoData{/*inputs=*/{}, {std::move(y)}};
           });
    return;
  }

  NodeProto node;
  node.set_op_type("Constant");
  node.add_output("y");

  Tensor value = Tensor::FromFloat("", {2, 3}, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});

  AttributeProto *attr = node.add_attribute();
  attr->set_name("value");
  attr->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto *t = attr->add_t();
  t->set_data_type(static_cast<DataType>(value.data_type));
  for (int64_t d : value.shape) {
    t->add_dims(static_cast<uint64_t>(d));
  }
  // Pack the value bytes as raw_data (row-major little-endian, matching the
  // layout used by ``Tensor::data``).
  t->set_raw_data(utils::ByteSpan(value.data));

  const OpsetId opset = DefaultOpset(13);
  Expect(registry, std::move(node), "test_cc_constant", {opset},
         [value = std::move(value)]() mutable -> IoData {
           const OpsetId opset = DefaultOpset(13);

           const KernelContext ctx_2{opset};
           const onnx_kernels::kernel::Constant kernel_2{ctx_2};

           Tensor y = kernel_2(std::move(value));
           return IoData{/*inputs=*/{}, {std::move(y)}};
         });

  // Upstream ONNX backend test case for the ``Constant`` operator (mirrors the
  // ``onnx.backend.test.case.node.constant.Constant`` Python class). The
  // upstream case uses ``np.random.randn(5, 5).astype(np.float32)`` as the
  // ``value`` attribute; the values below are the exact float32 bytes captured
  // by the upstream ``test_constant`` test data (``onnx/backend/test/data/node/
  // test_constant/test_data_set_0/output_0.pb``), so this case matches
  // upstream byte-for-byte (output and value attribute) without depending on
  // NumPy at generation time.
  //
  // From Constant.export():
  {
    const std::vector<int64_t> values_shape = {5, 5};
    const std::vector<float> values_data = {
        1.764052391f,   0.4001572132f, 0.9787380099f,  2.240893126f,   1.867558002f,
        -0.9772778749f, 0.9500884414f, -0.1513572037f, -0.1032188535f, 0.4105985165f,
        0.1440435648f,  1.454273462f,  0.7610377073f,  0.1216750145f,  0.4438632429f,
        0.3336743414f,  1.494079113f,  -0.2051582634f, 0.3130677044f,  -0.854095757f,
        -2.552989721f,  0.6536185741f, 0.8644362092f,  -0.742165029f,  2.269754648f,
    };
    Tensor values = Tensor::FromFloat("", values_shape, values_data);

    NodeProto upstream_node;
    upstream_node.set_op_type("Constant");
    upstream_node.add_output("values");

    AttributeProto *upstream_attr = upstream_node.add_attribute();
    upstream_attr->set_name("value");
    upstream_attr->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *ut = upstream_attr->add_t();
    ut->set_name("const_tensor");
    ut->set_data_type(static_cast<DataType>(values.data_type));
    for (int64_t d : values.shape) {
      ut->add_dims(static_cast<uint64_t>(d));
    }
    // Mirror the upstream Python helper ``onnx.helper.make_tensor`` which
    // stores FLOAT tensors in the typed ``float_data`` field rather than
    // ``raw_data``; this keeps the produced model byte-equivalent to the
    // upstream ``test_constant/model.onnx``.
    for (float v : values_data) {
      ut->add_float_data(v);
    }

    Expect(registry, std::move(upstream_node), "test_constant", {opset},
           [values_shape, values_data]() -> IoData {
             Tensor values = Tensor::FromFloat("", values_shape, values_data);

             const OpsetId opset = DefaultOpset(13);

             const KernelContext ctx_3{opset};
             const onnx_kernels::kernel::Constant kernel_3{ctx_3};

             Tensor y_upstream = kernel_3(std::move(values));
             return IoData{{}, {std::move(y_upstream)}};
           });
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
    Expect(registry, std::move(n), "test_cc_constant_value_float", {v12}, []() -> IoData {
      Tensor y = Tensor::FromFloat("", /*shape=*/{}, {3.5f});
      return IoData{{}, {std::move(y)}};
    });
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
    Expect(registry, std::move(n), "test_cc_constant_value_floats", {v12}, [vals]() -> IoData {
      Tensor y = Tensor::FromFloat("", {static_cast<int64_t>(vals.size())}, vals);
      return IoData{{}, {std::move(y)}};
    });
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
    Expect(registry, std::move(n), "test_cc_constant_value_int", {v12}, []() -> IoData {
      Tensor y = Tensor::FromInt64("", /*shape=*/{}, {static_cast<int64_t>(42)});
      return IoData{{}, {std::move(y)}};
    });
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
    Expect(registry, std::move(n), "test_cc_constant_value_ints", {v12}, [vals]() -> IoData {
      Tensor y = Tensor::FromInt64("", {static_cast<int64_t>(vals.size())}, vals);
      return IoData{{}, {std::move(y)}};
    });
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
    Expect(registry, std::move(n), "test_cc_constant_value_string", {v12}, []() -> IoData {
      Tensor y = Tensor::FromStrings("", /*shape=*/{}, {std::string("hello")});
      return IoData{{}, {std::move(y)}};
    });
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
    Expect(registry, std::move(n), "test_cc_constant_value_strings", {v12}, [vals]() -> IoData {
      Tensor y = Tensor::FromStrings("", {static_cast<int64_t>(vals.size())}, vals);
      return IoData{{}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
