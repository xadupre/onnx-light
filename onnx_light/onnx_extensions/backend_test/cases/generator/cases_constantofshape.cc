// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/generator/include_generator_cases.h"
#include "onnx_extensions/kernels/kernels/generator/include_generator_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Builds a single-node ``ConstantOfShape`` topology with the given
// ``shape`` input and ``value`` attribute and registers the case under
// ``case_name``.
void RegisterOneConstantOfShape(const std::string &case_name,
                                const std::vector<int64_t> &shape_values, const Tensor &value,
                                std::vector<TestCase> &registry, const OpsetId &opset) {
  NodeProto node;
  node.set_op_type("ConstantOfShape");
  node.add_input("x");
  node.add_output("y");

  AttributeProto *attr = node.add_attribute();
  attr->set_name("value");
  attr->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto *t = attr->add_t();
  Expect(registry, std::move(node), case_name, {opset}, [=]() -> IoData {
    t->set_data_type(static_cast<DataType>(value.data_type));
    for (int64_t d : value.shape) {
      t->add_dims(d);
    }
    t->set_raw_data(utils::ByteSpan(value.data));

    const Tensor shape_input =
        Tensor::FromInt64("x", {static_cast<int64_t>(shape_values.size())}, shape_values);

    const KernelContext ctx{opset};
    Tensor y = onnx_kernels::kernel::ConstantOfShape(ctx)(shape_input, value);

    return IoData{{std::move(shape_input)}, {std::move(y)}};
  });
}

} // namespace

// ---------------------------------------------------------------------------
// ConstantOfShape — produces a tensor of the shape given by the 1-D int64
// input filled with the (single) scalar value of the ``value`` TENSOR
// attribute. Defaults to a FLOAT zero when ``value`` is not present. Two
// cases mirror the upstream ``onnx.backend.test.case.node.constantofshape``
// exports (``test_constantofshape_int_zeros`` /
// ``test_constantofshape_float_ones``).
// ---------------------------------------------------------------------------
void RegisterConstantOfShapeCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(20);

  if (mode == TestMode::BENCHMARK) {
    const Tensor value = Tensor::FromFloat("", /*shape=*/{1}, {1.0f});
    const std::vector<int64_t> shape_values = {kBenchmarkElementwiseSize};
    NodeProto node;
    node.set_op_type("ConstantOfShape");
    node.add_input("x");
    node.add_output("y");

    AttributeProto *attr = node.add_attribute();
    attr->set_name("value");
    attr->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *t = attr->add_t();
    t->set_data_type(static_cast<DataType>(value.data_type));
    for (int64_t d : value.shape) {
      t->add_dims(d);
    }
    t->set_raw_data(utils::ByteSpan(value.data));

    const KernelContext ctx{opset};
    const onnx_kernels::kernel::ConstantOfShape constant_of_shape_kernel{ctx};
    Expect(registry, std::move(node), "test_constantofshape_float_ones_benchmark", {opset}, {1},
           {kBenchmarkElementwiseSize},
           [constant_of_shape_kernel, shape_values, value]() -> IoData {
             Tensor shape_input =
                 Tensor::FromInt64("x", {static_cast<int64_t>(shape_values.size())}, shape_values);
             Tensor y = constant_of_shape_kernel(shape_input, value);
             return IoData{{std::move(shape_input)}, {std::move(y)}};
           });
    return;
  }

  // Upstream ``test_constantofshape_float_ones``: shape ``[4, 3, 2]``,
  // ``value`` = FLOAT 1.0 -> output is a FLOAT 1-filled tensor.
  {
    const Tensor value = Tensor::FromFloat("", /*shape=*/{1}, {1.0f});
    RegisterOneConstantOfShape("test_constantofshape_float_ones", {4, 3, 2}, value, registry,
                               opset);
  }

  // Upstream ``test_constantofshape_int_zeros``: shape ``[10, 6]``,
  // ``value`` = INT32 0 -> output is an INT32 0-filled tensor.
  {
    const Tensor value = Tensor::FromInt32("", /*shape=*/{1}, {0});
    RegisterOneConstantOfShape("test_constantofshape_int_zeros", {10, 6}, value, registry, opset);
  }

  // Upstream ``test_constantofshape_int_shape_zero``: shape ``[0]`` ->
  // output is a length-0 INT32 1-filled tensor.
  {
    const Tensor value = Tensor::FromInt32("", /*shape=*/{1}, {1});
    RegisterOneConstantOfShape("test_constantofshape_int_shape_zero", {0}, value, registry, opset);
  }

  // Library-local case with a non-trivial fill value, verifying that the
  // attribute's element is broadcast to every output position.
  {
    const Tensor value = Tensor::FromInt64("", /*shape=*/{1}, {static_cast<int64_t>(42)});
    RegisterOneConstantOfShape("test_cc_constantofshape_int64_fortytwo", {2, 3}, value, registry,
                               opset);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
