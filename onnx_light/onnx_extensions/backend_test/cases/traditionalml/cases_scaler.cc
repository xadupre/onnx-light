// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

void AddFloatsAttr(NodeProto &node, const char *name, const std::vector<float> &values) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::FLOATS);
  for (float v : values) {
    attr->add_floats(v);
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Scaler — y = (x - offset) * scale, element-wise, with offset/scale either
// scalar (length 1) or broadcast over the last axis of the input. Output is
// always float. Mirrors the behaviour of the upstream ``ai.onnx.ml::Scaler``
// operator (since opset 1 in the ``ai.onnx.ml`` domain).
// ---------------------------------------------------------------------------
void RegisterScalerCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset("ai.onnx.ml", 1);
  const OpsetId default_opset = DefaultOpset(13);
  const auto scaler = MakeReferenceKernel<onnx_kernels::kernel::Scaler>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Scaler");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const std::vector<float> offset{0.5f, 1.0f, 1.5f};
    const std::vector<float> scale{2.0f, 0.5f, 1.0f};
    AddFloatsAttr(node, "offset", offset);
    AddFloatsAttr(node, "scale", scale);

    Expect(registry, std::move(node), "test_cc_scaler_float_benchmark", {default_opset, opset},
           {24576}, {24576}, [scaler, offset, scale]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {8192, 3}, 2601);
             Tensor y = scaler.Invoke([&](const auto &kernel) {
               return kernel.template operator()<float>(x, offset, scale);
             });
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // Per-feature offset/scale on a ``(2, 3)`` float input.
  {
    NodeProto node;
    node.set_op_type("Scaler");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");
    const std::vector<float> offset{0.5f, 1.0f, 1.5f};
    const std::vector<float> scale{2.0f, 0.5f, 1.0f};
    AddFloatsAttr(node, "offset", offset);
    AddFloatsAttr(node, "scale", scale);
    Expect(registry, std::move(node), "test_cc_scaler_float", {default_opset, opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromFloat("", {2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
             Tensor y = scaler.Invoke([&](const auto &kernel) {
               return kernel.template operator()<float>(x, offset, scale);
             });

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // Scalar (length-1) offset/scale broadcast to every element of an int64
  // input. Exercises the non-default input element type and the
  // float-output rule.
  {
    NodeProto node;
    node.set_op_type("Scaler");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");
    const std::vector<float> offset{1.0f};
    const std::vector<float> scale{0.5f};
    AddFloatsAttr(node, "offset", offset);
    AddFloatsAttr(node, "scale", scale);
    Expect(registry, std::move(node), "test_cc_scaler_int64", {default_opset, opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromInt64("", {5}, {0, 1, 2, 3, 4});
             Tensor y = scaler.Invoke([&](const auto &kernel) {
               return kernel.template operator()<int64_t>(x, offset, scale);
             });

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
