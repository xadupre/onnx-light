// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

void AddStringAttr(NodeProto &node, const char *name, const std::string &value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::STRING);
  attr->set_s(value);
}

} // namespace

// ---------------------------------------------------------------------------
// Normalizer — normalize each row of the input ([C] or [N,C]) along the last
// axis using one of three modes: ``MAX``, ``L1`` or ``L2``. Output is always
// float. Mirrors the upstream ``ai.onnx.ml::Normalizer`` operator (since
// opset 1 in the ``ai.onnx.ml`` domain).
// ---------------------------------------------------------------------------
void RegisterNormalizerCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset("ai.onnx.ml", 1);
  const OpsetId default_opset = DefaultOpset(13);
  const auto normalizer = MakeReferenceKernel<onnx_kernels::kernel::Normalizer>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Normalizer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");
    AddStringAttr(node, "norm", "L2");

    Expect(registry, std::move(node), "test_cc_normalizer_l2_float_benchmark",
           {default_opset, opset}, {524288}, {524288}, [normalizer]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {8192, 64}, 2621);
             Tensor y = normalizer.Invoke(
                 [&](const auto &kernel) { return kernel.template operator()<float>(x, "L2"); });
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // L2 normalization on a [2, 3] float input — rows normalized independently.
  {
    NodeProto node;
    node.set_op_type("Normalizer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");
    AddStringAttr(node, "norm", "L2");
    Expect(registry, std::move(node), "test_cc_normalizer_l2_float", {default_opset, opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 2.0f, 0.0f, 3.0f, 4.0f});
             Tensor y = normalizer.Invoke(
                 [&](const auto &kernel) { return kernel.template operator()<float>(x, "L2"); });

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // L1 normalization on a single-row [C] int64 input — exercises the rank-1
  // path and the non-float input element type.
  {
    NodeProto node;
    node.set_op_type("Normalizer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");
    AddStringAttr(node, "norm", "L1");
    Expect(registry, std::move(node), "test_cc_normalizer_l1_int64", {default_opset, opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromInt64("", {4}, {1, -1, 2, -2});
             Tensor y = normalizer.Invoke(
                 [&](const auto &kernel) { return kernel.template operator()<int64_t>(x, "L1"); });

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // MAX normalization on a [2, 3] double input — exercises the default
  // (``MAX``) mode with a per-row signed ``max(x)`` divisor.
  {
    NodeProto node;
    node.set_op_type("Normalizer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");
    AddStringAttr(node, "norm", "MAX");
    Expect(registry, std::move(node), "test_cc_normalizer_max_double", {default_opset, opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromDouble("", {2, 3}, {1.0, -3.0, 2.0, 0.0, 0.0, 0.0});
             Tensor y = normalizer.Invoke(
                 [&](const auto &kernel) { return kernel.template operator()<double>(x, "MAX"); });

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
