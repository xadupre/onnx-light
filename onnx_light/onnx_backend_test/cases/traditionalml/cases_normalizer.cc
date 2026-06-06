// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_kernels/test_case.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

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
void RegisterNormalizerCases(std::vector<TestCase> &registry) {
  const OpsetId opset("ai.onnx.ml", 1);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::Normalizer normalizer{ctx};

  // L2 normalization on a [2, 3] float input — rows normalized independently.
  {
    NodeProto node;
    node.set_op_type("Normalizer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");
    AddStringAttr(node, "norm", "L2");

    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 2.0f, 0.0f, 3.0f, 4.0f});
    Tensor y = normalizer.operator()<float>(x, "L2");

    Expect(node, {x}, {y}, "test_cc_normalizer_l2_float", {default_opset, opset}, "backend-test",
           registry);
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

    Tensor x = Tensor::FromInt64("", {4}, {1, -1, 2, -2});
    Tensor y = normalizer.operator()<int64_t>(x, "L1");

    Expect(node, {x}, {y}, "test_cc_normalizer_l1_int64", {default_opset, opset}, "backend-test",
           registry);
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

    Tensor x = Tensor::FromDouble("", {2, 3}, {1.0, -3.0, 2.0, 0.0, 0.0, 0.0});
    Tensor y = normalizer.operator()<double>(x, "MAX");

    Expect(node, {x}, {y}, "test_cc_normalizer_max_double", {default_opset, opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
