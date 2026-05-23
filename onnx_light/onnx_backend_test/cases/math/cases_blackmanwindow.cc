// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"

#include <cmath>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Builds an OperatorSetIdProto for the default ai.onnx domain.
OperatorSetIdProto DefaultOpset(int64_t version) {
  OperatorSetIdProto osid;
  osid.set_domain("");
  osid.set_version(version);
  return osid;
}

// Computes the Blackman window of a given size. ``divisor`` is ``size`` for the
// default periodic variant and ``size - 1`` for the symmetric variant. Matches
// the reference formula used by ``onnx_light/backend/test/case/node/blackmanwindow.py``.
std::vector<float> BlackmanWindowValues(int32_t size, double divisor) {
  constexpr double a0 = 0.42;
  constexpr double a1 = -0.5;
  constexpr double a2 = 0.08;
  constexpr double kPi = 3.14159265358979323846;
  std::vector<float> y(static_cast<size_t>(size));
  for (int32_t n = 0; n < size; ++n) {
    const double k = static_cast<double>(n) / divisor;
    y[static_cast<size_t>(n)] =
        static_cast<float>(a0 + a1 * std::cos(2.0 * kPi * k) + a2 * std::cos(4.0 * kPi * k));
  }
  return y;
}

} // namespace

// ---------------------------------------------------------------------------
// BlackmanWindow — generates a Blackman window of length ``size`` (since opset
// 17). Mirrors ``onnx_light/backend/test/case/node/blackmanwindow.py`` and
// registers both the default periodic case and the symmetric variant
// (``periodic = 0``).
// ---------------------------------------------------------------------------
void RegisterBlackmanWindowCases(std::vector<TestCase> &registry) {
  constexpr int32_t kSize = 10;

  // Default periodic variant.
  {
    NodeProto node;
    node.set_op_type("BlackmanWindow");
    node.add_input("x");
    node.add_output("y");

    std::vector<float> y = BlackmanWindowValues(kSize, /*divisor=*/static_cast<double>(kSize));

    Expect(node, {Tensor::FromInt32("x", {}, {kSize})}, {Tensor::FromFloat("y", {kSize}, y)},
           "test_cc_blackmanwindow", {DefaultOpset(17)}, "backend-test", registry);
  }

  // Symmetric variant (periodic = 0).
  {
    NodeProto node;
    node.set_op_type("BlackmanWindow");
    node.add_input("x");
    node.add_output("y");

    AttributeProto *attr = node.add_attribute();
    attr->set_name("periodic");
    attr->set_type(AttributeProto::AttributeType::INT);
    attr->set_i(0);

    std::vector<float> y = BlackmanWindowValues(kSize, /*divisor=*/static_cast<double>(kSize - 1));

    Expect(node, {Tensor::FromInt32("x", {}, {kSize})}, {Tensor::FromFloat("y", {kSize}, y)},
           "test_cc_blackmanwindow_symmetric", {DefaultOpset(17)}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
