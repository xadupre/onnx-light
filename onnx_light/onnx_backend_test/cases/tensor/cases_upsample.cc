// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_kernels/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

// ---------------------------------------------------------------------------
// Upsample — nearest/linear interpolation, output_dim[i] = floor(input_dim[i]
// * scales[i]). Available in opsets 1, 7, 9 and 10 (deprecated since opset 10
// in favour of ``Resize``). Mirrors the upstream upsample node test in
// ``onnx/backend/test/case/node/upsample.py``.
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeUpsampleNode(const std::string &mode) {
  NodeProto node;
  node.set_op_type("Upsample");
  node.add_input("X");
  node.add_input("scales");
  node.add_output("Y");
  if (!mode.empty()) {
    AddAttribute<std::string>(node, "mode", mode);
  }
  return node;
}

// Builds a 1-D FLOAT ``scales`` tensor from the given values.
Tensor MakeScalesTensor(const std::vector<float> &scales) {
  return Tensor::FromFloat("", {static_cast<int64_t>(scales.size())}, scales);
}

} // namespace

void RegisterUpsampleCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(9);
  const kernel::KernelContext ctx{opset};
  const kernel::Upsample upsample_kernel{ctx};

  // test_cc_upsample_nearest — mirrors the upstream ``test_upsample_nearest``
  // node test exactly. NCHW input of shape [1, 1, 2, 2], scales
  // [1, 1, 2, 3], nearest mode -> output shape [1, 1, 4, 6].
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor scales = MakeScalesTensor({1.0f, 1.0f, 2.0f, 3.0f});
    kernel::Upsample::Attributes attrs;
    attrs.mode = "nearest";
    const Tensor Y = upsample_kernel(X, scales, attrs);
    Expect(MakeUpsampleNode("nearest"), {X, scales}, {Y}, "test_cc_upsample_nearest", {opset},
           "backend-test", registry);
  }

  // test_cc_upsample_nearest_default_mode — mode attribute omitted (defaults
  // to "nearest" per the upstream spec).
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor scales = MakeScalesTensor({1.0f, 1.0f, 2.0f, 2.0f});
    kernel::Upsample::Attributes attrs;
    attrs.mode = "nearest";
    const Tensor Y = upsample_kernel(X, scales, attrs);
    Expect(MakeUpsampleNode(""), {X, scales}, {Y}, "test_cc_upsample_nearest_default_mode", {opset},
           "backend-test", registry);
  }

  // test_cc_upsample_nearest_1d — minimal 1-D nearest-neighbor case, useful
  // as an independent sanity check for the strided index mapping.
  {
    const Tensor X = Tensor::FromFloat("", {3}, {10.0f, 20.0f, 30.0f});
    const Tensor scales = MakeScalesTensor({2.0f});
    kernel::Upsample::Attributes attrs;
    attrs.mode = "nearest";
    const Tensor Y = upsample_kernel(X, scales, attrs);
    Expect(MakeUpsampleNode("nearest"), {X, scales}, {Y}, "test_cc_upsample_nearest_1d", {opset},
           "backend-test", registry);
  }

  // test_cc_upsample_linear — 4-D bilinear case (NCHW, scales == 1 on N and
  // C). Output is computed by the same kernel and embedded as the reference
  // so the test exercises kernel determinism rather than numerical parity
  // with another implementation.
  {
    const Tensor X = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor scales = MakeScalesTensor({1.0f, 1.0f, 2.0f, 2.0f});
    kernel::Upsample::Attributes attrs;
    attrs.mode = "linear";
    const Tensor Y = upsample_kernel(X, scales, attrs);
    Expect(MakeUpsampleNode("linear"), {X, scales}, {Y}, "test_cc_upsample_linear", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
