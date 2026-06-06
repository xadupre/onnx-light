// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeDepthToSpaceNode(int64_t blocksize, const std::string &mode) {
  NodeProto node;
  node.set_op_type("DepthToSpace");
  node.add_input("input");
  node.add_output("output");
  AddAttribute<int64_t>(node, "blocksize", blocksize);
  if (!mode.empty()) {
    AddAttribute<std::string>(node, "mode", mode);
  }
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// DepthToSpace — rearranges depth into spatial blocks (inverse of
// SpaceToDepth). Available since opset 1 in the ai.onnx domain; the ``mode``
// attribute (DCR / CRD) was added in opset 11.
// ---------------------------------------------------------------------------
void RegisterDepthToSpaceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::DepthToSpace d2s{ctx};

  // test_cc_depthtospace_dcr — N=1, C=8 (=2*2*2), H=2, W=3, blocksize=2.
  // Input values are simply [0, 1, ..., 47] so the output ordering is easy
  // to inspect by hand.
  {
    std::vector<float> values(48);
    for (int i = 0; i < 48; ++i) {
      values[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }
    const Tensor input = Tensor::FromFloat("", {1, 8, 2, 3}, values);
    kernel::DepthToSpace::Attributes attrs;
    attrs.blocksize = 2;
    attrs.mode = "DCR";
    const Tensor output = d2s(input, attrs);
    Expect(MakeDepthToSpaceNode(2, "DCR"), {input}, {output}, "test_cc_depthtospace_dcr", {opset},
           "backend-test", registry);
  }

  // test_cc_depthtospace_crd — same input, CRD mode.
  {
    std::vector<float> values(48);
    for (int i = 0; i < 48; ++i) {
      values[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }
    const Tensor input = Tensor::FromFloat("", {1, 8, 2, 3}, values);
    kernel::DepthToSpace::Attributes attrs;
    attrs.blocksize = 2;
    attrs.mode = "CRD";
    const Tensor output = d2s(input, attrs);
    Expect(MakeDepthToSpaceNode(2, "CRD"), {input}, {output}, "test_cc_depthtospace_crd", {opset},
           "backend-test", registry);
  }

  // test_cc_depthtospace_default_mode — mode attribute omitted (defaults to
  // DCR per the upstream spec).
  {
    std::vector<float> values(16);
    for (int i = 0; i < 16; ++i) {
      values[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }
    const Tensor input = Tensor::FromFloat("", {1, 4, 2, 2}, values);
    kernel::DepthToSpace::Attributes attrs;
    attrs.blocksize = 2;
    attrs.mode = "DCR";
    const Tensor output = d2s(input, attrs);
    Expect(MakeDepthToSpaceNode(2, ""), {input}, {output}, "test_cc_depthtospace_default_mode",
           {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
