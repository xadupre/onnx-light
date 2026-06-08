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

  // test_cc_depthtospace_example — mirrors ONNX ``test_depthtospace_example``
  // (DCR mode, blocksize=2, 1x8x2x3 input). Values use the same non-contiguous
  // step as ONNX so the output matches by inspection.
  {
    const Tensor input = Tensor::FromFloat(
        "x", {1, 8, 2, 3},
        {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
         18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
         36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
         54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f});
    kernel::DepthToSpace::Attributes attrs;
    attrs.blocksize = 2;
    attrs.mode = "DCR";
    const Tensor output = d2s(input, attrs);
    Expect(MakeDepthToSpaceNode(2, "DCR"), {input}, {output}, "test_cc_depthtospace_example",
           {opset}, "backend-test", registry);
  }

  // test_cc_depthtospace_crd_mode_example — mirrors ONNX
  // ``test_depthtospace_crd_mode_example`` (CRD mode, blocksize=2, same input).
  {
    const Tensor input = Tensor::FromFloat(
        "x", {1, 8, 2, 3},
        {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
         18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
         36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
         54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f});
    kernel::DepthToSpace::Attributes attrs;
    attrs.blocksize = 2;
    attrs.mode = "CRD";
    const Tensor output = d2s(input, attrs);
    Expect(MakeDepthToSpaceNode(2, "CRD"), {input}, {output},
           "test_cc_depthtospace_crd_mode_example", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
