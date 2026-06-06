// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeSpaceToDepthNode(int64_t blocksize) {
  NodeProto node;
  node.set_op_type("SpaceToDepth");
  node.add_input("input");
  node.add_output("output");
  AddAttribute<int64_t>(node, "blocksize", blocksize);
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// SpaceToDepth — rearranges blocks of spatial data into depth (inverse of
// DepthToSpace). Available since opset 1 in the ai.onnx domain; the type set
// was extended in opset 13.
// ---------------------------------------------------------------------------
void RegisterSpaceToDepthCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::SpaceToDepth s2d{ctx};

  // test_cc_spacetodepth_example — small N=1, C=2, H=2, W=4 input with
  // blocksize=2. Input values are simply [0, 1, ..., 15] so the output
  // ordering can be inspected by hand.
  {
    std::vector<float> values(16);
    for (int i = 0; i < 16; ++i) {
      values[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }
    const Tensor input = Tensor::FromFloat("", {1, 2, 2, 4}, values);
    kernel::SpaceToDepth::Attributes attrs;
    attrs.blocksize = 2;
    const Tensor output = s2d(input, attrs);
    Expect(MakeSpaceToDepthNode(2), {input}, {output}, "test_cc_spacetodepth_example", {opset},
           "backend-test", registry);
  }

  // test_cc_spacetodepth — larger N=2, C=3, H=4, W=6 input with blocksize=2.
  {
    const int64_t total = 2 * 3 * 4 * 6;
    std::vector<float> values(static_cast<std::size_t>(total));
    for (int64_t i = 0; i < total; ++i) {
      values[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }
    const Tensor input = Tensor::FromFloat("", {2, 3, 4, 6}, values);
    kernel::SpaceToDepth::Attributes attrs;
    attrs.blocksize = 2;
    const Tensor output = s2d(input, attrs);
    Expect(MakeSpaceToDepthNode(2), {input}, {output}, "test_cc_spacetodepth", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
