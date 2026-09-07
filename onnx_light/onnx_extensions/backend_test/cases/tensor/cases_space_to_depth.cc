// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

NodeProto MakeSpaceToDepthNode(int64_t blocksize, const std::string &mode = "") {
  NodeProto node;
  node.set_op_type("SpaceToDepth");
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
// SpaceToDepth — rearranges blocks of spatial data into depth (inverse of
// DepthToSpace). Available since opset 1 in the ai.onnx domain; the type set
// was extended in opset 13.
// ---------------------------------------------------------------------------
void RegisterSpaceToDepthCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeSpaceToDepthNode(2);
    Expect(registry, std::move(node), "test_cc_spacetodepth_example_benchmark", {opset}, {4194304},
           {4194304}, []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext s2d_ctx{opset};
             const onnx_kernels::kernel::SpaceToDepth s2d{s2d_ctx};

             Tensor input = RandnTensor(DataType::FLOAT, {1, 2, 1024, 2048}, 2001);
             onnx_kernels::kernel::SpaceToDepth::Attributes attrs;
             attrs.blocksize = 2;
             Tensor output = s2d(input, attrs);
             return IoData{{std::move(input)}, {std::move(output)}};
           });
    return;
  }

  for (const std::string layout : {"DCR", "CRD"}) {
    const std::string name = layout == "DCR" ? "dcr" : "crd";
    Expect(registry, MakeSpaceToDepthNode(2, layout), "test_spacetodepth_" + name + "_mode_example",
           {DefaultOpset(28)}, [layout]() -> IoData {
             const KernelContext ctx{DefaultOpset(28)};
             const onnx_kernels::kernel::SpaceToDepth kernel{ctx};
             const std::vector<float> values =
                 layout == "DCR"
                     ? std::vector<float>{0,  18, 1,  19, 2,  20, 36, 54, 37, 55, 38, 56,
                                          3,  21, 4,  22, 5,  23, 39, 57, 40, 58, 41, 59,
                                          9,  27, 10, 28, 11, 29, 45, 63, 46, 64, 47, 65,
                                          12, 30, 13, 31, 14, 32, 48, 66, 49, 67, 50, 68}
                     : std::vector<float>{0,  9,  1,  10, 2,  11, 18, 27, 19, 28, 20, 29,
                                          3,  12, 4,  13, 5,  14, 21, 30, 22, 31, 23, 32,
                                          36, 45, 37, 46, 38, 47, 54, 63, 55, 64, 56, 65,
                                          39, 48, 40, 49, 41, 50, 57, 66, 58, 67, 59, 68};
             Tensor input = Tensor::FromFloat("", {1, 2, 4, 6}, values);
             onnx_kernels::kernel::SpaceToDepth::Attributes attrs;
             attrs.blocksize = 2;
             attrs.mode = layout;
             Tensor output = kernel(input, attrs);
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }

  // test_cc_spacetodepth_example — small N=1, C=2, H=2, W=4 input with
  // blocksize=2. Input values are simply [0, 1, ..., 15] so the output
  // ordering can be inspected by hand.
  {
    Expect(registry, MakeSpaceToDepthNode(2), "test_cc_spacetodepth_example", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext s2d_ctx{opset};
             const onnx_kernels::kernel::SpaceToDepth s2d{s2d_ctx};

             std::vector<float> values(16);
             for (int i = 0; i < 16; ++i) {
               values[static_cast<std::size_t>(i)] = static_cast<float>(i);
             }
             const Tensor input = Tensor::FromFloat("", {1, 2, 2, 4}, values);
             onnx_kernels::kernel::SpaceToDepth::Attributes attrs;
             attrs.blocksize = 2;
             const Tensor output = s2d(input, attrs);
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }

  // test_cc_spacetodepth — larger N=2, C=3, H=4, W=6 input with blocksize=2.
  {
    Expect(registry, MakeSpaceToDepthNode(2), "test_cc_spacetodepth", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext s2d_ctx{opset};
      const onnx_kernels::kernel::SpaceToDepth s2d{s2d_ctx};

      const int64_t total = 2 * 3 * 4 * 6;
      std::vector<float> values(static_cast<std::size_t>(total));
      for (int64_t i = 0; i < total; ++i) {
        values[static_cast<std::size_t>(i)] = static_cast<float>(i);
      }
      const Tensor input = Tensor::FromFloat("", {2, 3, 4, 6}, values);
      onnx_kernels::kernel::SpaceToDepth::Attributes attrs;
      attrs.blocksize = 2;
      const Tensor output = s2d(input, attrs);
      return IoData{{std::move(input)}, {std::move(output)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
