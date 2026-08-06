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
void RegisterDepthToSpaceCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::DepthToSpace d2s{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeDepthToSpaceNode(2, "DCR");
    Expect(registry, std::move(node), "test_cc_depthtospace_dcr_benchmark", {opset},
           {1 * 8 * 512 * 1024}, {1 * 8 * 512 * 1024}, [d2s]() -> IoData {
             Tensor input =
                 Tensor::FromFloat("", {1, 8, 512, 1024}, Randn<float>({1, 8, 512, 1024}, 2001));
             onnx_kernels::kernel::DepthToSpace::Attributes attrs;
             attrs.blocksize = 2;
             attrs.mode = "DCR";
             Tensor output = d2s(input, attrs);
             return IoData{{std::move(input)}, {std::move(output)}};
           });
    return;
  }

  // test_cc_depthtospace_dcr — N=1, C=8 (=2*2*2), H=2, W=3, blocksize=2.
  // Input values are simply [0, 1, ..., 47] so the output ordering is easy
  // to inspect by hand.
  {
    Expect(registry, MakeDepthToSpaceNode(2, "DCR"), "test_cc_depthtospace_dcr", {opset},
           [=]() -> IoData {
             std::vector<float> values(48);
             for (int i = 0; i < 48; ++i) {
               values[static_cast<std::size_t>(i)] = static_cast<float>(i);
             }
             const Tensor input = Tensor::FromFloat("", {1, 8, 2, 3}, values);
             onnx_kernels::kernel::DepthToSpace::Attributes attrs;
             attrs.blocksize = 2;
             attrs.mode = "DCR";
             const Tensor output = d2s(input, attrs);
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }

  // test_cc_depthtospace_crd — same input, CRD mode.
  {
    Expect(registry, MakeDepthToSpaceNode(2, "CRD"), "test_cc_depthtospace_crd", {opset},
           [=]() -> IoData {
             std::vector<float> values(48);
             for (int i = 0; i < 48; ++i) {
               values[static_cast<std::size_t>(i)] = static_cast<float>(i);
             }
             const Tensor input = Tensor::FromFloat("", {1, 8, 2, 3}, values);
             onnx_kernels::kernel::DepthToSpace::Attributes attrs;
             attrs.blocksize = 2;
             attrs.mode = "CRD";
             const Tensor output = d2s(input, attrs);
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }

  // test_cc_depthtospace_default_mode — mode attribute omitted (defaults to
  // DCR per the upstream spec).
  {
    Expect(registry, MakeDepthToSpaceNode(2, ""), "test_cc_depthtospace_default_mode", {opset},
           [=]() -> IoData {
             std::vector<float> values(16);
             for (int i = 0; i < 16; ++i) {
               values[static_cast<std::size_t>(i)] = static_cast<float>(i);
             }
             const Tensor input = Tensor::FromFloat("", {1, 4, 2, 2}, values);
             onnx_kernels::kernel::DepthToSpace::Attributes attrs;
             attrs.blocksize = 2;
             attrs.mode = "DCR";
             const Tensor output = d2s(input, attrs);
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }

  // test_cc_depthtospace_example — mirrors ONNX ``test_depthtospace_example``
  // (DCR mode, blocksize=2, 1x8x2x3 input). Values use the same non-contiguous
  // step as ONNX so the output matches by inspection.
  {
    Expect(registry, MakeDepthToSpaceNode(2, "DCR"), "test_cc_depthtospace_example", {opset},
           [=]() -> IoData {
             const Tensor input = Tensor::FromFloat(
                 "x", {1, 8, 2, 3},
                 {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                  13.0f, 14.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 27.0f, 28.0f,
                  29.0f, 30.0f, 31.0f, 32.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f,
                  45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 54.0f, 55.0f, 56.0f, 57.0f,
                  58.0f, 59.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f});
             onnx_kernels::kernel::DepthToSpace::Attributes attrs;
             attrs.blocksize = 2;
             attrs.mode = "DCR";
             const Tensor output = d2s(input, attrs);
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }

  // test_cc_depthtospace_crd_mode_example — mirrors ONNX
  // ``test_depthtospace_crd_mode_example`` (CRD mode, blocksize=2, same input).
  {
    Expect(registry, MakeDepthToSpaceNode(2, "CRD"), "test_cc_depthtospace_crd_mode_example",
           {opset}, [=]() -> IoData {
             const Tensor input = Tensor::FromFloat(
                 "x", {1, 8, 2, 3},
                 {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                  13.0f, 14.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 27.0f, 28.0f,
                  29.0f, 30.0f, 31.0f, 32.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f,
                  45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 54.0f, 55.0f, 56.0f, 57.0f,
                  58.0f, 59.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f});
             onnx_kernels::kernel::DepthToSpace::Attributes attrs;
             attrs.blocksize = 2;
             attrs.mode = "CRD";
             const Tensor output = d2s(input, attrs);
             return IoData{{std::move(input)}, {std::move(output)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
