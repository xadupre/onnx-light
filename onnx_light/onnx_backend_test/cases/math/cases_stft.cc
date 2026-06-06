// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Build an STFT v17 node. ``with_window`` toggles wiring the optional
// ``window`` input; ``with_frame_length`` toggles wiring the optional
// ``frame_length`` input. ``onesided`` controls the corresponding attribute.
NodeProto MakeSTFTNode(bool with_window, bool with_frame_length, int64_t onesided) {
  NodeProto node;
  node.set_op_type("STFT");
  node.add_input("signal");
  node.add_input("frame_step");
  node.add_input(with_window ? "window" : "");
  if (with_frame_length) {
    node.add_input("frame_length");
  }
  node.add_output("output");
  AddAttribute<int64_t>(node, "onesided", onesided);
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// STFT — Short-time Fourier Transform (opset 17). Covers the basic real-valued
// signal path with and without a rectangular window, plus an explicit
// ``frame_length`` input.
// ---------------------------------------------------------------------------
void RegisterSTFTCases(std::vector<TestCase> &registry) {
  const OpsetId opset_v17 = DefaultOpset(17);
  const kernel::KernelContext ctx_v17{opset_v17};
  const kernel::STFT stft_v17{ctx_v17};

  // Real input of shape [1, 16, 1] (single batch, 16 real samples).
  const std::vector<float> samples = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
      8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
  };
  Tensor signal = Tensor::FromFloat("signal", {1, 16, 1}, samples);
  Tensor frame_step = Tensor::FromInt64("frame_step", {}, {4});
  Tensor frame_length = Tensor::FromInt64("frame_length", {}, {8});

  // --- STFT with frame_length input, no window, onesided=1.
  {
    Tensor y = stft_v17(signal, frame_step, /*window=*/nullptr, &frame_length, /*onesided=*/true);
    Expect(MakeSTFTNode(/*with_window=*/false, /*with_frame_length=*/true, /*onesided=*/1),
           {signal, frame_step, frame_length}, {y}, "test_cc_stft", {opset_v17}, "backend-test",
           registry);
    registry.back().atol = 1e-5;
  }

  // --- STFT with both window and frame_length, onesided=1.
  {
    Tensor window = Tensor::FromFloat(
        "window", {8}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}); // rectangular
    Tensor y = stft_v17(signal, frame_step, &window, &frame_length, /*onesided=*/true);
    Expect(MakeSTFTNode(/*with_window=*/true, /*with_frame_length=*/true, /*onesided=*/1),
           {signal, frame_step, window, frame_length}, {y}, "test_cc_stft_with_window", {opset_v17},
           "backend-test", registry);
    registry.back().atol = 1e-5;
  }

  // --- STFT with onesided=0 (two-sided, complex output).
  {
    Tensor y = stft_v17(signal, frame_step, /*window=*/nullptr, &frame_length, /*onesided=*/false);
    Expect(MakeSTFTNode(/*with_window=*/false, /*with_frame_length=*/true, /*onesided=*/0),
           {signal, frame_step, frame_length}, {y}, "test_cc_stft_twosided", {opset_v17},
           "backend-test", registry);
    registry.back().atol = 1e-5;
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
