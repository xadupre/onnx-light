// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

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
void RegisterSTFTCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset_v17 = DefaultOpset(17);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node =
        MakeSTFTNode(/*with_window=*/false, /*with_frame_length=*/true, /*onesided=*/1);
    const std::vector<int64_t> shape = {1, 65536, 1};
    const int64_t signal_count = 65536;
    const int64_t scalar_count = 1;
    const int64_t output_count = 1 * 127 * 513 * 2;
    Expect(registry, std::move(node), "test_cc_stft_benchmark", {opset_v17},
           {signal_count, scalar_count, scalar_count}, {output_count}, [shape]() -> IoData {
             const OpsetId opset_v17 = DefaultOpset(17);

             const KernelContext stft_v17_ctx{opset_v17};
             const onnx_kernels::kernel::STFT stft_v17{stft_v17_ctx};

             Tensor signal_b = RandnTensor(DataType::FLOAT, shape, 446);
             Tensor frame_step_b = Tensor::FromInt64("frame_step", {}, {512});
             Tensor frame_length_b = Tensor::FromInt64("frame_length", {}, {1024});
             Tensor y = stft_v17(signal_b, frame_step_b, /*window=*/nullptr, &frame_length_b,
                                 /*onesided=*/true);
             return IoData{
                 {std::move(signal_b), std::move(frame_step_b), std::move(frame_length_b)},
                 {std::move(y)}};
           });
    return;
  }

  // Real input of shape [1, 16, 1] (single batch, 16 real samples).
  const std::vector<float> samples = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
      8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
  };
  Tensor signal = Tensor::FromFloat("signal", {1, 16, 1}, samples);
  Tensor frame_step = Tensor::FromInt64("frame_step", {}, {4});
  Tensor frame_length = Tensor::FromInt64("frame_length", {}, {8});

  // --- STFT with frame_length defaulting to the signal length.
  {
    Expect(registry,
           MakeSTFTNode(/*with_window=*/false, /*with_frame_length=*/false, /*onesided=*/1),
           "test_cc_stft_default_frame_length", {opset_v17}, [samples]() -> IoData {
             Tensor signal = Tensor::FromFloat("signal", {1, 16, 1}, samples);
             Tensor frame_step = Tensor::FromInt64("frame_step", {}, {4});

             const OpsetId opset_v17 = DefaultOpset(17);
             const KernelContext stft_v17_ctx{opset_v17};
             const onnx_kernels::kernel::STFT stft_v17{stft_v17_ctx};

             Tensor y = stft_v17(signal, frame_step, /*window=*/nullptr,
                                 /*frame_length=*/nullptr, /*onesided=*/true);
             return IoData{{std::move(signal), std::move(frame_step)}, {std::move(y)}};
           });
    registry.back().atol = 1e-5;
  }

  // --- STFT with frame_length input, no window, onesided=1.
  {
    Expect(registry,
           MakeSTFTNode(/*with_window=*/false, /*with_frame_length=*/true, /*onesided=*/1),
           "test_cc_stft", {opset_v17}, [samples]() -> IoData {
             Tensor signal = Tensor::FromFloat("signal", {1, 16, 1}, samples);
             Tensor frame_step = Tensor::FromInt64("frame_step", {}, {4});
             Tensor frame_length = Tensor::FromInt64("frame_length", {}, {8});

             const OpsetId opset_v17 = DefaultOpset(17);

             const KernelContext stft_v17_ctx{opset_v17};
             const onnx_kernels::kernel::STFT stft_v17{stft_v17_ctx};

             Tensor y =
                 stft_v17(signal, frame_step, /*window=*/nullptr, &frame_length, /*onesided=*/true);
             return IoData{{std::move(signal), std::move(frame_step), std::move(frame_length)},
                           {std::move(y)}};
           });
    registry.back().atol = 1e-5;
  }

  // --- STFT with both window and frame_length, onesided=1.
  {
    Expect(registry, MakeSTFTNode(/*with_window=*/true, /*with_frame_length=*/true, /*onesided=*/1),
           "test_cc_stft_with_window", {opset_v17}, [samples]() -> IoData {
             Tensor signal = Tensor::FromFloat("signal", {1, 16, 1}, samples);
             Tensor frame_step = Tensor::FromInt64("frame_step", {}, {4});
             Tensor frame_length = Tensor::FromInt64("frame_length", {}, {8});

             const OpsetId opset_v17 = DefaultOpset(17);

             const KernelContext stft_v17_ctx{opset_v17};
             const onnx_kernels::kernel::STFT stft_v17{stft_v17_ctx};

             Tensor window = Tensor::FromFloat(
                 "window", {8}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}); // rectangular
             Tensor y = stft_v17(signal, frame_step, &window, &frame_length, /*onesided=*/true);
             return IoData{{std::move(signal), std::move(frame_step), std::move(window),
                            std::move(frame_length)},
                           {std::move(y)}};
           });
    registry.back().atol = 1e-5;
  }

  // --- STFT with onesided=0 (two-sided, complex output).
  {
    Expect(registry,
           MakeSTFTNode(/*with_window=*/false, /*with_frame_length=*/true, /*onesided=*/0),
           "test_cc_stft_twosided", {opset_v17}, [samples]() -> IoData {
             Tensor signal = Tensor::FromFloat("signal", {1, 16, 1}, samples);
             Tensor frame_step = Tensor::FromInt64("frame_step", {}, {4});
             Tensor frame_length = Tensor::FromInt64("frame_length", {}, {8});

             const OpsetId opset_v17 = DefaultOpset(17);

             const KernelContext stft_v17_ctx{opset_v17};
             const onnx_kernels::kernel::STFT stft_v17{stft_v17_ctx};

             Tensor y = stft_v17(signal, frame_step, /*window=*/nullptr, &frame_length,
                                 /*onesided=*/false);
             return IoData{{std::move(signal), std::move(frame_step), std::move(frame_length)},
                           {std::move(y)}};
           });
    registry.back().atol = 1e-5;
  }

  // --- STFT with complex-valued, batched input (onesided=0).
  //
  // Regression test mirroring onnxruntime/microsoft/onnxruntime#28961: the
  // pre-fix STFT kernel double-counted the per-sample component stride for
  // complex inputs, causing reads of one batch to leak into the next (and to
  // walk past the end of the allocation for the last batch). The expected
  // output below is the closed-form DFT of a DC-only complex signal whose
  // real value differs per batch — any cross-batch contamination breaks the
  // DC bin equality on at least one frame.
  {
    Expect(registry,
           MakeSTFTNode(/*with_window=*/false, /*with_frame_length=*/true, /*onesided=*/0),
           "test_cc_stft_complex_batched", {opset_v17}, []() -> IoData {
             const OpsetId opset_v17 = DefaultOpset(17);

             const KernelContext stft_v17_ctx{opset_v17};
             const onnx_kernels::kernel::STFT stft_v17{stft_v17_ctx};

             constexpr int64_t batch_size = 2;
             constexpr int64_t signal_size = 128;
             constexpr int64_t signal_components = 2; // complex: real + imag
             constexpr int64_t frame_length_v = 32;
             constexpr int64_t frame_step_v = 16;
             constexpr int64_t n_dfts = (signal_size - frame_length_v) / frame_step_v + 1; // 7
             constexpr int64_t dft_output_size = frame_length_v; // onesided=false
             constexpr int64_t output_components = 2;

             std::vector<float> signal_data(
                 static_cast<std::size_t>(batch_size * signal_size * signal_components), 0.0f);
             for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
               const float signal_value = batch_idx == 0 ? 1.0f : 99.0f;
               for (int64_t sample_idx = 0; sample_idx < signal_size; ++sample_idx) {
                 signal_data[static_cast<std::size_t>((batch_idx * signal_size + sample_idx) *
                                                      signal_components)] = signal_value;
               }
             }
             Tensor signal_complex = Tensor::FromFloat(
                 "signal", {batch_size, signal_size, signal_components}, signal_data);
             Tensor frame_step_b = Tensor::FromInt64("frame_step", {}, {frame_step_v});
             Tensor frame_length_b = Tensor::FromInt64("frame_length", {}, {frame_length_v});

             Tensor y_expected =
                 stft_v17(signal_complex, frame_step_b, /*window=*/nullptr, &frame_length_b,
                          /*onesided=*/false);

             // Sanity-check the kernel's output against the closed-form DFT of a
             // DC-only complex signal: the DC bin (k=0) of each frame must equal
             // ``frame_length × dc_value``, and all other bins must be zero. Any
             // cross-batch leak (the bug fixed by onnxruntime#28961) would jam
             // different constants into adjacent frames and break this equality.
             EXT_ENFORCE_INVALID(
                 y_expected.shape ==
                     std::vector<int64_t>({batch_size, n_dfts, dft_output_size, output_components}),
                 "test_cc_stft_complex_batched: unexpected output shape.");
             const float *y_ptr = y_expected.AsFloat();
             for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
               const float dc_value =
                   (batch_idx == 0 ? 1.0f : 99.0f) * static_cast<float>(frame_length_v);
               for (int64_t frame_idx = 0; frame_idx < n_dfts; ++frame_idx) {
                 const int64_t base =
                     ((batch_idx * n_dfts + frame_idx) * dft_output_size) * output_components;
                 EXT_ENFORCE_INVALID(std::fabs(y_ptr[base] - dc_value) < 1e-3f,
                                     "test_cc_stft_complex_batched: DC bin value mismatch.");
                 EXT_ENFORCE_INVALID(std::fabs(y_ptr[base + 1]) < 1e-3f,
                                     "test_cc_stft_complex_batched: DC bin imag must be zero.");
               }
             }

             return IoData{
                 {std::move(signal_complex), std::move(frame_step_b), std::move(frame_length_b)},
                 {std::move(y_expected)}};
           });
    registry.back().atol = 1e-3;
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
