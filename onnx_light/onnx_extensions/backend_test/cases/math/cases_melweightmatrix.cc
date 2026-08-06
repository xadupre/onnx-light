// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// MelWeightMatrix — generates the triangular Mel filter-bank weight matrix
// of shape [floor(dft_length/2) + 1, num_mel_bins] (since opset 17). Inputs
// are scalar tensors: ``num_mel_bins`` (int32), ``dft_length`` (int32),
// ``sample_rate`` (int32), ``lower_edge_hertz`` (float) and
// ``upper_edge_hertz`` (float).
// ---------------------------------------------------------------------------
void RegisterMelWeightMatrixCases(std::vector<TestCase> &registry, TestMode mode) {
  constexpr int32_t kNumMelBins = 8;
  constexpr int32_t kDftLength = 16;
  constexpr int32_t kSampleRate = 8192;
  constexpr float kLowerEdgeHertz = 0.0f;
  constexpr float kUpperEdgeHertz = 8192.0f / 2.0f;

  const OpsetId opset = DefaultOpset(17);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::MelWeightMatrix mel_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto bench_node;
    bench_node.set_op_type("MelWeightMatrix");
    bench_node.add_input("num_mel_bins");
    bench_node.add_input("dft_length");
    bench_node.add_input("sample_rate");
    bench_node.add_input("lower_edge_hertz");
    bench_node.add_input("upper_edge_hertz");
    bench_node.add_output("output");
    const int64_t output_count = (8192 / 2 + 1) * 2048;
    Expect(registry, std::move(bench_node), "test_cc_melweightmatrix_benchmark", {opset},
           {1, 1, 1, 1, 1}, {output_count}, [mel_kernel]() -> IoData {
             Tensor b_num_mel_bins = Tensor::FromInt32("", {}, {2048});
             Tensor b_dft_length = Tensor::FromInt32("", {}, {8192});
             Tensor b_sample_rate = Tensor::FromInt32("", {}, {16000});
             Tensor b_lower_edge_hertz = Tensor::FromFloat("", {}, {0.0f});
             Tensor b_upper_edge_hertz = Tensor::FromFloat("", {}, {8000.0f});
             Tensor b_output = mel_kernel(b_num_mel_bins, b_dft_length, b_sample_rate,
                                          b_lower_edge_hertz, b_upper_edge_hertz, DataType::FLOAT);
             return IoData{{std::move(b_num_mel_bins), std::move(b_dft_length),
                            std::move(b_sample_rate), std::move(b_lower_edge_hertz),
                            std::move(b_upper_edge_hertz)},
                           {std::move(b_output)}};
           });
    return;
  }

  NodeProto node;
  node.set_op_type("MelWeightMatrix");
  node.add_input("num_mel_bins");
  node.add_input("dft_length");
  node.add_input("sample_rate");
  node.add_input("lower_edge_hertz");
  node.add_input("upper_edge_hertz");
  node.add_output("output");

  Expect(registry, std::move(node), "test_cc_melweightmatrix", {opset}, [=]() -> IoData {
    Tensor num_mel_bins = Tensor::FromInt32("", {}, {kNumMelBins});
    Tensor dft_length = Tensor::FromInt32("", {}, {kDftLength});
    Tensor sample_rate = Tensor::FromInt32("", {}, {kSampleRate});
    Tensor lower_edge_hertz = Tensor::FromFloat("", {}, {kLowerEdgeHertz});
    Tensor upper_edge_hertz = Tensor::FromFloat("", {}, {kUpperEdgeHertz});
    Tensor output = mel_kernel(num_mel_bins, dft_length, sample_rate, lower_edge_hertz,
                               upper_edge_hertz, DataType::FLOAT);
    return IoData{{std::move(num_mel_bins), std::move(dft_length), std::move(sample_rate),
                   std::move(lower_edge_hertz), std::move(upper_edge_hertz)},
                  {std::move(output)}};
  });
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
