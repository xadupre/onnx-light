// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// MelWeightMatrix — generates the triangular Mel filter-bank weight matrix
// of shape [floor(dft_length/2) + 1, num_mel_bins] (since opset 17). Inputs
// are scalar tensors: ``num_mel_bins`` (int32), ``dft_length`` (int32),
// ``sample_rate`` (int32), ``lower_edge_hertz`` (float) and
// ``upper_edge_hertz`` (float).
// ---------------------------------------------------------------------------
void RegisterMelWeightMatrixCases(std::vector<TestCase> &registry) {
  constexpr int32_t kNumMelBins = 8;
  constexpr int32_t kDftLength = 16;
  constexpr int32_t kSampleRate = 8192;
  constexpr float kLowerEdgeHertz = 0.0f;
  constexpr float kUpperEdgeHertz = 8192.0f / 2.0f;

  const OpsetId opset = DefaultOpset(17);
  const kernel::KernelContext ctx{opset};
  const kernel::MelWeightMatrix mel_kernel{ctx};

  NodeProto node;
  node.set_op_type("MelWeightMatrix");
  node.add_input("num_mel_bins");
  node.add_input("dft_length");
  node.add_input("sample_rate");
  node.add_input("lower_edge_hertz");
  node.add_input("upper_edge_hertz");
  node.add_output("output");

  Tensor num_mel_bins = Tensor::FromInt32("", {}, {kNumMelBins});
  Tensor dft_length = Tensor::FromInt32("", {}, {kDftLength});
  Tensor sample_rate = Tensor::FromInt32("", {}, {kSampleRate});
  Tensor lower_edge_hertz = Tensor::FromFloat("", {}, {kLowerEdgeHertz});
  Tensor upper_edge_hertz = Tensor::FromFloat("", {}, {kUpperEdgeHertz});

  Tensor output = mel_kernel(num_mel_bins, dft_length, sample_rate, lower_edge_hertz,
                             upper_edge_hertz, DataType::FLOAT);

  Expect(node, {num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz},
         {output}, "test_cc_melweightmatrix", {opset}, "backend-test", registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
