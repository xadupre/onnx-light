// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// v17-style DFT node: axis/inverse/onesided are attributes; the only inputs
// are ``x`` (and optionally ``dft_length``).
NodeProto MakeDFTNodeV17(int64_t axis, bool inverse, bool onesided) {
  NodeProto node;
  node.set_op_type("DFT");
  node.add_input("x");
  node.add_output("y");
  AddAttribute<int64_t>(node, "axis", axis);
  if (inverse) {
    AddAttribute<int64_t>(node, "inverse", static_cast<int64_t>(1));
  }
  if (onesided) {
    AddAttribute<int64_t>(node, "onesided", static_cast<int64_t>(1));
  }
  return node;
}

// v20-style DFT node: axis becomes the third (optional) input. ``inverse``
// and ``onesided`` remain attributes.
NodeProto MakeDFTNodeV20(bool inverse, bool onesided) {
  NodeProto node;
  node.set_op_type("DFT");
  node.add_input("x");
  node.add_input(""); // dft_length omitted
  node.add_input("axis");
  node.add_output("y");
  if (inverse) {
    AddAttribute<int64_t>(node, "inverse", static_cast<int64_t>(1));
  }
  if (onesided) {
    AddAttribute<int64_t>(node, "onesided", static_cast<int64_t>(1));
  }
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// DFT — discrete Fourier transform. Provides coverage for the v17 (axis as
// attribute) and v20 (axis as input) signatures, including the forward,
// inverse, RFFT, and IRFFT modes.
// ---------------------------------------------------------------------------
void RegisterDFTCases(std::vector<TestCase> &registry) {
  const OpsetId opset_v17 = DefaultOpset(17);
  const OpsetId opset_v20 = DefaultOpset(20);
  const kernel::KernelContext ctx_v17{opset_v17};
  const kernel::KernelContext ctx_v20{opset_v20};
  const kernel::DFT dft_v17{ctx_v17};
  const kernel::DFT dft_v20{ctx_v20};

  // Simple real-valued input of shape [1, N, 1].
  Tensor x_real = Tensor::FromFloat("x", {1, 4, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
  // Simple complex-valued input of shape [1, N, 2] (interleaved re/im).
  Tensor x_cplx =
      Tensor::FromFloat("x", {1, 4, 2}, {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f});

  // DFT outputs include exact-zero imaginary parts (e.g. for a real input
  // with symmetric spectrum); reference implementations differ on the
  // residual rounding noise there (~1e-7) so loosen ``atol`` slightly to
  // tolerate the difference for values that are essentially zero.
  constexpr double kDFTAtol = 1e-5;
  // For the 10x10 ONNX fixtures the inputs span [0, 100); rounding noise
  // grows linearly with the magnitudes of the FFT terms, so a slightly
  // looser tolerance is required to cross-check against ORT/numpy.
  constexpr double kDFTAtolLarge = 1e-3;

  // --- v20: standard forward DFT (axis = 1).
  {
    Tensor axis = Tensor::FromInt64("axis", {}, {1});
    Tensor y = dft_v20(x_real, /*dft_length=*/nullptr, 1, /*onesided=*/false,
                       /*inverse=*/false);
    Expect(MakeDFTNodeV20(/*inverse=*/false, /*onesided=*/false), {x_real, axis}, {y},
           "test_cc_dft", {opset_v20}, "backend-test", registry);
    registry.back().atol = kDFTAtol;
  }

  // --- v20: inverse DFT (axis = 1).
  {
    Tensor axis = Tensor::FromInt64("axis", {}, {1});
    Tensor y = dft_v20(x_cplx, /*dft_length=*/nullptr, 1, /*onesided=*/false,
                       /*inverse=*/true);
    Expect(MakeDFTNodeV20(/*inverse=*/true, /*onesided=*/false), {x_cplx, axis}, {y},
           "test_cc_dft_inverse", {opset_v20}, "backend-test", registry);
    registry.back().atol = kDFTAtol;
  }

  // --- v20: RFFT (axis = 1, onesided=1).
  {
    Tensor axis = Tensor::FromInt64("axis", {}, {1});
    Tensor y = dft_v20(x_real, /*dft_length=*/nullptr, 1, /*onesided=*/true,
                       /*inverse=*/false);
    Expect(MakeDFTNodeV20(/*inverse=*/false, /*onesided=*/true), {x_real, axis}, {y},
           "test_cc_dft_rfft", {opset_v20}, "backend-test", registry);
    registry.back().atol = kDFTAtol;
  }

  // --- v17: standard forward DFT with axis attribute.
  {
    Tensor y = dft_v17(x_real, /*dft_length=*/nullptr, 1, /*onesided=*/false,
                       /*inverse=*/false);
    Expect(MakeDFTNodeV17(/*axis=*/1, /*inverse=*/false, /*onesided=*/false), {x_real}, {y},
           "test_cc_dft_opset17", {opset_v17}, "backend-test", registry);
    registry.back().atol = kDFTAtol;
  }

  // --- v17: inverse DFT.
  {
    Tensor y = dft_v17(x_cplx, /*dft_length=*/nullptr, 1, /*onesided=*/false,
                       /*inverse=*/true);
    Expect(MakeDFTNodeV17(/*axis=*/1, /*inverse=*/true, /*onesided=*/false), {x_cplx}, {y},
           "test_cc_dft_inverse_opset17", {opset_v17}, "backend-test", registry);
    registry.back().atol = kDFTAtol;
  }

  // -----------------------------------------------------------------------
  // ONNX-style fixtures using ``np.arange(0, 100).reshape(10, 10)``.
  // These mirror the ``test_dft*`` / ``test_dft_*_opset19`` cases in ONNX.
  // The expected output is computed by ``kernel::DFT`` so test data is
  // self-consistent with onnx-light's implementation.
  // -----------------------------------------------------------------------
  std::vector<float> arange100_real(100);
  for (int i = 0; i < 100; ++i) {
    arange100_real[static_cast<std::size_t>(i)] = static_cast<float>(i);
  }
  // Real-valued [1, 10, 10, 1] input.
  const Tensor x_real_10x10 = Tensor::FromFloat("x", {1, 10, 10, 1}, arange100_real);
  // Complex-valued [1, 10, 10, 2] input: ``np.arange(0, 100, dtype=complex64)``
  // i.e. real component equals the linear index, imaginary part is zero.
  std::vector<float> arange100_cplx(200, 0.0f);
  for (int i = 0; i < 100; ++i) {
    arange100_cplx[static_cast<std::size_t>(2 * i)] = static_cast<float>(i);
  }
  const Tensor x_cplx_10x10 = Tensor::FromFloat("x", {1, 10, 10, 2}, arange100_cplx);

  const kernel::KernelContext ctx_v19{DefaultOpset(19)};
  const kernel::DFT dft_v19{ctx_v19};
  const OpsetId opset_v19 = DefaultOpset(19);

  // --- v20: ``test_cc_dft_axis`` — forward DFT along axis=2.
  {
    Tensor axis = Tensor::FromInt64("axis", {}, {2});
    Tensor y = dft_v20(x_real_10x10, /*dft_length=*/nullptr, 2, /*onesided=*/false,
                       /*inverse=*/false);
    Expect(MakeDFTNodeV20(/*inverse=*/false, /*onesided=*/false), {x_real_10x10, axis}, {y},
           "test_cc_dft_axis", {opset_v20}, "backend-test", registry);
    registry.back().atol = kDFTAtolLarge;
  }

  // --- v20: ``test_cc_dft_irfft`` — inverse one-sided DFT producing a real
  // output of length 10 from 6 one-sided complex bins.
  {
    // Build the one-sided RFFT of ``np.arange(0, 100).reshape(10, 10)`` along
    // axis=0 by running the forward kernel with onesided=1, axis=1.
    Tensor x_onesided = dft_v20(x_real_10x10, /*dft_length=*/nullptr, 1, /*onesided=*/true,
                                /*inverse=*/false);
    x_onesided.name = "x";
    Tensor axis = Tensor::FromInt64("axis", {}, {1});
    Tensor y = dft_v20(x_onesided, /*dft_length=*/nullptr, 1, /*onesided=*/true,
                       /*inverse=*/true);
    Expect(MakeDFTNodeV20(/*inverse=*/true, /*onesided=*/true), {x_onesided, axis}, {y},
           "test_cc_dft_irfft", {opset_v20}, "backend-test", registry);
    registry.back().atol = kDFTAtolLarge;
  }

  // --- v19: ``test_cc_dft_opset19`` — forward DFT along axis=1.
  {
    Tensor y = dft_v19(x_real_10x10, /*dft_length=*/nullptr, 1, /*onesided=*/false,
                       /*inverse=*/false);
    Expect(MakeDFTNodeV17(/*axis=*/1, /*inverse=*/false, /*onesided=*/false), {x_real_10x10}, {y},
           "test_cc_dft_opset19", {opset_v19}, "backend-test", registry);
    registry.back().atol = kDFTAtolLarge;
  }

  // --- v19: ``test_cc_dft_axis_opset19`` — forward DFT along axis=2.
  {
    Tensor y = dft_v19(x_real_10x10, /*dft_length=*/nullptr, 2, /*onesided=*/false,
                       /*inverse=*/false);
    Expect(MakeDFTNodeV17(/*axis=*/2, /*inverse=*/false, /*onesided=*/false), {x_real_10x10}, {y},
           "test_cc_dft_axis_opset19", {opset_v19}, "backend-test", registry);
    registry.back().atol = kDFTAtolLarge;
  }

  // --- v19: ``test_cc_dft_inverse_opset19`` — inverse DFT along axis=1.
  {
    Tensor y = dft_v19(x_cplx_10x10, /*dft_length=*/nullptr, 1, /*onesided=*/false,
                       /*inverse=*/true);
    Expect(MakeDFTNodeV17(/*axis=*/1, /*inverse=*/true, /*onesided=*/false), {x_cplx_10x10}, {y},
           "test_cc_dft_inverse_opset19", {opset_v19}, "backend-test", registry);
    registry.back().atol = kDFTAtolLarge;
  }

  // --- v19: ``test_cc_dft_rfft_opset19`` — one-sided forward DFT (RFFT).
  {
    Tensor y = dft_v19(x_real_10x10, /*dft_length=*/nullptr, 1, /*onesided=*/true,
                       /*inverse=*/false);
    Expect(MakeDFTNodeV17(/*axis=*/1, /*inverse=*/false, /*onesided=*/true), {x_real_10x10}, {y},
           "test_cc_dft_rfft_opset19", {opset_v19}, "backend-test", registry);
    registry.back().atol = kDFTAtolLarge;
  }

  // --- v19: ``test_cc_dft_irfft_opset19`` — inverse one-sided DFT (IRFFT).
  {
    Tensor x_onesided = dft_v19(x_real_10x10, /*dft_length=*/nullptr, 1, /*onesided=*/true,
                                /*inverse=*/false);
    x_onesided.name = "x";
    Tensor y = dft_v19(x_onesided, /*dft_length=*/nullptr, 1, /*onesided=*/true,
                       /*inverse=*/true);
    Expect(MakeDFTNodeV17(/*axis=*/1, /*inverse=*/true, /*onesided=*/true), {x_onesided}, {y},
           "test_cc_dft_irfft_opset19", {opset_v19}, "backend-test", registry);
    registry.back().atol = kDFTAtolLarge;
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
