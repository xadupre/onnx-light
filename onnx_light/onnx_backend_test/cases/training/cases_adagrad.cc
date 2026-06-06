// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/training/include_training_cases.h"
#include "onnx_kernels/kernels/training/include_training_kernels.h"
#include "onnx_kernels/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

namespace {

// Opset id for the ``ai.onnx.preview.training`` domain at the only released
// version of Adagrad (v1). Duplicated rather than reused from
// ``onnx_op::training`` so this library does not need to depend on
// ``lib_onnx_op``.
constexpr const char *kOnnxPreviewTrainingDomain = "ai.onnx.preview.training";

OpsetId TrainingOpset(int64_t version) { return OpsetId(kOnnxPreviewTrainingDomain, version); }

} // namespace

// ---------------------------------------------------------------------------
// Adagrad — one iteration of the ADAGRAD stochastic gradient optimization
// algorithm (since opset 1 in the ``ai.onnx.preview.training`` domain).
//
// Mirrors the upstream Python cases produced by
// ``onnx.backend.test.case.node.adagrad.Adagrad``:
//
//   * ``test_adagrad`` — Single optimized rank-1 tensor of length 1.
//   * ``test_adagrad_multiple`` — Two optimized rank-1 tensors of lengths
//     1 and 2.
//
// Both cases set the three FLOAT attributes ``norm_coefficient``,
// ``epsilon`` and ``decay_factor`` and use the un-corrected learning-rate
// path (``T == 0``).
// ---------------------------------------------------------------------------
void RegisterAdagradCases(std::vector<TestCase> &registry) {
  const OpsetId opset = TrainingOpset(1);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::Adagrad adagrad{ctx};

  // From Adagrad.export_adagrad():
  {
    NodeProto node;
    node.set_op_type("Adagrad");
    node.set_domain(kOnnxPreviewTrainingDomain);
    AddInputs(node, {"R", "T", "X", "G", "H"});
    AddOutputs(node, {"X_new", "H_new"});

    const float norm_coefficient = 0.001f;
    const float epsilon = 1e-5f;
    const float decay_factor = 0.1f;
    AddFloatAttribute(node, "norm_coefficient", norm_coefficient);
    AddFloatAttribute(node, "epsilon", epsilon);
    AddFloatAttribute(node, "decay_factor", decay_factor);

    Tensor R = Tensor::FromFloat("", {}, {0.1f});
    Tensor T = Tensor::FromInt64("", {}, {0});
    Tensor X = Tensor::FromFloat("", {1}, {1.0f});
    Tensor G = Tensor::FromFloat("", {1}, {-1.0f});
    Tensor H = Tensor::FromFloat("", {1}, {2.0f});

    std::vector<Tensor> outs =
        adagrad(R, T, {X}, {G}, {H}, epsilon, decay_factor, norm_coefficient);
    Expect(node, {R, T, X, G, H}, {outs[0], outs[1]}, "test_adagrad", {default_opset, opset},
           "backend-test", registry);
  }

  // From Adagrad.export_adagrad_multiple():
  {
    NodeProto node;
    node.set_op_type("Adagrad");
    node.set_domain(kOnnxPreviewTrainingDomain);
    AddInputs(node, {"R", "T", "X1", "X2", "G1", "G2", "H1", "H2"});
    AddOutputs(node, {"X1_new", "X2_new", "H1_new", "H2_new"});

    const float norm_coefficient = 0.001f;
    const float epsilon = 1e-5f;
    const float decay_factor = 0.1f;
    AddFloatAttribute(node, "norm_coefficient", norm_coefficient);
    AddFloatAttribute(node, "epsilon", epsilon);
    AddFloatAttribute(node, "decay_factor", decay_factor);

    Tensor R = Tensor::FromFloat("", {}, {0.1f});
    Tensor T = Tensor::FromInt64("", {}, {0});
    Tensor X1 = Tensor::FromFloat("", {1}, {1.0f});
    Tensor X2 = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
    Tensor G1 = Tensor::FromFloat("", {1}, {-1.0f});
    Tensor G2 = Tensor::FromFloat("", {2}, {-1.0f, -3.0f});
    Tensor H1 = Tensor::FromFloat("", {1}, {2.0f});
    Tensor H2 = Tensor::FromFloat("", {2}, {4.0f, 1.0f});

    std::vector<Tensor> outs =
        adagrad(R, T, {X1, X2}, {G1, G2}, {H1, H2}, epsilon, decay_factor, norm_coefficient);
    Expect(node, {R, T, X1, X2, G1, G2, H1, H2}, {outs[0], outs[1], outs[2], outs[3]},
           "test_adagrad_multiple", {default_opset, opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
