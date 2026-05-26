// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/training/include_training_cases.h"
#include "onnx_backend_test/kernels/training/include_training_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Opset id for the ``ai.onnx.preview.training`` domain at the only released
// version of Adam (v1). Mirrors
// ``onnx_op::training::kOnnxPreviewTrainingDomain`` but is duplicated here so
// this library does not need to depend on ``lib_onnx_op``.
constexpr const char *kOnnxPreviewTrainingDomain = "ai.onnx.preview.training";

OpsetId TrainingOpset(int64_t version) { return OpsetId(kOnnxPreviewTrainingDomain, version); }

// Helper that appends a single FLOAT attribute (``name`` -> ``value``) to
// ``node``. Used to encode Adam's ``alpha``, ``beta``, ``epsilon``,
// ``norm_coefficient`` and ``norm_coefficient_post`` attributes.
void AddFloatAttribute(NodeProto &node, const char *name, float value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::FLOAT);
  attr->set_f(value);
}

// Helpers that append a batch of input/output names to ``node`` in a single
// call. Used to compactly wire Adam's variable-length ``{R, T, X*, G*, V*,
// H*}`` input list and ``{X*_new, V*_new, H*_new}`` output list without
// repeating ``node.add_input(...)`` once per name.
void AddInputs(NodeProto &node, std::initializer_list<const char *> names) {
  for (const char *name : names) {
    node.add_input(name);
  }
}

void AddOutputs(NodeProto &node, std::initializer_list<const char *> names) {
  for (const char *name : names) {
    node.add_output(name);
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Adam — one iteration of the Adam stochastic gradient optimization
// algorithm (since opset 1 in the ``ai.onnx.preview.training`` domain).
//
// Two cases are registered, both exercising the full ``norm_coefficient``,
// ``alpha``, ``beta``, ``epsilon`` and ``norm_coefficient_post`` pseudo-code
// from the operator's schema:
//
//   * ``test_cc_adam_single`` — Single optimized rank-1 tensor with the
//     un-corrected learning rate path (``T == 0``).
//   * ``test_cc_adam_multiple`` — Two optimized tensors of different ranks
//     with the bias-corrected learning rate path (``T > 0``).
//
// Inputs are small, fully deterministic tensors so this library does not
// depend on a PRNG.
// ---------------------------------------------------------------------------
void RegisterAdamCases(std::vector<TestCase> &registry) {
  const OpsetId opset = TrainingOpset(1);
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::Adam adam{kernel::KernelContext(opset)};

  // ----- Case 1: single optimized tensor, T == 0.
  {
    NodeProto node;
    node.set_op_type("Adam");
    node.set_domain(kOnnxPreviewTrainingDomain);
    AddInputs(node, {"R", "T", "X", "G", "V", "H"});
    AddOutputs(node, {"X_new", "V_new", "H_new"});

    kernel::Adam::Attributes attrs;
    attrs.alpha = 0.95f;
    attrs.beta = 0.9f;
    attrs.epsilon = 1e-2f;
    attrs.norm_coefficient = 0.001f;
    attrs.norm_coefficient_post = 0.0f;
    AddFloatAttribute(node, "alpha", attrs.alpha);
    AddFloatAttribute(node, "beta", attrs.beta);
    AddFloatAttribute(node, "epsilon", attrs.epsilon);
    AddFloatAttribute(node, "norm_coefficient", attrs.norm_coefficient);
    AddFloatAttribute(node, "norm_coefficient_post", attrs.norm_coefficient_post);

    Tensor R = Tensor::FromFloat("", {}, {0.1f});
    Tensor T = Tensor::FromInt64("", {}, {0});
    Tensor X = Tensor::FromFloat("", {3}, {1.0f, 2.0f, -1.0f});
    Tensor G = Tensor::FromFloat("", {3}, {0.5f, -0.5f, 0.25f});
    Tensor V = Tensor::FromFloat("", {3}, {0.0f, 0.0f, 0.0f});
    Tensor H = Tensor::FromFloat("", {3}, {0.0f, 0.0f, 0.0f});

    std::vector<Tensor> outs = adam(R, T, {X}, {G}, {V}, {H}, attrs);
    Expect(node, {R, T, X, G, V, H}, {outs[0], outs[1], outs[2]}, "test_cc_adam_single",
           {default_opset, opset}, "backend-test", registry);
  }

  // ----- Case 2: two optimized tensors of different ranks, T > 0 (uses the
  // bias-corrected learning rate path).
  {
    NodeProto node;
    node.set_op_type("Adam");
    node.set_domain(kOnnxPreviewTrainingDomain);
    AddInputs(node, {"R", "T", "X1", "X2", "G1", "G2", "V1", "V2", "H1", "H2"});
    AddOutputs(node, {"X1_new", "X2_new", "V1_new", "V2_new", "H1_new", "H2_new"});

    kernel::Adam::Attributes attrs;
    attrs.alpha = 0.9f;
    attrs.beta = 0.999f;
    attrs.epsilon = 1e-6f;
    attrs.norm_coefficient = 0.0f;
    attrs.norm_coefficient_post = 0.0f;
    AddFloatAttribute(node, "alpha", attrs.alpha);
    AddFloatAttribute(node, "beta", attrs.beta);
    AddFloatAttribute(node, "epsilon", attrs.epsilon);
    AddFloatAttribute(node, "norm_coefficient", attrs.norm_coefficient);
    AddFloatAttribute(node, "norm_coefficient_post", attrs.norm_coefficient_post);

    Tensor R = Tensor::FromFloat("", {}, {0.05f});
    Tensor T = Tensor::FromInt64("", {}, {5});
    Tensor X1 = Tensor::FromFloat("", {2}, {0.5f, -0.5f});
    Tensor X2 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor G1 = Tensor::FromFloat("", {2}, {0.1f, -0.2f});
    Tensor G2 = Tensor::FromFloat("", {2, 2}, {-0.5f, 0.25f, 0.75f, -1.0f});
    Tensor V1 = Tensor::FromFloat("", {2}, {0.01f, 0.02f});
    Tensor V2 = Tensor::FromFloat("", {2, 2}, {0.05f, 0.05f, -0.05f, 0.0f});
    Tensor H1 = Tensor::FromFloat("", {2}, {0.001f, 0.002f});
    Tensor H2 = Tensor::FromFloat("", {2, 2}, {0.01f, 0.02f, 0.03f, 0.04f});

    std::vector<Tensor> outs = adam(R, T, {X1, X2}, {G1, G2}, {V1, V2}, {H1, H2}, attrs);
    Expect(node, {R, T, X1, X2, G1, G2, V1, V2, H1, H2},
           {outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]}, "test_cc_adam_multiple",
           {default_opset, opset}, "backend-test", registry);
  }

  // ----- Upstream ONNX cases (mirror onnx.backend.test.case.node.adam.Adam).
  // These use the exact inputs and attributes from the upstream Python test
  // case, with expected outputs recomputed by ``kernel::Adam``. Only the four
  // attributes set by the upstream node (``norm_coefficient``, ``alpha``,
  // ``beta``, ``epsilon``) are written; ``norm_coefficient_post`` keeps its
  // schema default of 0.

  // From Adam.export_adam():
  {
    NodeProto node;
    node.set_op_type("Adam");
    node.set_domain(kOnnxPreviewTrainingDomain);
    AddInputs(node, {"R", "T", "X", "G", "V", "H"});
    AddOutputs(node, {"X_new", "V_new", "H_new"});

    kernel::Adam::Attributes attrs;
    attrs.alpha = 0.95f;
    attrs.beta = 0.1f;
    attrs.epsilon = 1e-7f;
    attrs.norm_coefficient = 0.001f;
    attrs.norm_coefficient_post = 0.0f;
    AddFloatAttribute(node, "norm_coefficient", attrs.norm_coefficient);
    AddFloatAttribute(node, "alpha", attrs.alpha);
    AddFloatAttribute(node, "beta", attrs.beta);
    AddFloatAttribute(node, "epsilon", attrs.epsilon);

    Tensor R = Tensor::FromFloat("", {}, {0.1f});
    Tensor T = Tensor::FromInt64("", {}, {0});
    Tensor X = Tensor::FromFloat("", {2}, {1.2f, 2.8f});
    Tensor G = Tensor::FromFloat("", {2}, {-0.94f, -2.5f});
    Tensor V = Tensor::FromFloat("", {2}, {1.7f, 3.6f});
    Tensor H = Tensor::FromFloat("", {2}, {0.1f, 0.1f});

    std::vector<Tensor> outs = adam(R, T, {X}, {G}, {V}, {H}, attrs);
    Expect(node, {R, T, X, G, V, H}, {outs[0], outs[1], outs[2]}, "test_adam",
           {default_opset, opset}, "backend-test", registry);
  }

  // From Adam.export_adam_multiple():
  {
    NodeProto node;
    node.set_op_type("Adam");
    node.set_domain(kOnnxPreviewTrainingDomain);
    AddInputs(node, {"R", "T", "X1", "X2", "G1", "G2", "V1", "V2", "H1", "H2"});
    AddOutputs(node, {"X1_new", "X2_new", "V1_new", "V2_new", "H1_new", "H2_new"});

    kernel::Adam::Attributes attrs;
    attrs.alpha = 0.95f;
    attrs.beta = 0.85f;
    attrs.epsilon = 1e-2f;
    attrs.norm_coefficient = 0.001f;
    attrs.norm_coefficient_post = 0.0f;
    AddFloatAttribute(node, "norm_coefficient", attrs.norm_coefficient);
    AddFloatAttribute(node, "alpha", attrs.alpha);
    AddFloatAttribute(node, "beta", attrs.beta);
    AddFloatAttribute(node, "epsilon", attrs.epsilon);

    Tensor R = Tensor::FromFloat("", {}, {0.1f});
    Tensor T = Tensor::FromInt64("", {}, {0});
    Tensor X1 = Tensor::FromFloat("", {1}, {1.0f});
    Tensor X2 = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
    Tensor G1 = Tensor::FromFloat("", {1}, {-1.0f});
    Tensor G2 = Tensor::FromFloat("", {2}, {-1.0f, -3.0f});
    Tensor V1 = Tensor::FromFloat("", {1}, {2.0f});
    Tensor V2 = Tensor::FromFloat("", {2}, {4.0f, 1.0f});
    Tensor H1 = Tensor::FromFloat("", {1}, {0.5f});
    Tensor H2 = Tensor::FromFloat("", {2}, {1.0f, 10.0f});

    std::vector<Tensor> outs = adam(R, T, {X1, X2}, {G1, G2}, {V1, V2}, {H1, H2}, attrs);
    Expect(node, {R, T, X1, X2, G1, G2, V1, V2, H1, H2},
           {outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]}, "test_adam_multiple",
           {default_opset, opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
