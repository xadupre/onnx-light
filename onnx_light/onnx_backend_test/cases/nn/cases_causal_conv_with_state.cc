// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// CausalConvWithState — stateful causal 1D depthwise convolution.
// Each case produces the expected outputs by invoking the local
// ``kernel::CausalConvWithState`` reference implementation so the recorded
// outputs are self-consistent with this library's kernel.
//
// The case names mirror the FLOAT variants of the upstream ONNX backend
// test suite for ``test_causal_conv_with_state_*``. ``fp16`` variants are
// intentionally omitted (the kernel is FLOAT-only) and remain listed in
// ``unittests/onnxl_vs_onnx/_backend_test_known_missing.txt``.
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeCausalConvNode(const std::vector<std::string> &inputs,
                             const std::vector<std::string> &outputs) {
  NodeProto node;
  node.set_op_type("CausalConvWithState");
  for (const auto &name : inputs) {
    node.add_input(name);
  }
  for (const auto &name : outputs) {
    node.add_output(name);
  }
  return node;
}

// Convenience: registers one case from the provided inputs / kernel attrs.
// The node template carries the activation attribute when it is non-default.
void RegisterCase(std::vector<TestCase> &registry, const std::string &name, const OpsetId &opset,
                  const kernel::CausalConvWithState &kernel, const Tensor &input,
                  const Tensor &weight, const Tensor *bias, const Tensor *past_state,
                  const std::string &activation) {
  kernel::CausalConvWithState::Attributes attrs;
  attrs.activation = activation;
  auto [output, present_state] = kernel(input, weight, bias != nullptr ? *bias : Tensor{},
                                        past_state != nullptr ? *past_state : Tensor{}, attrs);

  std::vector<std::string> input_names = {"input", "weight"};
  std::vector<Tensor> inputs = {input, weight};
  if (bias != nullptr) {
    input_names.push_back("bias");
    inputs.push_back(*bias);
  } else if (past_state != nullptr) {
    // Empty placeholder for the optional ``bias`` input so ``past_state``
    // remains positioned at index 3.
    input_names.push_back("");
  }
  if (past_state != nullptr) {
    input_names.push_back("past_state");
    inputs.push_back(*past_state);
  }
  NodeProto node = MakeCausalConvNode(input_names, {"output", "present_state"});
  if (activation != "none") {
    AddAttribute<std::string>(node, "activation", activation);
  }
  Expect(node, inputs, {output, present_state}, name, {opset}, "backend-test", registry);
}

} // namespace

void RegisterCausalConvWithStateCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(27);
  const kernel::KernelContext ctx{opset};
  const kernel::CausalConvWithState kernel{ctx};

  // Shared "basic" tensors used by several cases: B=1, C=2, L=4, K=3.
  Tensor X_basic =
      Tensor::FromFloat("input", {1, 2, 4}, {0.0f, 0.1f, 0.2f, 0.3f, 1.0f, 1.1f, 1.2f, 1.3f});
  Tensor W_basic =
      Tensor::FromFloat("weight", {2, 1, 3}, {0.5f, -0.25f, 0.125f, 1.0f, 0.5f, 0.25f});
  Tensor Bias = Tensor::FromFloat("bias", {2}, {0.1f, -0.2f});
  Tensor Past = Tensor::FromFloat("past_state", {1, 2, 2}, {0.5f, 0.6f, -0.5f, -0.6f});

  // Case: basic
  RegisterCase(registry, "test_cc_causal_conv_with_state_basic", opset, kernel, X_basic, W_basic,
               nullptr, nullptr, "none");

  // Case: with_bias
  RegisterCase(registry, "test_cc_causal_conv_with_state_with_bias", opset, kernel, X_basic,
               W_basic, &Bias, nullptr, "none");

  // Case: with_past_state
  RegisterCase(registry, "test_cc_causal_conv_with_state_with_past_state", opset, kernel, X_basic,
               W_basic, nullptr, &Past, "none");

  // Case: with_bias_and_past_state
  RegisterCase(registry, "test_cc_causal_conv_with_state_with_bias_and_past_state", opset, kernel,
               X_basic, W_basic, &Bias, &Past, "none");

  // Case: silu (activation = "silu")
  RegisterCase(registry, "test_cc_causal_conv_with_state_silu", opset, kernel, X_basic, W_basic,
               nullptr, nullptr, "silu");

  // Case: silu_with_past_state
  RegisterCase(registry, "test_cc_causal_conv_with_state_silu_with_past_state", opset, kernel,
               X_basic, W_basic, nullptr, &Past, "silu");

  // Case: swish_alias — activation = "swish" is an alias for "silu".
  RegisterCase(registry, "test_cc_causal_conv_with_state_swish_alias", opset, kernel, X_basic,
               W_basic, nullptr, nullptr, "swish");

  // Case: decode_step — L=1 with past_state (typical autoregressive step).
  {
    Tensor X = Tensor::FromFloat("input", {1, 2, 1}, {0.4f, 1.4f});
    RegisterCase(registry, "test_cc_causal_conv_with_state_decode_step", opset, kernel, X, W_basic,
                 nullptr, &Past, "none");
  }

  // Case: short_input_no_past_state — L=1 < K-1, no past_state. The kernel
  // must zero-pad on the left and the present_state contains a mix of the
  // implicit zero padding and the (only) input value.
  {
    Tensor X = Tensor::FromFloat("input", {1, 2, 1}, {0.4f, 1.4f});
    RegisterCase(registry, "test_cc_causal_conv_with_state_short_input_no_past_state", opset,
                 kernel, X, W_basic, nullptr, nullptr, "none");
  }

  // Case: kernel_size_one — K=1 so past_state degenerates to shape (B, C, 0)
  // and there is no padding. We exercise the no-past_state path; the
  // present_state output is an empty (B, C, 0) tensor.
  {
    Tensor X =
        Tensor::FromFloat("input", {1, 2, 4}, {0.0f, 0.1f, 0.2f, 0.3f, 1.0f, 1.1f, 1.2f, 1.3f});
    Tensor W = Tensor::FromFloat("weight", {2, 1, 1}, {0.5f, -1.0f});
    RegisterCase(registry, "test_cc_causal_conv_with_state_kernel_size_one", opset, kernel, X, W,
                 nullptr, nullptr, "none");
  }

  // Case: b1_c1_degenerate — minimal B=1, C=1, L=1, K=2 with past_state.
  {
    Tensor X = Tensor::FromFloat("input", {1, 1, 1}, {0.3f});
    Tensor W = Tensor::FromFloat("weight", {1, 1, 2}, {0.5f, -0.25f});
    Tensor P = Tensor::FromFloat("past_state", {1, 1, 1}, {0.7f});
    RegisterCase(registry, "test_cc_causal_conv_with_state_b1_c1_degenerate", opset, kernel, X, W,
                 nullptr, &P, "none");
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
