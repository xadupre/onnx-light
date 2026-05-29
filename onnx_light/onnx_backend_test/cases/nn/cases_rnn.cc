// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// RNN — single-direction (forward) one-layer RNN with the default Tanh
// activation, mirroring the upstream ``test_simple_rnn_*`` reference cases
// at opset 22:
//
//   * ``test_cc_simple_rnn_defaults`` — a tiny 1x3x2 input with X-only
//     inputs (W, R) and no bias / no initial hidden state.
//   * ``test_cc_simple_rnn_with_initial_bias`` — same shapes but exercising
//     the optional ``B`` (concatenated [Wb, Rb]) and ``initial_h`` inputs.
// ---------------------------------------------------------------------------
void RegisterRNNCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::RNN rnn_kernel{ctx};

  // ``simple_rnn_defaults``: seq_length=2, batch_size=3, input_size=2,
  // hidden_size=4. No bias, no initial_h.
  {
    NodeProto node;
    node.set_op_type("RNN");
    node.add_input("X");
    node.add_input("W");
    node.add_input("R");
    node.add_output("");
    node.add_output("Y_h");
    AddAttribute<int64_t>(node, "hidden_size", 4);

    const int64_t seq_length = 2;
    const int64_t batch_size = 3;
    const int64_t input_size = 2;
    const int64_t hidden_size = 4;
    std::vector<float> x_data(static_cast<size_t>(seq_length * batch_size * input_size));
    for (size_t i = 0; i < x_data.size(); ++i) {
      x_data[i] = static_cast<float>(i) * 0.1f - 0.5f;
    }
    std::vector<float> w_data(static_cast<size_t>(hidden_size * input_size));
    for (size_t i = 0; i < w_data.size(); ++i) {
      w_data[i] = static_cast<float>(i % 5) * 0.1f - 0.2f;
    }
    std::vector<float> r_data(static_cast<size_t>(hidden_size * hidden_size));
    for (size_t i = 0; i < r_data.size(); ++i) {
      r_data[i] = static_cast<float>(i % 7) * 0.05f - 0.15f;
    }
    Tensor x = Tensor::FromFloat("", {seq_length, batch_size, input_size}, x_data);
    Tensor w = Tensor::FromFloat("", {1, hidden_size, input_size}, w_data);
    Tensor r = Tensor::FromFloat("", {1, hidden_size, hidden_size}, r_data);

    auto [y_unused, y_h] = rnn_kernel(x, w, r);
    (void)y_unused; // Y is skipped (empty output name).

    Expect(node, {x, w, r}, {y_h}, "test_cc_simple_rnn_defaults", {opset}, "backend-test",
           registry);
  }

  // ``simple_rnn_with_initial_bias``: same shapes plus ``B`` and
  // ``initial_h``. Both Y and Y_h are produced.
  {
    NodeProto node;
    node.set_op_type("RNN");
    node.add_input("X");
    node.add_input("W");
    node.add_input("R");
    node.add_input("B");
    node.add_input(""); // sequence_lens omitted
    node.add_input("initial_h");
    node.add_output("Y");
    node.add_output("Y_h");
    AddAttribute<int64_t>(node, "hidden_size", 4);

    const int64_t seq_length = 2;
    const int64_t batch_size = 3;
    const int64_t input_size = 2;
    const int64_t hidden_size = 4;
    std::vector<float> x_data(static_cast<size_t>(seq_length * batch_size * input_size));
    for (size_t i = 0; i < x_data.size(); ++i) {
      x_data[i] = static_cast<float>(i) * 0.1f - 0.5f;
    }
    std::vector<float> w_data(static_cast<size_t>(hidden_size * input_size));
    for (size_t i = 0; i < w_data.size(); ++i) {
      w_data[i] = static_cast<float>(i % 5) * 0.1f - 0.2f;
    }
    std::vector<float> r_data(static_cast<size_t>(hidden_size * hidden_size));
    for (size_t i = 0; i < r_data.size(); ++i) {
      r_data[i] = static_cast<float>(i % 7) * 0.05f - 0.15f;
    }
    std::vector<float> b_data(static_cast<size_t>(2 * hidden_size));
    for (size_t i = 0; i < b_data.size(); ++i) {
      b_data[i] = static_cast<float>(i) * 0.02f - 0.05f;
    }
    std::vector<float> h0_data(static_cast<size_t>(batch_size * hidden_size));
    for (size_t i = 0; i < h0_data.size(); ++i) {
      h0_data[i] = static_cast<float>(i) * 0.03f - 0.1f;
    }
    Tensor x = Tensor::FromFloat("", {seq_length, batch_size, input_size}, x_data);
    Tensor w = Tensor::FromFloat("", {1, hidden_size, input_size}, w_data);
    Tensor r = Tensor::FromFloat("", {1, hidden_size, hidden_size}, r_data);
    Tensor b = Tensor::FromFloat("", {1, 2 * hidden_size}, b_data);
    Tensor h0 = Tensor::FromFloat("", {1, batch_size, hidden_size}, h0_data);

    auto [y, y_h] = rnn_kernel(x, w, r, b, h0);

    Expect(node, {x, w, r, b, h0}, {std::move(y), std::move(y_h)},
           "test_cc_simple_rnn_with_initial_bias", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
