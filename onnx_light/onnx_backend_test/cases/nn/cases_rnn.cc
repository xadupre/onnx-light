// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_kernels/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

// ---------------------------------------------------------------------------
// RNN — single-direction (forward) one-layer RNN with the default Tanh
// activation, mirroring the upstream ``test_simple_rnn_*`` /
// ``test_rnn_seq_length`` reference cases at opset 22:
//
//   * ``test_cc_simple_rnn_defaults`` — a tiny 1x3x2 input with X-only
//     inputs (W, R) and no bias / no initial hidden state.
//   * ``test_cc_simple_rnn_with_initial_bias`` — same shapes but exercising
//     the optional ``B`` (concatenated [Wb, Rb]) and ``initial_h`` inputs.
//   * ``test_cc_rnn_seq_length`` — multi-step (seq_length=2) RNN with the
//     optional ``B`` input, mirroring upstream ``test_rnn_seq_length``.
//   * ``test_cc_simple_rnn_batchwise`` — ``layout=1`` variant where ``X``
//     is laid out as ``[batch_size, seq_length, input_size]``, mirroring
//     upstream ``test_simple_rnn_batchwise``.
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

  // ``rnn_seq_length``: seq_length=2, batch_size=3, input_size=3,
  // hidden_size=5 with bias ``B`` only (no initial_h). Only ``Y_h`` is
  // produced, mirroring upstream ``test_rnn_seq_length``.
  {
    NodeProto node;
    node.set_op_type("RNN");
    node.add_input("X");
    node.add_input("W");
    node.add_input("R");
    node.add_input("B");
    node.add_output("");
    node.add_output("Y_h");
    AddAttribute<int64_t>(node, "hidden_size", 5);

    const int64_t seq_length = 2;
    const int64_t batch_size = 3;
    const int64_t input_size = 3;
    const int64_t hidden_size = 5;
    std::vector<float> x_data(static_cast<size_t>(seq_length * batch_size * input_size));
    for (size_t i = 0; i < x_data.size(); ++i) {
      x_data[i] = static_cast<float>(i) * 0.07f - 0.3f;
    }
    std::vector<float> w_data(static_cast<size_t>(hidden_size * input_size));
    for (size_t i = 0; i < w_data.size(); ++i) {
      w_data[i] = static_cast<float>(i % 6) * 0.08f - 0.2f;
    }
    std::vector<float> r_data(static_cast<size_t>(hidden_size * hidden_size));
    for (size_t i = 0; i < r_data.size(); ++i) {
      r_data[i] = static_cast<float>(i % 9) * 0.04f - 0.15f;
    }
    std::vector<float> b_data(static_cast<size_t>(2 * hidden_size));
    for (size_t i = 0; i < b_data.size(); ++i) {
      b_data[i] = static_cast<float>(i) * 0.03f - 0.07f;
    }
    Tensor x = Tensor::FromFloat("", {seq_length, batch_size, input_size}, x_data);
    Tensor w = Tensor::FromFloat("", {1, hidden_size, input_size}, w_data);
    Tensor r = Tensor::FromFloat("", {1, hidden_size, hidden_size}, r_data);
    Tensor b = Tensor::FromFloat("", {1, 2 * hidden_size}, b_data);

    auto [y_unused, y_h] = rnn_kernel(x, w, r, b);
    (void)y_unused; // Y is skipped (empty output name).

    Expect(node, {x, w, r, b}, {y_h}, "test_cc_rnn_seq_length", {opset}, "backend-test", registry);
  }

  // ``simple_rnn_batchwise``: ``layout=1`` variant with batch_size=3,
  // seq_length=1, input_size=2, hidden_size=4. The kernel itself only
  // implements ``layout=0`` so we run it on the axis-swapped ``X`` and
  // re-permute the outputs to the batchwise layout:
  //   * ``Y``   : [seq, 1, batch, hidden]  -> [batch, seq, 1, hidden]
  //   * ``Y_h`` : [1, batch, hidden]       -> [batch, 1, hidden]
  {
    NodeProto node;
    node.set_op_type("RNN");
    node.add_input("X");
    node.add_input("W");
    node.add_input("R");
    node.add_output("Y");
    node.add_output("Y_h");
    AddAttribute<int64_t>(node, "hidden_size", 4);
    AddAttribute<int64_t>(node, "layout", 1);

    const int64_t batch_size = 3;
    const int64_t seq_length = 1;
    const int64_t input_size = 2;
    const int64_t hidden_size = 4;

    // Input is [batch_size, seq_length, input_size] for layout=1.
    const std::vector<float> x_batchwise_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor x_batchwise =
        Tensor::FromFloat("", {batch_size, seq_length, input_size}, x_batchwise_data);

    // Same numerical contents but laid out for layout=0
    // ([seq_length, batch_size, input_size]); for seq_length=1 the axis
    // swap is a no-op on the underlying buffer.
    std::vector<float> x_layout0_data(x_batchwise_data.size());
    for (int64_t n = 0; n < batch_size; ++n) {
      for (int64_t t = 0; t < seq_length; ++t) {
        for (int64_t k = 0; k < input_size; ++k) {
          const size_t src = static_cast<size_t>((n * seq_length + t) * input_size + k);
          const size_t dst = static_cast<size_t>((t * batch_size + n) * input_size + k);
          x_layout0_data[dst] = x_batchwise_data[src];
        }
      }
    }
    Tensor x_layout0 = Tensor::FromFloat("", {seq_length, batch_size, input_size}, x_layout0_data);

    const float weight_scale = 0.5f;
    std::vector<float> w_data(static_cast<size_t>(hidden_size * input_size), weight_scale);
    std::vector<float> r_data(static_cast<size_t>(hidden_size * hidden_size), weight_scale);
    Tensor w = Tensor::FromFloat("", {1, hidden_size, input_size}, w_data);
    Tensor r = Tensor::FromFloat("", {1, hidden_size, hidden_size}, r_data);

    auto [y_layout0, y_h_layout0] = rnn_kernel(x_layout0, w, r);

    // Permute Y: [seq, 1, batch, hidden] -> [batch, seq, 1, hidden].
    std::vector<float> y_batchwise_data(static_cast<size_t>(batch_size * seq_length * hidden_size));
    const float *py0 = y_layout0.AsFloat();
    for (int64_t t = 0; t < seq_length; ++t) {
      for (int64_t n = 0; n < batch_size; ++n) {
        for (int64_t h = 0; h < hidden_size; ++h) {
          const size_t src = static_cast<size_t>((t * batch_size + n) * hidden_size + h);
          const size_t dst = static_cast<size_t>((n * seq_length + t) * hidden_size + h);
          y_batchwise_data[dst] = py0[src];
        }
      }
    }
    Tensor y_batchwise =
        Tensor::FromFloat("", {batch_size, seq_length, 1, hidden_size}, y_batchwise_data);

    // Permute Y_h: [1, batch, hidden] -> [batch, 1, hidden]. With
    // num_directions=1 the leading 1-axis simply moves to the middle, so
    // the underlying buffer contents are unchanged.
    const float *pyh0 = y_h_layout0.AsFloat();
    std::vector<float> y_h_batchwise_data(pyh0,
                                          pyh0 + static_cast<size_t>(batch_size * hidden_size));
    Tensor y_h_batchwise = Tensor::FromFloat("", {batch_size, 1, hidden_size}, y_h_batchwise_data);

    Expect(node, {x_batchwise, w, r}, {std::move(y_batchwise), std::move(y_h_batchwise)},
           "test_cc_simple_rnn_batchwise", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
