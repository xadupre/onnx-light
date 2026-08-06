// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/random.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// RNN — one-layer RNN with the default Tanh
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
void RegisterRNNCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::RNN rnn_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    const int64_t seq_length = 64;
    const int64_t batch_size = 32;
    const int64_t input_size = 128;
    const int64_t hidden_size = 128;
    NodeProto node;
    node.set_op_type("RNN");
    node.add_input("X");
    node.add_input("W");
    node.add_input("R");
    node.add_output("");
    node.add_output("Y_h");
    AddAttribute<int64_t>(node, "hidden_size", hidden_size);
    Expect(
        registry, std::move(node), "test_cc_rnn_benchmark", {opset},
        {seq_length * batch_size * input_size, hidden_size * input_size, hidden_size * hidden_size},
        {batch_size * hidden_size},
        [rnn_kernel, seq_length, batch_size, input_size, hidden_size]() -> IoData {
          const std::vector<int64_t> x_shape = {seq_length, batch_size, input_size};
          const std::vector<int64_t> w_shape = {1, hidden_size, input_size};
          const std::vector<int64_t> r_shape = {1, hidden_size, hidden_size};
          Tensor x = Tensor::FromFloat("", x_shape, Randn<float>(x_shape, 2001));
          Tensor w = Tensor::FromFloat("", w_shape, Randn<float>(w_shape, 2002));
          Tensor r = Tensor::FromFloat("", r_shape, Randn<float>(r_shape, 2003));
          auto [y_unused, y_h] = rnn_kernel(x, w, r);
          (void)y_unused;
          return IoData{{std::move(x), std::move(w), std::move(r)}, {std::move(y_h)}};
        });
    return;
  }

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
    Expect(registry, std::move(node), "test_cc_simple_rnn_defaults", {opset}, [=]() -> IoData {
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

      return IoData{{std::move(x), std::move(w), std::move(r)}, {std::move(y_h)}};
    });
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
    Expect(registry, std::move(node), "test_cc_simple_rnn_with_initial_bias", {opset},
           [=]() -> IoData {
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

             return IoData{{std::move(x), std::move(w), std::move(r), std::move(b), std::move(h0)},
                           {std::move(y), std::move(y_h)}};
           });
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
    Expect(registry, std::move(node), "test_cc_rnn_seq_length", {opset}, [=]() -> IoData {
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

      return IoData{{std::move(x), std::move(w), std::move(r), std::move(b)}, {std::move(y_h)}};
    });
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
    Expect(registry, std::move(node), "test_cc_simple_rnn_batchwise", {opset}, [=]() -> IoData {
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
      Tensor x_layout0 =
          Tensor::FromFloat("", {seq_length, batch_size, input_size}, x_layout0_data);

      const float weight_scale = 0.5f;
      std::vector<float> w_data(static_cast<size_t>(hidden_size * input_size), weight_scale);
      std::vector<float> r_data(static_cast<size_t>(hidden_size * hidden_size), weight_scale);
      Tensor w = Tensor::FromFloat("", {1, hidden_size, input_size}, w_data);
      Tensor r = Tensor::FromFloat("", {1, hidden_size, hidden_size}, r_data);

      auto [y_layout0, y_h_layout0] = rnn_kernel(x_layout0, w, r);

      // Permute Y: [seq, 1, batch, hidden] -> [batch, seq, 1, hidden].
      std::vector<float> y_batchwise_data(
          static_cast<size_t>(batch_size * seq_length * hidden_size));
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
      Tensor y_h_batchwise =
          Tensor::FromFloat("", {batch_size, 1, hidden_size}, y_h_batchwise_data);

      return IoData{{std::move(x_batchwise), std::move(w), std::move(r)},
                    {std::move(y_batchwise), std::move(y_h_batchwise)}};
    });
  }

  // ``simple_rnn_reverse``: single ``reverse`` direction, seq_length=3,
  // batch_size=1, input_size=2, hidden_size=5, mirroring upstream
  // ``test_simple_rnn_reverse``. Only ``Y_h`` is produced.
  {
    NodeProto node;
    node.set_op_type("RNN");
    node.add_input("X");
    node.add_input("W");
    node.add_input("R");
    node.add_output("");
    node.add_output("Y_h");
    AddAttribute<int64_t>(node, "hidden_size", 5);
    AddAttribute(node, "direction", std::string("reverse"));
    Expect(registry, std::move(node), "test_cc_simple_rnn_reverse", {opset}, [=]() -> IoData {
      const int64_t seq_length = 3;
      const int64_t batch_size = 1;
      const int64_t input_size = 2;
      const int64_t hidden_size = 5;
      const std::vector<float> x_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
      const float weight_scale = 0.1f;
      std::vector<float> w_data(static_cast<size_t>(hidden_size * input_size), weight_scale);
      std::vector<float> r_data(static_cast<size_t>(hidden_size * hidden_size), weight_scale);
      Tensor x = Tensor::FromFloat("", {seq_length, batch_size, input_size}, x_data);
      Tensor w = Tensor::FromFloat("", {1, hidden_size, input_size}, w_data);
      Tensor r = Tensor::FromFloat("", {1, hidden_size, hidden_size}, r_data);

      auto [y_unused, y_h] = rnn_kernel(x, w, r, Tensor{}, Tensor{}, 0, "reverse");
      (void)y_unused; // Y is skipped (empty output name).

      return IoData{{std::move(x), std::move(w), std::move(r)}, {std::move(y_h)}};
    });
  }

  // ``bidirectional_rnn``: ``bidirectional`` direction (num_directions=2),
  // seq_length=3, batch_size=1, input_size=2, hidden_size=5, mirroring
  // upstream ``test_simple_rnn_bidirectional``. Both ``Y`` and ``Y_h`` are
  // produced.
  {
    NodeProto node;
    node.set_op_type("RNN");
    node.add_input("X");
    node.add_input("W");
    node.add_input("R");
    node.add_output("Y");
    node.add_output("Y_h");
    AddAttribute<int64_t>(node, "hidden_size", 5);
    AddAttribute(node, "direction", std::string("bidirectional"));
    Expect(registry, std::move(node), "test_cc_simple_rnn_bidirectional", {opset}, [=]() -> IoData {
      const int64_t num_directions = 2;
      const int64_t seq_length = 3;
      const int64_t batch_size = 1;
      const int64_t input_size = 2;
      const int64_t hidden_size = 5;
      const std::vector<float> x_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
      // Distinct per-direction weight scales so the forward and reverse
      // passes produce different states.
      std::vector<float> w_data(static_cast<size_t>(num_directions * hidden_size * input_size));
      std::vector<float> r_data(static_cast<size_t>(num_directions * hidden_size * hidden_size));
      const float scales[2] = {0.1f, 0.2f};
      for (int64_t d = 0; d < num_directions; ++d) {
        for (int64_t i = 0; i < hidden_size * input_size; ++i) {
          w_data[static_cast<size_t>(d * hidden_size * input_size + i)] = scales[d];
        }
        for (int64_t i = 0; i < hidden_size * hidden_size; ++i) {
          r_data[static_cast<size_t>(d * hidden_size * hidden_size + i)] = scales[d];
        }
      }
      Tensor x = Tensor::FromFloat("", {seq_length, batch_size, input_size}, x_data);
      Tensor w = Tensor::FromFloat("", {num_directions, hidden_size, input_size}, w_data);
      Tensor r = Tensor::FromFloat("", {num_directions, hidden_size, hidden_size}, r_data);

      auto [y, y_h] = rnn_kernel(x, w, r, Tensor{}, Tensor{}, 0, "bidirectional");

      return IoData{{std::move(x), std::move(w), std::move(r)},
                    {std::move(y), std::move(y_h)}};
    });
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
