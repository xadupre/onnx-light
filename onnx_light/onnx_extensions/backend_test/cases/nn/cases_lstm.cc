// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// LSTM — one-layer LSTM with the default
// Sigmoid/Tanh/Tanh activations, mirroring the upstream ``test_lstm_*``
// reference cases:
//
//   * ``test_cc_lstm_defaults`` — tiny 1x3x2 input with X/W/R inputs only,
//     hidden_size=3.
//   * ``test_cc_lstm_with_initial_bias`` — exercises the optional ``B``
//     (concatenated ``[Wb, Rb]``) input, hidden_size=4 with a custom bias.
//   * ``test_cc_lstm_with_peepholes`` — full 8-input form
//     ``X, W, R, B, sequence_lens, initial_h, initial_c, P``, exercising
//     the optional peephole weights, hidden_size=3.
//   * ``test_cc_lstm_batchwise`` — ``layout=1`` variant where ``X`` is laid
//     out as ``[batch_size, seq_length, input_size]``, hidden_size=7.
// ---------------------------------------------------------------------------
void RegisterLSTMCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::LSTM lstm_kernel{ctx};

  constexpr int64_t kNumGates = 4;
  constexpr int64_t kNumPeepholes = 3;

  if (mode == TestMode::BENCHMARK) {
    const int64_t seq_length = 64;
    const int64_t batch_size = 32;
    const int64_t input_size = 128;
    const int64_t hidden_size = 128;
    NodeProto node;
    node.set_op_type("LSTM");
    node.add_input("X");
    node.add_input("W");
    node.add_input("R");
    node.add_output("");
    node.add_output("Y_h");
    AddAttribute<int64_t>(node, "hidden_size", hidden_size);
    Expect(registry, std::move(node), "test_cc_lstm_benchmark", {opset},
           {seq_length * batch_size * input_size, kNumGates * hidden_size * input_size,
            kNumGates * hidden_size * hidden_size},
           {batch_size * hidden_size},
           [lstm_kernel, seq_length, batch_size, input_size, hidden_size]() -> IoData {
             const std::vector<int64_t> x_shape = {seq_length, batch_size, input_size};
             const std::vector<int64_t> w_shape = {1, kNumGates * hidden_size, input_size};
             const std::vector<int64_t> r_shape = {1, kNumGates * hidden_size, hidden_size};
             Tensor x = RandnTensor(DataType::FLOAT, x_shape, 2001);
             Tensor w = RandnTensor(DataType::FLOAT, w_shape, 2002);
             Tensor r = RandnTensor(DataType::FLOAT, r_shape, 2003);
             auto [y_unused, y_h, y_c_unused] = lstm_kernel(x, w, r);
             (void)y_c_unused;
             (void)y_unused;
             return IoData{{std::move(x), std::move(w), std::move(r)}, {std::move(y_h)}};
           });
    return;
  }

  // ``lstm_defaults``: seq_length=1, batch_size=3, input_size=2,
  // hidden_size=3. No bias, no initial_h/c, no peepholes; only Y_h is
  // produced (Y is skipped via an empty output name).
  {
    NodeProto node;
    node.set_op_type("LSTM");
    node.add_input("X");
    node.add_input("W");
    node.add_input("R");
    node.add_output("");
    node.add_output("Y_h");
    AddAttribute<int64_t>(node, "hidden_size", 3);
    Expect(registry, std::move(node), "test_cc_lstm_defaults", {opset}, [=]() -> IoData {
      const int64_t seq_length = 1;
      const int64_t batch_size = 3;
      const int64_t input_size = 2;
      const int64_t hidden_size = 3;
      const float weight_scale = 0.1f;

      const std::vector<float> x_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
      std::vector<float> w_data(static_cast<size_t>(kNumGates * hidden_size * input_size),
                                weight_scale);
      std::vector<float> r_data(static_cast<size_t>(kNumGates * hidden_size * hidden_size),
                                weight_scale);
      Tensor x = Tensor::FromFloat("", {seq_length, batch_size, input_size}, x_data);
      Tensor w = Tensor::FromFloat("", {1, kNumGates * hidden_size, input_size}, w_data);
      Tensor r = Tensor::FromFloat("", {1, kNumGates * hidden_size, hidden_size}, r_data);

      auto [y_unused, y_h, y_c_unused] = lstm_kernel(x, w, r);
      (void)y_c_unused;
      (void)y_unused; // Y is skipped (empty output name).

      return IoData{{std::move(x), std::move(w), std::move(r)}, {std::move(y_h)}};
    });
  }

  // ``lstm_with_initial_bias``: seq_length=1, batch_size=3, input_size=3,
  // hidden_size=4 with a custom-valued ``B`` (Wb is all ``custom_bias``,
  // Rb is all zeros); only Y_h is produced.
  {
    NodeProto node;
    node.set_op_type("LSTM");
    node.add_input("X");
    node.add_input("W");
    node.add_input("R");
    node.add_input("B");
    node.add_output("");
    node.add_output("Y_h");
    AddAttribute<int64_t>(node, "hidden_size", 4);
    Expect(registry, std::move(node), "test_cc_lstm_with_initial_bias", {opset}, [=]() -> IoData {
      const int64_t seq_length = 1;
      const int64_t batch_size = 3;
      const int64_t input_size = 3;
      const int64_t hidden_size = 4;
      const float weight_scale = 0.1f;
      const float custom_bias = 0.1f;

      const std::vector<float> x_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
      std::vector<float> w_data(static_cast<size_t>(kNumGates * hidden_size * input_size),
                                weight_scale);
      std::vector<float> r_data(static_cast<size_t>(kNumGates * hidden_size * hidden_size),
                                weight_scale);
      // B = concat(Wb, Rb) along the last axis: Wb = custom_bias * ones,
      // Rb = zeros. Shape: [1, 8 * hidden_size].
      std::vector<float> b_data(static_cast<size_t>(2 * kNumGates * hidden_size), 0.0f);
      for (int64_t i = 0; i < kNumGates * hidden_size; ++i) {
        b_data[static_cast<size_t>(i)] = custom_bias;
      }
      Tensor x = Tensor::FromFloat("", {seq_length, batch_size, input_size}, x_data);
      Tensor w = Tensor::FromFloat("", {1, kNumGates * hidden_size, input_size}, w_data);
      Tensor r = Tensor::FromFloat("", {1, kNumGates * hidden_size, hidden_size}, r_data);
      Tensor b = Tensor::FromFloat("", {1, 2 * kNumGates * hidden_size}, b_data);

      auto [y_unused, y_h, y_c_unused] = lstm_kernel(x, w, r, b);
      (void)y_c_unused;
      (void)y_unused;

      return IoData{{std::move(x), std::move(w), std::move(r), std::move(b)}, {std::move(y_h)}};
    });
  }

  // ``lstm_with_peepholes``: full 8-input form
  // ``X, W, R, B, sequence_lens, initial_h, initial_c, P``. seq_length=1,
  // batch_size=2, input_size=4, hidden_size=3. ``B`` is all zeros,
  // ``sequence_lens`` is uniform (every batch row uses the full
  // seq_length), ``initial_h``/``initial_c`` are zero and ``P`` is
  // ``weight_scale * ones``. Only Y_h is produced.
  {
    NodeProto node;
    node.set_op_type("LSTM");
    node.add_input("X");
    node.add_input("W");
    node.add_input("R");
    node.add_input("B");
    node.add_input("sequence_lens");
    node.add_input("initial_h");
    node.add_input("initial_c");
    node.add_input("P");
    node.add_output("");
    node.add_output("Y_h");
    AddAttribute<int64_t>(node, "hidden_size", 3);
    Expect(registry, std::move(node), "test_cc_lstm_with_peepholes", {opset}, [=]() -> IoData {
      const int64_t seq_length = 1;
      const int64_t batch_size = 2;
      const int64_t input_size = 4;
      const int64_t hidden_size = 3;
      const float weight_scale = 0.1f;

      const std::vector<float> x_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
      std::vector<float> w_data(static_cast<size_t>(kNumGates * hidden_size * input_size),
                                weight_scale);
      std::vector<float> r_data(static_cast<size_t>(kNumGates * hidden_size * hidden_size),
                                weight_scale);
      std::vector<float> b_data(static_cast<size_t>(2 * kNumGates * hidden_size), 0.0f);
      std::vector<int32_t> seq_lens_data(static_cast<size_t>(batch_size),
                                         static_cast<int32_t>(seq_length));
      std::vector<float> h0_data(static_cast<size_t>(batch_size * hidden_size), 0.0f);
      std::vector<float> c0_data(static_cast<size_t>(batch_size * hidden_size), 0.0f);
      std::vector<float> p_data(static_cast<size_t>(kNumPeepholes * hidden_size), weight_scale);

      Tensor x = Tensor::FromFloat("", {seq_length, batch_size, input_size}, x_data);
      Tensor w = Tensor::FromFloat("", {1, kNumGates * hidden_size, input_size}, w_data);
      Tensor r = Tensor::FromFloat("", {1, kNumGates * hidden_size, hidden_size}, r_data);
      Tensor b = Tensor::FromFloat("", {1, 2 * kNumGates * hidden_size}, b_data);
      Tensor seq_lens = Tensor::FromInt32("", {batch_size}, seq_lens_data);
      Tensor h0 = Tensor::FromFloat("", {1, batch_size, hidden_size}, h0_data);
      Tensor c0 = Tensor::FromFloat("", {1, batch_size, hidden_size}, c0_data);
      Tensor p_tensor = Tensor::FromFloat("", {1, kNumPeepholes * hidden_size}, p_data);

      auto [y_unused, y_h, y_c_unused] = lstm_kernel(x, w, r, b, h0, c0, p_tensor);
      (void)y_c_unused;
      (void)y_unused;

      return IoData{{std::move(x), std::move(w), std::move(r), std::move(b), std::move(seq_lens),
                     std::move(h0), std::move(c0), std::move(p_tensor)},
                    {std::move(y_h)}};
    });
  }

  // ``lstm_batchwise``: ``layout=1`` variant with batch_size=3,
  // seq_length=1, input_size=2, hidden_size=7. The kernel itself only
  // implements ``layout=0`` so we run it on the axis-swapped ``X`` and
  // re-permute the outputs to the batchwise layout:
  //   * ``Y``   : [seq, 1, batch, hidden]  -> [batch, seq, 1, hidden]
  //   * ``Y_h`` : [1, batch, hidden]       -> [batch, 1, hidden]
  {
    NodeProto node;
    node.set_op_type("LSTM");
    node.add_input("X");
    node.add_input("W");
    node.add_input("R");
    node.add_output("Y");
    node.add_output("Y_h");
    AddAttribute<int64_t>(node, "hidden_size", 7);
    AddAttribute<int64_t>(node, "layout", 1);
    Expect(registry, std::move(node), "test_cc_lstm_batchwise", {opset}, [=]() -> IoData {
      const int64_t batch_size = 3;
      const int64_t seq_length = 1;
      const int64_t input_size = 2;
      const int64_t hidden_size = 7;
      const float weight_scale = 0.3f;

      // Input is [batch_size, seq_length, input_size] for layout=1.
      const std::vector<float> x_batchwise_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
      Tensor x_batchwise =
          Tensor::FromFloat("", {batch_size, seq_length, input_size}, x_batchwise_data);

      // Same contents but laid out for layout=0
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

      std::vector<float> w_data(static_cast<size_t>(kNumGates * hidden_size * input_size),
                                weight_scale);
      std::vector<float> r_data(static_cast<size_t>(kNumGates * hidden_size * hidden_size),
                                weight_scale);
      Tensor w = Tensor::FromFloat("", {1, kNumGates * hidden_size, input_size}, w_data);
      Tensor r = Tensor::FromFloat("", {1, kNumGates * hidden_size, hidden_size}, r_data);

      auto [y_layout0, y_h_layout0, y_c_layout0] = lstm_kernel(x_layout0, w, r);
      (void)y_c_layout0;

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

  // ``lstm_reverse``: single ``reverse`` direction, seq_length=3,
  // batch_size=1, input_size=2, hidden_size=3, mirroring upstream
  // ``test_lstm_reverse``. Both ``Y_h`` and ``Y_c`` are produced.
  {
    NodeProto node;
    node.set_op_type("LSTM");
    node.add_input("X");
    node.add_input("W");
    node.add_input("R");
    node.add_output("");
    node.add_output("Y_h");
    node.add_output("Y_c");
    AddAttribute<int64_t>(node, "hidden_size", 3);
    AddAttribute(node, "direction", std::string("reverse"));
    Expect(registry, std::move(node), "test_cc_lstm_reverse", {opset}, [=]() -> IoData {
      const int64_t seq_length = 3;
      const int64_t batch_size = 1;
      const int64_t input_size = 2;
      const int64_t hidden_size = 3;
      const std::vector<float> x_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
      const float weight_scale = 0.1f;
      std::vector<float> w_data(static_cast<size_t>(kNumGates * hidden_size * input_size),
                                weight_scale);
      std::vector<float> r_data(static_cast<size_t>(kNumGates * hidden_size * hidden_size),
                                weight_scale);
      Tensor x = Tensor::FromFloat("", {seq_length, batch_size, input_size}, x_data);
      Tensor w = Tensor::FromFloat("", {1, kNumGates * hidden_size, input_size}, w_data);
      Tensor r = Tensor::FromFloat("", {1, kNumGates * hidden_size, hidden_size}, r_data);

      auto [y_unused, y_h, y_c] =
          lstm_kernel(x, w, r, Tensor{}, Tensor{}, Tensor{}, Tensor{}, 0, "reverse");
      (void)y_unused; // Y is skipped (empty output name).

      return IoData{{std::move(x), std::move(w), std::move(r)}, {std::move(y_h), std::move(y_c)}};
    });
  }

  // ``lstm_bidirectional``: ``bidirectional`` direction (num_directions=2),
  // seq_length=3, batch_size=1, input_size=2, hidden_size=3, mirroring
  // upstream ``test_lstm_bidirectional``. ``Y``, ``Y_h`` and ``Y_c`` are all
  // produced.
  {
    NodeProto node;
    node.set_op_type("LSTM");
    node.add_input("X");
    node.add_input("W");
    node.add_input("R");
    node.add_output("Y");
    node.add_output("Y_h");
    node.add_output("Y_c");
    AddAttribute<int64_t>(node, "hidden_size", 3);
    AddAttribute(node, "direction", std::string("bidirectional"));
    Expect(registry, std::move(node), "test_cc_lstm_bidirectional", {opset}, [=]() -> IoData {
      const int64_t num_directions = 2;
      const int64_t seq_length = 3;
      const int64_t batch_size = 1;
      const int64_t input_size = 2;
      const int64_t hidden_size = 3;
      const std::vector<float> x_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
      std::vector<float> w_data(
          static_cast<size_t>(num_directions * kNumGates * hidden_size * input_size));
      std::vector<float> r_data(
          static_cast<size_t>(num_directions * kNumGates * hidden_size * hidden_size));
      const float scales[2] = {0.1f, 0.2f};
      const int64_t w_block = kNumGates * hidden_size * input_size;
      const int64_t r_block = kNumGates * hidden_size * hidden_size;
      for (int64_t d = 0; d < num_directions; ++d) {
        for (int64_t i = 0; i < w_block; ++i) {
          w_data[static_cast<size_t>(d * w_block + i)] = scales[d];
        }
        for (int64_t i = 0; i < r_block; ++i) {
          r_data[static_cast<size_t>(d * r_block + i)] = scales[d];
        }
      }
      Tensor x = Tensor::FromFloat("", {seq_length, batch_size, input_size}, x_data);
      Tensor w =
          Tensor::FromFloat("", {num_directions, kNumGates * hidden_size, input_size}, w_data);
      Tensor r =
          Tensor::FromFloat("", {num_directions, kNumGates * hidden_size, hidden_size}, r_data);

      auto [y, y_h, y_c] =
          lstm_kernel(x, w, r, Tensor{}, Tensor{}, Tensor{}, Tensor{}, 0, "bidirectional");

      return IoData{{std::move(x), std::move(w), std::move(r)},
                    {std::move(y), std::move(y_h), std::move(y_c)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
