// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/nn/include_nn_kernels.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Checks ``t`` is a FLOAT tensor and returns its data pointer (or nullptr
// when ``t`` is the sentinel default-constructed Tensor, used to indicate
// that the corresponding optional input is missing).
const float *AsFloatOrNull(const Tensor &t, const char *role) {
  if (t.shape.empty() && t.size_bytes() == 0) {
    return nullptr;
  }
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT),
                      std::string("kernel::LSTM: ") + role + " must be FLOAT.");
  return t.AsFloat();
}

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

} // namespace

std::pair<Tensor, Tensor> LSTM::operator()(const Tensor &x, const Tensor &w, const Tensor &r,
                                           const Tensor &b, const Tensor &initial_h,
                                           const Tensor &initial_c, const Tensor &p) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LSTM: X must be FLOAT.");
  EXT_ENFORCE_INVALID(w.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LSTM: W must be FLOAT.");
  EXT_ENFORCE_INVALID(r.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LSTM: R must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() == 3u, "kernel::LSTM: X must have rank 3.");
  EXT_ENFORCE_INVALID(w.shape.size() == 3u && w.shape[0] == 1,
                      "kernel::LSTM: W must have shape [1, 4 * hidden_size, input_size] "
                      "(single forward direction only).");
  EXT_ENFORCE_INVALID(r.shape.size() == 3u && r.shape[0] == 1,
                      "kernel::LSTM: R must have shape [1, 4 * hidden_size, hidden_size] "
                      "(single forward direction only).");

  const int64_t seq_length = x.shape[0];
  const int64_t batch_size = x.shape[1];
  const int64_t input_size = x.shape[2];
  const int64_t hidden_size = r.shape[2];

  EXT_ENFORCE_INVALID(w.shape[1] == 4 * hidden_size && w.shape[2] == input_size,
                      "kernel::LSTM: W must have shape [1, 4 * hidden_size, input_size].");
  EXT_ENFORCE_INVALID(r.shape[1] == 4 * hidden_size,
                      "kernel::LSTM: R must have shape [1, 4 * hidden_size, hidden_size].");

  const float *p_b = AsFloatOrNull(b, "B");
  if (p_b != nullptr) {
    EXT_ENFORCE_INVALID(b.shape.size() == 2u && b.shape[0] == 1 && b.shape[1] == 8 * hidden_size,
                        "kernel::LSTM: B must have shape [1, 8 * hidden_size].");
  }
  const float *p_initial_h = AsFloatOrNull(initial_h, "initial_h");
  if (p_initial_h != nullptr) {
    EXT_ENFORCE_INVALID(initial_h.shape.size() == 3u && initial_h.shape[0] == 1 &&
                            initial_h.shape[1] == batch_size && initial_h.shape[2] == hidden_size,
                        "kernel::LSTM: initial_h must have shape [1, batch_size, hidden_size].");
  }
  const float *p_initial_c = AsFloatOrNull(initial_c, "initial_c");
  if (p_initial_c != nullptr) {
    EXT_ENFORCE_INVALID(initial_c.shape.size() == 3u && initial_c.shape[0] == 1 &&
                            initial_c.shape[1] == batch_size && initial_c.shape[2] == hidden_size,
                        "kernel::LSTM: initial_c must have shape [1, batch_size, hidden_size].");
  }
  const float *p_p = AsFloatOrNull(p, "P");
  if (p_p != nullptr) {
    EXT_ENFORCE_INVALID(p.shape.size() == 2u && p.shape[0] == 1 && p.shape[1] == 3 * hidden_size,
                        "kernel::LSTM: P must have shape [1, 3 * hidden_size].");
  }

  const float *px = x.AsFloat();
  const float *pw = w.AsFloat();
  const float *pr = r.AsFloat();

  // Gate layout in W/R/B is ONNX's ``i, o, f, c`` order; convenience
  // base offsets into the gate axis (length 4 * hidden_size).
  const int64_t gI = 0;
  const int64_t gO = hidden_size;
  const int64_t gF = 2 * hidden_size;
  const int64_t gC = 3 * hidden_size;

  // Combined per-gate bias = Wb + Rb (length hidden_size each).
  std::vector<float> bias_i(static_cast<size_t>(hidden_size), 0.0f);
  std::vector<float> bias_o(static_cast<size_t>(hidden_size), 0.0f);
  std::vector<float> bias_f(static_cast<size_t>(hidden_size), 0.0f);
  std::vector<float> bias_c(static_cast<size_t>(hidden_size), 0.0f);
  if (p_b != nullptr) {
    const float *wb = p_b;                   // first half: Wb
    const float *rb = p_b + 4 * hidden_size; // second half: Rb
    for (int64_t h = 0; h < hidden_size; ++h) {
      bias_i[static_cast<size_t>(h)] = wb[gI + h] + rb[gI + h];
      bias_o[static_cast<size_t>(h)] = wb[gO + h] + rb[gO + h];
      bias_f[static_cast<size_t>(h)] = wb[gF + h] + rb[gF + h];
      bias_c[static_cast<size_t>(h)] = wb[gC + h] + rb[gC + h];
    }
  }

  // Peephole weights (length hidden_size each) in gate order i, o, f.
  std::vector<float> peep_i(static_cast<size_t>(hidden_size), 0.0f);
  std::vector<float> peep_o(static_cast<size_t>(hidden_size), 0.0f);
  std::vector<float> peep_f(static_cast<size_t>(hidden_size), 0.0f);
  if (p_p != nullptr) {
    for (int64_t h = 0; h < hidden_size; ++h) {
      peep_i[static_cast<size_t>(h)] = p_p[0 * hidden_size + h];
      peep_o[static_cast<size_t>(h)] = p_p[1 * hidden_size + h];
      peep_f[static_cast<size_t>(h)] = p_p[2 * hidden_size + h];
    }
  }

  // Output allocations.
  const std::vector<int64_t> y_shape{seq_length, 1, batch_size, hidden_size};
  const std::vector<int64_t> y_h_shape{1, batch_size, hidden_size};
  Tensor y("", static_cast<int32_t>(DataType::FLOAT), y_shape,
           std::vector<uint8_t>(static_cast<size_t>(seq_length * batch_size * hidden_size) *
                                sizeof(float)));
  Tensor y_h("", static_cast<int32_t>(DataType::FLOAT), y_h_shape,
             std::vector<uint8_t>(static_cast<size_t>(batch_size * hidden_size) * sizeof(float)));
  float *py = y.AsFloat();
  float *py_h = y_h.AsFloat();

  // Working buffers for H_{t-1}, C_{t-1}, H_t, C_t.
  std::vector<float> h_prev(static_cast<size_t>(batch_size * hidden_size), 0.0f);
  std::vector<float> c_prev(static_cast<size_t>(batch_size * hidden_size), 0.0f);
  if (p_initial_h != nullptr) {
    for (int64_t i = 0; i < batch_size * hidden_size; ++i) {
      h_prev[static_cast<size_t>(i)] = p_initial_h[i];
    }
  }
  if (p_initial_c != nullptr) {
    for (int64_t i = 0; i < batch_size * hidden_size; ++i) {
      c_prev[static_cast<size_t>(i)] = p_initial_c[i];
    }
  }
  std::vector<float> h_curr(static_cast<size_t>(batch_size * hidden_size), 0.0f);
  std::vector<float> c_curr(static_cast<size_t>(batch_size * hidden_size), 0.0f);

  // Reusable per-step accumulator: one row per gate (i, o, f, c).
  std::vector<float> acc_i(static_cast<size_t>(hidden_size), 0.0f);
  std::vector<float> acc_o(static_cast<size_t>(hidden_size), 0.0f);
  std::vector<float> acc_f(static_cast<size_t>(hidden_size), 0.0f);
  std::vector<float> acc_c(static_cast<size_t>(hidden_size), 0.0f);

  for (int64_t t = 0; t < seq_length; ++t) {
    const float *x_t = px + t * batch_size * input_size;
    for (int64_t n = 0; n < batch_size; ++n) {
      const float *x_row = x_t + n * input_size;
      const float *h_row = h_prev.data() + n * hidden_size;
      const float *c_row = c_prev.data() + n * hidden_size;
      float *h_out = h_curr.data() + n * hidden_size;
      float *c_out = c_curr.data() + n * hidden_size;

      // Compute per-gate pre-activations: X_t @ W^T + H_{t-1} @ R^T + bias.
      for (int64_t h = 0; h < hidden_size; ++h) {
        const float *wi_row = pw + (gI + h) * input_size;
        const float *wo_row = pw + (gO + h) * input_size;
        const float *wf_row = pw + (gF + h) * input_size;
        const float *wc_row = pw + (gC + h) * input_size;
        const float *ri_row = pr + (gI + h) * hidden_size;
        const float *ro_row = pr + (gO + h) * hidden_size;
        const float *rf_row = pr + (gF + h) * hidden_size;
        const float *rc_row = pr + (gC + h) * hidden_size;
        float ai = bias_i[static_cast<size_t>(h)];
        float ao = bias_o[static_cast<size_t>(h)];
        float af = bias_f[static_cast<size_t>(h)];
        float ac = bias_c[static_cast<size_t>(h)];
        for (int64_t k = 0; k < input_size; ++k) {
          const float xk = x_row[k];
          ai += xk * wi_row[k];
          ao += xk * wo_row[k];
          af += xk * wf_row[k];
          ac += xk * wc_row[k];
        }
        for (int64_t k = 0; k < hidden_size; ++k) {
          const float hk = h_row[k];
          ai += hk * ri_row[k];
          ao += hk * ro_row[k];
          af += hk * rf_row[k];
          ac += hk * rc_row[k];
        }
        acc_i[static_cast<size_t>(h)] = ai;
        acc_o[static_cast<size_t>(h)] = ao;
        acc_f[static_cast<size_t>(h)] = af;
        acc_c[static_cast<size_t>(h)] = ac;
      }

      // Apply activations + peepholes; update cell state then hidden state.
      // ``o`` depends on the *new* ``Ct``, so it is computed last.
      for (int64_t h = 0; h < hidden_size; ++h) {
        const float c_prev_h = c_row[h];
        const float it =
            Sigmoid(acc_i[static_cast<size_t>(h)] + peep_i[static_cast<size_t>(h)] * c_prev_h);
        const float ft =
            Sigmoid(acc_f[static_cast<size_t>(h)] + peep_f[static_cast<size_t>(h)] * c_prev_h);
        const float ct = std::tanh(acc_c[static_cast<size_t>(h)]);
        const float Ct = ft * c_prev_h + it * ct;
        const float ot =
            Sigmoid(acc_o[static_cast<size_t>(h)] + peep_o[static_cast<size_t>(h)] * Ct);
        const float Ht = ot * std::tanh(Ct);
        c_out[h] = Ct;
        h_out[h] = Ht;
      }
    }
    // Copy h_curr into Y at time-step t (num_directions=1) and swap into
    // h_prev / c_prev for the next iteration.
    float *y_t = py + t * batch_size * hidden_size;
    for (int64_t i = 0; i < batch_size * hidden_size; ++i) {
      y_t[i] = h_curr[static_cast<size_t>(i)];
    }
    h_prev.swap(h_curr);
    c_prev.swap(c_curr);
  }

  // Y_h is the last time step of Y (currently in h_prev after the final swap).
  for (int64_t i = 0; i < batch_size * hidden_size; ++i) {
    py_h[i] = h_prev[static_cast<size_t>(i)];
  }

  return std::pair<Tensor, Tensor>(std::move(y), std::move(y_h));
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
