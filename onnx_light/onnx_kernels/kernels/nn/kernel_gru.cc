// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Checks ``t`` is a FLOAT tensor and returns its data pointer (or nullptr
// when ``t`` is the sentinel default-constructed Tensor, used to indicate
// that the corresponding optional input is missing).
const float *AsFloatOrNull(const Tensor &t, const char *role) {
  if (t.shape.empty() && t.data.empty()) {
    return nullptr;
  }
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT),
                      std::string("kernel::GRU: ") + role + " must be FLOAT.");
  return t.AsFloat();
}

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

} // namespace

std::pair<Tensor, Tensor> GRU::operator()(const Tensor &x, const Tensor &w, const Tensor &r,
                                          const Tensor &b, const Tensor &initial_h,
                                          int64_t linear_before_reset) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GRU: X must be FLOAT.");
  EXT_ENFORCE_INVALID(w.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GRU: W must be FLOAT.");
  EXT_ENFORCE_INVALID(r.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GRU: R must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() == 3u, "kernel::GRU: X must have rank 3.");
  EXT_ENFORCE_INVALID(w.shape.size() == 3u && w.shape[0] == 1,
                      "kernel::GRU: W must have shape [1, 3 * hidden_size, input_size] "
                      "(single forward direction only).");
  EXT_ENFORCE_INVALID(r.shape.size() == 3u && r.shape[0] == 1,
                      "kernel::GRU: R must have shape [1, 3 * hidden_size, hidden_size] "
                      "(single forward direction only).");

  const int64_t seq_length = x.shape[0];
  const int64_t batch_size = x.shape[1];
  const int64_t input_size = x.shape[2];
  const int64_t hidden_size = r.shape[2];

  EXT_ENFORCE_INVALID(w.shape[1] == 3 * hidden_size && w.shape[2] == input_size,
                      "kernel::GRU: W must have shape [1, 3 * hidden_size, input_size].");
  EXT_ENFORCE_INVALID(r.shape[1] == 3 * hidden_size,
                      "kernel::GRU: R must have shape [1, 3 * hidden_size, hidden_size].");

  const float *p_b = AsFloatOrNull(b, "B");
  if (p_b != nullptr) {
    EXT_ENFORCE_INVALID(b.shape.size() == 2u && b.shape[0] == 1 && b.shape[1] == 6 * hidden_size,
                        "kernel::GRU: B must have shape [1, 6 * hidden_size].");
  }
  const float *p_initial_h = AsFloatOrNull(initial_h, "initial_h");
  if (p_initial_h != nullptr) {
    EXT_ENFORCE_INVALID(initial_h.shape.size() == 3u && initial_h.shape[0] == 1 &&
                            initial_h.shape[1] == batch_size && initial_h.shape[2] == hidden_size,
                        "kernel::GRU: initial_h must have shape [1, batch_size, hidden_size].");
  }

  const float *px = x.AsFloat();
  const float *pw = w.AsFloat();
  const float *pr = r.AsFloat();

  // Bias slices (each of length ``hidden_size``). Gate order is ``z, r, h``.
  // ``Wb*`` are the first 3 blocks of ``B``; ``Rb*`` are the last 3 blocks.
  const float *wbz = nullptr;
  const float *wbr = nullptr;
  const float *wbh = nullptr;
  const float *rbz = nullptr;
  const float *rbr = nullptr;
  const float *rbh = nullptr;
  if (p_b != nullptr) {
    wbz = p_b + 0 * hidden_size;
    wbr = p_b + 1 * hidden_size;
    wbh = p_b + 2 * hidden_size;
    rbz = p_b + 3 * hidden_size;
    rbr = p_b + 4 * hidden_size;
    rbh = p_b + 5 * hidden_size;
  }

  // Weight slice offsets within W / R (each ``hidden_size`` rows). Gate
  // order is ``z, r, h``.
  const float *wz = pw + 0 * hidden_size * input_size;
  const float *wr_ = pw + 1 * hidden_size * input_size;
  const float *wh = pw + 2 * hidden_size * input_size;
  const float *rz = pr + 0 * hidden_size * hidden_size;
  const float *rr_ = pr + 1 * hidden_size * hidden_size;
  const float *rh = pr + 2 * hidden_size * hidden_size;

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

  // Working buffers for H_{t-1} and H_t.
  std::vector<float> h_prev(static_cast<size_t>(batch_size * hidden_size), 0.0f);
  if (p_initial_h != nullptr) {
    for (int64_t i = 0; i < batch_size * hidden_size; ++i) {
      h_prev[static_cast<size_t>(i)] = p_initial_h[i];
    }
  }
  std::vector<float> h_curr(static_cast<size_t>(batch_size * hidden_size), 0.0f);

  // Per-time-step gate buffers (sized for one batch row at a time).
  std::vector<float> z_gate(static_cast<size_t>(hidden_size), 0.0f);
  std::vector<float> r_gate(static_cast<size_t>(hidden_size), 0.0f);
  std::vector<float> h_tilde(static_cast<size_t>(hidden_size), 0.0f);

  for (int64_t t = 0; t < seq_length; ++t) {
    const float *x_t = px + t * batch_size * input_size;
    for (int64_t n = 0; n < batch_size; ++n) {
      const float *x_row = x_t + n * input_size;
      const float *h_row = h_prev.data() + n * hidden_size;
      float *out_row = h_curr.data() + n * hidden_size;

      // z_t and r_t gates.
      for (int64_t h = 0; h < hidden_size; ++h) {
        const float *wz_row = wz + h * input_size;
        const float *rz_row = rz + h * hidden_size;
        float acc_z = (wbz != nullptr) ? wbz[h] + rbz[h] : 0.0f;
        for (int64_t k = 0; k < input_size; ++k) {
          acc_z += x_row[k] * wz_row[k];
        }
        for (int64_t k = 0; k < hidden_size; ++k) {
          acc_z += h_row[k] * rz_row[k];
        }
        z_gate[static_cast<size_t>(h)] = Sigmoid(acc_z);

        const float *wr_row = wr_ + h * input_size;
        const float *rr_row = rr_ + h * hidden_size;
        float acc_r = (wbr != nullptr) ? wbr[h] + rbr[h] : 0.0f;
        for (int64_t k = 0; k < input_size; ++k) {
          acc_r += x_row[k] * wr_row[k];
        }
        for (int64_t k = 0; k < hidden_size; ++k) {
          acc_r += h_row[k] * rr_row[k];
        }
        r_gate[static_cast<size_t>(h)] = Sigmoid(acc_r);
      }

      // h_tilde (candidate state). The reset gate is applied either to
      // H_{t-1} *before* the H @ Rh^T multiply (default,
      // ``linear_before_reset == 0``) or to the linear combination
      // ``H_{t-1} @ Rh^T + Rbh`` (``linear_before_reset != 0``).
      for (int64_t h = 0; h < hidden_size; ++h) {
        const float *wh_row = wh + h * input_size;
        const float *rh_row = rh + h * hidden_size;
        float acc_x = (wbh != nullptr) ? wbh[h] : 0.0f;
        for (int64_t k = 0; k < input_size; ++k) {
          acc_x += x_row[k] * wh_row[k];
        }
        float acc_h = 0.0f;
        if (linear_before_reset != 0) {
          float lin = (rbh != nullptr) ? rbh[h] : 0.0f;
          for (int64_t k = 0; k < hidden_size; ++k) {
            lin += h_row[k] * rh_row[k];
          }
          acc_h = r_gate[static_cast<size_t>(h)] * lin;
        } else {
          for (int64_t k = 0; k < hidden_size; ++k) {
            acc_h += (r_gate[static_cast<size_t>(k)] * h_row[k]) * rh_row[k];
          }
          if (rbh != nullptr) {
            acc_h += rbh[h];
          }
        }
        h_tilde[static_cast<size_t>(h)] = std::tanh(acc_x + acc_h);
      }

      // H_t = (1 - z) (.) h_tilde + z (.) H_{t-1}.
      for (int64_t h = 0; h < hidden_size; ++h) {
        const float z = z_gate[static_cast<size_t>(h)];
        out_row[h] = (1.0f - z) * h_tilde[static_cast<size_t>(h)] + z * h_row[h];
      }
    }

    // Copy h_curr into Y at time-step t (num_directions=1) and swap into
    // h_prev for the next iteration.
    float *y_t = py + t * batch_size * hidden_size;
    for (int64_t i = 0; i < batch_size * hidden_size; ++i) {
      y_t[i] = h_curr[static_cast<size_t>(i)];
    }
    h_prev.swap(h_curr);
  }

  // Y_h is the last time step of Y (which is currently in h_prev after the
  // final swap).
  for (int64_t i = 0; i < batch_size * hidden_size; ++i) {
    py_h[i] = h_prev[static_cast<size_t>(i)];
  }

  return std::pair<Tensor, Tensor>(std::move(y), std::move(y_h));
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
