// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_kernels/runtime_context.h"
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
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT), "kernel::RNN: ", role,
                      " must be FLOAT.");
  return t.AsFloat();
}

} // namespace

std::pair<Tensor, Tensor> RNN::operator()(const Tensor &x_in, const Tensor &w, const Tensor &r,
                                          const Tensor &b, const Tensor &initial_h_in,
                                          int64_t layout, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(layout == 0 || layout == 1, "kernel::RNN: layout must be 0 or 1, got ",
                      std::to_string(layout), ".");
  EXT_ENFORCE_INVALID(x_in.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RNN: X must be FLOAT.");
  EXT_ENFORCE_INVALID(w.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RNN: W must be FLOAT.");
  EXT_ENFORCE_INVALID(r.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RNN: R must be FLOAT.");
  EXT_ENFORCE_INVALID(x_in.shape.size() == 3u, "kernel::RNN: X must have rank 3.");
  EXT_ENFORCE_INVALID(w.shape.size() == 3u && w.shape[0] == 1,
                      "kernel::RNN: W must have shape [1, hidden_size, input_size] (single "
                      "forward direction only).");
  EXT_ENFORCE_INVALID(r.shape.size() == 3u && r.shape[0] == 1,
                      "kernel::RNN: R must have shape [1, hidden_size, hidden_size] (single "
                      "forward direction only).");

  // ``layout == 1`` permutes batch and time/direction axes on a subset
  // of inputs and outputs; the time-major kernel body below stays as
  // is. ``num_directions`` is always 1 (only ``forward`` is implemented)
  // which makes the state-tensor permutation a pure reshape on
  // contiguous storage.
  Tensor x_storage;
  Tensor initial_h_storage;
  const Tensor *x_p = &x_in;
  const Tensor *initial_h_p = &initial_h_in;
  if (layout == 1) {
    const int64_t batch_size = x_in.shape[0];
    const int64_t seq_length = x_in.shape[1];
    const int64_t input_size = x_in.shape[2];
    std::vector<float> x_data(static_cast<size_t>(batch_size * seq_length * input_size));
    const float *src = x_in.AsFloat();
    for (int64_t n = 0; n < batch_size; ++n) {
      for (int64_t s = 0; s < seq_length; ++s) {
        for (int64_t k = 0; k < input_size; ++k) {
          x_data[static_cast<size_t>((s * batch_size + n) * input_size + k)] =
              src[static_cast<size_t>((n * seq_length + s) * input_size + k)];
        }
      }
    }
    x_storage = Tensor::FromFloat("", {seq_length, batch_size, input_size}, std::move(x_data));
    x_p = &x_storage;

    if (!(initial_h_in.shape.empty() && initial_h_in.size_bytes() == 0)) {
      EXT_ENFORCE_INVALID(initial_h_in.shape.size() == 3u && initial_h_in.shape[1] == 1,
                          "kernel::RNN: initial_h must have shape [batch_size, num_directions=1, "
                          "hidden_size] for layout=1.");
      initial_h_storage = Tensor::FromFloat(
          "", {1, initial_h_in.shape[0], initial_h_in.shape[2]},
          std::vector<float>(initial_h_in.AsFloat(),
                             initial_h_in.AsFloat() + (initial_h_in.size_bytes() / sizeof(float))));
      initial_h_p = &initial_h_storage;
    }
  }

  const Tensor &x = *x_p;
  const Tensor &initial_h = *initial_h_p;

  const int64_t seq_length = x.shape[0];
  const int64_t batch_size = x.shape[1];
  const int64_t input_size = x.shape[2];
  const int64_t hidden_size = w.shape[1];

  EXT_ENFORCE_INVALID(w.shape[2] == input_size,
                      "kernel::RNN: W.shape[2] must equal X.shape[2] (input_size).");
  EXT_ENFORCE_INVALID(r.shape[1] == hidden_size && r.shape[2] == hidden_size,
                      "kernel::RNN: R must have shape [1, hidden_size, hidden_size].");

  const float *p_b = AsFloatOrNull(b, "B");
  if (p_b != nullptr) {
    EXT_ENFORCE_INVALID(b.shape.size() == 2u && b.shape[0] == 1 && b.shape[1] == 2 * hidden_size,
                        "kernel::RNN: B must have shape [1, 2 * hidden_size].");
  }
  const float *p_initial_h = AsFloatOrNull(initial_h, "initial_h");
  if (p_initial_h != nullptr) {
    EXT_ENFORCE_INVALID(initial_h.shape.size() == 3u && initial_h.shape[0] == 1 &&
                            initial_h.shape[1] == batch_size && initial_h.shape[2] == hidden_size,
                        "kernel::RNN: initial_h must have shape [1, batch_size, hidden_size].");
  }

  const float *px = x.AsFloat();
  const float *pw = w.AsFloat();
  const float *pr = r.AsFloat();

  // Per-hidden bias = Wb + Rb so we add it once per time step.
  std::vector<float> bias(static_cast<size_t>(hidden_size), 0.0f);
  if (p_b != nullptr) {
    for (int64_t h = 0; h < hidden_size; ++h) {
      bias[static_cast<size_t>(h)] = p_b[h] + p_b[hidden_size + h];
    }
  }

  // Output allocations.
  const onnx_kernels::Shape y_shape{seq_length, 1, batch_size, hidden_size};
  const onnx_kernels::Shape y_h_shape{1, batch_size, hidden_size};
  const size_t y_n_bytes =
      static_cast<size_t>(seq_length * batch_size * hidden_size) * sizeof(float);
  RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;
  Tensor y = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), y_shape, y_n_bytes, allocator);
  const size_t y_h_n_bytes = static_cast<size_t>(batch_size * hidden_size) * sizeof(float);
  Tensor y_h =
      MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), y_h_shape, y_h_n_bytes, allocator);
  float *py = y.AsFloat();
  float *py_h = y_h.AsFloat();

  // Working buffer for H_{t-1} and H_t.
  std::vector<float> h_prev(static_cast<size_t>(batch_size * hidden_size), 0.0f);
  if (p_initial_h != nullptr) {
    for (int64_t i = 0; i < batch_size * hidden_size; ++i) {
      h_prev[static_cast<size_t>(i)] = p_initial_h[i];
    }
  }
  std::vector<float> h_curr(static_cast<size_t>(batch_size * hidden_size), 0.0f);

  for (int64_t t = 0; t < seq_length; ++t) {
    const float *x_t = px + t * batch_size * input_size;
    // For each batch row and each output unit, accumulate X_t @ W^T plus
    // H_{t-1} @ R^T plus the per-unit bias, then apply tanh.
    for (int64_t n = 0; n < batch_size; ++n) {
      const float *x_row = x_t + n * input_size;
      const float *h_row = h_prev.data() + n * hidden_size;
      float *out_row = h_curr.data() + n * hidden_size;
      for (int64_t h = 0; h < hidden_size; ++h) {
        const float *w_row = pw + h * input_size;
        const float *r_row = pr + h * hidden_size;
        float acc = bias[static_cast<size_t>(h)];
        for (int64_t k = 0; k < input_size; ++k) {
          acc += x_row[k] * w_row[k];
        }
        for (int64_t k = 0; k < hidden_size; ++k) {
          acc += h_row[k] * r_row[k];
        }
        out_row[h] = std::tanh(acc);
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

  if (layout == 1) {
    // Permute Y from [seq, 1, batch, hidden] to [batch, seq, 1, hidden]
    // and reshape Y_h from [1, batch, hidden] to [batch, 1, hidden]
    // (num_directions == 1 makes the Y_h transform a pure reshape).
    std::vector<float> y_perm(static_cast<size_t>(seq_length * batch_size * hidden_size));
    const float *y_src = y.AsFloat();
    for (int64_t s = 0; s < seq_length; ++s) {
      for (int64_t n = 0; n < batch_size; ++n) {
        for (int64_t h = 0; h < hidden_size; ++h) {
          y_perm[static_cast<size_t>((n * seq_length + s) * hidden_size + h)] =
              y_src[static_cast<size_t>((s * batch_size + n) * hidden_size + h)];
        }
      }
    }
    y = Tensor::FromFloat("", {batch_size, seq_length, 1, hidden_size}, std::move(y_perm));
    y_h = Tensor::FromFloat(
        "", {batch_size, 1, hidden_size},
        std::vector<float>(y_h.AsFloat(), y_h.AsFloat() + (y_h.size_bytes() / sizeof(float))));
  }

  return std::pair<Tensor, Tensor>(std::move(y), std::move(y_h));
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
