// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/temporary_buffer.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Checks ``t`` is a FLOAT tensor and returns its data pointer (or nullptr
// when ``t`` is the sentinel default-constructed Tensor, used to indicate
// that the corresponding optional input is missing).
const float *AsFloatOrNull(const Tensor &t, const char *role) {
  if (t.shape.empty() && t.size_bytes() == 0) {
    return nullptr;
  }
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT), "kernel::GRU: ", role,
                      " must be FLOAT.");
  return t.AsFloat();
}

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

} // namespace

std::pair<Tensor, Tensor> GRU::operator()(const Tensor &x_in, const Tensor &w, const Tensor &r,
                                          const Tensor &b, const Tensor &initial_h_in,
                                          int64_t linear_before_reset, int64_t layout,
                                          RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(layout == 0 || layout == 1, "kernel::GRU: layout must be 0 or 1, got ",
                      std::to_string(layout), ".");
  EXT_ENFORCE_INVALID(x_in.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GRU: X must be FLOAT.");
  EXT_ENFORCE_INVALID(w.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GRU: W must be FLOAT.");
  EXT_ENFORCE_INVALID(r.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GRU: R must be FLOAT.");
  EXT_ENFORCE_INVALID(x_in.shape.size() == 3u, "kernel::GRU: X must have rank 3.");
  EXT_ENFORCE_INVALID(w.shape.size() == 3u && w.shape[0] == 1,
                      "kernel::GRU: W must have shape [1, 3 * hidden_size, input_size] "
                      "(single forward direction only).");
  EXT_ENFORCE_INVALID(r.shape.size() == 3u && r.shape[0] == 1,
                      "kernel::GRU: R must have shape [1, 3 * hidden_size, hidden_size] "
                      "(single forward direction only).");

  // ``layout == 1`` permutes batch and time/direction axes on a subset
  // of inputs and outputs; the time-major kernel body below stays as
  // is. ``num_directions`` is always 1 (only ``forward`` is implemented)
  // which makes the state-tensor permutation a pure reshape on
  // contiguous storage.
  Tensor x_storage;
  Tensor initial_h_storage;
  const Tensor *x_p = &x_in;
  const Tensor *initial_h_p = &initial_h_in;
  RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;
  if (layout == 1) {
    const int64_t batch_size = x_in.shape[0];
    const int64_t seq_length = x_in.shape[1];
    const int64_t input_size = x_in.shape[2];
    const size_t x_n_bytes =
        static_cast<size_t>(batch_size * seq_length * input_size) * sizeof(float);
    x_storage = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT),
                                 onnx_kernels::Shape{seq_length, batch_size, input_size}, x_n_bytes,
                                 allocator);
    float *x_data = x_storage.AsFloat();
    const float *src = x_in.AsFloat();
    for (int64_t n = 0; n < batch_size; ++n) {
      for (int64_t s = 0; s < seq_length; ++s) {
        for (int64_t k = 0; k < input_size; ++k) {
          x_data[static_cast<size_t>((s * batch_size + n) * input_size + k)] =
              src[static_cast<size_t>((n * seq_length + s) * input_size + k)];
        }
      }
    }
    x_p = &x_storage;

    if (!(initial_h_in.shape.empty() && initial_h_in.size_bytes() == 0)) {
      EXT_ENFORCE_INVALID(initial_h_in.shape.size() == 3u && initial_h_in.shape[1] == 1,
                          "kernel::GRU: initial_h must have shape [batch_size, num_directions=1, "
                          "hidden_size] for layout=1.");
      // The reshape from [batch_size, 1, hidden_size] to
      // [1, batch_size, hidden_size] is a pure reshape on contiguous
      // storage (num_directions == 1), so borrow the input buffer
      // instead of copying it.
      initial_h_storage = Tensor::Borrow("", static_cast<int32_t>(DataType::FLOAT),
                                         {1, initial_h_in.shape[0], initial_h_in.shape[2]},
                                         initial_h_in.bytes(), initial_h_in.size_bytes());
      initial_h_p = &initial_h_storage;
    }
  }

  const Tensor &x = *x_p;
  const Tensor &initial_h = *initial_h_p;

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
  const onnx_kernels::Shape y_shape{seq_length, 1, batch_size, hidden_size};
  const onnx_kernels::Shape y_h_shape{1, batch_size, hidden_size};
  const size_t y_n_bytes =
      static_cast<size_t>(seq_length * batch_size * hidden_size) * sizeof(float);
  Tensor y = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), y_shape, y_n_bytes, allocator);
  const size_t y_h_n_bytes = static_cast<size_t>(batch_size * hidden_size) * sizeof(float);
  Tensor y_h =
      MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), y_h_shape, y_h_n_bytes, allocator);
  float *py = y.AsFloat();
  float *py_h = y_h.AsFloat();

  // Working buffers for H_{t-1} and H_t. Drawn from the runtime allocator when
  // available (falling back to inline storage otherwise). Allocator-backed
  // buffers are not guaranteed zeroed, so ``h_prev`` is explicitly initialised.
  const size_t state_count = static_cast<size_t>(batch_size * hidden_size);
  detail::TemporaryTypedBuffer<float> h_prev_buf(state_count, allocator, "kernel::GRU h_prev");
  detail::TemporaryTypedBuffer<float> h_curr_buf(state_count, allocator, "kernel::GRU h_curr");
  float *h_prev = h_prev_buf.data();
  float *h_curr = h_curr_buf.data();
  if (p_initial_h != nullptr) {
    for (int64_t i = 0; i < batch_size * hidden_size; ++i) {
      h_prev[static_cast<size_t>(i)] = p_initial_h[i];
    }
  } else {
    std::fill(h_prev, h_prev + state_count, 0.0f);
  }

  // Per-time-step gate buffers (sized for one batch row at a time). Every
  // element is written before being read on each row, so no zero-fill needed.
  detail::TemporaryTypedBuffer<float> z_gate_buf(static_cast<size_t>(hidden_size), allocator,
                                                 "kernel::GRU z_gate");
  detail::TemporaryTypedBuffer<float> r_gate_buf(static_cast<size_t>(hidden_size), allocator,
                                                 "kernel::GRU r_gate");
  detail::TemporaryTypedBuffer<float> h_tilde_buf(static_cast<size_t>(hidden_size), allocator,
                                                  "kernel::GRU h_tilde");
  float *z_gate = z_gate_buf.data();
  float *r_gate = r_gate_buf.data();
  float *h_tilde = h_tilde_buf.data();

  for (int64_t t = 0; t < seq_length; ++t) {
    const float *x_t = px + t * batch_size * input_size;
    for (int64_t n = 0; n < batch_size; ++n) {
      const float *x_row = x_t + n * input_size;
      const float *h_row = h_prev + n * hidden_size;
      float *out_row = h_curr + n * hidden_size;

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
    // h_prev for the next iteration. The swap only exchanges the two raw
    // pointers; both TemporaryTypedBuffer objects stay alive (and their
    // storage valid) until the end of the function.
    float *y_t = py + t * batch_size * hidden_size;
    for (int64_t i = 0; i < batch_size * hidden_size; ++i) {
      y_t[i] = h_curr[static_cast<size_t>(i)];
    }
    std::swap(h_prev, h_curr);
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
    Tensor y_perm_t = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT),
                                       onnx_kernels::Shape{batch_size, seq_length, 1, hidden_size},
                                       y_n_bytes, allocator);
    float *y_perm = y_perm_t.AsFloat();
    const float *y_src = y.AsFloat();
    for (int64_t s = 0; s < seq_length; ++s) {
      for (int64_t n = 0; n < batch_size; ++n) {
        for (int64_t h = 0; h < hidden_size; ++h) {
          y_perm[static_cast<size_t>((n * seq_length + s) * hidden_size + h)] =
              y_src[static_cast<size_t>((s * batch_size + n) * hidden_size + h)];
        }
      }
    }
    y = std::move(y_perm_t);
    const size_t y_h_reshape_bytes = static_cast<size_t>(batch_size * hidden_size) * sizeof(float);
    Tensor y_h_reshape = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT),
                                          onnx_kernels::Shape{batch_size, 1, hidden_size},
                                          y_h_reshape_bytes, allocator);
    std::memcpy(y_h_reshape.mutable_bytes(), y_h.AsFloat(), y_h_reshape_bytes);
    y_h = std::move(y_h_reshape);
  }

  return std::pair<Tensor, Tensor>(std::move(y), std::move(y_h));
}

void GRU::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() < 3 || node.input_size() > 6), "RunNode: op '",
                      node.op_type(), "' expects between 3 and 6 input(s), got ", node.input_size(),
                      ".");
  EXT_ENFORCE_INVALID(!(node.output_size() < 1 || node.output_size() > 2), "RunNode: op '",
                      node.op_type(), "' expects 1 or 2 output(s), got ", node.output_size(), ".");

  // Unsupported attributes: only the default ``forward`` direction
  // with the default ``Sigmoid``/``Tanh`` activations and no
  // ``clip`` are implemented; ``layout=0`` and ``layout=1`` are
  // both supported.
  const std::string direction = GetAttributeStringOrDefault(node, "direction", "forward");
  EXT_ENFORCE_INVALID(direction == "forward",
                      "RunNode: op 'GRU' only supports direction='forward', got '", direction,
                      "'.");
  EXT_ENFORCE_INVALID(FindAttribute(node, "activations") == nullptr,
                      "RunNode: op 'GRU' does not support the 'activations' attribute.");
  EXT_ENFORCE_INVALID(!(FindAttribute(node, "activation_alpha") != nullptr ||
                        FindAttribute(node, "activation_beta") != nullptr),
                      "RunNode: op 'GRU' does not support 'activation_alpha'/'activation_beta'.");
  EXT_ENFORCE_INVALID(FindAttribute(node, "clip") == nullptr,
                      "RunNode: op 'GRU' does not support the 'clip' attribute.");
  const int64_t layout = GetAttributeIntOrDefault(node, "layout", 0);

  // ``sequence_lens`` (input #4) is not supported: it requires
  // per-batch sequence handling that the FLOAT kernel does not
  // implement.
  const Tensor *sequence_lens = GetOptionalInput(node, 4, rt.tensors());
  EXT_ENFORCE_INVALID(sequence_lens == nullptr,
                      "RunNode: op 'GRU' does not support the optional 'sequence_lens' input.");

  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &w = GetInput(node, 1, rt.tensors());
  const Tensor &r = GetInput(node, 2, rt.tensors());
  const Tensor *b = GetOptionalInput(node, 3, rt.tensors());
  const Tensor *initial_h = GetOptionalInput(node, 5, rt.tensors());

  const int64_t linear_before_reset = GetAttributeIntOrDefault(node, "linear_before_reset", 0);

  onnx_kernels::kernel::GRU kernel(rt.kernel_ctx());
  auto [y, y_h] = kernel(x, w, r, b != nullptr ? *b : Tensor{},
                         initial_h != nullptr ? *initial_h : Tensor{}, linear_before_reset, layout);

  auto set_optional_output = [&node, &rt](int index, Tensor output) {
    if (index >= node.output_size()) {
      return;
    }
    const std::string &name = node.output(index);
    if (name.empty()) {
      return;
    }
    output.name = name;
    rt.Put(name, std::move(output), RuntimeEventKind::kIntermediate);
  };
  set_optional_output(0, std::move(y));
  set_optional_output(1, std::move(y_h));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
