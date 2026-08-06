// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/temporary_buffer.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
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
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT), "kernel::LSTM: ", role,
                      " must be FLOAT.");
  return t.AsFloat();
}

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

} // namespace

std::pair<Tensor, Tensor> LSTM::operator()(const Tensor &x_in, const Tensor &w, const Tensor &r,
                                           const Tensor &b, const Tensor &initial_h_in,
                                           const Tensor &initial_c_in, const Tensor &p,
                                           int64_t layout, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(layout == 0 || layout == 1, "kernel::LSTM: layout must be 0 or 1, got ",
                      std::to_string(layout), ".");
  EXT_ENFORCE_INVALID(x_in.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LSTM: X must be FLOAT.");
  EXT_ENFORCE_INVALID(w.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LSTM: W must be FLOAT.");
  EXT_ENFORCE_INVALID(r.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LSTM: R must be FLOAT.");
  EXT_ENFORCE_INVALID(x_in.shape.size() == 3u, "kernel::LSTM: X must have rank 3.");
  EXT_ENFORCE_INVALID(w.shape.size() == 3u && w.shape[0] == 1,
                      "kernel::LSTM: W must have shape [1, 4 * hidden_size, input_size] "
                      "(single forward direction only).");
  EXT_ENFORCE_INVALID(r.shape.size() == 3u && r.shape[0] == 1,
                      "kernel::LSTM: R must have shape [1, 4 * hidden_size, hidden_size] "
                      "(single forward direction only).");

  // Scratch and result buffers are drawn from the runtime allocator (when one
  // is attached) so no storage is acquired outside the runtime context; they
  // fall back to inline ``std::vector`` storage otherwise.
  RawBufferAllocator *allocator = rt != nullptr ? rt->allocator() : nullptr;

  // ``layout == 1`` permutes batch and time/direction axes on a subset
  // of inputs and outputs; the time-major kernel body below stays as
  // is. ``num_directions`` is always 1 (only ``forward`` is implemented)
  // which makes the state-tensor permutations a pure reshape on
  // contiguous storage.
  Tensor x_storage;
  Tensor initial_h_storage;
  Tensor initial_c_storage;
  const Tensor *x_p = &x_in;
  const Tensor *initial_h_p = &initial_h_in;
  const Tensor *initial_c_p = &initial_c_in;
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

    auto reshape_initial_state = [allocator](const Tensor &t, const char *role) {
      EXT_ENFORCE_INVALID(
          t.shape.size() == 3u && t.shape[1] == 1, "kernel::LSTM: ", role,
          " must have shape [batch_size, num_directions=1, hidden_size] for layout=1.");
      const size_t n_bytes = t.size_bytes();
      Tensor out =
          MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT),
                           onnx_kernels::Shape{1, t.shape[0], t.shape[2]}, n_bytes, allocator);
      std::memcpy(out.mutable_bytes(), t.AsFloat(), n_bytes);
      return out;
    };
    if (!(initial_h_in.shape.empty() && initial_h_in.size_bytes() == 0)) {
      initial_h_storage = reshape_initial_state(initial_h_in, "initial_h");
      initial_h_p = &initial_h_storage;
    }
    if (!(initial_c_in.shape.empty() && initial_c_in.size_bytes() == 0)) {
      initial_c_storage = reshape_initial_state(initial_c_in, "initial_c");
      initial_c_p = &initial_c_storage;
    }
  }

  const Tensor &x = *x_p;
  const Tensor &initial_h = *initial_h_p;
  const Tensor &initial_c = *initial_c_p;

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
  const std::size_t hidden_count = static_cast<std::size_t>(hidden_size);
  detail::TemporaryTypedBuffer<float> bias_i_buf(hidden_count, allocator, "kernel::LSTM bias_i");
  detail::TemporaryTypedBuffer<float> bias_o_buf(hidden_count, allocator, "kernel::LSTM bias_o");
  detail::TemporaryTypedBuffer<float> bias_f_buf(hidden_count, allocator, "kernel::LSTM bias_f");
  detail::TemporaryTypedBuffer<float> bias_c_buf(hidden_count, allocator, "kernel::LSTM bias_c");
  float *bias_i = bias_i_buf.data();
  float *bias_o = bias_o_buf.data();
  float *bias_f = bias_f_buf.data();
  float *bias_c = bias_c_buf.data();
  // Allocator-backed buffers are not guaranteed zeroed; the gate biases are
  // read for every step even when ``B`` is absent, so zero-fill explicitly.
  std::fill(bias_i, bias_i + hidden_count, 0.0f);
  std::fill(bias_o, bias_o + hidden_count, 0.0f);
  std::fill(bias_f, bias_f + hidden_count, 0.0f);
  std::fill(bias_c, bias_c + hidden_count, 0.0f);
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
  detail::TemporaryTypedBuffer<float> peep_i_buf(hidden_count, allocator, "kernel::LSTM peep_i");
  detail::TemporaryTypedBuffer<float> peep_o_buf(hidden_count, allocator, "kernel::LSTM peep_o");
  detail::TemporaryTypedBuffer<float> peep_f_buf(hidden_count, allocator, "kernel::LSTM peep_f");
  float *peep_i = peep_i_buf.data();
  float *peep_o = peep_o_buf.data();
  float *peep_f = peep_f_buf.data();
  // Peepholes are read for every step even when ``P`` is absent; zero-fill the
  // (potentially uninitialized) allocator-backed storage explicitly.
  std::fill(peep_i, peep_i + hidden_count, 0.0f);
  std::fill(peep_o, peep_o + hidden_count, 0.0f);
  std::fill(peep_f, peep_f + hidden_count, 0.0f);
  if (p_p != nullptr) {
    for (int64_t h = 0; h < hidden_size; ++h) {
      peep_i[static_cast<size_t>(h)] = p_p[0 * hidden_size + h];
      peep_o[static_cast<size_t>(h)] = p_p[1 * hidden_size + h];
      peep_f[static_cast<size_t>(h)] = p_p[2 * hidden_size + h];
    }
  }

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

  // Working buffers for H_{t-1}, C_{t-1}, H_t, C_t.
  const std::size_t state_count = static_cast<std::size_t>(batch_size * hidden_size);
  detail::TemporaryTypedBuffer<float> h_prev_buf(state_count, allocator, "kernel::LSTM h_prev");
  detail::TemporaryTypedBuffer<float> c_prev_buf(state_count, allocator, "kernel::LSTM c_prev");
  detail::TemporaryTypedBuffer<float> h_curr_buf(state_count, allocator, "kernel::LSTM h_curr");
  detail::TemporaryTypedBuffer<float> c_curr_buf(state_count, allocator, "kernel::LSTM c_curr");
  float *h_prev = h_prev_buf.data();
  float *c_prev = c_prev_buf.data();
  float *h_curr = h_curr_buf.data();
  float *c_curr = c_curr_buf.data();
  // H_0 / C_0 default to zero; allocator-backed storage is not guaranteed
  // zeroed, so fill before optionally copying in the provided initial states.
  std::fill(h_prev, h_prev + state_count, 0.0f);
  std::fill(c_prev, c_prev + state_count, 0.0f);
  if (p_initial_h != nullptr) {
    for (int64_t i = 0; i < batch_size * hidden_size; ++i) {
      h_prev[i] = p_initial_h[i];
    }
  }
  if (p_initial_c != nullptr) {
    for (int64_t i = 0; i < batch_size * hidden_size; ++i) {
      c_prev[i] = p_initial_c[i];
    }
  }

  // Reusable per-step accumulator: one row per gate (i, o, f, c). Each entry is
  // fully assigned before it is read within a step, so no zero-fill is needed.
  detail::TemporaryTypedBuffer<float> acc_i_buf(hidden_count, allocator, "kernel::LSTM acc_i");
  detail::TemporaryTypedBuffer<float> acc_o_buf(hidden_count, allocator, "kernel::LSTM acc_o");
  detail::TemporaryTypedBuffer<float> acc_f_buf(hidden_count, allocator, "kernel::LSTM acc_f");
  detail::TemporaryTypedBuffer<float> acc_c_buf(hidden_count, allocator, "kernel::LSTM acc_c");
  float *acc_i = acc_i_buf.data();
  float *acc_o = acc_o_buf.data();
  float *acc_f = acc_f_buf.data();
  float *acc_c = acc_c_buf.data();

  for (int64_t t = 0; t < seq_length; ++t) {
    const float *x_t = px + t * batch_size * input_size;
    for (int64_t n = 0; n < batch_size; ++n) {
      const float *x_row = x_t + n * input_size;
      const float *h_row = h_prev + n * hidden_size;
      const float *c_row = c_prev + n * hidden_size;
      float *h_out = h_curr + n * hidden_size;
      float *c_out = c_curr + n * hidden_size;

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
      y_t[i] = h_curr[i];
    }
    std::swap(h_prev, h_curr);
    std::swap(c_prev, c_curr);
  }

  // Y_h is the last time step of Y (currently in h_prev after the final swap).
  for (int64_t i = 0; i < batch_size * hidden_size; ++i) {
    py_h[i] = h_prev[i];
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

void LSTM::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() < 3 || node.input_size() > 8), "RunNode: op '",
                      node.op_type(), "' expects between 3 and 8 input(s), got ", node.input_size(),
                      ".");
  EXT_ENFORCE_INVALID(!(node.output_size() < 1 || node.output_size() > 3), "RunNode: op '",
                      node.op_type(), "' expects between 1 and 3 output(s), got ",
                      node.output_size(), ".");

  // Unsupported attributes: only the default ``forward`` direction
  // with the default ``Sigmoid``/``Tanh``/``Tanh`` activations, no
  // ``clip``, ``input_forget == 0``, and ``layout == 0`` are
  // implemented.
  const std::string direction = GetAttributeStringOrDefault(node, "direction", "forward");
  EXT_ENFORCE_INVALID(direction == "forward",
                      "RunNode: op 'LSTM' only supports direction='forward', got '", direction,
                      "'.");
  EXT_ENFORCE_INVALID(FindAttribute(node, "activations") == nullptr,
                      "RunNode: op 'LSTM' does not support the 'activations' attribute.");
  EXT_ENFORCE_INVALID(!(FindAttribute(node, "activation_alpha") != nullptr ||
                        FindAttribute(node, "activation_beta") != nullptr),
                      "RunNode: op 'LSTM' does not support 'activation_alpha'/'activation_beta'.");
  EXT_ENFORCE_INVALID(FindAttribute(node, "clip") == nullptr,
                      "RunNode: op 'LSTM' does not support the 'clip' attribute.");
  EXT_ENFORCE_INVALID(GetAttributeIntOrDefault(node, "input_forget", 0) == 0,
                      "RunNode: op 'LSTM' only supports input_forget=0.");
  const int64_t layout = GetAttributeIntOrDefault(node, "layout", 0);

  // The current kernel only produces (Y, Y_h); the optional third
  // output ``Y_c`` (final cell state) is not implemented.
  EXT_ENFORCE_INVALID(!(node.output_size() >= 3 && !node.output(2).empty()),
                      "RunNode: op 'LSTM' does not support the optional third output 'Y_c'.");

  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &w = GetInput(node, 1, rt.tensors());
  const Tensor &r = GetInput(node, 2, rt.tensors());
  const Tensor *b = GetOptionalInput(node, 3, rt.tensors());
  const Tensor *initial_h = GetOptionalInput(node, 5, rt.tensors());
  const Tensor *initial_c = GetOptionalInput(node, 6, rt.tensors());
  const Tensor *p = GetOptionalInput(node, 7, rt.tensors());

  // ``sequence_lens`` (input #4) requires per-batch sequence
  // handling that the FLOAT kernel does not implement; accept it
  // only when it degenerates to a no-op (every batch row uses the
  // full ``seq_length`` so masking would not change the output).
  // ``seq_length`` is read from ``X`` at axis 0 for ``layout=0``
  // and axis 1 for ``layout=1``.
  const Tensor *sequence_lens = GetOptionalInput(node, 4, rt.tensors());
  if (sequence_lens != nullptr) {
    EXT_ENFORCE_INVALID(!(sequence_lens->data_type != static_cast<int32_t>(DataType::INT32)),
                        "RunNode: op 'LSTM' expects 'sequence_lens' to be INT32.");
    const size_t seq_axis = layout == 1 ? 1u : 0u;
    const int64_t seq_length = x.shape.size() > seq_axis ? x.shape[seq_axis] : 0;
    const int64_t n = sequence_lens->element_count();
    const int32_t *seq_data = sequence_lens->AsInt32();
    for (int64_t i = 0; i < n; ++i) {
      EXT_ENFORCE_INVALID(!(static_cast<int64_t>(seq_data[i]) != seq_length),
                          "RunNode: op 'LSTM' does not support the optional 'sequence_lens' "
                          "input unless every entry equals the full seq_length.");
    }
  }

  onnx_kernels::kernel::LSTM kernel(rt.kernel_ctx());
  auto [y, y_h] =
      kernel(x, w, r, b != nullptr ? *b : Tensor{}, initial_h != nullptr ? *initial_h : Tensor{},
             initial_c != nullptr ? *initial_c : Tensor{}, p != nullptr ? *p : Tensor{}, layout);

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
