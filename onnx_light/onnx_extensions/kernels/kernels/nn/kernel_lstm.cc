// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernels/nn/recurrent_common.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
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

std::tuple<Tensor, Tensor, Tensor>
LSTM::operator()(const Tensor &x_in, const Tensor &w, const Tensor &r, const Tensor &b,
                 const Tensor &initial_h_in, const Tensor &initial_c_in, const Tensor &p,
                 int64_t layout, const std::string &direction, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(layout == 0 || layout == 1, "kernel::LSTM: layout must be 0 or 1, got ",
                      std::to_string(layout), ".");
  // ``direction`` selects the recurrence order: ``forward`` and ``reverse``
  // use a single direction (``num_directions == 1``); ``bidirectional``
  // concatenates a forward and a reverse pass (``num_directions == 2``).
  const int64_t num_directions = recurrent::RecurrentNumDirections("LSTM", direction);
  EXT_ENFORCE_INVALID(x_in.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LSTM: X must be FLOAT.");
  EXT_ENFORCE_INVALID(w.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LSTM: W must be FLOAT.");
  EXT_ENFORCE_INVALID(r.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LSTM: R must be FLOAT.");
  EXT_ENFORCE_INVALID(x_in.shape.size() == 3u, "kernel::LSTM: X must have rank 3.");
  EXT_ENFORCE_INVALID(w.shape.size() == 3u && w.shape[0] == num_directions,
                      "kernel::LSTM: W must have shape [num_directions, 4 * hidden_size, "
                      "input_size].");
  EXT_ENFORCE_INVALID(r.shape.size() == 3u && r.shape[0] == num_directions,
                      "kernel::LSTM: R must have shape [num_directions, 4 * hidden_size, "
                      "hidden_size].");

  // Scratch and result buffers are drawn from the runtime allocator (when one
  // is attached) so no storage is acquired outside the runtime context; they
  // fall back to inline ``std::vector`` storage otherwise.
  RawBufferAllocator *allocator = rt != nullptr ? rt->allocator() : nullptr;

  // ``layout == 1`` permutes batch and time/direction axes on a subset
  // of inputs and outputs; the time-major kernel body below stays as is.
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

    if (!(initial_h_in.shape.empty() && initial_h_in.size_bytes() == 0)) {
      initial_h_storage =
          recurrent::RecurrentTransposeInitialState(initial_h_in, num_directions, allocator);
      initial_h_p = &initial_h_storage;
    }
    if (!(initial_c_in.shape.empty() && initial_c_in.size_bytes() == 0)) {
      initial_c_storage =
          recurrent::RecurrentTransposeInitialState(initial_c_in, num_directions, allocator);
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
                      "kernel::LSTM: W must have shape [num_directions, 4 * hidden_size, "
                      "input_size].");
  EXT_ENFORCE_INVALID(r.shape[1] == 4 * hidden_size,
                      "kernel::LSTM: R must have shape [num_directions, 4 * hidden_size, "
                      "hidden_size].");

  const float *p_b = AsFloatOrNull(b, "B");
  if (p_b != nullptr) {
    EXT_ENFORCE_INVALID(b.shape.size() == 2u && b.shape[0] == num_directions &&
                            b.shape[1] == 8 * hidden_size,
                        "kernel::LSTM: B must have shape [num_directions, 8 * hidden_size].");
  }
  const float *p_initial_h = AsFloatOrNull(initial_h, "initial_h");
  if (p_initial_h != nullptr) {
    EXT_ENFORCE_INVALID(initial_h.shape.size() == 3u && initial_h.shape[0] == num_directions &&
                            initial_h.shape[1] == batch_size && initial_h.shape[2] == hidden_size,
                        "kernel::LSTM: initial_h must have shape [num_directions, batch_size, "
                        "hidden_size].");
  }
  const float *p_initial_c = AsFloatOrNull(initial_c, "initial_c");
  if (p_initial_c != nullptr) {
    EXT_ENFORCE_INVALID(initial_c.shape.size() == 3u && initial_c.shape[0] == num_directions &&
                            initial_c.shape[1] == batch_size && initial_c.shape[2] == hidden_size,
                        "kernel::LSTM: initial_c must have shape [num_directions, batch_size, "
                        "hidden_size].");
  }
  const float *p_p = AsFloatOrNull(p, "P");
  if (p_p != nullptr) {
    EXT_ENFORCE_INVALID(p.shape.size() == 2u && p.shape[0] == num_directions &&
                            p.shape[1] == 3 * hidden_size,
                        "kernel::LSTM: P must have shape [num_directions, 3 * hidden_size].");
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

  // Per-gate bias, peephole and state buffers are reused across directions.
  const std::size_t hidden_count = static_cast<std::size_t>(hidden_size);
  detail::TemporaryTypedBuffer<float> bias_i_buf(hidden_count, allocator, "kernel::LSTM bias_i");
  detail::TemporaryTypedBuffer<float> bias_o_buf(hidden_count, allocator, "kernel::LSTM bias_o");
  detail::TemporaryTypedBuffer<float> bias_f_buf(hidden_count, allocator, "kernel::LSTM bias_f");
  detail::TemporaryTypedBuffer<float> bias_c_buf(hidden_count, allocator, "kernel::LSTM bias_c");
  float *bias_i = bias_i_buf.data();
  float *bias_o = bias_o_buf.data();
  float *bias_f = bias_f_buf.data();
  float *bias_c = bias_c_buf.data();

  // Peephole weights (length hidden_size each) in gate order i, o, f.
  detail::TemporaryTypedBuffer<float> peep_i_buf(hidden_count, allocator, "kernel::LSTM peep_i");
  detail::TemporaryTypedBuffer<float> peep_o_buf(hidden_count, allocator, "kernel::LSTM peep_o");
  detail::TemporaryTypedBuffer<float> peep_f_buf(hidden_count, allocator, "kernel::LSTM peep_f");
  float *peep_i = peep_i_buf.data();
  float *peep_o = peep_o_buf.data();
  float *peep_f = peep_f_buf.data();

  // Output allocations. ``Y`` is [seq, num_directions, batch, hidden];
  // ``Y_h`` and ``Y_c`` are [num_directions, batch, hidden].
  const onnx_kernels::Shape y_shape{seq_length, num_directions, batch_size, hidden_size};
  const onnx_kernels::Shape state_shape{num_directions, batch_size, hidden_size};
  const size_t y_n_bytes =
      static_cast<size_t>(seq_length * num_directions * batch_size * hidden_size) * sizeof(float);
  Tensor y = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), y_shape, y_n_bytes, allocator);
  const size_t state_n_bytes =
      static_cast<size_t>(num_directions * batch_size * hidden_size) * sizeof(float);
  Tensor y_h = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), state_shape, state_n_bytes,
                                allocator);
  Tensor y_c = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), state_shape, state_n_bytes,
                                allocator);
  float *py = y.AsFloat();
  float *py_h = y_h.AsFloat();
  float *py_c = y_c.AsFloat();

  // Working buffers for H_{t-1}, C_{t-1}, H_t, C_t.
  const std::size_t state_count = static_cast<std::size_t>(batch_size * hidden_size);
  detail::TemporaryTypedBuffer<float> h_prev_buf(state_count, allocator, "kernel::LSTM h_prev");
  detail::TemporaryTypedBuffer<float> c_prev_buf(state_count, allocator, "kernel::LSTM c_prev");
  detail::TemporaryTypedBuffer<float> h_curr_buf(state_count, allocator, "kernel::LSTM h_curr");
  detail::TemporaryTypedBuffer<float> c_curr_buf(state_count, allocator, "kernel::LSTM c_curr");

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

  for (int64_t d = 0; d < num_directions; ++d) {
    // For ``bidirectional`` the second direction (d == 1) runs in reverse;
    // ``reverse`` runs its single direction in reverse.
    const bool reverse = (direction == "reverse") || (direction == "bidirectional" && d == 1);
    const float *wd = pw + d * 4 * hidden_size * input_size;
    const float *rd = pr + d * 4 * hidden_size * hidden_size;

    // Combined per-gate bias = Wb + Rb (length hidden_size each).
    // Allocator-backed buffers are not guaranteed zeroed; the gate biases are
    // read for every step even when ``B`` is absent, so zero-fill explicitly.
    std::fill(bias_i, bias_i + hidden_count, 0.0f);
    std::fill(bias_o, bias_o + hidden_count, 0.0f);
    std::fill(bias_f, bias_f + hidden_count, 0.0f);
    std::fill(bias_c, bias_c + hidden_count, 0.0f);
    if (p_b != nullptr) {
      const float *bd = p_b + d * 8 * hidden_size;
      const float *wb = bd;                   // first half: Wb
      const float *rb = bd + 4 * hidden_size; // second half: Rb
      for (int64_t h = 0; h < hidden_size; ++h) {
        bias_i[static_cast<size_t>(h)] = wb[gI + h] + rb[gI + h];
        bias_o[static_cast<size_t>(h)] = wb[gO + h] + rb[gO + h];
        bias_f[static_cast<size_t>(h)] = wb[gF + h] + rb[gF + h];
        bias_c[static_cast<size_t>(h)] = wb[gC + h] + rb[gC + h];
      }
    }

    // Peepholes are read for every step even when ``P`` is absent; zero-fill
    // the (potentially uninitialized) allocator-backed storage explicitly.
    std::fill(peep_i, peep_i + hidden_count, 0.0f);
    std::fill(peep_o, peep_o + hidden_count, 0.0f);
    std::fill(peep_f, peep_f + hidden_count, 0.0f);
    if (p_p != nullptr) {
      const float *pd = p_p + d * 3 * hidden_size;
      for (int64_t h = 0; h < hidden_size; ++h) {
        peep_i[static_cast<size_t>(h)] = pd[0 * hidden_size + h];
        peep_o[static_cast<size_t>(h)] = pd[1 * hidden_size + h];
        peep_f[static_cast<size_t>(h)] = pd[2 * hidden_size + h];
      }
    }

    float *h_prev = h_prev_buf.data();
    float *c_prev = c_prev_buf.data();
    float *h_curr = h_curr_buf.data();
    float *c_curr = c_curr_buf.data();
    // H_0 / C_0 default to zero; allocator-backed storage is not guaranteed
    // zeroed, so fill before optionally copying in the provided initial states.
    std::fill(h_prev, h_prev + state_count, 0.0f);
    std::fill(c_prev, c_prev + state_count, 0.0f);
    if (p_initial_h != nullptr) {
      const float *h0 = p_initial_h + d * batch_size * hidden_size;
      for (std::size_t i = 0; i < state_count; ++i) {
        h_prev[i] = h0[i];
      }
    }
    if (p_initial_c != nullptr) {
      const float *c0 = p_initial_c + d * batch_size * hidden_size;
      for (std::size_t i = 0; i < state_count; ++i) {
        c_prev[i] = c0[i];
      }
    }

    for (int64_t step = 0; step < seq_length; ++step) {
      const int64_t t = reverse ? seq_length - 1 - step : step;
      const float *x_t = px + t * batch_size * input_size;
      for (int64_t n = 0; n < batch_size; ++n) {
        const float *x_row = x_t + n * input_size;
        const float *h_row = h_prev + n * hidden_size;
        const float *c_row = c_prev + n * hidden_size;
        float *h_out = h_curr + n * hidden_size;
        float *c_out = c_curr + n * hidden_size;

        // Compute per-gate pre-activations: X_t @ W^T + H_{t-1} @ R^T + bias.
        for (int64_t h = 0; h < hidden_size; ++h) {
          const float *wi_row = wd + (gI + h) * input_size;
          const float *wo_row = wd + (gO + h) * input_size;
          const float *wf_row = wd + (gF + h) * input_size;
          const float *wc_row = wd + (gC + h) * input_size;
          const float *ri_row = rd + (gI + h) * hidden_size;
          const float *ro_row = rd + (gO + h) * hidden_size;
          const float *rf_row = rd + (gF + h) * hidden_size;
          const float *rc_row = rd + (gC + h) * hidden_size;
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
      // Copy h_curr into Y at [t, d] and swap into h_prev/c_prev for the next
      // step.
      float *y_t = py + ((t * num_directions + d) * batch_size) * hidden_size;
      for (std::size_t i = 0; i < state_count; ++i) {
        y_t[i] = h_curr[i];
      }
      std::swap(h_prev, h_curr);
      std::swap(c_prev, c_curr);
    }

    // Y_h[d] / Y_c[d] are the last processed step (in h_prev/c_prev after the
    // final swap).
    float *y_h_d = py_h + d * batch_size * hidden_size;
    float *y_c_d = py_c + d * batch_size * hidden_size;
    for (std::size_t i = 0; i < state_count; ++i) {
      y_h_d[i] = h_prev[i];
      y_c_d[i] = c_prev[i];
    }
  }

  if (layout == 1) {
    // Permute Y [seq, D, batch, hidden] -> [batch, seq, D, hidden] and
    // Y_h / Y_c [D, batch, hidden] -> [batch, D, hidden].
    y = recurrent::RecurrentPermuteYLayout1(y, seq_length, num_directions, batch_size, hidden_size,
                                            allocator);
    y_h = recurrent::RecurrentPermuteStateLayout1(y_h, num_directions, batch_size, hidden_size,
                                                  allocator);
    y_c = recurrent::RecurrentPermuteStateLayout1(y_c, num_directions, batch_size, hidden_size,
                                                  allocator);
  }

  return std::tuple<Tensor, Tensor, Tensor>(std::move(y), std::move(y_h), std::move(y_c));
}

void LSTM::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() < 3 || node.input_size() > 8), "RunNode: op '",
                      node.op_type(), "' expects between 3 and 8 input(s), got ", node.input_size(),
                      ".");
  EXT_ENFORCE_INVALID(!(node.output_size() < 1 || node.output_size() > 3), "RunNode: op '",
                      node.op_type(), "' expects between 1 and 3 output(s), got ",
                      node.output_size(), ".");

  // Unsupported attributes: only the default ``Sigmoid``/``Tanh``/``Tanh``
  // activations, no ``clip`` and ``input_forget == 0`` are implemented;
  // ``direction`` may be ``forward``, ``reverse`` or ``bidirectional``;
  // ``layout=0`` and ``layout=1`` are both supported.
  const std::string direction = GetAttributeStringOrDefault(node, "direction", "forward");
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
  auto [y, y_h, y_c] =
      kernel(x, w, r, b != nullptr ? *b : Tensor{}, initial_h != nullptr ? *initial_h : Tensor{},
             initial_c != nullptr ? *initial_c : Tensor{}, p != nullptr ? *p : Tensor{}, layout,
             direction);

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
  set_optional_output(2, std::move(y_c));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
