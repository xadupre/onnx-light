// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/temporary_buffer.h"
#include "onnx_extensions/kernels/kernels/nn/recurrent_common.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

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
                                          int64_t layout, const std::string &direction,
                                          RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(layout == 0 || layout == 1, "kernel::RNN: layout must be 0 or 1, got ",
                      std::to_string(layout), ".");
  // ``direction`` selects the recurrence order: ``forward`` and ``reverse``
  // use a single direction (``num_directions == 1``); ``bidirectional``
  // concatenates a forward and a reverse pass (``num_directions == 2``).
  const int64_t num_directions = recurrent::RecurrentNumDirections("RNN", direction);
  EXT_ENFORCE_INVALID(x_in.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RNN: X must be FLOAT.");
  EXT_ENFORCE_INVALID(w.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RNN: W must be FLOAT.");
  EXT_ENFORCE_INVALID(r.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RNN: R must be FLOAT.");
  EXT_ENFORCE_INVALID(x_in.shape.size() == 3u, "kernel::RNN: X must have rank 3.");
  EXT_ENFORCE_INVALID(w.shape.size() == 3u && w.shape[0] == num_directions,
                      "kernel::RNN: W must have shape [num_directions, hidden_size, input_size].");
  EXT_ENFORCE_INVALID(r.shape.size() == 3u && r.shape[0] == num_directions,
                      "kernel::RNN: R must have shape [num_directions, hidden_size, hidden_size].");

  // Scratch and result buffers are drawn from the runtime allocator (when one
  // is attached) so no storage is acquired outside the runtime context; they
  // fall back to inline ``std::vector`` storage otherwise.
  RawBufferAllocator *allocator = rt != nullptr ? rt->allocator() : nullptr;

  // ``layout == 1`` permutes batch and time/direction axes on a subset
  // of inputs and outputs; the time-major kernel body below stays as is.
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
    x_storage =
        Tensor::FromFloat("", {seq_length, batch_size, input_size}, std::move(x_data), allocator);
    x_p = &x_storage;

    if (!(initial_h_in.shape.empty() && initial_h_in.size_bytes() == 0)) {
      initial_h_storage =
          recurrent::RecurrentTransposeInitialState(initial_h_in, num_directions, allocator);
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
                      "kernel::RNN: R must have shape [num_directions, hidden_size, hidden_size].");

  const float *p_b = AsFloatOrNull(b, "B");
  if (p_b != nullptr) {
    EXT_ENFORCE_INVALID(b.shape.size() == 2u && b.shape[0] == num_directions &&
                            b.shape[1] == 2 * hidden_size,
                        "kernel::RNN: B must have shape [num_directions, 2 * hidden_size].");
  }
  const float *p_initial_h = AsFloatOrNull(initial_h, "initial_h");
  if (p_initial_h != nullptr) {
    EXT_ENFORCE_INVALID(initial_h.shape.size() == 3u && initial_h.shape[0] == num_directions &&
                            initial_h.shape[1] == batch_size && initial_h.shape[2] == hidden_size,
                        "kernel::RNN: initial_h must have shape [num_directions, batch_size, "
                        "hidden_size].");
  }

  const float *px = x.AsFloat();
  const float *pw = w.AsFloat();
  const float *pr = r.AsFloat();

  // Output allocations. ``Y`` is [seq, num_directions, batch, hidden];
  // ``Y_h`` is [num_directions, batch, hidden].
  const onnx_kernels::Shape y_shape{seq_length, num_directions, batch_size, hidden_size};
  const onnx_kernels::Shape y_h_shape{num_directions, batch_size, hidden_size};
  const size_t y_n_bytes =
      static_cast<size_t>(seq_length * num_directions * batch_size * hidden_size) * sizeof(float);
  Tensor y = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), y_shape, y_n_bytes, allocator);
  const size_t y_h_n_bytes =
      static_cast<size_t>(num_directions * batch_size * hidden_size) * sizeof(float);
  Tensor y_h =
      MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), y_h_shape, y_h_n_bytes, allocator);
  float *py = y.AsFloat();
  float *py_h = y_h.AsFloat();

  // Per-direction working buffers for H_{t-1} and H_t, plus the combined
  // per-hidden bias = Wb + Rb.
  const std::size_t hidden_count = static_cast<std::size_t>(hidden_size);
  const std::size_t state_count = static_cast<std::size_t>(batch_size * hidden_size);
  detail::TemporaryTypedBuffer<float> bias_buf(hidden_count, allocator, "kernel::RNN bias");
  detail::TemporaryTypedBuffer<float> h_prev_buf(state_count, allocator, "kernel::RNN h_prev");
  detail::TemporaryTypedBuffer<float> h_curr_buf(state_count, allocator, "kernel::RNN h_curr");
  float *bias = bias_buf.data();

  for (int64_t d = 0; d < num_directions; ++d) {
    // For ``bidirectional`` the second direction (d == 1) runs in reverse;
    // ``reverse`` runs its single direction in reverse.
    const bool reverse = (direction == "reverse") || (direction == "bidirectional" && d == 1);
    const float *wd = pw + d * hidden_size * input_size;
    const float *rd = pr + d * hidden_size * hidden_size;

    std::fill(bias, bias + hidden_count, 0.0f);
    if (p_b != nullptr) {
      const float *bd = p_b + d * 2 * hidden_size;
      for (int64_t h = 0; h < hidden_size; ++h) {
        bias[static_cast<size_t>(h)] = bd[h] + bd[hidden_size + h];
      }
    }

    float *h_prev = h_prev_buf.data();
    float *h_curr = h_curr_buf.data();
    // H_0 defaults to zero; allocator-backed storage is not guaranteed zeroed.
    std::fill(h_prev, h_prev + state_count, 0.0f);
    if (p_initial_h != nullptr) {
      const float *h0 = p_initial_h + d * batch_size * hidden_size;
      for (std::size_t i = 0; i < state_count; ++i) {
        h_prev[i] = h0[i];
      }
    }

    for (int64_t step = 0; step < seq_length; ++step) {
      const int64_t t = reverse ? seq_length - 1 - step : step;
      const float *x_t = px + t * batch_size * input_size;
      // For each batch row and each output unit, accumulate X_t @ W^T plus
      // H_{t-1} @ R^T plus the per-unit bias, then apply tanh.
      for (int64_t n = 0; n < batch_size; ++n) {
        const float *x_row = x_t + n * input_size;
        const float *h_row = h_prev + n * hidden_size;
        float *out_row = h_curr + n * hidden_size;
        for (int64_t h = 0; h < hidden_size; ++h) {
          const float *w_row = wd + h * input_size;
          const float *r_row = rd + h * hidden_size;
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
      // Copy h_curr into Y at [t, d] and swap into h_prev for the next step.
      float *y_t = py + ((t * num_directions + d) * batch_size) * hidden_size;
      for (std::size_t i = 0; i < state_count; ++i) {
        y_t[i] = h_curr[i];
      }
      std::swap(h_prev, h_curr);
    }

    // Y_h[d] is the last processed time step (in h_prev after the final swap).
    float *y_h_d = py_h + d * batch_size * hidden_size;
    for (std::size_t i = 0; i < state_count; ++i) {
      y_h_d[i] = h_prev[i];
    }
  }

  if (layout == 1) {
    // Permute Y [seq, D, batch, hidden] -> [batch, seq, D, hidden] and
    // Y_h [D, batch, hidden] -> [batch, D, hidden].
    y = recurrent::RecurrentPermuteYLayout1(y, seq_length, num_directions, batch_size, hidden_size,
                                            allocator);
    y_h = recurrent::RecurrentPermuteStateLayout1(y_h, num_directions, batch_size, hidden_size,
                                                  allocator);
  }

  return std::pair<Tensor, Tensor>(std::move(y), std::move(y_h));
}

void RNN::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() < 3 || node.input_size() > 6), "RunNode: op '",
                      node.op_type(), "' expects between 3 and 6 input(s), got ", node.input_size(),
                      ".");
  EXT_ENFORCE_INVALID(!(node.output_size() < 1 || node.output_size() > 2), "RunNode: op '",
                      node.op_type(), "' expects 1 or 2 output(s), got ", node.output_size(), ".");

  // Only the default ``Tanh`` activation and no ``clip`` are
  // implemented; ``direction`` may be ``forward``, ``reverse`` or
  // ``bidirectional``; ``layout=0`` and ``layout=1`` are both supported.
  const std::string direction = GetAttributeStringOrDefault(node, "direction", "forward");
  if (const AttributeProto *activations = FindAttribute(node, "activations");
      activations != nullptr) {
    const std::vector<std::string> values = GetAttributeStringsOrDefault(node, "activations", {});
    EXT_ENFORCE_INVALID(!(values.size() != 1 || values[0] != "Tanh"),
                        "RunNode: op 'RNN' only supports the default activations=['Tanh'].");
  }
  EXT_ENFORCE_INVALID(!(FindAttribute(node, "activation_alpha") != nullptr ||
                        FindAttribute(node, "activation_beta") != nullptr),
                      "RunNode: op 'RNN' does not support 'activation_alpha'/'activation_beta'.");
  EXT_ENFORCE_INVALID(FindAttribute(node, "clip") == nullptr,
                      "RunNode: op 'RNN' does not support the 'clip' attribute.");
  const int64_t layout = GetAttributeIntOrDefault(node, "layout", 0);

  // ``sequence_lens`` (input #4) is not supported: it requires
  // per-batch sequence handling that the FLOAT kernel does not
  // implement.
  const Tensor *sequence_lens = GetOptionalInput(node, 4, rt.tensors());
  EXT_ENFORCE_INVALID(sequence_lens == nullptr,
                      "RunNode: op 'RNN' does not support the optional 'sequence_lens' input.");

  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &w = GetInput(node, 1, rt.tensors());
  const Tensor &r = GetInput(node, 2, rt.tensors());
  const Tensor *b = GetOptionalInput(node, 3, rt.tensors());
  const Tensor *initial_h = GetOptionalInput(node, 5, rt.tensors());

  onnx_kernels::kernel::RNN kernel(rt.kernel_ctx());
  auto [y, y_h] = kernel(x, w, r, b != nullptr ? *b : Tensor{},
                         initial_h != nullptr ? *initial_h : Tensor{}, layout, direction);

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

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
