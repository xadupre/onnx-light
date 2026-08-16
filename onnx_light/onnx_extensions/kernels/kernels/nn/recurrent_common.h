// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"
#include <cstdint>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {
namespace recurrent {

// Returns ``num_directions`` for a recurrent operator (RNN / GRU / LSTM) given
// its ``direction`` attribute: ``1`` for ``"forward"`` / ``"reverse"`` and
// ``2`` for ``"bidirectional"``. Any other value is rejected. ``op`` is used
// only to build the diagnostic message.
int64_t RecurrentNumDirections(const char *op, const std::string &direction);

// Transposes an initial state tensor from the ``layout=1`` shape
// ``[batch_size, num_directions, hidden_size]`` to the time-major
// ``[num_directions, batch_size, hidden_size]`` used by the core loop.
core::runtime::Tensor RecurrentTransposeInitialState(const core::runtime::Tensor &state,
                                                     int64_t num_directions,
                                                     core::runtime::RawBufferAllocator *allocator);

// Permutes a ``Y`` output from the time-major layout
// ``[seq_length, num_directions, batch_size, hidden_size]`` to the batch-major
// ``layout=1`` shape ``[batch_size, seq_length, num_directions, hidden_size]``.
core::runtime::Tensor RecurrentPermuteYLayout1(const core::runtime::Tensor &y, int64_t seq_length,
                                               int64_t num_directions, int64_t batch_size,
                                               int64_t hidden_size,
                                               core::runtime::RawBufferAllocator *allocator);

// Permutes a hidden/cell state output (``Y_h`` or ``Y_c``) from the time-major
// layout ``[num_directions, batch_size, hidden_size]`` to the batch-major
// ``layout=1`` shape ``[batch_size, num_directions, hidden_size]``.
core::runtime::Tensor RecurrentPermuteStateLayout1(const core::runtime::Tensor &state,
                                                   int64_t num_directions, int64_t batch_size,
                                                   int64_t hidden_size,
                                                   core::runtime::RawBufferAllocator *allocator);

} // namespace recurrent
} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
