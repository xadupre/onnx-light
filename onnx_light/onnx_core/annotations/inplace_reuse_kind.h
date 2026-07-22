// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_helpers.h"

#include <cstdint>

/**
 * @file inplace_reuse_kind.h
 * @brief Lightweight descriptors for in-place reuse opportunities.
 *
 * This header only defines the small POD types :cpp:enum:`InPlaceReuseKind`
 * and :cpp:struct:`InPlaceReuse`. It is deliberately kept free of the heavy
 * shape-inference includes pulled in by ``inplace_reuse.h`` so that headers
 * that merely need to store an :cpp:struct:`InPlaceReuse` (such as
 * ``execute_action.h``) do not transitively depend on ``expressions.h``,
 * ``shapes_context.h`` or ``onnx.h``.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace annotations {

/**
 * Classifies how an input buffer compares in size with the output that
 * reuses it:
 *
 *   - :cpp:enumerator:`kEqual`: the input and output buffers have the same
 *     byte size (e.g. a Transpose or a same-total-size Reshape).
 *     This is the preferred, space-optimal reuse.
 *   - :cpp:enumerator:`kGreater`: the input buffer is strictly larger in
 *     bytes than the output, so the output still fits but leaves part of
 *     the buffer unused.
 */
enum class InPlaceReuseKind {
  kEqual,
  kGreater,
};

/**
 * A single in-place reuse opportunity for one node: the output at
 * position :cpp:var:`output_index` may reuse the buffer of the input at
 * position :cpp:var:`input_index` (both indices refer to the node's
 * ``output()`` / ``input()`` lists). :cpp:var:`kind` records whether the
 * input buffer has the same size as the output
 * (:cpp:enumerator:`InPlaceReuseKind::kEqual`) or is strictly larger
 * (:cpp:enumerator:`InPlaceReuseKind::kGreater`).
 */
struct InPlaceReuse {
  int64_t output_index = -1;
  int64_t input_index = -1;
  InPlaceReuseKind kind = InPlaceReuseKind::kEqual;

  bool operator==(const InPlaceReuse &other) const noexcept {
    return output_index == other.output_index && input_index == other.input_index &&
           kind == other.kind;
  }
  bool operator!=(const InPlaceReuse &other) const noexcept { return !(*this == other); }
};

} // namespace annotations
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
