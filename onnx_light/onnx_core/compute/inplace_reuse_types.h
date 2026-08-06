// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "onnx_proto/onnx.h"

/**
 * @file inplace_reuse_types.h
 * @brief Shared primitive types for the in-place reuse analysis.
 *
 * This lightweight header declares :cpp:enum:`InPlaceReuseKind`,
 * :cpp:struct:`InPlaceReuse`, and the metadata key constants that are shared
 * between :file:`inplace_reuse.h` and :file:`compute_context.h`.  Keeping
 * these types in their own header avoids the circular include that would
 * otherwise arise between the two files.
 */

namespace ONNX_LIGHT_NAMESPACE::core::compute {

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

/**
 * Metadata key under which :cpp:func:`ComputeContext::WriteToMetadata`
 * records a node's in-place reuse opportunities. The associated value is a
 * string with one ``output_index:input_index:kind`` triplet per opportunity
 * (``kind`` is ``equal`` or ``greater``), triplets separated by ``;``.
 */
constexpr const char *kInPlaceReuseMetadataKey = "onnx_light.inplace_reuse";

/**
 * Metadata key under which :cpp:func:`ComputeContext::WriteToMetadata`
 * records, for every node, which referenced values reach their last use at
 * this node and can therefore be released after the node runs. The associated
 * value is a ``;``-separated list of value names.
 */
constexpr const char *kReleaseAfterMetadataKey = "onnx_light.release_after";

/**
 * Metadata key under which :cpp:func:`ComputeContext::WriteToMetadata`
 * records, for every node, which declared graph inputs or initializers reach
 * their last use at that node. Those names are not released by the runtime
 * (their lifetime is owned by the caller / model) but this key still exposes
 * the "last read" information. The associated value is a ``;``-separated list
 * of value names.
 */
constexpr const char *kNotUsedAfterMetadataKey = "onnx_light.not_used_after";

/**
 * Metadata key under which :cpp:func:`ComputeContext::WriteToMetadata`
 * records, for every node, the subset of releasable values that carry the
 * ``"shape"`` value tag (i.e. tensors that represent tensor-shape metadata
 * rather than activation data). The associated value is a ``;``-separated
 * list of value names — a strict subset of the
 * :cpp:var:`kReleaseAfterMetadataKey` entry for the same node. The key is
 * omitted for nodes that have no shape-tagged releasable values.
 */
constexpr const char *kReleaseAfterShapeTagMetadataKey = "onnx_light.release_after_shape_tag";

} // namespace ONNX_LIGHT_NAMESPACE::core::compute
