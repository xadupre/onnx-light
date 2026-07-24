// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file op_schema_info.h
 * @brief Namespace-stable operator-schema description used by GraphBuilder.
 *
 * :cpp:class:`core::builder::GraphBuilder` needs a little schema information
 * (the versioned ``since_version`` and the output-count bounds) to resolve the
 * opset of a node and to validate its outputs. The full
 * :cpp:class:`core::schema::LightOpSchema` cannot be used here: the ``onnx_op``
 * library that produces the built-in schemas is compiled with the
 * ``ONNX_LIGHT_NAMESPACE`` token left as a literal identifier (its headers do
 * not pull in the macro definition), so a ``LightOpSchema`` object crossing
 * between an ``onnx_op`` translation unit and an ``onnx_light`` one would be two
 * distinct C++ types with the same layout. To keep the schema provider usable
 * from both worlds, this header declares :cpp:struct:`OpSchemaInfo` in the
 * *literal* ``onnx_light`` namespace (spelled out, not via the macro) so every
 * translation unit agrees on its identity regardless of the macro's value.
 */

#pragma once

#include <string>
#include <vector>

// The namespace is spelled literally (not through ``ONNX_LIGHT_NAMESPACE``) on
// purpose: see the file comment above. In ``onnx_light`` translation units the
// macro also expands to ``onnx_light``, so this is the same namespace.
namespace onnx_light {
namespace core {
namespace builder {

/// Minimal description of one versioned operator schema.
struct OpSchemaInfo {
  /// Opset version at which this schema was introduced.
  int since_version = 0;
  /// Minimum number of outputs the operator produces.
  int min_output = 0;
  /// Maximum number of outputs the operator produces.
  int max_output = 0;
  /// Operator domain (empty string denotes the default ONNX domain).
  std::string domain;
};

} // namespace builder
} // namespace core
} // namespace onnx_light
