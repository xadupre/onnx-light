// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// IMPORTANT: ``operator_sets.h`` is included first, before any header that
// pulls in the ``ONNX_LIGHT_NAMESPACE`` macro definition. The ``onnx_op``
// library is compiled with that token left as a literal identifier, so its
// ``GetAllOnnxOpSchemasWithHistory`` symbol lives in namespace
// ``ONNX_LIGHT_NAMESPACE`` spelled literally. Keeping the include order below
// ensures this translation unit references the very same symbol.
#include "onnx_op/operator_sets.h"

#include "onnx_op/default_onnx_schema.h"

namespace onnx_light {
namespace core {
namespace builder {

std::vector<::onnx_light::core::schema::OpSchemaInfo>
DefaultOnnxSchemaLookup(const std::string &op_type) {
  std::vector<::onnx_light::core::schema::OpSchemaInfo> result;
  const std::vector<ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema> schemas =
      ONNX_LIGHT_NAMESPACE::onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, /*init_doc=*/false);
  result.reserve(schemas.size());
  for (const ONNX_LIGHT_NAMESPACE::core::schema::LightOpSchema &schema : schemas) {
    result.push_back(schema.op_schema_info());
  }
  return result;
}

} // namespace builder
} // namespace core
} // namespace onnx_light
