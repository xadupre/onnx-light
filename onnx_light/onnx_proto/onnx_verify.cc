#include "onnx_verify.h"

#include <string>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE {

namespace {

/** Returns @p s converted to a plain std::string, treating an unset optional as empty. */
inline std::string ToStdString(const utils::OptionalString &s) { return std::string(s.sv()); }
/** Returns @p s converted to a plain std::string. */
inline std::string ToStdString(const utils::String &s) { return std::string(s.sv()); }

} // namespace

void VerifyValueInfo(const ValueInfoProto &value_info, bool is_main_graph) {
  EXT_ENFORCE_INVALID(!value_info.name().empty(),
                      "ValueInfoProto is missing a non-empty 'name' field.");

  if (!is_main_graph) {
    // Shapes and types are optional for subgraph inputs/outputs (e.g. Loop/If/Scan bodies).
    return;
  }

  EXT_ENFORCE_INVALID(value_info.has_type(), "ValueInfoProto '", value_info.name(),
                      "' is missing its 'type' field.");
  const TypeProto &type = value_info.type();
  switch (type.value_case()) {
  case TypeProto::kTensorType: {
    const auto &tensor_type = type.tensor_type();
    EXT_ENFORCE_INVALID(tensor_type.has_elem_type(), "ValueInfoProto '", value_info.name(),
                        "' tensor_type is missing 'elem_type'.");
    EXT_ENFORCE_INVALID(tensor_type.elem_type() != TensorProto::UNDEFINED, "ValueInfoProto '",
                        value_info.name(), "' tensor_type.elem_type must not be UNDEFINED.");
    EXT_ENFORCE_INVALID(tensor_type.has_shape(), "ValueInfoProto '", value_info.name(),
                        "' tensor_type is missing 'shape'.");
    break;
  }
  case TypeProto::kSparseTensorType: {
    const auto &sparse_type = type.sparse_tensor_type();
    EXT_ENFORCE_INVALID(sparse_type.has_elem_type(), "ValueInfoProto '", value_info.name(),
                        "' sparse_tensor_type is missing 'elem_type'.");
    EXT_ENFORCE_INVALID(sparse_type.elem_type() != TensorProto::UNDEFINED, "ValueInfoProto '",
                        value_info.name(), "' sparse_tensor_type.elem_type must not be UNDEFINED.");
    EXT_ENFORCE_INVALID(sparse_type.has_shape(), "ValueInfoProto '", value_info.name(),
                        "' sparse_tensor_type is missing 'shape'.");
    break;
  }
  case TypeProto::kSequenceType:
    EXT_ENFORCE_INVALID(type.sequence_type().has_elem_type(), "ValueInfoProto '", value_info.name(),
                        "' sequence_type is missing 'elem_type'.");
    break;
  case TypeProto::kOptionalType:
    EXT_ENFORCE_INVALID(type.optional_type().has_elem_type(), "ValueInfoProto '", value_info.name(),
                        "' optional_type is missing 'elem_type'.");
    break;
  case TypeProto::kMapType:
    EXT_ENFORCE_INVALID(type.map_type().has_key_type(), "ValueInfoProto '", value_info.name(),
                        "' map_type is missing 'key_type'.");
    EXT_ENFORCE_INVALID(type.map_type().has_value_type(), "ValueInfoProto '", value_info.name(),
                        "' map_type is missing 'value_type'.");
    break;
  case TypeProto::kOpaqueType:
    // domain/name are both optional per spec; nothing further to check.
    break;
  case TypeProto::kUndefined:
  default:
    EXT_THROW_INVALID("ValueInfoProto '", value_info.name(),
                      "' has an unrecognized or unset type.");
  }
}

void VerifyTensor(const TensorProto &tensor) {
  EXT_ENFORCE_INVALID(tensor.data_type() != TensorProto::UNDEFINED, "TensorProto '", tensor.name(),
                      "' has data_type UNDEFINED.");

  const bool has_float = !tensor.float_data().empty();
  const bool has_int32 = !tensor.int32_data().empty();
  const bool has_string = !tensor.string_data().empty();
  const bool has_int64 = !tensor.int64_data().empty();
  const bool has_raw = !tensor.raw_data().empty();
  const bool has_double = !tensor.double_data().empty();
  const bool has_uint64 = !tensor.uint64_data().empty();
  const int num_value_fields = static_cast<int>(has_float) + static_cast<int>(has_int32) +
                               static_cast<int>(has_string) + static_cast<int>(has_int64) +
                               static_cast<int>(has_raw) + static_cast<int>(has_double) +
                               static_cast<int>(has_uint64);

  const bool stored_externally =
      tensor.has_data_location() && tensor.data_location() == TensorProto::EXTERNAL;
  if (stored_externally) {
    EXT_ENFORCE_INVALID(num_value_fields == 0, "TensorProto '", tensor.name(),
                        "' is stored externally but also carries inline data.");
    bool has_location = false;
    for (const auto &entry : tensor.external_data()) {
      if (entry.has_key() && entry.key() == "location" && entry.has_value() &&
          !entry.value().empty()) {
        has_location = true;
        break;
      }
    }
    EXT_ENFORCE_INVALID(has_location, "TensorProto '", tensor.name(),
                        "' is stored externally but is missing a non-empty 'location' entry.");
    return;
  }

  int64_t nelem = 1;
  for (auto d : tensor.dims()) {
    nelem *= static_cast<int64_t>(d);
  }

  if (nelem == 0) {
    EXT_ENFORCE_INVALID(num_value_fields == 0, "TensorProto '", tensor.name(),
                        "' declares zero elements but carries data.");
    return;
  }

  EXT_ENFORCE_INVALID(num_value_fields == 1, "TensorProto '", tensor.name(),
                      "' must carry exactly one data field, found ", num_value_fields, ".");

  if (has_raw) {
    EXT_ENFORCE_INVALID(tensor.data_type() != TensorProto::STRING, "TensorProto '", tensor.name(),
                        "' stores STRING data in 'raw_data', which is not allowed.");
    // Sanity-check that raw_data is large enough for packed sub-byte types.
    int64_t expected_bytes = 0;
    switch (tensor.data_type()) {
    case TensorProto::UINT4:
    case TensorProto::INT4:
    case TensorProto::FLOAT4E2M1:
      expected_bytes = (nelem + 1) / 2; // 2 elements per byte, ceiling division
      break;
    case TensorProto::UINT2:
    case TensorProto::INT2:
      expected_bytes = (nelem + 3) / 4; // 4 elements per byte, ceiling division
      break;
    case TensorProto::FLOAT6E2M3:
    case TensorProto::FLOAT6E3M2:
      expected_bytes = nelem / 4 * 3 + (nelem % 4 * 6 + 7) / 8;
      if (expected_bytes > 0 && static_cast<int64_t>(tensor.raw_data().size()) >= expected_bytes) {
        const auto used_bits = static_cast<uint8_t>((nelem % 4 * 6) % 8);
        if (used_bits != 0) {
          const auto last_byte = static_cast<uint8_t>(tensor.raw_data()[expected_bytes - 1]);
          const auto unused_bits_mask = static_cast<uint8_t>(0xFFU << used_bits);
          EXT_ENFORCE_INVALID((last_byte & unused_bits_mask) == 0, "TensorProto '", tensor.name(),
                              "' has non-zero padding bits in its packed FLOAT6 raw_data.");
        }
      }
      break;
    default:
      break;
    }
    EXT_ENFORCE_INVALID(
        expected_bytes == 0 || static_cast<int64_t>(tensor.raw_data().size()) >= expected_bytes,
        "TensorProto '", tensor.name(), "' raw_data size (", tensor.raw_data().size(),
        " bytes) is too small for the declared shape and packed type (", expected_bytes,
        " bytes required).");
    return;
  }

  switch (tensor.data_type()) {
  case TensorProto::FLOAT:
  case TensorProto::COMPLEX64:
    EXT_ENFORCE_INVALID(has_float, "TensorProto '", tensor.name(),
                        "' data_type requires data to be stored in 'float_data'.");
    break;
  case TensorProto::DOUBLE:
  case TensorProto::COMPLEX128:
    EXT_ENFORCE_INVALID(has_double, "TensorProto '", tensor.name(),
                        "' data_type requires data to be stored in 'double_data'.");
    break;
  case TensorProto::INT64:
    EXT_ENFORCE_INVALID(has_int64, "TensorProto '", tensor.name(),
                        "' data_type requires data to be stored in 'int64_data'.");
    break;
  case TensorProto::UINT32:
  case TensorProto::UINT64:
    EXT_ENFORCE_INVALID(has_uint64, "TensorProto '", tensor.name(),
                        "' data_type requires data to be stored in 'uint64_data'.");
    break;
  case TensorProto::STRING:
    EXT_ENFORCE_INVALID(has_string, "TensorProto '", tensor.name(),
                        "' data_type requires data to be stored in 'string_data'.");
    break;
  case TensorProto::INT32:
    EXT_ENFORCE_INVALID(has_int32, "TensorProto '", tensor.name(),
                        "' data_type requires data to be stored in 'int32_data'.");
    break;
  case TensorProto::UINT8:
  case TensorProto::INT8:
  case TensorProto::UINT16:
  case TensorProto::INT16:
  case TensorProto::BOOL:
  case TensorProto::FLOAT16:
  case TensorProto::BFLOAT16:
  case TensorProto::FLOAT8E4M3FN:
  case TensorProto::FLOAT8E4M3FNUZ:
  case TensorProto::FLOAT8E5M2:
  case TensorProto::FLOAT8E5M2FNUZ:
  case TensorProto::FLOAT8E8M0:
  case TensorProto::FLOAT6E2M3:
  case TensorProto::FLOAT6E3M2:
    EXT_ENFORCE_INVALID(has_int32, "TensorProto '", tensor.name(),
                        "' data_type requires data to be stored in 'int32_data'.");
    // These types are not packed: each element occupies one int32_data entry.
    EXT_ENFORCE_INVALID(static_cast<int64_t>(tensor.int32_data().size()) >= nelem, "TensorProto '",
                        tensor.name(), "' int32_data size (", tensor.int32_data().size(),
                        ") is too small for the declared shape (", nelem,
                        " int32 values required).");
    if (tensor.data_type() == TensorProto::FLOAT6E2M3 ||
        tensor.data_type() == TensorProto::FLOAT6E3M2) {
      for (const auto value : tensor.int32_data()) {
        EXT_ENFORCE_INVALID(value >= 0 && value <= 0x3F, "TensorProto '", tensor.name(),
                            "' FLOAT6 int32_data values must use only bits 0-5.");
      }
    }
    break;
  case TensorProto::UINT4:
  case TensorProto::INT4:
  case TensorProto::FLOAT4E2M1: {
    EXT_ENFORCE_INVALID(has_int32, "TensorProto '", tensor.name(),
                        "' data_type requires data to be stored in 'int32_data'.");
    // Each int32 packs 8 4-bit elements.
    const int64_t expected_int32s = (nelem + 7) / 8;
    EXT_ENFORCE_INVALID(static_cast<int64_t>(tensor.int32_data().size()) >= expected_int32s,
                        "TensorProto '", tensor.name(), "' int32_data size (",
                        tensor.int32_data().size(),
                        ") is too small for the declared shape and packed type (", expected_int32s,
                        " int32 values required).");
    break;
  }
  case TensorProto::UINT2:
  case TensorProto::INT2: {
    EXT_ENFORCE_INVALID(has_int32, "TensorProto '", tensor.name(),
                        "' data_type requires data to be stored in 'int32_data'.");
    // Each int32 packs 16 2-bit elements.
    const int64_t expected_int32s = (nelem + 15) / 16;
    EXT_ENFORCE_INVALID(static_cast<int64_t>(tensor.int32_data().size()) >= expected_int32s,
                        "TensorProto '", tensor.name(), "' int32_data size (",
                        tensor.int32_data().size(),
                        ") is too small for the declared shape and packed type (", expected_int32s,
                        " int32 values required).");
    break;
  }
  default:
    EXT_THROW_INVALID("TensorProto '", tensor.name(), "' has an unrecognized data_type (",
                      static_cast<int>(tensor.data_type()), ").");
  }
}

void VerifySparseTensor(const SparseTensorProto &sparse_tensor) {
  const TensorProto &values = sparse_tensor.values();
  EXT_ENFORCE_INVALID(!values.name().empty(),
                      "SparseTensorProto 'values' must have a non-empty name.");
  VerifyTensor(values);

  const TensorProto &indices = sparse_tensor.indices();
  VerifyTensor(indices);
  EXT_ENFORCE_INVALID(indices.data_type() == TensorProto::INT64, "SparseTensorProto '",
                      values.name(), "' indices must use data_type INT64.");
}

void VerifyAttribute(const AttributeProto &attribute, bool in_function_body,
                     const std::unordered_set<std::string> &scope) {
  EXT_ENFORCE_INVALID(!attribute.name().empty(),
                      "AttributeProto is missing a non-empty 'name' field.");

  if (!attribute.ref_attr_name().empty()) {
    EXT_ENFORCE_INVALID(in_function_body, "AttributeProto '", attribute.name(),
                        "' uses 'ref_attr_name' outside of a function body, which is not allowed.");
    // A reference attribute carries no data of its own; nothing further to check.
    return;
  }

  const int num_set =
      static_cast<int>(attribute.has_f()) + static_cast<int>(attribute.has_i()) +
      static_cast<int>(attribute.has_s()) + static_cast<int>(attribute.has_t()) +
      static_cast<int>(attribute.has_g()) + static_cast<int>(attribute.has_sparse_tensor()) +
      static_cast<int>(attribute.has_tp()) + static_cast<int>(!attribute.floats().empty()) +
      static_cast<int>(!attribute.ints().empty()) + static_cast<int>(!attribute.strings().empty()) +
      static_cast<int>(!attribute.tensors().empty()) +
      static_cast<int>(!attribute.sparse_tensors().empty()) +
      static_cast<int>(!attribute.graphs().empty()) +
      static_cast<int>(!attribute.type_protos().empty());
  EXT_ENFORCE_INVALID(num_set <= 1, "AttributeProto '", attribute.name(),
                      "' must set at most one value field, found ", num_set, ".");

  switch (attribute.type()) {
  case AttributeProto::FLOAT:
    EXT_ENFORCE_INVALID(attribute.has_f(), "AttributeProto '", attribute.name(),
                        "' has type FLOAT but 'f' is not set.");
    break;
  case AttributeProto::INT:
    EXT_ENFORCE_INVALID(attribute.has_i(), "AttributeProto '", attribute.name(),
                        "' has type INT but 'i' is not set.");
    break;
  case AttributeProto::STRING:
    EXT_ENFORCE_INVALID(attribute.has_s(), "AttributeProto '", attribute.name(),
                        "' has type STRING but 's' is not set.");
    break;
  case AttributeProto::TENSOR:
    EXT_ENFORCE_INVALID(attribute.has_t(), "AttributeProto '", attribute.name(),
                        "' has type TENSOR but 't' is not set.");
    VerifyTensor(attribute.t());
    break;
  case AttributeProto::GRAPH:
    EXT_ENFORCE_INVALID(attribute.has_g(), "AttributeProto '", attribute.name(),
                        "' has type GRAPH but 'g' is not set.");
    VerifyGraph(attribute.g(), /*is_main_graph=*/false, in_function_body, &scope);
    break;
  case AttributeProto::SPARSE_TENSOR:
    EXT_ENFORCE_INVALID(attribute.has_sparse_tensor(), "AttributeProto '", attribute.name(),
                        "' has type SPARSE_TENSOR but 'sparse_tensor' is not set.");
    VerifySparseTensor(attribute.sparse_tensor());
    break;
  case AttributeProto::TYPE_PROTO:
    EXT_ENFORCE_INVALID(attribute.has_tp(), "AttributeProto '", attribute.name(),
                        "' has type TYPE_PROTO but 'tp' is not set.");
    break;
  case AttributeProto::FLOATS:
  case AttributeProto::INTS:
  case AttributeProto::STRINGS:
  case AttributeProto::TYPE_PROTOS:
    // An empty repeated value is a valid (if unusual) attribute value.
    break;
  case AttributeProto::TENSORS:
    for (const auto &t : attribute.tensors()) {
      VerifyTensor(t);
    }
    break;
  case AttributeProto::SPARSE_TENSORS:
    for (const auto &t : attribute.sparse_tensors()) {
      VerifySparseTensor(t);
    }
    break;
  case AttributeProto::GRAPHS:
    for (const auto &g : attribute.graphs()) {
      VerifyGraph(g, /*is_main_graph=*/false, in_function_body, &scope);
    }
    break;
  case AttributeProto::UNDEFINED:
  default:
    EXT_THROW_INVALID("AttributeProto '", attribute.name(), "' has an unrecognized or unset type.");
  }
}

void VerifyNode(const NodeProto &node, bool in_function_body,
                const std::unordered_set<std::string> &scope) {
  EXT_ENFORCE_INVALID(!node.op_type().empty(), "NodeProto '", node.name(),
                      "' is missing a non-empty 'op_type'.");
  EXT_ENFORCE_INVALID(!(node.ref_input().empty() && node.ref_output().empty()),
                      "NodeProto (name: ", node.name(), ", op_type: ", node.op_type(),
                      ") has zero input and zero output.");

  std::unordered_set<std::string> seen_attr_names;
  for (const auto &attr : node.attribute()) {
    EXT_ENFORCE_INVALID(!attr.name().empty(), "NodeProto '", node.name(),
                        "' has an attribute without a name.");
    EXT_ENFORCE_INVALID(seen_attr_names.insert(ToStdString(attr.name())).second, "NodeProto '",
                        node.name(), "' has attribute '", attr.name(), "' more than once.");
    VerifyAttribute(attr, in_function_body, scope);
  }
}

void VerifyGraph(const GraphProto &graph, bool is_main_graph, bool in_function_body,
                 const std::unordered_set<std::string> *outer_scope) {
  std::unordered_set<std::string> defined;
  if (outer_scope != nullptr) {
    defined = *outer_scope;
  }

  for (const auto &value_info : graph.input()) {
    VerifyValueInfo(value_info, is_main_graph);
    EXT_ENFORCE_INVALID(defined.insert(ToStdString(value_info.name())).second, "Graph '",
                        graph.name(), "' has input '", value_info.name(),
                        "' defined more than once (SSA violation).");
  }

  std::unordered_set<std::string> initializer_names;
  for (const auto &init : graph.initializer()) {
    EXT_ENFORCE_INVALID(!init.name().empty(), "Graph '", graph.name(),
                        "' has an initializer without a name.");
    EXT_ENFORCE_INVALID(initializer_names.insert(ToStdString(init.name())).second, "Graph '",
                        graph.name(), "' initializer '", init.name(), "' is not unique.");
    VerifyTensor(init);
    defined.insert(ToStdString(init.name()));
  }
  for (const auto &sparse_init : graph.sparse_initializer()) {
    const auto &name = sparse_init.values().name();
    EXT_ENFORCE_INVALID(!name.empty(), "Graph '", graph.name(),
                        "' has a sparse initializer without a name.");
    EXT_ENFORCE_INVALID(initializer_names.insert(ToStdString(name)).second, "Graph '", graph.name(),
                        "' sparse initializer '", name,
                        "' is not unique across initializers and sparse_initializers.");
    VerifySparseTensor(sparse_init);
    defined.insert(ToStdString(name));
  }

  for (const auto &node : graph.node()) {
    for (const auto &input : node.input()) {
      if (input.empty()) {
        continue; // Explicit optional input left unset.
      }
      EXT_ENFORCE_INVALID(defined.count(ToStdString(input)) > 0, "Graph '", graph.name(),
                          "': node '", node.name(), "' (op_type '", node.op_type(), "') consumes '",
                          input, "' before it is produced; nodes must be topologically sorted.");
    }

    VerifyNode(node, in_function_body, defined);

    for (const auto &output : node.output()) {
      if (output.empty()) {
        continue; // Explicit optional output left unset.
      }
      EXT_ENFORCE_INVALID(defined.insert(ToStdString(output)).second, "Graph '", graph.name(),
                          "' output '", output, "' is produced more than once (SSA violation).");
    }
  }

  for (const auto &value_info : graph.output()) {
    VerifyValueInfo(value_info, is_main_graph);
    EXT_ENFORCE_INVALID(defined.count(ToStdString(value_info.name())) > 0, "Graph '", graph.name(),
                        "' output '", value_info.name(),
                        "' is not produced by any node, input, or initializer.");
  }
}

void VerifyFunction(const FunctionProto &function) {
  EXT_ENFORCE_INVALID(!function.name().empty(),
                      "FunctionProto is missing a non-empty 'name' field.");

  std::unordered_set<std::string> defined;
  for (const auto &input : function.input()) {
    EXT_ENFORCE_INVALID(!input.empty(), "FunctionProto '", function.name(),
                        "' has an empty input name.");
    EXT_ENFORCE_INVALID(defined.insert(ToStdString(input)).second, "FunctionProto '",
                        function.name(), "' input '", input, "' is declared more than once.");
  }

  for (const auto &node : function.node()) {
    for (const auto &input : node.input()) {
      if (input.empty()) {
        continue;
      }
      EXT_ENFORCE_INVALID(defined.count(ToStdString(input)) > 0, "FunctionProto '", function.name(),
                          "': node '", node.name(), "' consumes '", input,
                          "' before it is produced; nodes must be topologically sorted.");
    }

    VerifyNode(node, /*in_function_body=*/true, defined);

    for (const auto &output : node.output()) {
      if (output.empty()) {
        continue;
      }
      EXT_ENFORCE_INVALID(defined.insert(ToStdString(output)).second, "FunctionProto '",
                          function.name(), "' output '", output, "' is produced more than once.");
    }
  }

  for (const auto &output : function.output()) {
    EXT_ENFORCE_INVALID(defined.count(ToStdString(output)) > 0, "FunctionProto '", function.name(),
                        "' output '", output,
                        "' is not produced by any node or declared as an input.");
  }
}

void VerifyModel(const ModelProto &model) {
  EXT_ENFORCE_INVALID(model.has_graph(), "ModelProto is missing its 'graph' field.");
  EXT_ENFORCE_INVALID(!model.ref_opset_import().empty(),
                      "ModelProto must import at least one operator set.");

  std::unordered_set<std::string> opset_domains;
  for (const auto &opset : model.opset_import()) {
    EXT_ENFORCE_INVALID(opset_domains.insert(ToStdString(opset.domain())).second, "ModelProto",
                        " imports domain '", opset.domain(), "' more than once.");
  }

  VerifyGraph(model.graph(), /*is_main_graph=*/true);

  std::unordered_set<std::string> function_ids;
  for (const auto &function : model.functions()) {
    VerifyFunction(function);
    const std::string id = onnx_light_helpers::MakeString(function.domain(), "::", function.name(),
                                                          "::", function.overload());
    EXT_ENFORCE_INVALID(function_ids.insert(id).second, "ModelProto declares function '", id,
                        "' more than once.");
  }
}

} // namespace ONNX_LIGHT_NAMESPACE
