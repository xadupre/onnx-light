#include "onnx_verify.h"
#include "onnx_encoded_value_view.h"
#include "onnx_helper.h"

#include <algorithm>
#include <charconv>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

namespace {

/** Returns @p s converted to a plain std::string, treating an unset optional as empty. */
inline std::string ToStdString(const utils::OptionalString &s) { return std::string(s.sv()); }
/** Returns @p s converted to a plain std::string. */
inline std::string ToStdString(const utils::String &s) { return std::string(s.sv()); }

/**
 * Raises ``std::invalid_argument`` with the shared onnx-light prefix.
 *
 * Structured-type and encoded-value checks use this out-of-line helper and
 * build their messages with plain string concatenation rather than the
 * stringstream-based EXT_ENFORCE_INVALID: dozens of validation branches would
 * otherwise inline one stringstream per call site and dominate the compiled
 * size of ``lib_onnx_proto``, which PR02 keeps within an explicit budget.
 */
[[noreturn]] void Invalid(const std::string &message) {
  throw std::invalid_argument("[onnx-light] " + message);
}

/** Returns "<kind> '<name>' " used as a message prefix. */
std::string Named(const char *kind, const utils::OptionalString &name) {
  return std::string(kind) + " '" + std::string(name.sv()) + "' ";
}

/** Raises when @p condition is false, prefixing the message with the value name. */
void Require(bool condition, const char *kind, const utils::OptionalString &name,
             const char *message) {
  if (!condition) {
    Invalid(Named(kind, name) + message);
  }
}

/** Multiplies with overflow detection. Returns false when the product does not fit. */
bool CheckedMultiply(uint64_t left, uint64_t right, uint64_t &out) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    return false;
  }
  out = left * right;
  return true;
}

/** Adds with overflow detection. Returns false when the sum does not fit. */
bool CheckedAdd(uint64_t left, uint64_t right, uint64_t &out) {
  if (left > std::numeric_limits<uint64_t>::max() - right) {
    return false;
  }
  out = left + right;
  return true;
}

/** Returns ceil(bits / 8) without ever forming bits + 7, which overflows near UINT64_MAX. */
uint64_t CeilBytes(uint64_t bits) { return bits / 8 + (bits % 8 == 0 ? 0 : 1); }

/** Returns true when @p key_type is one of the scalar types ONNX allows as a map key. */
bool IsAllowedMapKeyType(int32_t key_type) {
  switch (static_cast<TensorProto::DataType>(key_type)) {
  case TensorProto::INT8:
  case TensorProto::INT16:
  case TensorProto::INT32:
  case TensorProto::INT64:
  case TensorProto::UINT8:
  case TensorProto::UINT16:
  case TensorProto::UINT32:
  case TensorProto::UINT64:
  case TensorProto::STRING:
    return true;
  default:
    return false;
  }
}

/** Stores a failure cause when the caller asked for one. Always returns false. */
bool Fail(std::string *reason, const char *message) {
  if (reason != nullptr) {
    *reason = message;
  }
  return false;
}

/** Returns true when @p data_type is a floating-point type usable for affine scales. */
bool IsFloatingType(TensorProto::DataType data_type) {
  return data_type == TensorProto::FLOAT || data_type == TensorProto::FLOAT16 ||
         data_type == TensorProto::DOUBLE || data_type == TensorProto::BFLOAT16;
}

/**
 * Returns the number of elements described by a tensor type, requiring a
 * present shape with concrete non-negative dimensions.
 */
bool ConcreteElementCount(const TypeProto::Tensor &tensor, uint64_t &count, std::string *reason) {
  if (!tensor.has_shape()) {
    return Fail(reason, "a tensor leaf needs concrete dimensions");
  }
  count = 1;
  for (const auto &dim : tensor.ref_shape().ref_dim()) {
    if (!dim.has_dim_value() || dim.ref_dim_value() < 0) {
      return Fail(reason, "a tensor leaf needs concrete dimensions");
    }
    if (!CheckedMultiply(count, static_cast<uint64_t>(dim.ref_dim_value()), count)) {
      return Fail(reason, "the element count of a tensor leaf overflows uint64");
    }
  }
  return true;
}

/** Returns the total width in bits of one bit-packing group. */
bool GroupBits(const StructTypeProto::BitPacking &packing, uint64_t &bits, std::string *reason) {
  bits = 0;
  for (const auto &component : packing.ref_component()) {
    if (!CheckedAdd(bits, static_cast<uint64_t>(component.ref_bit_width()), bits)) {
      return Fail(reason, "the width of a bit-packing group overflows uint64");
    }
  }
  return true;
}

/** Role a StructTypeProto plays where it appears, which selects the applicable rules. */
enum class StructRole {
  /** Entry of ModelProto.struct_types. */
  kDeclaration,
  /** Nested inside a TypeProto, where an unset kind is an unconstrained category. */
  kNested,
  /** Layout of an EncodedValueProto, where an unset kind is invalid. */
  kEncodedLayout,
};

/** Memoized outcome of measuring or validating one catalogue identity. */
struct TypeMemo {
  uint64_t type_id = 0;
  /** 0: not measured yet, 1: fixed size known, 2: no fixed inline layout. */
  uint8_t size_state = 0;
  uint64_t bits = 0;
  /** True once the declaration passed structural validation in this walk. */
  bool validated = false;
};

/**
 * Recursion context shared by the size and validation walkers.
 *
 * A catalogue is a directed acyclic graph, so a naive walk revisits shared
 * subtypes once per incoming edge and costs exponential time on a diamond
 * chain. Results are therefore memoized per identity (tri-state, so a
 * "not fixed size" answer is cached too), and the expansion depth is capped so
 * a long reference chain cannot exhaust the stack.
 */
struct Walk {
  const StructTypeCatalogue *catalogue = nullptr;
  /** Identities currently being expanded, used to reject reference cycles. */
  std::vector<uint64_t> active;
  /** Per-identity memo; catalogues are small, so a linear scan beats a hash map. */
  std::vector<TypeMemo> memo;
  /** When false, an unresolvable reference suspends the walk instead of failing. */
  bool require_resolution = true;
  /** Current number of nested type_ref expansions. */
  uint32_t depth = 0;

  /** Returns true when @p identity is already being expanded. */
  bool expanding(uint64_t identity) const {
    return std::find(active.begin(), active.end(), identity) != active.end();
  }

  /** Returns the memo entry for @p identity, creating it on first use. */
  TypeMemo &entry(uint64_t identity) {
    for (auto &item : memo) {
      if (item.type_id == identity) {
        return item;
      }
    }
    memo.push_back(TypeMemo{identity, 0, 0, false});
    return memo.back();
  }
};

/** Scoped depth counter rejecting reference chains deeper than kMaxStructTypeDepth. */
class DepthGuard {
public:
  explicit DepthGuard(Walk &walk) : walk_(walk) { ++walk_.depth; }
  ~DepthGuard() { --walk_.depth; }
  DepthGuard(const DepthGuard &) = delete;
  DepthGuard &operator=(const DepthGuard &) = delete;

  /** Returns true when the current depth is still within the limit. */
  bool valid() const { return walk_.depth <= kMaxStructTypeDepth; }

private:
  Walk &walk_;
};

bool BitSizeOfStruct(Walk &walk, const StructTypeProto &type, uint64_t &bits, std::string *reason);

/** Computes the fixed size in bits of a TypeProto leaf. */
bool BitSizeOfType(Walk &walk, const TypeProto &type, uint64_t &bits, std::string *reason) {
  DepthGuard guard(walk);
  if (!guard.valid()) {
    return Fail(reason, "the type is nested too deeply");
  }
  switch (type.value_case()) {
  case TypeProto::kTensorType: {
    const TypeProto::Tensor &tensor = type.ref_tensor_type();
    if (!tensor.has_elem_type()) {
      return Fail(reason, "a tensor leaf needs an element type");
    }
    const uint32_t width = FixedBitWidth(tensor.elem_type());
    if (width == 0) {
      return Fail(reason, "a tensor leaf needs an element type with a fixed inline width");
    }
    uint64_t count = 0;
    if (!ConcreteElementCount(tensor, count, reason)) {
      return false;
    }
    if (!CheckedMultiply(count, static_cast<uint64_t>(width), bits)) {
      return Fail(reason, "the size of a tensor leaf overflows uint64");
    }
    return true;
  }
  case TypeProto::kStructType:
    return BitSizeOfStruct(walk, type.ref_struct_type(), bits, reason);
  case TypeProto::kSequenceType:
    return Fail(reason, "a sequence field has no fixed inline size");
  case TypeProto::kMapType:
    return Fail(reason, "a map field has no fixed inline size");
  case TypeProto::kOptionalType:
    return Fail(reason, "an optional field has no fixed inline size");
  case TypeProto::kSparseTensorType:
    return Fail(reason, "a sparse tensor field has no fixed inline size");
  case TypeProto::kOpaqueType:
    return Fail(reason, "an opaque field has no fixed inline size");
  default:
    return Fail(reason, "an unset type has no fixed inline size");
  }
}

/** Computes the fixed size in bits of a StructTypeProto. */
bool BitSizeOfStruct(Walk &walk, const StructTypeProto &type, uint64_t &bits, std::string *reason) {
  switch (type.kind_case()) {
  case StructTypeProto::kTypeRef: {
    const uint64_t identity = type.ref_type_ref();
    const StructTypeProto *declaration = walk.catalogue->Find(identity);
    if (declaration == nullptr) {
      return Fail(reason, "a type_ref points at an identity the model does not declare");
    }
    if (walk.expanding(identity)) {
      return Fail(reason, "a type_ref takes part in a reference cycle");
    }
    TypeMemo &memo = walk.entry(identity);
    if (memo.size_state == 1) {
      bits = memo.bits;
      return true;
    }
    if (memo.size_state == 2) {
      return Fail(reason, "a referenced type has no fixed inline layout");
    }
    DepthGuard guard(walk);
    if (!guard.valid()) {
      return Fail(reason, "the type_ref chain is nested too deeply");
    }
    walk.active.push_back(identity);
    const bool ok = BitSizeOfStruct(walk, *declaration, bits, reason);
    walk.active.pop_back();
    TypeMemo &stored = walk.entry(identity);
    stored.size_state = ok ? uint8_t(1) : uint8_t(2);
    stored.bits = ok ? bits : 0;
    return ok;
  }
  case StructTypeProto::kArray: {
    const StructTypeProto::Array &array = type.ref_array();
    if (!array.has_element_type()) {
      return Fail(reason, "an array needs an element type");
    }
    uint64_t element_bits = 0;
    if (!BitSizeOfType(walk, array.ref_element_type(), element_bits, reason)) {
      return false;
    }
    if (!CheckedMultiply(element_bits, array.ref_dimension(), bits)) {
      return Fail(reason, "the size of an array overflows uint64");
    }
    return true;
  }
  case StructTypeProto::kBitPacking: {
    const StructTypeProto::BitPacking &packing = type.ref_bit_packing();
    uint64_t group = 0;
    if (!GroupBits(packing, group, reason)) {
      return false;
    }
    if (!CheckedMultiply(group, packing.ref_dimension(), bits)) {
      return Fail(reason, "the size of a bit packing overflows uint64");
    }
    return true;
  }
  case StructTypeProto::kStructure: {
    bits = 0;
    for (const auto &field : type.ref_structure().ref_field()) {
      if (field.content_case() == StructTypeProto::Structure::Field::kConstant) {
        continue;
      }
      if (field.content_case() != StructTypeProto::Structure::Field::kType) {
        return Fail(reason, "a structure field has no content");
      }
      uint64_t field_bits = 0;
      if (!BitSizeOfType(walk, field.ref_type(), field_bits, reason)) {
        return false;
      }
      if (!CheckedAdd(bits, field_bits, bits)) {
        return Fail(reason, "the size of a structure overflows uint64");
      }
    }
    return true;
  }
  default:
    return Fail(reason, "a StructTypeProto with no kind has no fixed inline layout");
  }
}

void ValidateStructType(Walk &walk, const StructTypeProto &type, StructRole role);

/**
 * Validates a TypeProto and, recursively, every type nested in it.
 *
 * Tensor element types are deliberately not required here: VerifyValueInfo()
 * owns that rule for graph inputs and outputs, and struct declarations add it
 * separately. What this walker owns is the map key contract and reaching every
 * nested ``struct_type``, including the ones hidden inside sequence, map and
 * optional element types, so no reference escapes catalogue resolution.
 */
void ValidateTypeProto(Walk &walk, const TypeProto &type) {
  DepthGuard guard(walk);
  if (!guard.valid()) {
    Invalid("A TypeProto is nested too deeply.");
  }
  switch (type.value_case()) {
  case TypeProto::kTensorType:
    if (!type.ref_tensor_type().has_elem_type() ||
        type.ref_tensor_type().elem_type() == TensorProto::UNDEFINED) {
      Invalid("A tensor type is missing a defined 'elem_type'.");
    }
    break;
  case TypeProto::kSparseTensorType:
    if (!type.ref_sparse_tensor_type().has_elem_type() ||
        type.ref_sparse_tensor_type().elem_type() == TensorProto::UNDEFINED) {
      Invalid("A sparse tensor type is missing a defined 'elem_type'.");
    }
    break;
  case TypeProto::kOpaqueType:
    break;
  case TypeProto::kSequenceType:
    if (!type.ref_sequence_type().has_elem_type()) {
      Invalid("A sequence type is missing its 'elem_type'.");
    }
    ValidateTypeProto(walk, type.ref_sequence_type().ref_elem_type());
    break;
  case TypeProto::kOptionalType:
    if (!type.ref_optional_type().has_elem_type()) {
      Invalid("An optional type is missing its 'elem_type'.");
    }
    ValidateTypeProto(walk, type.ref_optional_type().ref_elem_type());
    break;
  case TypeProto::kMapType:
    if (!IsAllowedMapKeyType(type.ref_map_type().ref_key_type())) {
      Invalid("A map type must use an integral or STRING 'key_type', got " +
              std::to_string(type.ref_map_type().ref_key_type()) + ".");
    }
    if (!type.ref_map_type().has_value_type()) {
      Invalid("A map type is missing its 'value_type'.");
    }
    ValidateTypeProto(walk, type.ref_map_type().ref_value_type());
    break;
  case TypeProto::kStructType:
    ValidateStructType(walk, type.ref_struct_type(), StructRole::kNested);
    break;
  default:
    Invalid("A TypeProto has no value set.");
  }
}

/**
 * Validates a shared format constant: concrete dimensions and a payload holding
 * exactly the declared number of elements.
 *
 * ``VerifyTensor`` only checks that the payload is large enough; a format
 * constant is baked into the type identity, so a short or padded payload has to
 * be rejected outright. Element counts and packed byte counts use checked
 * arithmetic.
 */
void ValidateConstant(const StructTypeProto::Structure::Field &field) {
  const TensorProto &constant = field.ref_constant();
  const char *kind = "Structure field";
  const utils::OptionalString &name = field.name();
  VerifyTensor(constant);
  Require(!constant.has_data_location() || constant.data_location() != TensorProto::EXTERNAL, kind,
          name, "constant must store its data inline.");
  uint64_t elements = 1;
  for (const auto dim : constant.ref_dims()) {
    Require(dim >= 0, kind, name, "constant needs concrete dimensions.");
    Require(CheckedMultiply(elements, static_cast<uint64_t>(dim), elements), kind, name,
            "constant element count overflows uint64.");
  }

  const TensorProto::DataType data_type = constant.data_type();
  uint64_t stored = 0;
  uint64_t expected = elements;
  if (constant.has_raw_data()) {
    const uint32_t width = FixedBitWidth(data_type);
    Require(width != 0, kind, name, "constant cannot store this element type in 'raw_data'.");
    uint64_t bits = 0;
    Require(CheckedMultiply(elements, static_cast<uint64_t>(width), bits), kind, name,
            "constant payload size overflows uint64.");
    stored = constant.ref_raw_data().size();
    expected = CeilBytes(bits);
  } else {
    switch (data_type) {
    case TensorProto::FLOAT:
      stored = constant.ref_float_data().size();
      break;
    case TensorProto::COMPLEX64:
      stored = constant.ref_float_data().size();
      Require(CheckedMultiply(elements, 2, expected), kind, name,
              "constant element count overflows uint64.");
      break;
    case TensorProto::DOUBLE:
      stored = constant.ref_double_data().size();
      break;
    case TensorProto::COMPLEX128:
      stored = constant.ref_double_data().size();
      Require(CheckedMultiply(elements, 2, expected), kind, name,
              "constant element count overflows uint64.");
      break;
    case TensorProto::INT64:
      stored = constant.ref_int64_data().size();
      break;
    case TensorProto::UINT32:
    case TensorProto::UINT64:
      stored = constant.ref_uint64_data().size();
      break;
    case TensorProto::STRING:
      stored = constant.ref_string_data().size();
      break;
    case TensorProto::UINT4:
    case TensorProto::INT4:
    case TensorProto::FLOAT4E2M1:
      // int32_data packs eight 4-bit elements per entry.
      stored = constant.ref_int32_data().size();
      expected = elements / 8 + (elements % 8 == 0 ? 0 : 1);
      break;
    case TensorProto::UINT2:
    case TensorProto::INT2:
      // int32_data packs sixteen 2-bit elements per entry.
      stored = constant.ref_int32_data().size();
      expected = elements / 16 + (elements % 16 == 0 ? 0 : 1);
      break;
    case TensorProto::UNDEFINED:
      Invalid(Named(kind, name) + "constant has data_type UNDEFINED.");
      break;
    default:
      stored = constant.ref_int32_data().size();
      break;
    }
  }
  if (stored != expected) {
    Invalid(Named(kind, name) + "constant declares " + std::to_string(elements) +
            " elements but stores " + std::to_string(stored) + " payload entries instead of " +
            std::to_string(expected) + ".");
  }
}

/** Validates one structure declaration. */
void ValidateStructure(Walk &walk, const StructTypeProto::Structure &structure) {
  if (structure.ref_field().empty()) {
    Invalid("A StructTypeProto structure must declare at least one field.");
  }
  const auto &fields = structure.ref_field();
  for (size_t i = 0; i < fields.size(); ++i) {
    const auto &field = fields[i];
    if (field.name().empty()) {
      Invalid("A structure field is missing a non-empty 'name'.");
    }
    for (size_t j = 0; j < i; ++j) {
      if (fields[j].ref_name() == field.ref_name()) {
        Invalid(Named("Structure field", field.name()) + "is declared more than once.");
      }
    }
    if (field.has_type() && field.has_constant()) {
      Invalid(Named("Structure field", field.name()) + "sets both 'type' and 'constant'.");
    }
    switch (field.content_case()) {
    case StructTypeProto::Structure::Field::kType:
      if (field.ref_type().value_case() == TypeProto::kTensorType &&
          (!field.ref_type().ref_tensor_type().has_elem_type() ||
           field.ref_type().ref_tensor_type().elem_type() == TensorProto::UNDEFINED)) {
        Invalid(Named("Structure field", field.name()) + "needs a defined tensor 'elem_type'.");
      }
      ValidateTypeProto(walk, field.ref_type());
      break;
    case StructTypeProto::Structure::Field::kConstant:
      ValidateConstant(field);
      break;
    default:
      Invalid(Named("Structure field", field.name()) + "must select either 'type' or 'constant'.");
    }
  }
}

/** Validates one bit-packing declaration. */
void ValidateBitPacking(const StructTypeProto::BitPacking &packing) {
  if (packing.ref_component().empty()) {
    Invalid("A StructTypeProto bit packing must declare at least one component.");
  }
  if (packing.ref_dimension() == 0) {
    Invalid("A StructTypeProto bit packing needs a strictly positive dimension.");
  }
  const auto &components = packing.ref_component();
  for (size_t i = 0; i < components.size(); ++i) {
    const auto &component = components[i];
    if (component.name().empty()) {
      Invalid("A bit-packing component is missing a non-empty 'name'.");
    }
    for (size_t j = 0; j < i; ++j) {
      if (components[j].ref_name() == component.ref_name()) {
        Invalid(Named("Bit-packing component", component.name()) + "is declared more than once.");
      }
    }
    if (component.ref_bit_width() == 0 || component.ref_bit_width() > 64) {
      Invalid(Named("Bit-packing component", component.name()) +
              "needs a 'bit_width' between 1 and 64.");
    }
  }
}

/** Validates a StructTypeProto for the role it plays where it appears. */
void ValidateStructType(Walk &walk, const StructTypeProto &type, StructRole role) {
  const bool is_declaration = role == StructRole::kDeclaration;
  if (type.kind_case() == StructTypeProto::kTypeRef) {
    const uint64_t identity = type.ref_type_ref();
    if (is_declaration) {
      Invalid("A catalogue entry must declare a kind, not a bare type_ref.");
    }
    if (identity == 0) {
      Invalid("A StructTypeProto type_ref must be a nonzero identity.");
    }
    if (type.has_type_id() || type.has_decoder() || type.has_encoder() || type.has_name() ||
        type.has_doc_string() || !type.ref_metadata_props().empty()) {
      Invalid("A StructTypeProto type_ref carries only the referenced ID; declaration fields, "
              "metadata and codecs must be absent.");
    }
    const StructTypeProto *declaration = walk.catalogue->Find(identity);
    if (declaration == nullptr) {
      if (walk.require_resolution) {
        Invalid("A StructTypeProto references type_id " + std::to_string(identity) +
                " which the model does not declare.");
      }
      return;
    }
    if (walk.expanding(identity)) {
      Invalid("StructTypeProto type_id " + std::to_string(identity) +
              " takes part in a reference cycle.");
    }
    if (walk.entry(identity).validated) {
      return;
    }
    DepthGuard guard(walk);
    if (!guard.valid()) {
      Invalid("StructTypeProto type_ref chain starting at type_id " + std::to_string(identity) +
              " exceeds the maximum nesting depth of " + std::to_string(kMaxStructTypeDepth) + ".");
    }
    walk.active.push_back(identity);
    ValidateStructType(walk, *declaration, StructRole::kDeclaration);
    walk.active.pop_back();
    walk.entry(identity).validated = true;
    return;
  }

  if (is_declaration) {
    if (!type.has_type_id() || type.ref_type_id() == 0) {
      Invalid("A catalogue entry needs a nonzero 'type_id'.");
    }
  } else if (type.has_type_id()) {
    Invalid("An inline StructTypeProto declaration must not carry a 'type_id'; only catalogue "
            "entries own an identity.");
  }

  switch (type.kind_case()) {
  case StructTypeProto::kArray:
    if (!type.ref_array().has_element_type()) {
      Invalid("A StructTypeProto array is missing its 'element_type'.");
    }
    if (type.ref_array().ref_dimension() == 0) {
      Invalid("A StructTypeProto array needs a strictly positive dimension.");
    }
    ValidateTypeProto(walk, type.ref_array().ref_element_type());
    break;
  case StructTypeProto::kStructure:
    ValidateStructure(walk, type.ref_structure());
    break;
  case StructTypeProto::kBitPacking:
    ValidateBitPacking(type.ref_bit_packing());
    break;
  default:
    if (role != StructRole::kNested) {
      Invalid("A StructTypeProto with no kind is an unconstrained category, permitted only "
              "inside a TypeProto, never as a declaration or a payload layout.");
    }
    if (type.has_decoder() || type.has_encoder()) {
      Invalid("An unconstrained StructTypeProto must not carry a decoder or encoder.");
    }
    break;
  }
}

/** Returns the value of an external_data entry, or null when the key is absent. */
const utils::OptionalString *FindExternalEntry(const EncodedValueProto &value, const char *key) {
  for (const auto &entry : value.ref_external_data()) {
    if (entry.ref_key() == key) {
      return &entry.ref_value();
    }
  }
  return nullptr;
}

/** Rejects empty and repeated external_data keys, which make the metadata ambiguous. */
void ValidateExternalKeys(const EncodedValueProto &value) {
  const utils::RepeatedProtoField<StringStringEntryProto> &entries = value.ref_external_data();
  for (size_t i = 0; i < entries.size(); ++i) {
    Require(entries[i].has_key() && !entries[i].key().empty(), "EncodedValueProto", value.name(),
            "has an external_data entry without a key.");
    for (size_t j = 0; j < i; ++j) {
      if (entries[j].ref_key() == entries[i].ref_key()) {
        Invalid(Named("EncodedValueProto", value.name()) + "repeats the external_data key '" +
                std::string(entries[i].key().sv()) + "'.");
      }
    }
  }
}

/** Parses a non-negative integer stored as a string in external_data. */
bool ParseExternalInteger(const utils::OptionalString *text, uint64_t &out) {
  if (text == nullptr) {
    return false;
  }
  const char *begin = text->data();
  const char *end = begin + text->size();
  if (begin == end) {
    return false;
  }
  const auto parsed = std::from_chars(begin, end, out);
  return parsed.ec == std::errc() && parsed.ptr == end;
}

/** Validates the payload location of an encoded value and returns its byte extent. */
uint64_t ValidatePayloadLocation(const EncodedValueProto &value, bool &external) {
  const char *kind = "EncodedValueProto";
  const bool has_raw = value.has_raw_data();
  const bool has_external = !value.ref_external_data().empty();
  Require(!(has_raw && has_external), kind, value.name(),
          "sets both 'raw_data' and 'external_data', which are mutually exclusive.");
  external = value.has_data_location() && value.data_location() == TensorProto::EXTERNAL;
  if (!external) {
    Require(!has_external, kind, value.name(),
            "carries 'external_data' without data_location EXTERNAL.");
    return value.ref_raw_data().size();
  }
  Require(!has_raw, kind, value.name(), "is stored externally but also carries an inline payload.");
  ValidateExternalKeys(value);
  const utils::OptionalString *location = FindExternalEntry(value, "location");
  Require(location != nullptr && !location->empty(), kind, value.name(),
          "is stored externally but is missing a non-empty 'location' entry.");
  Require(IsSafeExternalDataLocation(location->sv()), kind, value.name(),
          "external 'location' must be a relative path that cannot escape the model directory.");
  uint64_t bytes = 0;
  Require(ParseExternalInteger(FindExternalEntry(value, "length"), bytes), kind, value.name(),
          "is stored externally but is missing a numeric explicit 'length' entry.");
  const utils::OptionalString *offset = FindExternalEntry(value, "offset");
  if (offset != nullptr) {
    uint64_t parsed_offset = 0;
    Require(ParseExternalInteger(offset, parsed_offset), kind, value.name(),
            "has a non-numeric external 'offset' entry.");
    uint64_t end = 0;
    Require(CheckedAdd(parsed_offset, bytes, end), kind, value.name(),
            "external 'offset' plus 'length' overflows uint64.");
    Require(end <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), kind, value.name(),
            "external 'offset' plus 'length' exceeds the addressable file range.");
  }
  return bytes;
}

/** Returns the dimensions of a tensor type, requiring concrete non-negative values. */
std::vector<uint64_t> ConcreteDims(const TypeProto::Tensor &tensor, const char *kind,
                                   const utils::OptionalString &name, const char *what) {
  std::vector<uint64_t> dims;
  Require(tensor.has_shape(), kind, name, what);
  for (const auto &dim : tensor.ref_shape().ref_dim()) {
    Require(dim.has_dim_value() && dim.ref_dim_value() >= 0, kind, name, what);
    dims.push_back(static_cast<uint64_t>(dim.ref_dim_value()));
  }
  return dims;
}

/** Returns the dimensions of a parameter tensor, requiring non-negative values. */
std::vector<uint64_t> ParameterDims(const TensorProto &tensor, const char *kind,
                                    const utils::OptionalString &name, const char *what) {
  std::vector<uint64_t> dims;
  for (const auto dim : tensor.ref_dims()) {
    Require(dim >= 0, kind, name, what);
    dims.push_back(static_cast<uint64_t>(dim));
  }
  return dims;
}

/** Renders a dimension list for an error message. */
std::string DescribeDims(const std::vector<uint64_t> &dims) {
  std::string text = "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) {
      text += ", ";
    }
    text += std::to_string(dims[i]);
  }
  return text + "]";
}

/**
 * Checks the affine parameter geometry against the QuantizeLinear /
 * DequantizeLinear contract: a scalar per-tensor scale, a one-dimensional
 * per-axis scale whose length matches the quantized axis, or a blocked scale
 * with the logical rank whose axis length is ceil(dim / block_size).
 */
void ValidateAffineParameterShape(const AffineLayoutProto &affine,
                                  const std::vector<uint64_t> &logical_dims,
                                  const std::vector<uint64_t> &scale_dims, const char *kind,
                                  const utils::OptionalString &name) {
  if (!affine.has_axis()) {
    Require(scale_dims.empty(), kind, name,
            "per-tensor affine quantization needs a scalar 'scale'.");
    return;
  }

  const int64_t rank = static_cast<int64_t>(logical_dims.size());
  const int64_t declared_axis = affine.ref_axis();
  Require(rank > 0 && declared_axis >= -rank && declared_axis < rank, kind, name,
          "affine 'axis' is out of range for the logical rank.");
  const size_t axis = static_cast<size_t>(declared_axis < 0 ? declared_axis + rank : declared_axis);

  if (!affine.has_block_size()) {
    Require(scale_dims.size() == 1, kind, name,
            "per-axis affine quantization needs a one-dimensional 'scale'.");
    if (scale_dims[0] != logical_dims[axis]) {
      Invalid(Named(kind, name) + "per-axis affine 'scale' must hold " +
              std::to_string(logical_dims[axis]) + " values, got " + std::to_string(scale_dims[0]) +
              ".");
    }
    return;
  }

  const uint64_t block_size = affine.ref_block_size();
  Require(scale_dims.size() == logical_dims.size(), kind, name,
          "blocked affine 'scale' must have the same rank as the logical type.");
  const uint64_t blocks =
      logical_dims[axis] / block_size + (logical_dims[axis] % block_size == 0 ? 0 : 1);
  for (size_t i = 0; i < logical_dims.size(); ++i) {
    const uint64_t expected = i == axis ? blocks : logical_dims[i];
    if (scale_dims[i] != expected) {
      Invalid(Named(kind, name) + "blocked affine 'scale' shape " + DescribeDims(scale_dims) +
              " does not match the expected block geometry for logical shape " +
              DescribeDims(logical_dims) + " with block_size " + std::to_string(block_size) + ".");
    }
  }
}

/** Requires an affine parameter payload to contain exactly its declared values. */
void ValidateAffineParameterPayload(const TensorProto &tensor, const char *kind,
                                    const utils::OptionalString &name, const char *what) {
  uint64_t elements = 1;
  for (const int64_t dim : tensor.ref_dims()) {
    Require(dim >= 0 && CheckedMultiply(elements, static_cast<uint64_t>(dim), elements), kind, name,
            "affine parameter element count is invalid or overflows uint64.");
  }
  uint64_t expected = elements;
  uint64_t stored = 0;
  if (tensor.has_raw_data()) {
    uint64_t bits = 0;
    Require(CheckedMultiply(elements, FixedBitWidth(tensor.data_type()), bits), kind, name,
            "affine parameter payload size overflows uint64.");
    expected = CeilBytes(bits);
    stored = tensor.ref_raw_data().size();
  } else {
    switch (tensor.data_type()) {
    case TensorProto::FLOAT:
      stored = tensor.ref_float_data().size();
      break;
    case TensorProto::DOUBLE:
      stored = tensor.ref_double_data().size();
      break;
    case TensorProto::UINT4:
    case TensorProto::INT4:
      stored = tensor.ref_int32_data().size();
      expected = elements / 8 + (elements % 8 == 0 ? 0 : 1);
      break;
    default:
      stored = tensor.ref_int32_data().size();
      break;
    }
  }
  Require(stored == expected, kind, name, what);
}

/** Validates the frozen affine branch and fills the layout. */
void ValidateAffineLayout(const EncodedValueProto &value, EncodedValueLayout &layout) {
  const char *kind = "EncodedValueProto";
  const utils::OptionalString &name = value.name();
  const AffineLayoutProto &affine = value.ref_affine();
  layout.affine = &affine;
  Require(affine.has_storage_type(), kind, name, "affine layout is missing its 'storage_type'.");
  const TensorProto::DataType storage = affine.storage_type();
  Require(IsAffineStorageType(storage), kind, name,
          "affine 'storage_type' must be INT8, UINT8, INT4 or UINT4.");
  Require(affine.has_scale(), kind, name, "affine layout is missing its 'scale'.");
  VerifyTensor(affine.ref_scale());
  Require(IsFloatingType(affine.ref_scale().data_type()), kind, name,
          "affine 'scale' must have a floating element type.");
  ValidateAffineParameterPayload(affine.ref_scale(), kind, name,
                                 "affine 'scale' payload does not match its declared shape.");
  if (affine.has_block_size()) {
    Require(affine.has_axis(), kind, name,
            "affine 'block_size' is valid only together with an 'axis'.");
    Require(affine.ref_block_size() > 0, kind, name,
            "affine 'block_size' must be strictly positive.");
  }

  Require(value.has_logical_type(), kind, name, "affine layout requires a 'logical_type'.");
  Require(value.ref_logical_type().value_case() == TypeProto::kTensorType, kind, name,
          "affine 'logical_type' must be a tensor type.");
  const TypeProto::Tensor &logical = value.ref_logical_type().ref_tensor_type();
  Require(logical.has_elem_type() && logical.elem_type() != TensorProto::UNDEFINED, kind, name,
          "affine 'logical_type' is missing its element type.");
  const std::vector<uint64_t> logical_dims =
      ConcreteDims(logical, kind, name, "affine 'logical_type' needs concrete dimensions.");
  uint64_t elements = 1;
  for (const uint64_t dim : logical_dims) {
    Require(CheckedMultiply(elements, dim, elements), kind, name,
            "affine logical element count overflows uint64.");
  }

  const std::vector<uint64_t> scale_dims = ParameterDims(
      affine.ref_scale(), kind, name, "affine 'scale' needs non-negative dimensions.");
  ValidateAffineParameterShape(affine, logical_dims, scale_dims, kind, name);
  if (affine.has_zero_point()) {
    VerifyTensor(affine.ref_zero_point());
    Require(affine.ref_zero_point().data_type() == storage, kind, name,
            "affine 'zero_point' must use 'storage_type'.");
    ValidateAffineParameterPayload(
        affine.ref_zero_point(), kind, name,
        "affine 'zero_point' payload does not match its declared shape.");
    const std::vector<uint64_t> zero_point_dims = ParameterDims(
        affine.ref_zero_point(), kind, name, "affine 'zero_point' needs non-negative dimensions.");
    if (zero_point_dims != scale_dims) {
      Invalid(Named(kind, name) + "affine 'zero_point' shape " + DescribeDims(zero_point_dims) +
              " must match the 'scale' shape " + DescribeDims(scale_dims) + ".");
    }
  }

  const uint32_t width = FixedBitWidth(storage);
  uint64_t bits = 0;
  Require(CheckedMultiply(elements, static_cast<uint64_t>(width), bits), kind, name,
          "affine code payload size overflows uint64.");
  // CeilBytes never forms bits + 7, which wraps for a payload near UINT64_MAX.
  const uint64_t expected = CeilBytes(bits);
  if (layout.payload_bytes != expected) {
    Invalid(Named(kind, name) + "affine payload must hold exactly " + std::to_string(expected) +
            " code bytes, got " + std::to_string(layout.payload_bytes) + ".");
  }
  if (!layout.external && width == 4 && elements % 2 == 1 && expected > 0) {
    Require((value.ref_raw_data()[expected - 1] & 0xF0U) == 0, kind, name,
            "leaves a non-zero unused high nibble in its packed 4-bit payload.");
  }
  layout.element_bits = width;
  layout.record_count = elements;
}

/** Splits @p path on the first dot and returns the leading segment. */
std::string_view SplitPath(std::string_view path, std::string_view &rest) {
  const size_t dot = path.find('.');
  if (dot == std::string_view::npos) {
    rest = std::string_view();
    return path;
  }
  rest = path.substr(dot + 1);
  return path.substr(0, dot);
}

bool LocateInStruct(const StructTypeCatalogue &catalogue, const StructTypeProto &type,
                    std::string_view path, uint64_t base_bits, EncodedFieldRef &out);

/** Describes a fixed-width leaf reachable through a TypeProto. */
bool LocateInType(const StructTypeCatalogue &catalogue, const TypeProto &type,
                  std::string_view path, uint64_t base_bits, EncodedFieldRef &out) {
  if (type.value_case() == TypeProto::kStructType) {
    return LocateInStruct(catalogue, type.ref_struct_type(), path, base_bits, out);
  }
  if (!path.empty() || type.value_case() != TypeProto::kTensorType) {
    return false;
  }
  const TypeProto::Tensor &tensor = type.ref_tensor_type();
  if (!tensor.has_elem_type()) {
    return false;
  }
  const uint32_t width = FixedBitWidth(tensor.elem_type());
  uint64_t count = 0;
  if (width == 0 || !ConcreteElementCount(tensor, count, nullptr)) {
    return false;
  }
  out.bit_offset = base_bits;
  out.bit_stride = width;
  out.bit_width = width;
  out.count = count;
  out.elem_type = tensor.elem_type();
  return true;
}

/** Walks @p path inside @p type, accumulating bit offsets in declaration order. */
bool LocateInStruct(const StructTypeCatalogue &catalogue, const StructTypeProto &type,
                    std::string_view path, uint64_t base_bits, EncodedFieldRef &out) {
  const StructTypeProto &resolved = catalogue.Resolve(type);
  switch (resolved.kind_case()) {
  case StructTypeProto::kStructure: {
    std::string_view rest;
    const std::string_view head = SplitPath(path, rest);
    if (head.empty()) {
      return false;
    }
    Walk walk;
    walk.catalogue = &catalogue;
    uint64_t offset = base_bits;
    for (const auto &field : resolved.ref_structure().ref_field()) {
      if (field.name().sv() == head) {
        if (field.content_case() != StructTypeProto::Structure::Field::kType) {
          return false;
        }
        return LocateInType(catalogue, field.ref_type(), rest, offset, out);
      }
      if (field.content_case() == StructTypeProto::Structure::Field::kConstant) {
        continue;
      }
      uint64_t field_bits = 0;
      if (!BitSizeOfType(walk, field.ref_type(), field_bits, nullptr) ||
          !CheckedAdd(offset, field_bits, offset)) {
        return false;
      }
    }
    return false;
  }
  case StructTypeProto::kBitPacking: {
    std::string_view rest;
    const std::string_view head = SplitPath(path, rest);
    if (head.empty() || !rest.empty()) {
      return false;
    }
    const StructTypeProto::BitPacking &packing = resolved.ref_bit_packing();
    uint64_t group = 0;
    if (!GroupBits(packing, group, nullptr)) {
      return false;
    }
    uint64_t offset = 0;
    for (const auto &component : packing.ref_component()) {
      if (component.name().sv() == head) {
        out.bit_offset = base_bits + offset;
        out.bit_stride = group;
        out.bit_width = component.ref_bit_width();
        out.count = packing.ref_dimension();
        out.elem_type = TensorProto::UNDEFINED;
        return true;
      }
      offset += component.ref_bit_width();
    }
    return false;
  }
  case StructTypeProto::kArray: {
    if (!path.empty() || !resolved.ref_array().has_element_type()) {
      return false;
    }
    EncodedFieldRef element;
    if (!LocateInType(catalogue, resolved.ref_array().ref_element_type(), std::string_view(),
                      base_bits, element)) {
      return false;
    }
    uint64_t total = 0;
    if (!CheckedMultiply(element.count, resolved.ref_array().ref_dimension(), total)) {
      return false;
    }
    out = element;
    out.count = total;
    return true;
  }
  default:
    return false;
  }
}

} // namespace

void StructTypeCatalogue::Build(const ModelProto &model) {
  // Validate through a probe catalogue so a rejected model never leaves this
  // catalogue bound to declarations that failed validation.
  model_ = nullptr;
  StructTypeCatalogue probe;
  probe.model_ = &model;
  const utils::RepeatedProtoField<StructTypeProto> &declarations = model.ref_struct_types();
  for (size_t i = 0; i < declarations.size(); ++i) {
    const StructTypeProto &declaration = declarations[i];
    if (!declaration.has_type_id() || declaration.ref_type_id() == 0) {
      Invalid(Named("ModelProto.struct_types entry", declaration.name()) +
              "needs a nonzero 'type_id'.");
    }
    for (size_t j = 0; j < i; ++j) {
      if (declarations[j].has_type_id() &&
          declarations[j].ref_type_id() == declaration.ref_type_id()) {
        Invalid("ModelProto declares type_id " + std::to_string(declaration.ref_type_id()) +
                " more than once.");
      }
    }
  }
  Walk walk;
  walk.catalogue = &probe;
  for (size_t i = 0; i < declarations.size(); ++i) {
    walk.active.assign(1, declarations[i].ref_type_id());
    ValidateStructType(walk, declarations[i], StructRole::kDeclaration);
    walk.entry(declarations[i].ref_type_id()).validated = true;
  }
  model_ = &model;
}

bool StructTypeCatalogue::FixedBitSize(const StructTypeProto &type, uint64_t &bits,
                                       std::string *reason) const {
  Walk walk;
  walk.catalogue = this;
  bits = 0;
  return BitSizeOfStruct(walk, type, bits, reason);
}

void StructTypeCatalogue::ValidateType(const TypeProto &type) const {
  Walk walk;
  walk.catalogue = this;
  ValidateTypeProto(walk, type);
}

EncodedValueLayout
StructTypeCatalogue::ValidateEncodedValue(const EncodedValueProto &value,
                                          bool require_resolved_reference) const {
  const char *kind = "EncodedValueProto";
  EncodedValueLayout layout;
  Require(value.has_layout(), kind, value.name(),
          "must select either an affine or a structured layout.");
  layout.payload_bytes = ValidatePayloadLocation(value, layout.external);
  // Only an inline payload lets the bytes themselves be checked; an external
  // payload is validated as metadata alone (see EncodedValueLayout).
  layout.content_verified = !layout.external;

  if (value.layout_case() == EncodedValueProto::kAffine) {
    ValidateAffineLayout(value, layout);
    return layout;
  }

  const StructTypeProto &declared = value.ref_struct_type();
  Walk walk;
  walk.catalogue = this;
  walk.require_resolution = require_resolved_reference;
  ValidateStructType(walk, declared, StructRole::kEncodedLayout);
  if (!require_resolved_reference && declared.kind_case() == StructTypeProto::kTypeRef &&
      Find(declared.ref_type_ref()) == nullptr) {
    layout.unresolved_reference = true;
    return layout;
  }
  const StructTypeProto &root = Resolve(declared);
  layout.root = &root;
  if (value.has_logical_type()) {
    ValidateType(value.ref_logical_type());
  }

  std::string reason;
  uint64_t element_bytes = 0;
  if (!IsByteEncodable(root, element_bytes, &reason)) {
    Invalid(Named(kind, value.name()) + "cannot use a flat encoded layout: " + reason + ".");
  }
  layout.element_bits = element_bytes * 8;
  if (layout.payload_bytes % element_bytes != 0) {
    Invalid(Named(kind, value.name()) + "payload of " + std::to_string(layout.payload_bytes) +
            " bytes is not a whole number of " + std::to_string(element_bytes) + "-byte records.");
  }
  layout.record_count = layout.payload_bytes / element_bytes;
  return layout;
}

bool FindEncodedField(const StructTypeCatalogue &catalogue, const StructTypeProto &root,
                      std::string_view path, EncodedFieldRef &out) {
  out = EncodedFieldRef();
  return LocateInStruct(catalogue, root, path, 0, out);
}

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
  case TypeProto::kStructType: {
    // A structured value may be an unconstrained category, a reference or an
    // inline declaration; only an identity claim is rejected here because a
    // value type never owns a catalogue entry.
    const StructTypeProto &struct_type = type.struct_type();
    EXT_ENFORCE_INVALID(!struct_type.has_type_id(), "ValueInfoProto '", value_info.name(),
                        "' struct_type must not carry a 'type_id'; only ModelProto.struct_types "
                        "entries own an identity.");
    EXT_ENFORCE_INVALID(struct_type.kind_case() != StructTypeProto::kTypeRef ||
                            struct_type.ref_type_ref() != 0,
                        "ValueInfoProto '", value_info.name(),
                        "' struct_type uses type_ref 0, which is never a valid identity.");
    break;
  }
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
  VerifyAttribute(/*struct_types=*/nullptr, attribute, in_function_body, scope);
}

void VerifyAttribute(const StructTypeCatalogue *struct_types, const AttributeProto &attribute,
                     bool in_function_body, const std::unordered_set<std::string> &scope) {
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
    VerifyGraph(struct_types, attribute.g(), /*is_main_graph=*/false, in_function_body, &scope);
    break;
  case AttributeProto::SPARSE_TENSOR:
    EXT_ENFORCE_INVALID(attribute.has_sparse_tensor(), "AttributeProto '", attribute.name(),
                        "' has type SPARSE_TENSOR but 'sparse_tensor' is not set.");
    VerifySparseTensor(attribute.sparse_tensor());
    break;
  case AttributeProto::TYPE_PROTO:
    EXT_ENFORCE_INVALID(attribute.has_tp(), "AttributeProto '", attribute.name(),
                        "' has type TYPE_PROTO but 'tp' is not set.");
    if (struct_types != nullptr) {
      struct_types->ValidateType(attribute.tp());
    }
    break;
  case AttributeProto::FLOATS:
  case AttributeProto::INTS:
  case AttributeProto::STRINGS:
    // An empty repeated value is a valid (if unusual) attribute value.
    break;
  case AttributeProto::TYPE_PROTOS:
    if (struct_types != nullptr) {
      for (const auto &type : attribute.type_protos()) {
        struct_types->ValidateType(type);
      }
    }
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
      VerifyGraph(struct_types, g, /*is_main_graph=*/false, in_function_body, &scope);
    }
    break;
  case AttributeProto::UNDEFINED:
  default:
    EXT_THROW_INVALID("AttributeProto '", attribute.name(), "' has an unrecognized or unset type.");
  }
}

void VerifyNode(const NodeProto &node, bool in_function_body,
                const std::unordered_set<std::string> &scope) {
  VerifyNode(/*struct_types=*/nullptr, node, in_function_body, scope);
}

void VerifyNode(const StructTypeCatalogue *struct_types, const NodeProto &node,
                bool in_function_body, const std::unordered_set<std::string> &scope) {
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
    VerifyAttribute(struct_types, attr, in_function_body, scope);
  }
}

void VerifyGraph(const GraphProto &graph, bool is_main_graph, bool in_function_body,
                 const std::unordered_set<std::string> *outer_scope) {
  VerifyGraph(/*struct_types=*/nullptr, graph, is_main_graph, in_function_body, outer_scope);
}

void VerifyGraph(const StructTypeCatalogue *struct_types, const GraphProto &graph,
                 bool is_main_graph, bool in_function_body,
                 const std::unordered_set<std::string> *outer_scope) {
  const StructTypeCatalogue empty_catalogue;
  const StructTypeCatalogue &catalogue = struct_types == nullptr ? empty_catalogue : *struct_types;
  std::unordered_set<std::string> defined;
  if (outer_scope != nullptr) {
    defined = *outer_scope;
  }

  const auto verify_value_type = [struct_types](const ValueInfoProto &value_info) {
    if (struct_types != nullptr && value_info.has_type()) {
      struct_types->ValidateType(value_info.type());
    }
  };

  for (const auto &value_info : graph.input()) {
    VerifyValueInfo(value_info, is_main_graph);
    verify_value_type(value_info);
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
  for (const auto &encoded : graph.encoded_initializer()) {
    EXT_ENFORCE_INVALID(!encoded.name().empty(), "Graph '", graph.name(),
                        "' has an encoded initializer without a name.");
    EXT_ENFORCE_INVALID(initializer_names.insert(ToStdString(encoded.name())).second, "Graph '",
                        graph.name(), "' encoded initializer '", encoded.name(),
                        "' is not unique across initializers, sparse_initializers and "
                        "encoded_initializers.");
    catalogue.ValidateEncodedValue(encoded, /*require_resolved_reference=*/struct_types != nullptr);
    defined.insert(ToStdString(encoded.name()));
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

    VerifyNode(struct_types, node, in_function_body, defined);

    for (const auto &output : node.output()) {
      if (output.empty()) {
        continue; // Explicit optional output left unset.
      }
      EXT_ENFORCE_INVALID(defined.insert(ToStdString(output)).second, "Graph '", graph.name(),
                          "' output '", output, "' is produced more than once (SSA violation).");
    }
  }

  for (const auto &value_info : graph.value_info()) {
    verify_value_type(value_info);
  }

  for (const auto &value_info : graph.output()) {
    VerifyValueInfo(value_info, is_main_graph);
    verify_value_type(value_info);
    EXT_ENFORCE_INVALID(defined.count(ToStdString(value_info.name())) > 0, "Graph '", graph.name(),
                        "' output '", value_info.name(),
                        "' is not produced by any node, input, or initializer.");
  }
}

void VerifyFunction(const FunctionProto &function) {
  VerifyFunction(/*struct_types=*/nullptr, function);
}

void VerifyFunction(const StructTypeCatalogue *struct_types, const FunctionProto &function) {
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

    VerifyNode(struct_types, node, /*in_function_body=*/true, defined);

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

  if (struct_types != nullptr) {
    for (const auto &value_info : function.value_info()) {
      if (value_info.has_type()) {
        struct_types->ValidateType(value_info.type());
      }
    }
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

  StructTypeCatalogue struct_types;
  struct_types.Build(model);

  VerifyGraph(&struct_types, model.graph(), /*is_main_graph=*/true,
              /*in_function_body=*/false, /*outer_scope=*/nullptr);

  std::unordered_set<std::string> function_ids;
  for (const auto &function : model.functions()) {
    VerifyFunction(&struct_types, function);
    const std::string id = onnx_light_helpers::MakeString(function.domain(), "::", function.name(),
                                                          "::", function.overload());
    EXT_ENFORCE_INVALID(function_ids.insert(id).second, "ModelProto declares function '", id,
                        "' more than once.");
  }
}

} // namespace ONNX_LIGHT_NAMESPACE
