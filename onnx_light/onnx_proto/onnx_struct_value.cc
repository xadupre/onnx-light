#include "onnx_struct_value.h"
#include "onnx_helper.h"
#include "onnx_verify.h"

#include <algorithm>
#include <charconv>
#include <limits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

namespace {

/**
 * Raises ``std::invalid_argument`` with the shared onnx-light prefix.
 *
 * Every failure in this file funnels through this out-of-line helper and
 * builds its message with plain string concatenation rather than the
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

bool StructTypeCatalogue::FindField(const StructTypeProto &root, std::string_view path,
                                    EncodedFieldRef &out) const {
  out = EncodedFieldRef();
  return LocateInStruct(*this, root, path, 0, out);
}

} // namespace ONNX_LIGHT_NAMESPACE
