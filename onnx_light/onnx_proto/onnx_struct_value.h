#pragma once

#include "onnx.h"

#include <cstring>
#include <limits>
#include <string>
#include <string_view>

/**
 * @file onnx_struct_value.h
 * @brief Catalogue resolution, byte-layout arithmetic and payload access for
 *        ``StructTypeProto`` / ``EncodedValueProto``.
 *
 * The helpers declared here answer three questions and nothing more:
 *
 * - which declaration does a ``type_id`` denote (model-scoped catalogue,
 *   unique nonzero identities, no cycles);
 * - what is the fixed physical size of a struct, and is it byte-encodable at
 *   all (checked ``uint64_t`` arithmetic, explicit padding, no hidden
 *   alignment);
 * - where does a leaf live inside one encoded record, and how are its bits
 *   read back with bounds checks.
 *
 * Format-specific decoders, codec execution and registries deliberately stay
 * outside ``lib_onnx_proto``: loading or validating a model never runs codec
 * code. Validation lives in the library because ``VerifyModel`` needs it; the
 * payload reader stays header-only because it is pure, allocation-free
 * arithmetic over an already validated value.
 *
 * Every function raises ``std::invalid_argument`` (through
 * ``EXT_THROW_INVALID`` / ``EXT_ENFORCE_INVALID``) on the first violation,
 * except the ones documented as returning ``false``.
 */

namespace ONNX_LIGHT_NAMESPACE {

/**
 * Returns the physical width in bits of a fixed-width ONNX element type, or 0
 * when the type has no fixed inline width (``UNDEFINED``, ``STRING`` or an
 * unknown value).
 */
inline constexpr uint32_t FixedBitWidth(TensorProto::DataType data_type) {
  switch (data_type) {
  case TensorProto::UINT2:
  case TensorProto::INT2:
    return 2;
  case TensorProto::UINT4:
  case TensorProto::INT4:
  case TensorProto::FLOAT4E2M1:
    return 4;
  case TensorProto::FLOAT6E2M3:
  case TensorProto::FLOAT6E3M2:
    return 6;
  case TensorProto::UINT8:
  case TensorProto::INT8:
  case TensorProto::BOOL:
  case TensorProto::FLOAT8E4M3FN:
  case TensorProto::FLOAT8E4M3FNUZ:
  case TensorProto::FLOAT8E5M2:
  case TensorProto::FLOAT8E5M2FNUZ:
  case TensorProto::FLOAT8E8M0:
    return 8;
  case TensorProto::UINT16:
  case TensorProto::INT16:
  case TensorProto::FLOAT16:
  case TensorProto::BFLOAT16:
    return 16;
  case TensorProto::UINT32:
  case TensorProto::INT32:
  case TensorProto::FLOAT:
    return 32;
  case TensorProto::UINT64:
  case TensorProto::INT64:
  case TensorProto::DOUBLE:
  case TensorProto::COMPLEX64:
    return 64;
  case TensorProto::COMPLEX128:
    return 128;
  default:
    return 0;
  }
}

/** Returns true when @p data_type is one of the affine storage types frozen by PR01. */
inline constexpr bool IsAffineStorageType(TensorProto::DataType data_type) {
  return data_type == TensorProto::INT8 || data_type == TensorProto::UINT8 ||
         data_type == TensorProto::INT4 || data_type == TensorProto::UINT4;
}

/** Locates one physical leaf inside a byte-encoded record. */
struct EncodedFieldRef {
  /** Bit offset of element 0 relative to the start of the record. */
  uint64_t bit_offset = 0;
  /** Distance in bits between two consecutive elements. */
  uint64_t bit_stride = 0;
  /** Number of addressable elements. */
  uint64_t count = 0;
  /** Width in bits of a single element. */
  uint32_t bit_width = 0;
  /** Element type, ``UNDEFINED`` for a bit-packing component. */
  TensorProto::DataType elem_type = TensorProto::UNDEFINED;
};

/** Describes the validated payload geometry of one ``EncodedValueProto``. */
struct EncodedValueLayout {
  /** Resolved structured root, or null for the affine branch. */
  const StructTypeProto *root = nullptr;
  /** Affine descriptor, or null for the structured branch. */
  const AffineLayoutProto *affine = nullptr;
  /** Size of one record in bits: the element size, or the storage width for the affine branch. */
  uint64_t element_bits = 0;
  /** Byte extent of the payload, inline or declared by the external metadata. */
  uint64_t payload_bytes = 0;
  /** Number of records derived from the payload extent, or logical elements for affine. */
  uint64_t record_count = 0;
  /** True when the payload lives outside the message. */
  bool external = false;
  /** True when a ``type_ref`` was left unresolved because no catalogue was available. */
  bool unresolved_reference = false;
  /**
   * True when the payload bytes themselves were inspected, which only an inline
   * ``raw_data`` payload allows. For an external payload only the metadata is
   * validated: ``payload_bytes`` is the *declared* extent, the backing file is
   * neither opened nor measured, and content rules that need the bytes (the
   * affine 4-bit nibble padding in particular) are not checked. Load the payload
   * and re-validate before treating an external value as fully checked.
   */
  bool content_verified = false;
};

/**
 * Maximum nesting depth accepted while walking a declaration. Every nested
 * ``TypeProto`` and every ``type_ref`` expansion counts as one level, so a
 * hostile or accidentally recursive catalogue cannot exhaust the stack.
 */
inline constexpr uint32_t kMaxStructTypeDepth = 64;

/**
 * Resolves ``type_id`` references declared by ``ModelProto.struct_types``.
 *
 * The catalogue borrows the model it was built from: it must not outlive that
 * model, and the model must not be mutated while it is in use. A catalogue
 * holds a handful of declarations, so lookups scan them linearly instead of
 * paying for a hash map.
 */
class ONNX_LIGHT_PROTO_API StructTypeCatalogue {
public:
  /** Builds an empty catalogue resolving no identity. */
  StructTypeCatalogue() = default;

  /**
   * Indexes and validates every declaration of @p model.
   *
   * Each entry must carry a nonzero ``type_id`` unique in the model and a
   * concrete kind; references must resolve and must not form a cycle.
   *
   * @param model Model owning the declarations, borrowed by the catalogue.
   *
   * @throws std::invalid_argument Thrown when a declaration is malformed.
   */
  void Build(const ModelProto &model);

  /** Returns the declaration for @p type_id, or null when the model declares no such identity. */
  inline const StructTypeProto *Find(uint64_t type_id) const {
    if (model_ == nullptr || type_id == 0) {
      return nullptr;
    }
    const utils::RepeatedProtoField<StructTypeProto> &declarations = model_->ref_struct_types();
    for (size_t i = 0; i < declarations.size(); ++i) {
      if (declarations[i].has_type_id() && declarations[i].ref_type_id() == type_id) {
        return &declarations[i];
      }
    }
    return nullptr;
  }

  /** Returns the number of indexed declarations. */
  inline size_t size() const { return model_ == nullptr ? 0 : model_->ref_struct_types().size(); }

  /** Returns true when the catalogue indexes no declaration. */
  inline bool empty() const { return size() == 0; }

  /**
   * Returns the declaration @p type denotes, or @p type itself when it is not a reference.
   *
   * @throws std::invalid_argument Thrown when the reference does not resolve.
   */
  inline const StructTypeProto &Resolve(const StructTypeProto &type) const {
    if (type.kind_case() != StructTypeProto::kTypeRef) {
      return type;
    }
    const StructTypeProto *declaration = Find(type.ref_type_ref());
    EXT_ENFORCE_INVALID(declaration != nullptr,
                        "StructTypeProto references a type_id the model does not declare.");
    return *declaration;
  }

  /**
   * Computes the fixed size of @p type in bits with checked ``uint64_t`` arithmetic.
   *
   * Constants contribute zero bits, arrays and bit packings are tight and
   * fields follow declaration order.
   *
   * @param type   Type whose layout is measured.
   * @param bits   Receives the size in bits on success.
   * @param reason Optional destination for a human-readable failure cause.
   *
   * Returns: True when the type has a fixed inline layout, false otherwise.
   */
  bool FixedBitSize(const StructTypeProto &type, uint64_t &bits,
                    std::string *reason = nullptr) const;

  /**
   * Returns true when @p type can be the root of a byte-encoded payload, that
   * is when its size is fixed, strictly positive and a whole number of bytes.
   *
   * @param type   Type to test.
   * @param bytes  Receives the size of one record in bytes on success.
   * @param reason Optional destination for a human-readable failure cause.
   */
  inline bool IsByteEncodable(const StructTypeProto &type, uint64_t &bytes,
                              std::string *reason = nullptr) const {
    uint64_t bits = 0;
    if (!FixedBitSize(type, bits, reason)) {
      return false;
    }
    if (bits == 0 || bits % 8 != 0) {
      if (reason != nullptr) {
        *reason = "the encoded root needs a strictly positive size divisible by eight";
      }
      return false;
    }
    bytes = bits / 8;
    return true;
  }

  /**
   * Validates a ``TypeProto``, including any nested ``struct_type``.
   *
   * Dynamic tensor dimensions, sequences, maps and optional values stay valid
   * types here; only byte encoding requires a fixed layout.
   *
   * @param type Type to validate.
   *
   * @throws std::invalid_argument Thrown when validation fails.
   */
  void ValidateType(const TypeProto &type) const;

  /**
   * Validates one ``EncodedValueProto`` and returns its payload geometry.
   *
   * Checks the layout choice, raw/external exclusivity, external metadata,
   * byte-encodability of the structured root and payload divisibility, or the
   * frozen affine contract (storage type, parameter types, axis and block
   * size, logical type and shape, code byte length and nibble padding).
   *
   * @param value                       Value to validate.
   * @param require_resolved_reference  When false, a ``type_ref`` that this
   *        catalogue cannot resolve suspends the size checks instead of
   *        failing; used to validate a graph before its model catalogue is
   *        known.
   *
   * Returns: The validated layout.
   *
   * @throws std::invalid_argument Thrown when validation fails.
   */
  EncodedValueLayout ValidateEncodedValue(const EncodedValueProto &value,
                                          bool require_resolved_reference = true) const;

  /**
   * Locates a physical leaf inside one encoded record of @p root.
   *
   * @p path is a dot-separated sequence of structure field names ending on a
   * fixed-width leaf: a tensor field, an array of fixed-width elements or a
   * bit-packing component. A constant field is not physical and is reported as
   * not found: read it from the declaration instead.
   *
   * @param root Root type of the encoded element, resolved or not.
   * @param path Dot-separated field path, empty for an array root.
   * @param out  Receives the located leaf on success.
   *
   * Returns: True when the path denotes a physical leaf.
   *
   * @throws std::invalid_argument Thrown when the type itself is malformed.
   */
  bool FindField(const StructTypeProto &root, std::string_view path, EncodedFieldRef &out) const;

private:
  const ModelProto *model_ = nullptr;
};

/**
 * Bounds-checked read access to the records of an ``EncodedValueProto``.
 *
 * The view borrows the payload bytes: the value (and whatever owner token its
 * ``raw_data`` holds) must outlive the view.
 */
class ONNX_LIGHT_PROTO_API EncodedValueView {
public:
  /**
   * Validates @p value against @p catalogue and binds the view to its payload.
   *
   * @param catalogue Catalogue resolving the value's references.
   * @param value     Value to read.
   *
   * @throws std::invalid_argument Thrown when the value is invalid.
   */
  inline EncodedValueView(const StructTypeCatalogue &catalogue, const EncodedValueProto &value)
      : layout_(catalogue.ValidateEncodedValue(value)) {
    if (!layout_.external) {
      payload_ = value.ref_raw_data().data();
      payload_size_ = value.ref_raw_data().size();
    }
  }

  /** Returns the number of records derived from the payload extent. */
  inline uint64_t record_count() const { return layout_.record_count; }

  /** Returns the size of one record in bits. */
  inline uint64_t record_bits() const { return layout_.element_bits; }

  /** Returns the validated payload geometry. */
  inline const EncodedValueLayout &layout() const { return layout_; }

  /** Returns the number of readable payload bytes, zero for an external payload. */
  inline size_t payload_size() const { return payload_size_; }

  /**
   * Reads @p bit_width bits starting at @p bit_offset inside record @p record.
   *
   * Bits run from the least to the most significant bit and multi-byte values
   * are little-endian.
   *
   * @param record     Record index.
   * @param bit_offset Bit offset inside the record.
   * @param bit_width  Number of bits to read, between 1 and 64.
   *
   * Returns: The requested bits, zero-extended.
   *
   * @throws std::invalid_argument Thrown when the read falls outside the payload.
   */
  inline uint64_t ReadBits(uint64_t record, uint64_t bit_offset, uint32_t bit_width) const {
    EXT_ENFORCE_INVALID(!layout_.external,
                        "EncodedValueView cannot read an external payload; load it first.");
    EXT_ENFORCE_INVALID(bit_width >= 1 && bit_width <= 64,
                        "EncodedValueView reads between 1 and 64 bits at a time.");
    EXT_ENFORCE_INVALID(record < layout_.record_count,
                        "EncodedValueView record index is out of range.");
    // Subtraction-based bounds: bit_offset + bit_width and record * element_bits
    // both overflow for a hostile EncodedFieldRef, so never form those sums.
    EXT_ENFORCE_INVALID(bit_width <= layout_.element_bits &&
                            bit_offset <= layout_.element_bits - bit_width,
                        "EncodedValueView read leaves the record.");
    EXT_ENFORCE_INVALID(static_cast<uint64_t>(payload_size_) <=
                            std::numeric_limits<uint64_t>::max() / 8,
                        "EncodedValueView payload is too large to address in bits.");
    const uint64_t payload_bits = static_cast<uint64_t>(payload_size_) * 8;
    EXT_ENFORCE_INVALID(layout_.element_bits != 0 && record < payload_bits / layout_.element_bits,
                        "EncodedValueView read leaves the payload.");
    // record * element_bits <= payload_bits - element_bits and
    // bit_offset + bit_width <= element_bits, so neither sum below can overflow.
    const uint64_t absolute = record * layout_.element_bits + bit_offset;
    const uint64_t first = absolute / 8;
    const uint64_t last = (absolute + bit_width - 1) / 8;
    EXT_ENFORCE_INVALID(last < payload_size_, "EncodedValueView read leaves the payload.");
    uint32_t shift = static_cast<uint32_t>(absolute % 8);
    uint64_t value = 0;
    uint32_t produced = 0;
    for (uint64_t index = first; index <= last && produced < 64; ++index) {
      value |= (static_cast<uint64_t>(payload_[index]) >> shift) << produced;
      produced += 8 - shift;
      shift = 0;
      if (produced >= bit_width) {
        break;
      }
    }
    if (bit_width < 64) {
      value &= (uint64_t(1) << bit_width) - 1;
    }
    return value;
  }

  /** Reads element @p index of @p ref inside record @p record, zero-extended. */
  inline uint64_t ReadElement(const EncodedFieldRef &ref, uint64_t record, uint64_t index) const {
    EXT_ENFORCE_INVALID(index < ref.count, "EncodedValueView element index is out of range.");
    EXT_ENFORCE_INVALID(ref.bit_width >= 1 && ref.bit_width <= 64,
                        "EncodedValueView reads between 1 and 64 bits at a time.");
    // index * bit_stride and bit_offset + that product both overflow for a
    // hostile EncodedFieldRef; bound each step against the record size instead.
    EXT_ENFORCE_INVALID(ref.bit_stride == 0 || index <= layout_.element_bits / ref.bit_stride,
                        "EncodedValueView element offset leaves the record.");
    const uint64_t relative = index * ref.bit_stride;
    EXT_ENFORCE_INVALID(ref.bit_offset <= layout_.element_bits - relative,
                        "EncodedValueView element offset leaves the record.");
    return ReadBits(record, ref.bit_offset + relative, ref.bit_width);
  }

  /** Reads element @p index of @p ref as a two's-complement signed integer. */
  inline int64_t ReadSignedElement(const EncodedFieldRef &ref, uint64_t record,
                                   uint64_t index) const {
    const uint64_t bits = ReadElement(ref, record, index);
    if (ref.bit_width >= 64) {
      return static_cast<int64_t>(bits);
    }
    const uint64_t sign_bit = uint64_t(1) << (ref.bit_width - 1);
    return static_cast<int64_t>((bits ^ sign_bit) - sign_bit);
  }

  /** Reads element @p index of @p ref as a 32-bit floating-point value. */
  inline float ReadFloatElement(const EncodedFieldRef &ref, uint64_t record, uint64_t index) const {
    EXT_ENFORCE_INVALID(ref.bit_width == 32,
                        "EncodedValueView needs a 32-bit leaf to read a float.");
    const uint32_t bits = static_cast<uint32_t>(ReadElement(ref, record, index));
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

private:
  EncodedValueLayout layout_;
  const uint8_t *payload_ = nullptr;
  size_t payload_size_ = 0;
};

} // namespace ONNX_LIGHT_NAMESPACE
