#pragma once

#include "onnx_verify.h"

#include <cstring>
#include <limits>
#include <string_view>

/**
 * @file onnx_encoded_value_view.h
 * @brief Field location and bounds-checked payload access for ``EncodedValueProto``.
 *
 * The helpers declared here locate fixed-width leaves within a validated
 * structured layout and read their bits from an inline encoded payload.
 *
 * Every function raises ``std::invalid_argument`` (through
 * ``EXT_THROW_INVALID`` / ``EXT_ENFORCE_INVALID``) on the first violation,
 * except the ones documented as returning ``false``.
 */

namespace ONNX_LIGHT_NAMESPACE {

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

/**
 * Locates a physical leaf inside one encoded record of @p root.
 *
 * @p path is a dot-separated sequence of structure field names ending on a
 * fixed-width leaf: a tensor field, an array of fixed-width elements or a
 * bit-packing component. A constant field is not physical and is reported as
 * not found: read it from the declaration instead.
 *
 * @param catalogue Catalogue resolving references in @p root.
 * @param root Root type of the encoded element, resolved or not.
 * @param path Dot-separated field path, empty for an array root.
 * @param out Receives the located leaf on success.
 *
 * Returns: True when the path denotes a physical leaf.
 *
 * @throws std::invalid_argument Thrown when the type itself is malformed.
 */
ONNX_LIGHT_PROTO_API bool FindEncodedField(const StructTypeCatalogue &catalogue,
                                           const StructTypeProto &root, std::string_view path,
                                           EncodedFieldRef &out);

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
   * @param value Value to read.
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
   * @param record Record index.
   * @param bit_offset Bit offset inside the record.
   * @param bit_width Number of bits to read, between 1 and 64.
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
