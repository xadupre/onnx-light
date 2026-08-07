#pragma once

#include "stream_class.h"
#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <tuple>
#include <type_traits>
#include <vector>
// #define DEBUG_READ

#if defined(DEBUG_READ)
#define DEBUG_PRINT(s) printf("%s\n", s);
#define DEBUG_PRINT2(s1, s2) printf("%s%s\n", s1, s2);
#else
#define DEBUG_PRINT(s)
#define DEBUG_PRINT2(s1, s2)
#endif

using namespace onnx_light_helpers;

namespace ONNX_LIGHT_NAMESPACE {

/**
 * @brief Scoped guard that tracks protobuf sub-message nesting depth.
 *
 * Increments ParseOptions::_recursion_depth on construction and enforces that
 * it stays within ParseOptions::max_recursion_depth, throwing otherwise. The
 * counter is decremented on destruction so the depth is restored even when an
 * exception unwinds through the parser. This protects the parser against stack
 * overflow / out-of-memory from deeply nested messages.
 */
class RecursionGuard {
public:
  explicit RecursionGuard(ParseOptions &options) : options_(options) {
    ++options_._recursion_depth;
  }
  ~RecursionGuard() { --options_._recursion_depth; }
  RecursionGuard(const RecursionGuard &) = delete;
  RecursionGuard &operator=(const RecursionGuard &) = delete;

  /// Throws when the current nesting depth exceeds the configured limit. Kept
  /// separate from the constructor so that, if it throws, this fully constructed
  /// guard's destructor still runs and restores the depth counter while the
  /// exception unwinds the parser. Uses ParseLimitExceeded (not a plain
  /// std::runtime_error) since this is a deliberate, configurable resource guard
  /// and must keep propagating out of ParseFromString/ParseFromArray rather than
  /// being swallowed as a generic parse failure.
  void validate() const {
    EXT_ENFORCE_LIMIT(options_._recursion_depth <= options_.max_recursion_depth,
                      "Protobuf message nesting depth (", options_._recursion_depth,
                      ") exceeds the maximum allowed recursion depth (",
                      options_.max_recursion_depth,
                      "); the message is nested too deeply. Increase "
                      "ParseOptions::max_recursion_depth if this nesting is intentional.");
  }

private:
  ParseOptions &options_;
};

/**
 * @brief Scoped guard that sets ParseOptions::_current_graph while a GraphProto is being parsed.
 *
 * Stores the graph on construction so the parse raw_data_callback can locate the parent graph of
 * each tensor, and restores the previous value on destruction — even when an exception unwinds
 * through the parser — so nested subgraphs correctly restore the enclosing graph pointer.
 */
class CurrentGraphGuard {
public:
  CurrentGraphGuard(ParseOptions &options, GraphProto *graph)
      : options_(options), previous_graph_(options._current_graph) {
    options_._current_graph = graph;
  }
  ~CurrentGraphGuard() { options_._current_graph = previous_graph_; }
  CurrentGraphGuard(const CurrentGraphGuard &) = delete;
  CurrentGraphGuard &operator=(const CurrentGraphGuard &) = delete;

private:
  ParseOptions &options_;
  GraphProto *previous_graph_;
};

/**
 * @brief Raises an error when the requested byte count exceeds the configured
 *        per-tensor allocation limit.
 *
 * Called immediately before any large heap allocation during parsing to
 * prevent OOM from maliciously or accidentally large size prefixes in the wire
 * format.  The check is a no-op when
 * ``ParseOptions::max_tensor_size_bytes == 0`` (the default).
 *
 * @param len      Byte count that is about to be allocated.
 * @param options  Active parse options carrying the limit.
 * @param name     Field name, included in the error message for diagnostics.
 * @param location Short description of the call site, e.g. "read_field<ByteSpan>".
 */
inline void CheckAllocationLimit(uint64_t len, const ParseOptions &options, const char *name,
                                 const char *location) {
  if (options.max_tensor_size_bytes > 0 &&
      len > static_cast<uint64_t>(options.max_tensor_size_bytes)) {
    // ParseLimitExceeded (not a plain std::runtime_error): this is a deliberate,
    // configurable resource guard, not wire-format corruption, so it must keep
    // propagating out of ParseFromString/ParseFromArray rather than being
    // swallowed as a generic parse failure.
    EXT_THROW_LIMIT(
        location, ": tensor field '", name, "' requests ", len,
        " bytes which exceeds ParseOptions::max_tensor_size_bytes=", options.max_tensor_size_bytes,
        ". Increase max_tensor_size_bytes or set it to 0 to disable the limit.");
  }
}

/**
 * @brief Skips a single field's on-wire payload according to *wire_type*, without interpreting it.
 *
 * The protobuf wire format requires decoders to accept messages that contain a known field number
 * paired with a wire type that does not match what the schema declares for that field (e.g. a
 * newer/older/adversarial writer, or bit-flipped data). Per the protobuf spec, such a field must be
 * treated exactly like an unrecognized field number: its bytes are skipped based on the wire type
 * actually present in the tag, and parsing continues with the next field. This mirrors the
 * behavior of the protobuf-generated C++ parser (google::protobuf), which stores such
 * fields in an UnknownFieldSet rather than failing to parse.
 *
 * Only throws when *wire_type* is not one of the four wire types protobuf defines for scalar/
 * length-delimited encoding (varint, 64-bit, length-delimited, 32-bit); such a value cannot be
 * skipped unambiguously and indicates a corrupted stream.
 *
 * @param stream   Stream positioned right after the field's tag (field_number/wire_type), i.e.
 *                 at the start of the field's value.
 * @param wire_type Wire type read from the tag (FIELD_VARINT / FIELD_FIXED64 / FIELD_FIXED_SIZE /
 *                 FIELD_FIXED32).
 * @param name     Field or message name, included in the error message when *wire_type* is
 *                 itself invalid.
 */
inline void SkipFieldByWireType(utils::BinaryStream &stream, uint64_t wire_type, const char *name) {
  switch (wire_type) {
  case FIELD_VARINT:
    stream.next_uint64();
    break;
  case FIELD_FIXED64:
    stream.skip_bytes(8);
    break;
  case FIELD_FIXED_SIZE: {
    uint64_t len = stream.next_uint64();
    stream.CanRead(len, "[SkipFieldByWireType] length exceeds stream bounds");
    stream.skip_bytes(static_cast<utils::offset_t>(len));
    break;
  }
  case FIELD_FIXED32:
    stream.skip_bytes(4);
    break;
  default:
    EXT_THROW("SkipFieldByWireType: cannot skip field '", name,
              "', unsupported wire_type=", wire_type, " at position '", stream.tell_around(), "'");
  }
}

/**
 * @brief Skips the current field and returns from the enclosing (void) function when *cond* is
 * false.
 *
 * Used at the top of the per-type read_field/read_enum_field/read_repeated_field overloads that
 * are dispatched by field number from the READ_FIELD family of macros: when the actual wire type
 * on the wire does not match what the field's C++ type requires, the field is treated like an
 * unknown field (skipped via SkipFieldByWireType) instead of raising a hard parse error, matching
 * protobuf's documented wire-format compatibility rules. See SkipFieldByWireType for rationale.
 */
#define SKIP_IF_WRONG_WIRE_TYPE(cond, stream, wire_type, name)                                     \
  if (!(cond)) {                                                                                   \
    SkipFieldByWireType(stream, static_cast<uint64_t>(wire_type), name);                           \
    return;                                                                                        \
  }

/**
 * @brief Scoped guard that restores a BinaryStream's previous read limit.
 *
 * Calls BinaryStream::Restore() on destruction, so the limit pushed by the
 * matching LimitToNext() is popped both on normal return and while an exception
 * unwinds the parser. Using RAII instead of a try/catch keeps the recursive
 * read_next_field_in_shortended_stream frame small: on MSVC an explicit
 * try/catch (...) with a rethrow materializes exception state and enlarges the
 * frame, and since that function is on the per-nesting-level recursion path the
 * extra footprint can overflow the smaller (1 MB) default stack on Windows
 * before the recursion-depth guard can reject a deeply nested message.
 */
class StreamLimitGuard {
public:
  explicit StreamLimitGuard(utils::BinaryStream &stream) : stream_(stream) {}
  ~StreamLimitGuard() { stream_.Restore(); }
  StreamLimitGuard(const StreamLimitGuard &) = delete;
  StreamLimitGuard &operator=(const StreamLimitGuard &) = delete;

private:
  utils::BinaryStream &stream_;
};

template <typename T>
void read_next_field_in_shortended_stream(utils::BinaryStream &stream, const char *,
                                          ParseOptions &options, T &field) {
  RecursionGuard recursion_guard(options);
  recursion_guard.validate();
  uint64_t length = stream.next_uint64();
  stream.LimitToNext(length);
  StreamLimitGuard limit_guard(stream);
  field.ParseFromStream(stream, options);
}

template <typename T>
void read_field(utils::BinaryStream &stream, int wire_type, T &field, const char *name,
                ParseOptions &options) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_FIXED_SIZE, stream, wire_type, name);
  read_next_field_in_shortended_stream(stream, name, options, field);
}

template <typename T>
void read_optional_proto_field(utils::BinaryStream &stream, int wire_type,
                               utils::OptionalField<T> &field, const char *name,
                               ParseOptions &options) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_FIXED_SIZE, stream, wire_type, name);
  field.set_empty_value();
  read_next_field_in_shortended_stream(stream, name, options, *field);
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::RefString &field,
                const char *name, ParseOptions &) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_FIXED_SIZE, stream, wire_type, name);
  field = stream.next_string();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::String &field, const char *name,
                ParseOptions &) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_FIXED_SIZE, stream, wire_type, name);
  field = stream.next_string();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::OptionalString &field,
                const char *name, ParseOptions &) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_FIXED_SIZE, stream, wire_type, name);
  utils::RefString ref = stream.next_string();
  if (ref.data() == nullptr)
    field.emplace();
  else
    field.emplace(ref.data(), ref.size());
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::OptionalField<int64_t> &field,
                const char *name, ParseOptions &) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_VARINT, stream, wire_type, name);
  field = stream.next_int64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::OptionalField<int32_t> &field,
                const char *name, ParseOptions &) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_VARINT, stream, wire_type, name);
  field = stream.next_int32();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::OptionalField<float> &field,
                const char *name, ParseOptions &) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_FIXED_SIZE || wire_type == FIELD_FIXED32, stream,
                          wire_type, name);
  field = stream.next_float();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, uint64_t &field, const char *name,
                ParseOptions &) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_VARINT, stream, wire_type, name);
  field = stream.next_uint64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, int64_t &field, const char *name,
                ParseOptions &) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_VARINT, stream, wire_type, name);
  field = stream.next_int64();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, int32_t &field, const char *name,
                ParseOptions &) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_VARINT, stream, wire_type, name);
  field = stream.next_int32();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, float &field, const char *name,
                ParseOptions &) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_FIXED_SIZE || wire_type == FIELD_FIXED32, stream,
                          wire_type, name);
  field = stream.next_float();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, double &field, const char *name,
                ParseOptions &) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_FIXED_SIZE, stream, wire_type, name);
  field = stream.next_double();
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, std::vector<uint8_t> &field,
                const char *name, ParseOptions &options) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_FIXED_SIZE, stream, wire_type, name);
  uint64_t len = stream.next_uint64();
  stream.CanRead(len, "[read_field<vector<uint8_t>>] length exceeds stream bounds");
  CheckAllocationLimit(len, options, name, "read_field<vector<uint8_t>>");
  field.resize(len);
  stream.read_bytes(len, field.data());
}

template <>
void read_field(utils::BinaryStream &stream, int wire_type, utils::ByteSpan &field,
                const char *name, ParseOptions &options) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_FIXED_SIZE, stream, wire_type, name);
  uint64_t len = stream.next_uint64();
  stream.CanRead(len, "[read_field<ByteSpan>] length exceeds stream bounds");
  CheckAllocationLimit(len, options, name, "read_field<ByteSpan>");
  field.resize(len);
  stream.read_bytes(len, field.data());
}

void read_field_limit_parallel(utils::BinaryStream &stream, int wire_type,
                               std::vector<uint8_t> &field, const char *name,
                               ParseOptions &options) {
  if (!options.skip_raw_data && !options.is_parallel()) {
    read_field(stream, wire_type, field, name, options);
  } else {
    SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_FIXED_SIZE, stream, wire_type, name);
    uint64_t len = stream.next_uint64();
    stream.CanRead(len, "[read_field_limit_parallel] length exceeds stream bounds");
    if (!options.skip_raw_data || static_cast<int64_t>(len) < options.raw_data_threshold) {
      CheckAllocationLimit(len, options, name, "read_field_limit_parallel");
      field.resize(len);
      if (options.is_parallel() && static_cast<int64_t>(len) >= options.min_parallel_block_size) {
        utils::DelayedBlock block;
        block.size = len;
        block.data = field.data();
        block.offset = stream.tell();
        stream.ReadDelayedBlock(block);
      } else {
        stream.read_bytes(len, field.data());
      }
    } else {
      stream.skip_bytes(len);
    }
  }
}

/** Variant of read_field_limit_parallel that supports zero-copy parsing.
 *  When options.no_copy is true and the stream supports it (stream.CanNoCopy()), the
 *  raw data is NOT copied into field.  Instead field is set to borrowed mode pointing
 *  directly into the stream's backing buffer.  The caller must keep the underlying buffer
 *  alive for as long as field.data() is used.
 *  Falls back to ordinary copy behaviour for file-backed streams.
 *  When options.alignment > 1 (and zero-copy is not in use), the buffer is allocated with
 *  ByteSpan::resize_aligned() so that field.data() is aligned to options.alignment bytes. */
void read_field_limit_parallel_nc(utils::BinaryStream &stream, int wire_type,
                                  utils::ByteSpan &field, const char *name, ParseOptions &options) {
  onnx_light_helpers::ValidateAlignmentOption(options.alignment, "ParseOptions.alignment");
  const bool use_zero_copy = options.no_copy && stream.CanNoCopy();
  // Fast path: no special modes — delegate to the plain byte reader.
  if (!options.skip_raw_data && !options.is_parallel() && !use_zero_copy &&
      options.alignment <= 1) {
    read_field(stream, wire_type, field, name, options);
    return;
  }
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_FIXED_SIZE, stream, wire_type, name);
  uint64_t len = stream.next_uint64();
  stream.CanRead(len, "[read_field_limit_parallel_nc] length exceeds stream bounds");
  if (!options.skip_raw_data || static_cast<int64_t>(len) < options.raw_data_threshold) {
    CheckAllocationLimit(len, options, name, "read_field_limit_parallel_nc");
    if (use_zero_copy) {
      if (options.alignment > 1) {
        const utils::offset_t raw_data_offset = stream.tell();
        // ParseLimitExceeded: this is a deliberate alignment/no_copy configuration
        // guard, not wire-format corruption, so it must keep propagating out of
        // ParseFromString/ParseFromArray rather than being swallowed as a generic
        // parse failure.
        EXT_ENFORCE_LIMIT(raw_data_offset % options.alignment == 0, "Raw data field '", name,
                          "' offset ", raw_data_offset,
                          " is incompatible with ParseOptions.alignment=", options.alignment,
                          " when no_copy=true. Disable no_copy or use a compatible alignment.");
      }
      const uint8_t *ptr = stream.read_bytes(static_cast<utils::offset_t>(len), nullptr);
      field.assign_borrowed(ptr, static_cast<size_t>(len), stream.zero_copy_owner());
    } else if (!options.is_parallel() && options.alignment > 1) {
      // Alignment-only fast path: no thread pool overhead.
      field.resize_aligned(static_cast<size_t>(len), static_cast<size_t>(options.alignment));
      stream.read_bytes(len, field.data());
    } else {
      if (options.alignment > 1) {
        field.resize_aligned(static_cast<size_t>(len), static_cast<size_t>(options.alignment));
      } else {
        field.resize(len);
      }
      if (options.is_parallel() && static_cast<int64_t>(len) >= options.min_parallel_block_size) {
        utils::DelayedBlock block;
        block.size = len;
        block.data = field.data();
        block.offset = stream.tell();
        stream.ReadDelayedBlock(block);
      } else {
        stream.read_bytes(len, field.data());
      }
    }
  } else {
    stream.skip_bytes(len);
  }
}

template <typename T>
void read_enum_field(utils::BinaryStream &stream, int wire_type, T &field, const char *name,
                     ParseOptions &) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_VARINT, stream, wire_type, name);
  field = static_cast<T>(static_cast<int32_t>(stream.next_uint64()));
}

template <typename T>
void read_optional_enum_field(utils::BinaryStream &stream, int wire_type,
                              utils::OptionalEnumField<T> &field, const char *name,
                              ParseOptions &) {
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_VARINT, stream, wire_type, name);
  field = static_cast<T>(stream.next_uint64());
}

// repeated fields

template <typename T>
void read_repeated_field(utils::BinaryStream &stream, int wire_type, utils::RepeatedField<T> &field,
                         const char *name, bool is_packed, ParseOptions &options) {
  read_repeated_field(stream, wire_type, field.mutable_values(), name, is_packed, options);
}

template <typename T>
void read_repeated_field(utils::BinaryStream &stream, int wire_type,
                         utils::RepeatedProtoField<T> &field, const char *name, bool is_packed,
                         ParseOptions &options) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field name '", name,
              "' at position '", stream.tell_around(), "'");
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_FIXED_SIZE, stream, wire_type, name);
  T &elem = field.add();
  read_next_field_in_shortended_stream(stream, name, options, elem);
}

template <typename T>
void read_repeated_field(utils::BinaryStream &stream, int wire_type, std::vector<T> &field,
                         const char *name, bool is_packed, ParseOptions &options) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field name '", name, "'");
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_FIXED_SIZE, stream, wire_type, name);
  // Construct the element directly in place to avoid an extra default construct
  // + copy/move of T into the vector.
  T &elem = field.emplace_back();
  read_next_field_in_shortended_stream(stream, name, options, elem);
}

template <>
void read_repeated_field(utils::BinaryStream &stream, int wire_type,
                         std::vector<utils::String> &field, const char *name, bool is_packed,
                         ParseOptions &) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field name '", name, "'");
  SKIP_IF_WRONG_WIRE_TYPE(wire_type == FIELD_FIXED_SIZE, stream, wire_type, name);
  field.emplace_back(utils::String(stream.next_string()));
}

// unpacked numbers

template <typename T> T read_unpacked_number_float(utils::BinaryStream &stream, int wire_type) {
  // Non-packed wire format for fixed-width floating-point fields:
  // float uses FIELD_FIXED32 (wire type 5), double uses FIELD_FIXED64 (wire type 1).
  // Older writers may have used FIELD_FIXED_SIZE (wire type 2) by mistake; accept it
  // for read backward compatibility.
  EXT_ENFORCE(wire_type == (sizeof(T) == 4 ? FIELD_FIXED32 : FIELD_FIXED64) ||
                  wire_type == FIELD_FIXED_SIZE,
              "unexpected wire_type=", wire_type);
  T value;
  stream.next_packed_element(value);
  return value;
}

template <typename T> T read_unpacked_number_int(utils::BinaryStream &stream, int wire_type) {
  EXT_ENFORCE(wire_type == FIELD_VARINT, "unexpected wire_type=", wire_type);
  uint64_t i = stream.next_uint64();
  return static_cast<T>(i);
}

template <typename T> T read_unpacked_number(utils::BinaryStream &stream, int wire_type);

#define READ_UNPACKED_NUMBER_FLOAT(type)                                                           \
  template <> type read_unpacked_number(utils::BinaryStream &stream, int wire_type) {              \
    return read_unpacked_number_float<type>(stream, wire_type);                                    \
  }

READ_UNPACKED_NUMBER_FLOAT(float)
READ_UNPACKED_NUMBER_FLOAT(double)

#define READ_UNPACKED_NUMBER_INT(type)                                                             \
  template <> type read_unpacked_number(utils::BinaryStream &stream, int wire_type) {              \
    return read_unpacked_number_int<type>(stream, wire_type);                                      \
  }

READ_UNPACKED_NUMBER_INT(uint64_t)
READ_UNPACKED_NUMBER_INT(int64_t)
READ_UNPACKED_NUMBER_INT(int32_t)

// packed numbers

template <typename T>
void read_repeated_field_packed_numerical_float(utils::BinaryStream &stream, int wire_type,
                                                std::vector<T> &field, const char *name, bool,
                                                ParseOptions &options) {
  DEBUG_PRINT2("    read packed", name);
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '",
              name, "' at position '", stream.tell_around(), "'");
  uint64_t size = stream.next_uint64();
  EXT_ENFORCE(size % sizeof(T) == 0, "unexpected size ", size, ", it is not a multiple of sizeof(",
              typeid(T).name(), ") for field '", name, "' at position '", stream.tell_around(),
              "'");
  stream.CanRead(size, "[read_repeated_field_packed_numerical_float] length exceeds stream bounds");
  CheckAllocationLimit(size, options, name, "read_repeated_field_packed_numerical_float");
  size /= sizeof(T);
  field.resize(size);
  // Bulk read: a single read_bytes call replaces per-element virtual dispatch.
  // Protocol Buffers packed repeated fields store fixed-width values in their
  // native little-endian wire encoding; this matches the in-memory layout on
  // all platforms that onnx-light targets (x86/ARM little-endian).
  stream.read_bytes(static_cast<utils::offset_t>(size * sizeof(T)),
                    reinterpret_cast<uint8_t *>(field.data()));
}

template <typename T>
void read_repeated_field_packed_numerical_int(utils::BinaryStream &stream, int wire_type,
                                              std::vector<T> &field, const char *name, bool,
                                              ParseOptions &options) {
  DEBUG_PRINT2("    read packed", name);
  EXT_ENFORCE(wire_type == FIELD_FIXED_SIZE, "unexpected wire_type=", wire_type, " for field '",
              name, "' at position '", stream.tell_around(), "'");

  uint64_t length = stream.next_uint64();
  stream.CanRead(length, "[read_repeated_field_packed_numerical_int] length exceeds stream bounds");
  CheckAllocationLimit(length, options, name, "read_repeated_field_packed_numerical_int");
  // Each varint encodes at least 1 byte, so `length` is a strict upper bound
  // on the element count. Pre-reserving avoids the O(log n) reallocations
  // a plain push_back loop would cause on large packed tensors (shapes,
  // indices, etc.).
  field.reserve(field.size() + static_cast<size_t>(length));
  stream.LimitToNext(length);
  while (stream.NotEnd()) {
    field.push_back(static_cast<T>(stream.next_uint64()));
  }
  stream.Restore();
}

template <typename T>
void read_repeated_field_packed_numerical(utils::BinaryStream &stream, int wire_type,
                                          std::vector<T> &field, const char *name, bool is_packed,
                                          ParseOptions &options);

#define READ_PACKED_NUMBER_REPEAT_FLOAT(type)                                                      \
  template <>                                                                                      \
  void read_repeated_field_packed_numerical(utils::BinaryStream &stream, int wire_type,            \
                                            std::vector<type> &field, const char *name,            \
                                            bool is_packed, ParseOptions &options) {               \
    read_repeated_field_packed_numerical_float(stream, wire_type, field, name, is_packed,          \
                                               options);                                           \
  }

READ_PACKED_NUMBER_REPEAT_FLOAT(float)
READ_PACKED_NUMBER_REPEAT_FLOAT(double)

#define READ_PACKED_NUMBER_REPEAT_INT(type)                                                        \
  template <>                                                                                      \
  void read_repeated_field_packed_numerical(utils::BinaryStream &stream, int wire_type,            \
                                            std::vector<type> &field, const char *name,            \
                                            bool is_packed, ParseOptions &options) {               \
    read_repeated_field_packed_numerical_int(stream, wire_type, field, name, is_packed, options);  \
  }

READ_PACKED_NUMBER_REPEAT_INT(uint64_t)
READ_PACKED_NUMBER_REPEAT_INT(int64_t)
READ_PACKED_NUMBER_REPEAT_INT(int32_t)

// main function to read repeated numerical numbers

template <typename T>
void read_repeated_field_numerical(utils::BinaryStream &stream, int wire_type,
                                   std::vector<T> &field, const char *name, bool is_packed,
                                   ParseOptions &options) {
  // The protobuf spec requires a decoder to accept both the packed and the
  // unpacked wire format for any repeated numeric field, regardless of what
  // the schema declares. For integer fields the encoding is unambiguous: a
  // length-delimited wire type (FIELD_FIXED_SIZE) always denotes a packed
  // block, while a varint (wire type 0) denotes a single unpacked value, so we
  // can safely decode either (see gh_issue_24203, which packs
  // TensorProto.dims). Floating-point fields are handled by the branch below:
  // for them wire type 2 is ambiguous because a legacy writer emitted a single
  // element as raw fixed-size bytes without a length prefix, so we keep relying
  // on the schema's is_packed flag and the unpacked reader's legacy handling.
  bool packed = is_packed;
  if constexpr (std::is_integral_v<T>) {
    packed = packed || wire_type == FIELD_FIXED_SIZE;
  }
  if (packed) {
    read_repeated_field_packed_numerical(stream, wire_type, field, name, packed, options);
  } else {
    DEBUG_PRINT2("    read unpacked", name);
    field.push_back(read_unpacked_number<T>(stream, wire_type));
  }
}

#define READ_REPEATED_FIELD_IMPL(type)                                                             \
  template <>                                                                                      \
  void read_repeated_field(utils::BinaryStream &stream, int wire_type, std::vector<type> &field,   \
                           const char *name, bool is_packed, ParseOptions &options) {              \
    read_repeated_field_numerical(stream, wire_type, field, name, is_packed, options);             \
  }

READ_REPEATED_FIELD_IMPL(double)
READ_REPEATED_FIELD_IMPL(float)
READ_REPEATED_FIELD_IMPL(uint64_t)
READ_REPEATED_FIELD_IMPL(int64_t)
READ_REPEATED_FIELD_IMPL(int32_t)

} // namespace ONNX_LIGHT_NAMESPACE
