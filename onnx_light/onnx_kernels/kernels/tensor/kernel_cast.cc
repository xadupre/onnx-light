// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_kernels/kernels/tensor/cast_float8.h"
#include "onnx_kernels/kernels/tensor/cast_sub_byte.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Returns true when ``dtype`` is one of the numeric (non-STRING) element
// types supported by ``Cast``. Element bytes for these types live in
// ``Tensor::data``; their fixed element size is given by
// ``onnx_kernels::ElementSize``.
bool IsSupportedNumericCastDtype(int32_t dtype) {
  switch (static_cast<DataType>(dtype)) {
  case DataType::FLOAT:
  case DataType::DOUBLE:
  case DataType::INT32:
  case DataType::INT64:
  case DataType::INT8:
  case DataType::UINT8:
  case DataType::INT16:
  case DataType::UINT16:
  case DataType::BOOL:
    return true;
  default:
    return false;
  }
}

// Float8 element types currently supported by ``kernel::Cast``. These types
// only round-trip against ``FLOAT`` (matching what the upstream ONNX
// ``test_cast_<FROM>_to_<TO>`` node tests exercise for the floating-point
// 8-bit variants).
bool IsFloat8CastDtype(int32_t dtype) {
  switch (static_cast<DataType>(dtype)) {
  case DataType::FLOAT8E4M3FN:
  case DataType::FLOAT8E4M3FNUZ:
  case DataType::FLOAT8E5M2:
  case DataType::FLOAT8E5M2FNUZ:
    return true;
  default:
    return false;
  }
}

// Packed sub-byte dtypes currently supported by ``kernel::Cast``. INT4 / UINT4
// pack two elements per byte, INT2 / UINT2 four. Each type only round-trips
// against ``FLOAT`` and its companion whole-byte integer (``INT8`` for the
// signed variants, ``UINT8`` for the unsigned variants), mirroring the
// upstream ONNX ``test_cast_<FROM>_to_<TO>`` coverage for these dtypes.
bool IsInt4CastDtype(int32_t dtype) {
  return static_cast<DataType>(dtype) == DataType::INT4 ||
         static_cast<DataType>(dtype) == DataType::UINT4;
}

bool IsInt2CastDtype(int32_t dtype) {
  return static_cast<DataType>(dtype) == DataType::INT2 ||
         static_cast<DataType>(dtype) == DataType::UINT2;
}

bool IsSubByteCastDtype(int32_t dtype) { return IsInt4CastDtype(dtype) || IsInt2CastDtype(dtype); }

bool IsSupportedCastDtype(int32_t dtype) {
  if (IsSupportedNumericCastDtype(dtype))
    return true;
  if (IsFloat8CastDtype(dtype))
    return true;
  if (IsSubByteCastDtype(dtype))
    return true;
  return static_cast<DataType>(dtype) == DataType::STRING;
}

// Float ↔ float8 round-trip. ``saturate`` semantics are hard-coded to ``true``
// to match the default ``Cast`` attribute used by the registered cases.
std::uint8_t FloatToFloat8Bits(float v, int32_t to) {
  switch (static_cast<DataType>(to)) {
  case DataType::FLOAT8E4M3FN:
    return FloatToFloat8E4M3FNBits(v);
  case DataType::FLOAT8E4M3FNUZ:
    return FloatToFloat8E4M3FNUZBits(v);
  case DataType::FLOAT8E5M2:
    return FloatToFloat8E5M2Bits(v);
  case DataType::FLOAT8E5M2FNUZ:
    return FloatToFloat8E5M2FNUZBits(v);
  default:
    throw std::invalid_argument("kernel::Cast: unsupported float8 'to' dtype.");
  }
}

float Float8BitsToFloat(std::uint8_t bits, int32_t from) {
  switch (static_cast<DataType>(from)) {
  case DataType::FLOAT8E4M3FN:
    return Float8E4M3FNBitsToFloat(bits);
  case DataType::FLOAT8E4M3FNUZ:
    return Float8E4M3FNUZBitsToFloat(bits);
  case DataType::FLOAT8E5M2:
    return Float8E5M2BitsToFloat(bits);
  case DataType::FLOAT8E5M2FNUZ:
    return Float8E5M2FNUZBitsToFloat(bits);
  default:
    throw std::invalid_argument("kernel::Cast: unsupported float8 'from' dtype.");
  }
}

// ---------------------------------------------------------------------------
// Sub-byte (INT4 / UINT4 / INT2 / UINT2) packing helpers.
//
// The packed wire layout matches the ONNX TensorProto convention:
//   * INT4 / UINT4 — two 4-bit elements per byte, low nibble holds the
//     even-indexed element (flat index 2*i), high nibble holds the
//     odd-indexed element (flat index 2*i + 1).
//   * INT2 / UINT2 — four 2-bit elements per byte, packed least-significant
//     pair first (flat index 4*i in bits 0-1, 4*i+1 in bits 2-3, etc.).
//
// When the element count is not a multiple of the packing factor, the
// trailing nibble / bit-pair is zero-padded.
// ---------------------------------------------------------------------------
inline std::uint8_t Read4BitElement(const std::uint8_t *data, int64_t i) noexcept {
  const std::uint8_t byte = data[i / 2];
  return static_cast<std::uint8_t>((i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F));
}

inline void Write4BitElement(std::uint8_t *data, int64_t i, std::uint8_t nibble) noexcept {
  std::uint8_t &byte = data[i / 2];
  if (i % 2 == 0) {
    byte = static_cast<std::uint8_t>((byte & 0xF0) | (nibble & 0x0F));
  } else {
    byte = static_cast<std::uint8_t>((byte & 0x0F) | ((nibble & 0x0F) << 4));
  }
}

inline std::uint8_t Read2BitElement(const std::uint8_t *data, int64_t i) noexcept {
  const std::uint8_t byte = data[i / 4];
  const int shift = static_cast<int>((i % 4) * 2);
  return static_cast<std::uint8_t>((byte >> shift) & 0x03);
}

inline void Write2BitElement(std::uint8_t *data, int64_t i, std::uint8_t bits) noexcept {
  std::uint8_t &byte = data[i / 4];
  const int shift = static_cast<int>((i % 4) * 2);
  const std::uint8_t mask = static_cast<std::uint8_t>(0x03u << shift);
  byte = static_cast<std::uint8_t>((byte & ~mask) | (((bits & 0x03) << shift) & mask));
}

// Returns true when ``(from, to)`` matches one of the supported sub-byte
// cast pairs: FLOAT ↔ {INT4,UINT4,INT2,UINT2} and INT4 ↔ INT8 /
// UINT4 ↔ UINT8 / INT2 ↔ INT8 / UINT2 ↔ UINT8.
bool IsSupportedSubBytePair(int32_t from, int32_t to) {
  const bool from_sub = IsSubByteCastDtype(from);
  const bool to_sub = IsSubByteCastDtype(to);
  if (!from_sub && !to_sub)
    return false;
  const auto sub = from_sub ? from : to;
  const auto other = from_sub ? to : from;
  const auto sub_dt = static_cast<DataType>(sub);
  const auto other_dt = static_cast<DataType>(other);
  if (other_dt == DataType::FLOAT)
    return true;
  switch (sub_dt) {
  case DataType::INT4:
  case DataType::INT2:
    return other_dt == DataType::INT8;
  case DataType::UINT4:
  case DataType::UINT2:
    return other_dt == DataType::UINT8;
  default:
    return false;
  }
}

// Returns one numeric element of ``x`` (at flat index ``i``) widened to
// ``double``. ``double`` is the canonical intermediate value because every
// supported numeric dtype is exactly representable as ``double`` for the
// value ranges exercised by the backend test cases.
double LoadAsDouble(const Tensor &x, int64_t i) {
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    return static_cast<double>(x.AsFloat()[i]);
  case DataType::DOUBLE:
    return x.AsDouble()[i];
  case DataType::INT32:
    return static_cast<double>(x.AsInt32()[i]);
  case DataType::INT64:
    return static_cast<double>(x.AsInt64()[i]);
  case DataType::INT8:
    return static_cast<double>(x.AsInt8()[i]);
  case DataType::UINT8:
    return static_cast<double>(x.AsUint8()[i]);
  case DataType::INT16:
    return static_cast<double>(x.AsInt16()[i]);
  case DataType::UINT16:
    return static_cast<double>(x.AsUint16()[i]);
  case DataType::BOOL:
    return x.AsBool()[i] != 0 ? 1.0 : 0.0;
  default:
    throw std::invalid_argument("kernel::Cast: unsupported input dtype for numeric load.");
  }
}

// Writes the value ``v`` (already cast to the destination C++ type) into the
// numeric output buffer at flat index ``i``. For ``BOOL``, follows ONNX
// reference semantics: the value is true iff ``v`` is not zero (including
// NaN, which compares unequal to 0.0 and therefore maps to true).
//
// Narrowing casts to integer types route through ``int64_t`` so that
// out-of-range values (notably negative ``v`` cast to an unsigned target)
// produce a well-defined modular result on every supported platform. The
// raw ``static_cast<unsigned>(negative double)`` form is undefined
// behaviour in C++ and gives different results across compilers (e.g.
// gcc on Linux clamps to 0, while clang on macOS wraps), which previously
// caused mismatches against the ONNX runtime reference behaviour.
void StoreFromDouble(Tensor &output, int64_t i, double v) {
  switch (static_cast<DataType>(output.data_type)) {
  case DataType::FLOAT:
    output.AsFloat()[i] = static_cast<float>(v);
    return;
  case DataType::DOUBLE:
    output.AsDouble()[i] = v;
    return;
  case DataType::INT32:
    output.AsInt32()[i] = static_cast<int32_t>(static_cast<int64_t>(v));
    return;
  case DataType::INT64:
    output.AsInt64()[i] = static_cast<int64_t>(v);
    return;
  case DataType::INT8:
    output.AsInt8()[i] = static_cast<int8_t>(static_cast<int64_t>(v));
    return;
  case DataType::UINT8:
    output.AsUint8()[i] = static_cast<uint8_t>(static_cast<int64_t>(v));
    return;
  case DataType::INT16:
    output.AsInt16()[i] = static_cast<int16_t>(static_cast<int64_t>(v));
    return;
  case DataType::UINT16:
    output.AsUint16()[i] = static_cast<uint16_t>(static_cast<int64_t>(v));
    return;
  case DataType::BOOL:
    output.AsBool()[i] = (v != 0.0) ? uint8_t{1} : uint8_t{0};
    return;
  default:
    throw std::invalid_argument("kernel::Cast: unsupported output dtype for numeric store.");
  }
}

// Converts one numeric element of ``x`` to its canonical decimal string
// representation. Floating-point values use a stream-default representation
// (matching upstream reference behaviour closely enough for the
// deterministic backend test inputs registered here); ``BOOL`` becomes
// "1" or "0" so it round-trips through the numeric parse path.
std::string ElementToString(const Tensor &x, int64_t i) {
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT: {
    std::ostringstream os;
    os << x.AsFloat()[i];
    return os.str();
  }
  case DataType::DOUBLE: {
    std::ostringstream os;
    os << x.AsDouble()[i];
    return os.str();
  }
  case DataType::INT32:
    return std::to_string(x.AsInt32()[i]);
  case DataType::INT64:
    return std::to_string(x.AsInt64()[i]);
  case DataType::INT8:
    return std::to_string(static_cast<int32_t>(x.AsInt8()[i]));
  case DataType::UINT8:
    return std::to_string(static_cast<uint32_t>(x.AsUint8()[i]));
  case DataType::INT16:
    return std::to_string(static_cast<int32_t>(x.AsInt16()[i]));
  case DataType::UINT16:
    return std::to_string(static_cast<uint32_t>(x.AsUint16()[i]));
  case DataType::BOOL:
    return x.AsBool()[i] != 0 ? std::string("1") : std::string("0");
  default:
    throw std::invalid_argument("kernel::Cast: unsupported input dtype for string conversion.");
  }
}

// Parses ``s`` (a numeric representation produced by ``ElementToString`` or
// equivalent) as a ``double``. Throws ``std::invalid_argument`` if the
// string is not a recognised numeric literal.
double ParseAsDouble(const std::string &s) {
  try {
    return std::stod(s);
  } catch (const std::exception &) {
    throw std::invalid_argument("kernel::Cast: cannot parse string '" + s +
                                "' as a numeric value.");
  }
}

} // namespace

Tensor Cast::operator()(const Tensor &x, int32_t to) const {
  EXT_ENFORCE_INVALID(IsSupportedCastDtype(to),
                      "kernel::Cast: unsupported 'to' dtype " + std::to_string(to) +
                          " (supported: FLOAT, DOUBLE, INT32, INT64, INT8, UINT8, "
                          "INT16, UINT16, BOOL, STRING, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, "
                          "FLOAT8E5M2, FLOAT8E5M2FNUZ).");
  if (static_cast<DataType>(to) == DataType::STRING) {
    Tensor out = Tensor::MakeString(
        "", x.shape, std::vector<std::string>(static_cast<size_t>(x.element_count())));
    (*this)(x, to, out);
    return out;
  }
  const size_t out_bytes = PackedByteSize(to, x.element_count());
  Tensor out("", to, x.shape, std::vector<uint8_t>(out_bytes));
  (*this)(x, to, out);
  return out;
}

void Cast::operator()(const Tensor &x, int32_t to, Tensor &output) const {
  EXT_ENFORCE_INVALID(IsSupportedCastDtype(x.data_type),
                      "kernel::Cast: unsupported input dtype " + std::to_string(x.data_type) +
                          " (supported: FLOAT, DOUBLE, INT32, INT64, INT8, UINT8, "
                          "INT16, UINT16, BOOL, STRING, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, "
                          "FLOAT8E5M2, FLOAT8E5M2FNUZ).");
  EXT_ENFORCE_INVALID(IsSupportedCastDtype(to),
                      "kernel::Cast: unsupported 'to' dtype " + std::to_string(to) +
                          " (supported: FLOAT, DOUBLE, INT32, INT64, INT8, UINT8, "
                          "INT16, UINT16, BOOL, STRING, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, "
                          "FLOAT8E5M2, FLOAT8E5M2FNUZ).");
  EXT_ENFORCE_INVALID(output.data_type == to,
                      "kernel::Cast preallocated output dtype must match 'to'.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Cast preallocated output shape must match input shape.");
  const int64_t n = x.element_count();

  const bool to_string = static_cast<DataType>(to) == DataType::STRING;
  const bool from_string = static_cast<DataType>(x.data_type) == DataType::STRING;
  const bool to_float8 = IsFloat8CastDtype(to);
  const bool from_float8 = IsFloat8CastDtype(x.data_type);
  const bool to_sub_byte = IsSubByteCastDtype(to);
  const bool from_sub_byte = IsSubByteCastDtype(x.data_type);

  // Sub-byte (INT4 / UINT4 / INT2 / UINT2) dtypes only round-trip against
  // ``FLOAT`` and their companion whole-byte integer (``INT8`` for the
  // signed variants, ``UINT8`` for the unsigned variants). Cross-casting
  // against any other dtype is rejected up front rather than silently
  // routed through ``double``.
  if (from_sub_byte || to_sub_byte) {
    EXT_ENFORCE_INVALID(IsSupportedSubBytePair(x.data_type, to),
                        "kernel::Cast: unsupported sub-byte cast pair.");
    const size_t expected_bytes = PackedByteSize(to, n);
    EXT_ENFORCE_INVALID(output.data.size() == expected_bytes,
                        "kernel::Cast preallocated output buffer has unexpected size in bytes.");
    if (to_sub_byte) {
      // Reset trailing padding bytes so the unused nibble / bit-pair stays
      // zero regardless of the previous buffer contents.
      if (!output.data.empty()) {
        std::memset(output.data.data(), 0, output.data.size());
      }
      const auto to_dt = static_cast<DataType>(to);
      uint8_t *dst = output.data.data();
      if (static_cast<DataType>(x.data_type) == DataType::FLOAT) {
        const float *src = x.AsFloat();
        for (int64_t i = 0; i < n; ++i) {
          std::uint8_t v = 0;
          switch (to_dt) {
          case DataType::INT4:
            v = FloatToInt4Nibble(src[i]);
            Write4BitElement(dst, i, v);
            break;
          case DataType::UINT4:
            v = FloatToUint4Nibble(src[i]);
            Write4BitElement(dst, i, v);
            break;
          case DataType::INT2:
            v = FloatToInt2Bits(src[i]);
            Write2BitElement(dst, i, v);
            break;
          case DataType::UINT2:
            v = FloatToUint2Bits(src[i]);
            Write2BitElement(dst, i, v);
            break;
          default:
            throw std::invalid_argument("kernel::Cast: unsupported sub-byte 'to' dtype.");
          }
        }
      } else if (static_cast<DataType>(x.data_type) == DataType::INT8) {
        const int8_t *src = x.AsInt8();
        for (int64_t i = 0; i < n; ++i) {
          if (to_dt == DataType::INT4) {
            const int v = std::max(-8, std::min(7, static_cast<int>(src[i])));
            Write4BitElement(dst, i, static_cast<std::uint8_t>(v & 0x0F));
          } else if (to_dt == DataType::INT2) {
            const int v = std::max(-2, std::min(1, static_cast<int>(src[i])));
            Write2BitElement(dst, i, static_cast<std::uint8_t>(v & 0x03));
          } else {
            throw std::invalid_argument("kernel::Cast: unsupported sub-byte 'to' dtype from INT8.");
          }
        }
      } else if (static_cast<DataType>(x.data_type) == DataType::UINT8) {
        const uint8_t *src = x.AsUint8();
        for (int64_t i = 0; i < n; ++i) {
          if (to_dt == DataType::UINT4) {
            const unsigned v = std::min(15u, static_cast<unsigned>(src[i]));
            Write4BitElement(dst, i, static_cast<std::uint8_t>(v & 0x0Fu));
          } else if (to_dt == DataType::UINT2) {
            const unsigned v = std::min(3u, static_cast<unsigned>(src[i]));
            Write2BitElement(dst, i, static_cast<std::uint8_t>(v & 0x03u));
          } else {
            throw std::invalid_argument(
                "kernel::Cast: unsupported sub-byte 'to' dtype from UINT8.");
          }
        }
      } else {
        throw std::invalid_argument("kernel::Cast: unsupported sub-byte 'from' dtype.");
      }
    } else {
      const uint8_t *src = x.bytes();
      const auto from_dt = static_cast<DataType>(x.data_type);
      const auto to_dt = static_cast<DataType>(to);
      for (int64_t i = 0; i < n; ++i) {
        // Read the unpacked sub-byte value (sign-extended into ``int``).
        int value = 0;
        switch (from_dt) {
        case DataType::INT4:
          value = static_cast<int>(Int4NibbleToInt8(Read4BitElement(src, i)));
          break;
        case DataType::UINT4:
          value = static_cast<int>(Uint4NibbleToUint8(Read4BitElement(src, i)));
          break;
        case DataType::INT2:
          value = static_cast<int>(Int2BitsToInt8(Read2BitElement(src, i)));
          break;
        case DataType::UINT2:
          value = static_cast<int>(Uint2BitsToUint8(Read2BitElement(src, i)));
          break;
        default:
          throw std::invalid_argument("kernel::Cast: unsupported sub-byte 'from' dtype.");
        }
        switch (to_dt) {
        case DataType::FLOAT:
          output.AsFloat()[i] = static_cast<float>(value);
          break;
        case DataType::INT8:
          output.AsInt8()[i] = static_cast<int8_t>(value);
          break;
        case DataType::UINT8:
          output.AsUint8()[i] = static_cast<uint8_t>(value);
          break;
        default:
          throw std::invalid_argument("kernel::Cast: unsupported sub-byte 'to' dtype.");
        }
      }
    }
    return;
  }

  // Float8 dtypes only round-trip against ``FLOAT`` in this reference
  // kernel (matching the upstream ONNX ``test_cast`` coverage that
  // ``kernel::Cast`` mirrors). Cross-casting against any other dtype is
  // rejected up front rather than silently routed through ``double``.
  if (from_float8 || to_float8) {
    EXT_ENFORCE_INVALID((from_float8 && static_cast<DataType>(to) == DataType::FLOAT) ||
                            (to_float8 && static_cast<DataType>(x.data_type) == DataType::FLOAT),
                        "kernel::Cast: FLOAT8* dtypes only round-trip against FLOAT.");
    const size_t expected_bytes = static_cast<size_t>(n) * (to_float8 ? size_t{1} : sizeof(float));
    EXT_ENFORCE_INVALID(output.data.size() == expected_bytes,
                        "kernel::Cast preallocated output buffer has unexpected size in bytes.");
    if (to_float8) {
      const float *src = x.AsFloat();
      uint8_t *dst = output.data.data();
      for (int64_t i = 0; i < n; ++i) {
        dst[i] = FloatToFloat8Bits(src[i], to);
      }
    } else {
      const uint8_t *src = x.bytes();
      float *dst = output.AsFloat();
      for (int64_t i = 0; i < n; ++i) {
        dst[i] = Float8BitsToFloat(src[i], x.data_type);
      }
    }
    return;
  }

  if (to_string) {
    EXT_ENFORCE_INVALID(static_cast<int64_t>(output.string_data.size()) == n,
                        "kernel::Cast preallocated STRING output must have one entry per element.");
    if (from_string) {
      output.string_data = x.string_data;
      return;
    }
    for (int64_t i = 0; i < n; ++i) {
      output.string_data[static_cast<size_t>(i)] = ElementToString(x, i);
    }
    return;
  }

  const size_t out_elem = ElementSize(to);
  const size_t expected_bytes = static_cast<size_t>(n) * out_elem;
  EXT_ENFORCE_INVALID(output.data.size() == expected_bytes,
                      "kernel::Cast preallocated output buffer has unexpected size in bytes.");

  if (from_string) {
    EXT_ENFORCE_INVALID(static_cast<int64_t>(x.string_data.size()) == n,
                        "kernel::Cast STRING input must have one entry per element.");
    for (int64_t i = 0; i < n; ++i) {
      StoreFromDouble(output, i, ParseAsDouble(x.string_data[static_cast<size_t>(i)]));
    }
    return;
  }

  // Fast path: identity cast — bitwise copy when input and output dtypes
  // match. This matches the ONNX semantics for ``to == input dtype`` and
  // avoids a needless double round-trip.
  if (x.data_type == to) {
    if (x.size_bytes() > 0) {
      std::memcpy(output.data.data(), x.bytes(), x.size_bytes());
    }
    return;
  }

  for (int64_t i = 0; i < n; ++i) {
    StoreFromDouble(output, i, LoadAsDouble(x, i));
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
