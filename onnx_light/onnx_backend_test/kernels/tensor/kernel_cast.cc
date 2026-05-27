// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Returns true when ``dtype`` is one of the numeric (non-STRING) element
// types supported by ``Cast``. Element bytes for these types live in
// ``Tensor::data``; their fixed element size is given by
// ``onnx_backend_test::ElementSize``.
bool IsSupportedNumericCastDtype(int32_t dtype) {
  switch (static_cast<TensorProto::DataType>(dtype)) {
  case TensorProto::DataType::FLOAT:
  case TensorProto::DataType::DOUBLE:
  case TensorProto::DataType::INT32:
  case TensorProto::DataType::INT64:
  case TensorProto::DataType::INT8:
  case TensorProto::DataType::UINT8:
  case TensorProto::DataType::INT16:
  case TensorProto::DataType::UINT16:
  case TensorProto::DataType::BOOL:
    return true;
  default:
    return false;
  }
}

bool IsSupportedCastDtype(int32_t dtype) {
  if (IsSupportedNumericCastDtype(dtype))
    return true;
  return static_cast<TensorProto::DataType>(dtype) == TensorProto::DataType::STRING;
}

// Returns one numeric element of ``x`` (at flat index ``i``) widened to
// ``double``. ``double`` is the canonical intermediate value because every
// supported numeric dtype is exactly representable as ``double`` for the
// value ranges exercised by the backend test cases.
double LoadAsDouble(const Tensor &x, int64_t i) {
  switch (static_cast<TensorProto::DataType>(x.data_type)) {
  case TensorProto::DataType::FLOAT:
    return static_cast<double>(x.AsFloat()[i]);
  case TensorProto::DataType::DOUBLE:
    return x.AsDouble()[i];
  case TensorProto::DataType::INT32:
    return static_cast<double>(x.AsInt32()[i]);
  case TensorProto::DataType::INT64:
    return static_cast<double>(x.AsInt64()[i]);
  case TensorProto::DataType::INT8:
    return static_cast<double>(x.AsInt8()[i]);
  case TensorProto::DataType::UINT8:
    return static_cast<double>(x.AsUint8()[i]);
  case TensorProto::DataType::INT16:
    return static_cast<double>(x.AsInt16()[i]);
  case TensorProto::DataType::UINT16:
    return static_cast<double>(x.AsUint16()[i]);
  case TensorProto::DataType::BOOL:
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
  switch (static_cast<TensorProto::DataType>(output.data_type)) {
  case TensorProto::DataType::FLOAT:
    output.AsFloat()[i] = static_cast<float>(v);
    return;
  case TensorProto::DataType::DOUBLE:
    output.AsDouble()[i] = v;
    return;
  case TensorProto::DataType::INT32:
    output.AsInt32()[i] = static_cast<int32_t>(static_cast<int64_t>(v));
    return;
  case TensorProto::DataType::INT64:
    output.AsInt64()[i] = static_cast<int64_t>(v);
    return;
  case TensorProto::DataType::INT8:
    output.AsInt8()[i] = static_cast<int8_t>(static_cast<int64_t>(v));
    return;
  case TensorProto::DataType::UINT8:
    output.AsUint8()[i] = static_cast<uint8_t>(static_cast<int64_t>(v));
    return;
  case TensorProto::DataType::INT16:
    output.AsInt16()[i] = static_cast<int16_t>(static_cast<int64_t>(v));
    return;
  case TensorProto::DataType::UINT16:
    output.AsUint16()[i] = static_cast<uint16_t>(static_cast<int64_t>(v));
    return;
  case TensorProto::DataType::BOOL:
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
  switch (static_cast<TensorProto::DataType>(x.data_type)) {
  case TensorProto::DataType::FLOAT: {
    std::ostringstream os;
    os << x.AsFloat()[i];
    return os.str();
  }
  case TensorProto::DataType::DOUBLE: {
    std::ostringstream os;
    os << x.AsDouble()[i];
    return os.str();
  }
  case TensorProto::DataType::INT32:
    return std::to_string(x.AsInt32()[i]);
  case TensorProto::DataType::INT64:
    return std::to_string(x.AsInt64()[i]);
  case TensorProto::DataType::INT8:
    return std::to_string(static_cast<int32_t>(x.AsInt8()[i]));
  case TensorProto::DataType::UINT8:
    return std::to_string(static_cast<uint32_t>(x.AsUint8()[i]));
  case TensorProto::DataType::INT16:
    return std::to_string(static_cast<int32_t>(x.AsInt16()[i]));
  case TensorProto::DataType::UINT16:
    return std::to_string(static_cast<uint32_t>(x.AsUint16()[i]));
  case TensorProto::DataType::BOOL:
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
                          "INT16, UINT16, BOOL, STRING).");
  if (static_cast<TensorProto::DataType>(to) == TensorProto::DataType::STRING) {
    Tensor out = Tensor::MakeString(
        "", x.shape, std::vector<std::string>(static_cast<size_t>(x.element_count())));
    (*this)(x, to, out);
    return out;
  }
  const size_t out_elem = ElementSize(to);
  Tensor out("", to, x.shape,
             std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * out_elem));
  (*this)(x, to, out);
  return out;
}

void Cast::operator()(const Tensor &x, int32_t to, Tensor &output) const {
  EXT_ENFORCE_INVALID(IsSupportedCastDtype(x.data_type),
                      "kernel::Cast: unsupported input dtype " + std::to_string(x.data_type) +
                          " (supported: FLOAT, DOUBLE, INT32, INT64, INT8, UINT8, "
                          "INT16, UINT16, BOOL, STRING).");
  EXT_ENFORCE_INVALID(IsSupportedCastDtype(to),
                      "kernel::Cast: unsupported 'to' dtype " + std::to_string(to) +
                          " (supported: FLOAT, DOUBLE, INT32, INT64, INT8, UINT8, "
                          "INT16, UINT16, BOOL, STRING).");
  EXT_ENFORCE_INVALID(output.data_type == to,
                      "kernel::Cast preallocated output dtype must match 'to'.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Cast preallocated output shape must match input shape.");
  const int64_t n = x.element_count();

  const bool to_string = static_cast<TensorProto::DataType>(to) == TensorProto::DataType::STRING;
  const bool from_string =
      static_cast<TensorProto::DataType>(x.data_type) == TensorProto::DataType::STRING;

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
    if (!x.data.empty()) {
      std::memcpy(output.data.data(), x.data.data(), x.data.size());
    }
    return;
  }

  for (int64_t i = 0; i < n; ++i) {
    StoreFromDouble(output, i, LoadAsDouble(x, i));
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
