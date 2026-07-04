#pragma once

// Compatibility shim that provides google::protobuf namespace aliases
// mapped to onnx-light's own implementations. This allows code written
// against the protobuf API (e.g., onnxruntime) to compile without
// linking libprotobuf.

#include "fields.h"
#include "stream.h"
#include <cstdint>
#include <istream>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

namespace google {
namespace protobuf {

// --- Scalar integer typedefs (protobuf compat) ---
// The protobuf runtime historically exposed these fixed-width integer aliases
// in the google::protobuf namespace; some consumers (e.g. onnxruntime) still
// reference them.  Map them onto the standard <cstdint> types.
typedef std::int32_t int32;
typedef std::int64_t int64;
typedef std::uint32_t uint32;
typedef std::uint64_t uint64;

// --- Repeated field types ---

template <typename T> using RepeatedField = ONNX_LIGHT_NAMESPACE::utils::RepeatedField<T>;

template <typename T> using RepeatedPtrField = ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<T>;

// --- RepeatedFieldBackInserter ---

/** Output iterator that appends to a RepeatedField via push_back.
 *  onnx-light provides the concrete implementation; this is a pure alias. */
template <typename T>
using RepeatedFieldBackInsertIterator =
    ONNX_LIGHT_NAMESPACE::utils::RepeatedFieldBackInsertIterator<T>;

/** Creates a back-insert iterator for a RepeatedField. */
template <typename T>
RepeatedFieldBackInsertIterator<T> RepeatedFieldBackInserter(RepeatedField<T> *field) {
  return ONNX_LIGHT_NAMESPACE::utils::RepeatedFieldBackInserter(field);
}

// --- Lifecycle ---

/** No-op: onnx-light has no global protobuf state to shut down. */
inline void ShutdownProtobufLibrary() {}

// --- I/O streams ---
//
// Every class below is a pure alias to a concrete onnx-light stream class
// (defined in stream.h, namespace ONNX_LIGHT_NAMESPACE::utils). onnx-light owns
// the implementations and its own stream hierarchy implements the protobuf
// ZeroCopyInputStream / ZeroCopyOutputStream interfaces (Next / BackUp /
// ByteCount / Flush), so no thin wrapper layer is needed here.

namespace io {

/** Zero-copy input stream over an in-memory buffer.
 *  onnx-light's StringStream (a concrete BinaryStream) implements the protobuf
 *  ZeroCopyInputStream interface and accepts a (const void*, int) buffer. */
using ArrayInputStream = ONNX_LIGHT_NAMESPACE::utils::StringStream;

/** Minimal coded input stream wrapping an ArrayInputStream. */
using CodedInputStream = ONNX_LIGHT_NAMESPACE::utils::CodedInputStream;

/** Zero-copy output stream that appends to a std::string. */
using StringOutputStream = ONNX_LIGHT_NAMESPACE::utils::StdStringWriteStream;

/** Zero-copy output stream wrapping a file descriptor. */
using FileOutputStream = ONNX_LIGHT_NAMESPACE::utils::FdWriteStream;

/** Zero-copy input stream that owns a copy of a std::istream's contents. */
using IstreamInputStream = ONNX_LIGHT_NAMESPACE::utils::IstreamStream;

/** Zero-copy output stream wrapping a std::ostream. */
using OstreamOutputStream = ONNX_LIGHT_NAMESPACE::utils::OstreamWriteStream;

} // namespace io
} // namespace protobuf
} // namespace google
