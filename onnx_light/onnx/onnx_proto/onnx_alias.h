/**
 * @file
 * @brief Provides backwards-compatible type aliases for ONNX protobuf enumerations.
 *
 * The original protobuf-generated C++ code exposed flat names such as
 * `TensorProto_DataType`.  The onnx-light implementation uses scoped enumerations
 * (e.g. `TensorProto::DataType`).  The aliases defined here restore the flat names
 * so that code written against the original protobuf headers continues to compile
 * without modification.
 */

#pragma once

namespace onnx {

/**
 * Alias for `TensorProto::DataType` that preserves the flat protobuf-style name.
 *
 * Use `TensorProto_DataType` wherever the original protobuf-generated API expects the
 * flat enumeration name.  Fully-qualified values such as
 * `TensorProto_DataType::TensorProto_DataType_FLOAT` are available through the
 * `TensorProto_DataType_*` macros defined in `onnx.h`.
 */
using TensorProto_DataType = TensorProto::DataType;

} // namespace onnx
