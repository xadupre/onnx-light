/**
 * @file
 * @brief Defines backwards-compatible type aliases for ONNX protobuf enumerations
 *        and nested message types.
 *
 * The original protobuf-generated C++ code exposed flat names such as
 * `TensorProto_DataType` and `TypeProto_Tensor`.  The onnx-light implementation
 * uses scoped types (e.g. `TensorProto::DataType`, `TypeProto::Tensor`).  The
 * aliases defined here restore the flat names so that code written against the
 * original protobuf headers continues to compile without modification.
 */

#pragma once

namespace ONNX_LIGHT_NAMESPACE {

/**
 * Aliases `TensorProto::DataType` under the flat protobuf-style name.
 *
 * Use `TensorProto_DataType` wherever the original protobuf-generated API expects the
 * flat enumeration name.  Fully-qualified values such as
 * `TensorProto_DataType::TensorProto_DataType_FLOAT` are available through the
 * `TensorProto_DataType_*` macros defined in `onnx.h`.
 */
using TensorProto_DataType = TensorProto::DataType;

/**
 * Flat aliases for the nested message types inside `TypeProto`.
 *
 * In protobuf-generated C++ code these would be top-level type aliases
 * (e.g. `using TypeProto_Tensor = TypeProto::Tensor`). In onnx-light these
 * nested classes exist inside `TypeProto`, so the aliases are provided here
 * to keep vendored ONNX sources compiling unchanged.
 */
using TypeProto_Tensor = TypeProto::Tensor;
using TypeProto_SparseTensor = TypeProto::SparseTensor;
using TypeProto_Sequence = TypeProto::Sequence;
using TypeProto_Map = TypeProto::Map;
using TypeProto_Optional = TypeProto::Optional;

} // namespace ONNX_LIGHT_NAMESPACE
