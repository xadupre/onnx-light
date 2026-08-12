/**
 * @file
 * @brief Defines backwards-compatible type aliases for ONNX protobuf enumerations.
 *
 * The original protobuf-generated C++ code exposed flat names such as
 * `TensorProto_DataType`.  The onnx-light implementation uses scoped enumerations
 * (e.g. `TensorProto::DataType`).  The aliases defined here restore the flat names
 * so that code written against the original protobuf headers continues to compile
 * without modification.
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

using TensorProto_DataLocation = TensorProto::DataLocation;

using TensorShapeProto_Dimension = TensorShapeProto::Dimension;

using AttributeProto_AttributeType = AttributeProto::AttributeType;

using TypeProto_Map = TypeProto::Map;

using TypeProto_Opaque = TypeProto::Opaque;

using TypeProto_Optional = TypeProto::Optional;

using TypeProto_Sequence = TypeProto::Sequence;

using TypeProto_SparseTensor = TypeProto::SparseTensor;

using TypeProto_Tensor = TypeProto::Tensor;

#define TensorProto_DataType_Name TensorProto::DataType_Name

// Defined as free functions (rather than a macro like TensorProto_DataType_Name
// above) because some consumers (e.g. onnxruntime's provider bridge) declare a
// class member function named `TensorProto_DataType_IsValid`; a macro would
// rewrite that declaration/definition and fail to compile.
inline constexpr bool TensorProto_DataType_IsValid(TensorProto::DataType t) {
  return TensorProto::DataType_IsValid(t);
}
inline constexpr bool TensorProto_DataType_IsValid(int t) {
  return TensorProto::DataType_IsValid(t);
}

// Flat protobuf-style enumerator constants exposed at namespace scope.
//
// The flat names are also available as members of the scoped enums (added to
// TensorProto::DataType / AttributeProto::AttributeType / TensorProto::DataLocation
// in onnx.h) so that both `ONNX_NAMESPACE::TensorProto_DataType_FLOAT` and
// `ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT` resolve, as
// they do with the protobuf-generated header.  These are constants (not macros)
// so they do not interfere with the enum member names when onnx.h is parsed.
inline constexpr TensorProto::DataType TensorProto_DataType_UNDEFINED = TensorProto::UNDEFINED;
inline constexpr TensorProto::DataType TensorProto_DataType_FLOAT = TensorProto::FLOAT;
inline constexpr TensorProto::DataType TensorProto_DataType_UINT8 = TensorProto::UINT8;
inline constexpr TensorProto::DataType TensorProto_DataType_INT8 = TensorProto::INT8;
inline constexpr TensorProto::DataType TensorProto_DataType_UINT16 = TensorProto::UINT16;
inline constexpr TensorProto::DataType TensorProto_DataType_INT16 = TensorProto::INT16;
inline constexpr TensorProto::DataType TensorProto_DataType_INT32 = TensorProto::INT32;
inline constexpr TensorProto::DataType TensorProto_DataType_INT64 = TensorProto::INT64;
inline constexpr TensorProto::DataType TensorProto_DataType_STRING = TensorProto::STRING;
inline constexpr TensorProto::DataType TensorProto_DataType_BOOL = TensorProto::BOOL;
inline constexpr TensorProto::DataType TensorProto_DataType_FLOAT16 = TensorProto::FLOAT16;
inline constexpr TensorProto::DataType TensorProto_DataType_DOUBLE = TensorProto::DOUBLE;
inline constexpr TensorProto::DataType TensorProto_DataType_UINT32 = TensorProto::UINT32;
inline constexpr TensorProto::DataType TensorProto_DataType_UINT64 = TensorProto::UINT64;
inline constexpr TensorProto::DataType TensorProto_DataType_COMPLEX64 = TensorProto::COMPLEX64;
inline constexpr TensorProto::DataType TensorProto_DataType_COMPLEX128 = TensorProto::COMPLEX128;
inline constexpr TensorProto::DataType TensorProto_DataType_BFLOAT16 = TensorProto::BFLOAT16;
inline constexpr TensorProto::DataType TensorProto_DataType_FLOAT8E4M3FN =
    TensorProto::FLOAT8E4M3FN;
inline constexpr TensorProto::DataType TensorProto_DataType_FLOAT8E4M3FNUZ =
    TensorProto::FLOAT8E4M3FNUZ;
inline constexpr TensorProto::DataType TensorProto_DataType_FLOAT8E5M2 = TensorProto::FLOAT8E5M2;
inline constexpr TensorProto::DataType TensorProto_DataType_FLOAT8E5M2FNUZ =
    TensorProto::FLOAT8E5M2FNUZ;
inline constexpr TensorProto::DataType TensorProto_DataType_UINT4 = TensorProto::UINT4;
inline constexpr TensorProto::DataType TensorProto_DataType_INT4 = TensorProto::INT4;
inline constexpr TensorProto::DataType TensorProto_DataType_FLOAT4E2M1 = TensorProto::FLOAT4E2M1;
inline constexpr TensorProto::DataType TensorProto_DataType_FLOAT8E8M0 = TensorProto::FLOAT8E8M0;
inline constexpr TensorProto::DataType TensorProto_DataType_UINT2 = TensorProto::UINT2;
inline constexpr TensorProto::DataType TensorProto_DataType_INT2 = TensorProto::INT2;

// protobuf sentinel constants for the DataType enum range.
inline constexpr int TensorProto_DataType_DataType_MIN = 0;
inline constexpr int TensorProto_DataType_DataType_MAX = 26;
inline constexpr int TensorProto_DataType_DataType_ARRAYSIZE = 27;

inline constexpr TensorProto::DataLocation TensorProto_DataLocation_DEFAULT = TensorProto::DEFAULT;
inline constexpr TensorProto::DataLocation TensorProto_DataLocation_EXTERNAL =
    TensorProto::EXTERNAL;

inline constexpr AttributeProto::AttributeType AttributeProto_AttributeType_UNDEFINED =
    AttributeProto::UNDEFINED;
inline constexpr AttributeProto::AttributeType AttributeProto_AttributeType_FLOAT =
    AttributeProto::FLOAT;
inline constexpr AttributeProto::AttributeType AttributeProto_AttributeType_INT =
    AttributeProto::INT;
inline constexpr AttributeProto::AttributeType AttributeProto_AttributeType_STRING =
    AttributeProto::STRING;
inline constexpr AttributeProto::AttributeType AttributeProto_AttributeType_TENSOR =
    AttributeProto::TENSOR;
inline constexpr AttributeProto::AttributeType AttributeProto_AttributeType_GRAPH =
    AttributeProto::GRAPH;
inline constexpr AttributeProto::AttributeType AttributeProto_AttributeType_SPARSE_TENSOR =
    AttributeProto::SPARSE_TENSOR;
inline constexpr AttributeProto::AttributeType AttributeProto_AttributeType_TYPE_PROTO =
    AttributeProto::TYPE_PROTO;
inline constexpr AttributeProto::AttributeType AttributeProto_AttributeType_FLOATS =
    AttributeProto::FLOATS;
inline constexpr AttributeProto::AttributeType AttributeProto_AttributeType_INTS =
    AttributeProto::INTS;
inline constexpr AttributeProto::AttributeType AttributeProto_AttributeType_STRINGS =
    AttributeProto::STRINGS;
inline constexpr AttributeProto::AttributeType AttributeProto_AttributeType_TENSORS =
    AttributeProto::TENSORS;
inline constexpr AttributeProto::AttributeType AttributeProto_AttributeType_GRAPHS =
    AttributeProto::GRAPHS;
inline constexpr AttributeProto::AttributeType AttributeProto_AttributeType_SPARSE_TENSORS =
    AttributeProto::SPARSE_TENSORS;
inline constexpr AttributeProto::AttributeType AttributeProto_AttributeType_TYPE_PROTOS =
    AttributeProto::TYPE_PROTOS;

using OptionalProto_DataType = OptionalProto::DataType;

// OptionalProto_DataType flat enumerator constants.
inline constexpr OptionalProto::DataType OptionalProto_DataType_UNDEFINED =
    OptionalProto::UNDEFINED;
inline constexpr OptionalProto::DataType OptionalProto_DataType_TENSOR = OptionalProto::TENSOR;
inline constexpr OptionalProto::DataType OptionalProto_DataType_SPARSE_TENSOR =
    OptionalProto::SPARSE_TENSOR;
inline constexpr OptionalProto::DataType OptionalProto_DataType_SEQUENCE = OptionalProto::SEQUENCE;
inline constexpr OptionalProto::DataType OptionalProto_DataType_MAP = OptionalProto::MAP;
inline constexpr OptionalProto::DataType OptionalProto_DataType_OPTIONAL = OptionalProto::OPTIONAL;

// ONNX IR version enum.  The protobuf-generated headers expose this at
// namespace scope (via onnx-data.pb.h); mirror it here so that consumers that
// only include <onnx/onnx_pb.h> (e.g. onnxruntime) can reference
// ONNX_NAMESPACE::Version and ONNX_NAMESPACE::IR_VERSION.  Guarded so the copy
// in onnx_lib/onnx-data.pb.h does not clash when both are included.
#ifndef ONNX_LIGHT_VERSION_ENUM_DEFINED
#define ONNX_LIGHT_VERSION_ENUM_DEFINED
enum Version {
  START_VERSION = 0,
  IR_VERSION_2017_10_10 = 1,
  IR_VERSION_2017_10_30 = 2,
  IR_VERSION_2017_11_3 = 3,
  IR_VERSION_2019_1_22 = 4,
  IR_VERSION_2019_3_18 = 5,
  IR_VERSION_2019_9_19 = 6,
  IR_VERSION_2020_5_8 = 7,
  IR_VERSION_2021_7_30 = 8,
  IR_VERSION_2023_5_5 = 9,
  IR_VERSION_2024_3_25 = 10,
  IR_VERSION_2025_05_12 = 11,
  IR_VERSION_2025_08_26 = 12,
  IR_VERSION = 13,
};
#endif // ONNX_LIGHT_VERSION_ENUM_DEFINED

} // namespace ONNX_LIGHT_NAMESPACE
