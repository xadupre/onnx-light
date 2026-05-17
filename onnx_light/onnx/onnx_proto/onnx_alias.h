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

using TensorShapeProto_Dimension = TensorShapeProto::Dimension;

using TypeProto_Map = TypeProto::Map;

using TypeProto_Optional = TypeProto::Optional;

using TypeProto_Sequence = TypeProto::Sequence;

using TypeProto_SparseTensor = TypeProto::SparseTensor;

using TypeProto_Tensor = TypeProto::Tensor;

#define TensorProto_DataType_Name TensorProto::DataType_Name

#define TensorProto_DataType_UNDEFINED TensorProto::DataType::UNDEFINED
#define TensorProto_DataType_FLOAT TensorProto::DataType::FLOAT
#define TensorProto_DataType_UINT8 TensorProto::DataType::UINT8
#define TensorProto_DataType_INT8 TensorProto::DataType::INT8
#define TensorProto_DataType_UINT16 TensorProto::DataType::UINT16
#define TensorProto_DataType_INT16 TensorProto::DataType::INT16
#define TensorProto_DataType_INT32 TensorProto::DataType::INT32
#define TensorProto_DataType_INT64 TensorProto::DataType::INT64
#define TensorProto_DataType_STRING TensorProto::DataType::STRING
#define TensorProto_DataType_BOOL TensorProto::DataType::BOOL
#define TensorProto_DataType_FLOAT16 TensorProto::DataType::FLOAT16
#define TensorProto_DataType_DOUBLE TensorProto::DataType::DOUBLE
#define TensorProto_DataType_UINT32 TensorProto::DataType::UINT32
#define TensorProto_DataType_UINT64 TensorProto::DataType::UINT64
#define TensorProto_DataType_COMPLEX64 TensorProto::DataType::COMPLEX64
#define TensorProto_DataType_COMPLEX128 TensorProto::DataType::COMPLEX128
#define TensorProto_DataType_BFLOAT16 TensorProto::DataType::BFLOAT16
#define TensorProto_DataType_FLOAT8E4M3FN TensorProto::DataType::FLOAT8E4M3FN
#define TensorProto_DataType_FLOAT8E4M3FNUZ TensorProto::DataType::FLOAT8E4M3FNUZ
#define TensorProto_DataType_FLOAT8E5M2 TensorProto::DataType::FLOAT8E5M2
#define TensorProto_DataType_FLOAT8E5M2FNUZ TensorProto::DataType::FLOAT8E5M2FNUZ
#define TensorProto_DataType_UINT4 TensorProto::DataType::UINT4
#define TensorProto_DataType_INT4 TensorProto::DataType::INT4
#define TensorProto_DataType_FLOAT4E2M1 TensorProto::DataType::FLOAT4E2M1
#define TensorProto_DataType_FLOAT8E8M0 TensorProto::DataType::FLOAT8E8M0
#define TensorProto_DataType_UINT2 TensorProto::DataType::UINT2
#define TensorProto_DataType_INT2 TensorProto::DataType::INT2

#define AttributeProto_AttributeType_INT AttributeProto::AttributeType::INT
#define AttributeProto_AttributeType_INTS AttributeProto::AttributeType::INTS
#define AttributeProto_AttributeType_FLOAT AttributeProto::AttributeType::FLOAT
#define AttributeProto_AttributeType_FLOATS AttributeProto::AttributeType::FLOATS
#define AttributeProto_AttributeType_GRAPH AttributeProto::AttributeType::GRAPH
#define AttributeProto_AttributeType_GRAPHS AttributeProto::AttributeType::GRAPHS
#define AttributeProto_AttributeType_STRING AttributeProto::AttributeType::STRING
#define AttributeProto_AttributeType_STRINGS AttributeProto::AttributeType::STRINGS
#define AttributeProto_AttributeType_TENSOR AttributeProto::AttributeType::TENSOR
#define AttributeProto_AttributeType_TENSORS AttributeProto::AttributeType::TENSORS
#define AttributeProto_AttributeType_TYPE_PROTO AttributeProto::AttributeType::TYPE_PROTO
#define AttributeProto_AttributeType_TYPE_PROTOS AttributeProto::AttributeType::TYPE_PROTOS

} // namespace ONNX_LIGHT_NAMESPACE
