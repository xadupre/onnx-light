// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/generator/shape_generator.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace generator {

namespace {

// Maps a TensorProto::DataType to the matching ``onnx_op::TensorType``
// enumerator used by ``OptimTensor``. Only the data types reachable
// through one of the Constant ``value*`` attributes are listed; every
// other input maps to ``kUndefined``.
TensorType DataTypeToTensorType(TensorProto::DataType dtype) {
  switch (dtype) {
  case TensorProto::DataType::BOOL:
    return TensorType::kBool;
  case TensorProto::DataType::UINT8:
    return TensorType::kUint8;
  case TensorProto::DataType::UINT16:
    return TensorType::kUint16;
  case TensorProto::DataType::UINT32:
    return TensorType::kUint32;
  case TensorProto::DataType::UINT64:
    return TensorType::kUint64;
  case TensorProto::DataType::INT8:
    return TensorType::kInt8;
  case TensorProto::DataType::INT16:
    return TensorType::kInt16;
  case TensorProto::DataType::INT32:
    return TensorType::kInt32;
  case TensorProto::DataType::INT64:
    return TensorType::kInt64;
  case TensorProto::DataType::FLOAT16:
    return TensorType::kFloat16;
  case TensorProto::DataType::BFLOAT16:
    return TensorType::kBfloat16;
  case TensorProto::DataType::FLOAT:
    return TensorType::kFloat;
  case TensorProto::DataType::DOUBLE:
    return TensorType::kDouble;
  case TensorProto::DataType::STRING:
    return TensorType::kString;
  case TensorProto::DataType::COMPLEX64:
    return TensorType::kComplex64;
  case TensorProto::DataType::COMPLEX128:
    return TensorType::kComplex128;
  case TensorProto::DataType::FLOAT8E4M3FN:
    return TensorType::kFloat8e4m3fn;
  case TensorProto::DataType::FLOAT8E4M3FNUZ:
    return TensorType::kFloat8e4m3fnuz;
  case TensorProto::DataType::FLOAT8E5M2:
    return TensorType::kFloat8e5m2;
  case TensorProto::DataType::FLOAT8E5M2FNUZ:
    return TensorType::kFloat8e5m2fnuz;
  case TensorProto::DataType::FLOAT8E8M0:
    return TensorType::kFloat8e8m0;
  case TensorProto::DataType::FLOAT4E2M1:
    return TensorType::kFloat4e2m1;
  case TensorProto::DataType::UINT4:
    return TensorType::kUint4;
  case TensorProto::DataType::INT4:
    return TensorType::kInt4;
  case TensorProto::DataType::UINT2:
    return TensorType::kUint2;
  case TensorProto::DataType::INT2:
    return TensorType::kInt2;
  default:
    return TensorType::kUndefined;
  }
}

// Builds an ``OptimShape`` from the ``dims`` repeated field of a
// ``TensorProto`` (which uses ``uint64_t`` storage but encodes
// non-negative shape values).
OptimShape ShapeFromTensorProtoDims(const TensorProto &tensor_proto) {
  OptimShape shape;
  for (int i = 0; i < tensor_proto.dims().size(); ++i) {
    shape.PushBack(OptimDim(static_cast<int64_t>(tensor_proto.dims()[i])));
  }
  return shape;
}

// Returns ``true`` when ``t`` is an integer scalar/element type for
// which ``ValueAsShape`` is meaningful (i.e. the tensor's content can
// legitimately be interpreted as shape dimensions).
bool IsIntegerTensorType(TensorType t) {
  switch (t) {
  case TensorType::kInt8:
  case TensorType::kInt16:
  case TensorType::kInt32:
  case TensorType::kInt64:
  case TensorType::kUint8:
  case TensorType::kUint16:
  case TensorType::kUint32:
  case TensorType::kUint64:
    return true;
  default:
    return false;
  }
}

// Extracts the integer values of ``tensor_proto`` into ``out``. Reads
// from the type-specific repeated field when available, otherwise
// falls back to ``raw_data`` (little-endian, as required by the ONNX
// spec). Only the integer data types accepted by
// :cpp:func:`IsIntegerTensorType` are supported; ``out`` is left
// untouched and the function returns ``false`` when the underlying
// data is not present in any recognised location.
bool ReadIntegerValues(const TensorProto &tensor_proto, std::vector<int64_t> &out) {
  const auto dtype = tensor_proto.data_type();
  out.clear();

  // Type-specific storage takes precedence over raw_data when populated.
  if (dtype == TensorProto::DataType::INT64 && tensor_proto.int64_data().size() > 0) {
    out.reserve(tensor_proto.int64_data().size());
    for (int i = 0; i < tensor_proto.int64_data().size(); ++i) {
      out.push_back(tensor_proto.int64_data()[i]);
    }
    return true;
  }
  if ((dtype == TensorProto::DataType::INT32 || dtype == TensorProto::DataType::INT16 ||
       dtype == TensorProto::DataType::INT8 || dtype == TensorProto::DataType::UINT16 ||
       dtype == TensorProto::DataType::UINT8) &&
      tensor_proto.int32_data().size() > 0) {
    out.reserve(tensor_proto.int32_data().size());
    for (int i = 0; i < tensor_proto.int32_data().size(); ++i) {
      out.push_back(static_cast<int64_t>(tensor_proto.int32_data()[i]));
    }
    return true;
  }
  if ((dtype == TensorProto::DataType::UINT64 || dtype == TensorProto::DataType::UINT32) &&
      tensor_proto.uint64_data().size() > 0) {
    out.reserve(tensor_proto.uint64_data().size());
    for (int i = 0; i < tensor_proto.uint64_data().size(); ++i) {
      out.push_back(static_cast<int64_t>(tensor_proto.uint64_data()[i]));
    }
    return true;
  }

  // Fall back to raw_data (little-endian fixed-width).
  if (!tensor_proto.is_raw_data()) {
    return false;
  }
  const utils::ByteSpan &raw = tensor_proto.raw_data();
  const uint8_t *bytes = raw.data();
  const size_t nbytes = raw.size();
  auto read_le = [&](size_t element_bytes, bool is_signed, size_t count) {
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      uint64_t u = 0;
      for (size_t b = 0; b < element_bytes; ++b) {
        u |= static_cast<uint64_t>(bytes[i * element_bytes + b]) << (8 * b);
      }
      int64_t v;
      if (is_signed) {
        // Sign-extend ``element_bytes``-wide value.
        const uint64_t sign_bit = uint64_t{1} << (element_bytes * 8 - 1);
        if (u & sign_bit) {
          // Fill the high bits with 1s.
          const uint64_t mask = ~((uint64_t{1} << (element_bytes * 8)) - 1);
          v = static_cast<int64_t>(u | mask);
        } else {
          v = static_cast<int64_t>(u);
        }
      } else {
        v = static_cast<int64_t>(u);
      }
      out.push_back(v);
    }
  };
  switch (dtype) {
  case TensorProto::DataType::INT64:
    if (nbytes % 8 != 0)
      return false;
    read_le(8, /*is_signed=*/true, nbytes / 8);
    return true;
  case TensorProto::DataType::UINT64:
    if (nbytes % 8 != 0)
      return false;
    read_le(8, /*is_signed=*/false, nbytes / 8);
    return true;
  case TensorProto::DataType::INT32:
    if (nbytes % 4 != 0)
      return false;
    read_le(4, /*is_signed=*/true, nbytes / 4);
    return true;
  case TensorProto::DataType::UINT32:
    if (nbytes % 4 != 0)
      return false;
    read_le(4, /*is_signed=*/false, nbytes / 4);
    return true;
  case TensorProto::DataType::INT16:
    if (nbytes % 2 != 0)
      return false;
    read_le(2, /*is_signed=*/true, nbytes / 2);
    return true;
  case TensorProto::DataType::UINT16:
    if (nbytes % 2 != 0)
      return false;
    read_le(2, /*is_signed=*/false, nbytes / 2);
    return true;
  case TensorProto::DataType::INT8:
    read_le(1, /*is_signed=*/true, nbytes);
    return true;
  case TensorProto::DataType::UINT8:
    read_le(1, /*is_signed=*/false, nbytes);
    return true;
  default:
    return false;
  }
}

// If ``tensor`` carries small integer content compatible with a shape
// value (rank ≤ 1, fewer than ``kConstantValueAsShapeMaxElements``
// elements, integer dtype), stamps it with a ``ValueAsShape``
// annotation reusing the integer values as dims.
void MaybeSetValueAsShapeFromTensorProto(OptimTensor &tensor, const TensorProto &tensor_proto) {
  if (!IsIntegerTensorType(tensor.Dtype())) {
    return;
  }
  if (tensor.Shape().Rank() > 1) {
    return;
  }
  int64_t count = 1;
  for (std::size_t i = 0; i < tensor.Shape().Rank(); ++i) {
    count *= tensor.Shape()[i].AsInt();
  }
  if (count < 0 || count >= kConstantValueAsShapeMaxElements) {
    return;
  }
  std::vector<int64_t> values;
  if (!ReadIntegerValues(tensor_proto, values)) {
    return;
  }
  // For a 0-D scalar the tensor proto stores a single element; for a
  // 1-D tensor the number of stored elements must match the dimension.
  if (static_cast<int64_t>(values.size()) != count) {
    return;
  }
  OptimShape value_shape;
  for (int64_t v : values) {
    value_shape.PushBack(OptimDim(v));
  }
  tensor.SetValueAsShape(std::move(value_shape));
}

// Counts how many of the supported Constant ``value*`` attributes are
// present on ``node``. Stores pointers to the first one of each kind
// in the corresponding out-parameter (``nullptr`` when absent).
int CollectConstantValueAttributes(
    const NodeProto &node, const AttributeProto *&value, const AttributeProto *&sparse_value,
    const AttributeProto *&value_int, const AttributeProto *&value_ints,
    const AttributeProto *&value_float, const AttributeProto *&value_floats,
    const AttributeProto *&value_string, const AttributeProto *&value_strings) {
  value = sparse_value = value_int = value_ints = value_float = value_floats = value_string =
      value_strings = nullptr;
  for (int i = 0; i < node.attribute().size(); ++i) {
    const AttributeProto &attr = node.attribute()[i];
    const std::string name = attr.name().as_string();
    if (name == "value" && value == nullptr) {
      value = &attr;
    } else if (name == "sparse_value" && sparse_value == nullptr) {
      sparse_value = &attr;
    } else if (name == "value_int" && value_int == nullptr) {
      value_int = &attr;
    } else if (name == "value_ints" && value_ints == nullptr) {
      value_ints = &attr;
    } else if (name == "value_float" && value_float == nullptr) {
      value_float = &attr;
    } else if (name == "value_floats" && value_floats == nullptr) {
      value_floats = &attr;
    } else if (name == "value_string" && value_string == nullptr) {
      value_string = &attr;
    } else if (name == "value_strings" && value_strings == nullptr) {
      value_strings = &attr;
    }
  }
  int count = 0;
  for (const AttributeProto *a : {value, sparse_value, value_int, value_ints, value_float,
                                  value_floats, value_string, value_strings}) {
    if (a != nullptr) {
      ++count;
    }
  }
  return count;
}

} // namespace

void ComputeShapeConstant(ShapesContext &ctx, const NodeProto &node) {
  if (node.op_type() != "Constant") {
    throw std::invalid_argument("ComputeShapeConstant expects op_type='Constant', got '" +
                                node.op_type().as_string() + "'.");
  }
  if (node.output_size() < 1) {
    throw std::invalid_argument("ComputeShapeConstant: node has no output.");
  }

  const AttributeProto *value = nullptr;
  const AttributeProto *sparse_value = nullptr;
  const AttributeProto *value_int = nullptr;
  const AttributeProto *value_ints = nullptr;
  const AttributeProto *value_float = nullptr;
  const AttributeProto *value_floats = nullptr;
  const AttributeProto *value_string = nullptr;
  const AttributeProto *value_strings = nullptr;
  const int non_null =
      CollectConstantValueAttributes(node, value, sparse_value, value_int, value_ints, value_float,
                                     value_floats, value_string, value_strings);
  if (non_null != 1) {
    throw std::invalid_argument(
        "ComputeShapeConstant: exactly one of the attributes 'value', 'sparse_value', "
        "'value_int', 'value_ints', 'value_float', 'value_floats', 'value_string' or "
        "'value_strings' must be specified for a Constant node.");
  }

  OptimTensor output;

  if (value != nullptr) {
    if (!value->has_t()) {
      throw std::invalid_argument(
          "ComputeShapeConstant: attribute 'value' must carry a tensor value.");
    }
    const TensorProto &tensor_proto = value->t();
    const TensorType dtype = DataTypeToTensorType(tensor_proto.data_type());
    OptimShape shape = ShapeFromTensorProtoDims(tensor_proto);
    output = OptimTensor(nullptr, dtype, std::move(shape));
    MaybeSetValueAsShapeFromTensorProto(output, tensor_proto);
  } else if (value_int != nullptr) {
    // Scalar INT64.
    output = OptimTensor(nullptr, TensorType::kInt64, OptimShape{});
    if (kConstantValueAsShapeMaxElements > 1) {
      OptimShape value_shape;
      value_shape.PushBack(OptimDim(value_int->i()));
      output.SetValueAsShape(std::move(value_shape));
    }
  } else if (value_ints != nullptr) {
    const auto &ints = value_ints->ints();
    OptimShape shape;
    shape.PushBack(OptimDim(static_cast<int64_t>(ints.size())));
    output = OptimTensor(nullptr, TensorType::kInt64, std::move(shape));
    if (static_cast<int64_t>(ints.size()) < kConstantValueAsShapeMaxElements) {
      OptimShape value_shape;
      for (int i = 0; i < ints.size(); ++i) {
        value_shape.PushBack(OptimDim(ints[i]));
      }
      output.SetValueAsShape(std::move(value_shape));
    }
  } else if (value_float != nullptr) {
    output = OptimTensor(nullptr, TensorType::kFloat, OptimShape{});
  } else if (value_floats != nullptr) {
    OptimShape shape;
    shape.PushBack(OptimDim(static_cast<int64_t>(value_floats->floats().size())));
    output = OptimTensor(nullptr, TensorType::kFloat, std::move(shape));
  } else if (value_string != nullptr) {
    output = OptimTensor(nullptr, TensorType::kString, OptimShape{});
  } else if (value_strings != nullptr) {
    OptimShape shape;
    shape.PushBack(OptimDim(static_cast<int64_t>(value_strings->strings().size())));
    output = OptimTensor(nullptr, TensorType::kString, std::move(shape));
  } else { // sparse_value
    if (!sparse_value->has_sparse_tensor()) {
      throw std::invalid_argument(
          "ComputeShapeConstant: attribute 'sparse_value' must carry a sparse tensor value.");
    }
    const SparseTensorProto &sparse = sparse_value->sparse_tensor();
    const TensorType dtype = DataTypeToTensorType(sparse.values().data_type());
    OptimShape shape;
    for (int i = 0; i < sparse.dims().size(); ++i) {
      shape.PushBack(OptimDim(static_cast<int64_t>(sparse.dims()[i])));
    }
    output = OptimTensor(nullptr, dtype, std::move(shape));
  }

  ctx.Set(node.output(0), std::move(output));
}

} // namespace generator
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
