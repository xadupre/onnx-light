// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file optim_tensor.cc
 * @brief Out-of-line implementation of :cpp:class:`OptimShape`.
 *
 * Most of the ``onnx_optim`` public API is small enough to be defined
 * inline in ``optim_tensor.h``. Only the few :cpp:class:`OptimShape`
 * members that perform bounds checking, iteration over the stored
 * dimensions, or arithmetic on integer dimensions live here so that
 * the header stays free of ``<stdexcept>``.
 *
 * @see optim_tensor.h
 */

#include "onnx_optim/optim_tensor.h"

#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {

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

TensorProto::DataType TensorTypeToDataType(TensorType t) {
  switch (t) {
  case TensorType::kBool:
    return TensorProto::DataType::BOOL;
  case TensorType::kString:
    return TensorProto::DataType::STRING;
  case TensorType::kUint8:
    return TensorProto::DataType::UINT8;
  case TensorType::kUint16:
    return TensorProto::DataType::UINT16;
  case TensorType::kUint32:
    return TensorProto::DataType::UINT32;
  case TensorType::kUint64:
    return TensorProto::DataType::UINT64;
  case TensorType::kInt8:
    return TensorProto::DataType::INT8;
  case TensorType::kInt16:
    return TensorProto::DataType::INT16;
  case TensorType::kInt32:
    return TensorProto::DataType::INT32;
  case TensorType::kInt64:
    return TensorProto::DataType::INT64;
  case TensorType::kFloat16:
    return TensorProto::DataType::FLOAT16;
  case TensorType::kBfloat16:
    return TensorProto::DataType::BFLOAT16;
  case TensorType::kFloat:
    return TensorProto::DataType::FLOAT;
  case TensorType::kDouble:
    return TensorProto::DataType::DOUBLE;
  case TensorType::kComplex64:
    return TensorProto::DataType::COMPLEX64;
  case TensorType::kComplex128:
    return TensorProto::DataType::COMPLEX128;
  case TensorType::kFloat8e4m3fn:
    return TensorProto::DataType::FLOAT8E4M3FN;
  case TensorType::kFloat8e4m3fnuz:
    return TensorProto::DataType::FLOAT8E4M3FNUZ;
  case TensorType::kFloat8e5m2:
    return TensorProto::DataType::FLOAT8E5M2;
  case TensorType::kFloat8e5m2fnuz:
    return TensorProto::DataType::FLOAT8E5M2FNUZ;
  case TensorType::kFloat8e8m0:
    return TensorProto::DataType::FLOAT8E8M0;
  case TensorType::kFloat4e2m1:
    return TensorProto::DataType::FLOAT4E2M1;
  case TensorType::kUint4:
    return TensorProto::DataType::UINT4;
  case TensorType::kInt4:
    return TensorProto::DataType::INT4;
  case TensorType::kUint2:
    return TensorProto::DataType::UINT2;
  case TensorType::kInt2:
    return TensorProto::DataType::INT2;
  default:
    return TensorProto::DataType::UNDEFINED;
  }
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

/**
 * Constructs an :cpp:class:`OptimShape` from a brace-enclosed list of
 * dimensions.
 *
 * @param dims Dimensions to copy into the new shape, in order.
 * @throws std::length_error if ``dims.size() > kMaxOptimRank``.
 */
OptimShape::OptimShape(std::initializer_list<OptimDim> dims) {
  if (dims.size() > kMaxOptimRank) {
    throw std::length_error("OptimShape exceeds maximum rank");
  }
  dims_.reserve(dims.size());
  for (const auto &d : dims) {
    dims_.push_back(d);
  }
}

/**
 * Constructs an :cpp:class:`OptimShape` by copying an existing
 * ``std::vector`` of :cpp:class:`OptimDim`.
 *
 * @param dims Source dimensions to copy.
 * @throws std::length_error if ``dims.size() > kMaxOptimRank``.
 */
OptimShape::OptimShape(const std::vector<OptimDim> &dims) {
  if (dims.size() > kMaxOptimRank) {
    throw std::length_error("OptimShape exceeds maximum rank");
  }
  dims_ = dims;
}

/**
 * Appends a dimension to the shape.
 *
 * @param dim Dimension to append; may be integer or symbolic.
 * @throws std::length_error if the shape already contains
 *         ``kMaxOptimRank`` dimensions.
 */
void OptimShape::PushBack(OptimDim dim) {
  if (dims_.size() >= kMaxOptimRank) {
    throw std::length_error("OptimShape exceeds maximum rank");
  }
  dims_.push_back(std::move(dim));
}

/**
 * Returns ``true`` when every dimension in the shape is a concrete
 * integer. A rank-0 (empty) shape is considered fully known.
 */
bool OptimShape::IsFullyKnown() const noexcept {
  for (const auto &d : dims_) {
    if (!d.IsInt()) {
      return false;
    }
  }
  return true;
}

/**
 * Computes the product of every integer dimension.
 *
 * Returns ``1`` for a rank-0 (empty) shape, matching the standard
 * scalar element-count semantic.
 *
 * @throws std::runtime_error if any dimension is symbolic; check
 *         :cpp:func:`IsFullyKnown` first if the shape may be
 *         partially symbolic.
 */
int64_t OptimShape::NumElements() const {
  int64_t total = 1;
  for (const auto &d : dims_) {
    if (!d.IsInt()) {
      throw std::runtime_error("OptimShape::NumElements requires a fully-known shape");
    }
    total *= d.AsInt();
  }
  return total;
}

/**
 * Renders the dimension as a short human-readable token. Integer dimensions
 * are formatted with :cpp:func:`std::to_string`; symbolic dimensions are
 * rendered using their stored expression.
 */
std::string OptimDim::ToString() const { return IsInt() ? std::to_string(AsInt()) : AsExpr(); }

/**
 * Renders the tensor's dtype and shape as a short human-readable string,
 * suitable for diagnostic messages emitted by shape inference and other
 * optimisation passes.
 */
std::string OptimTensor::ToString() const {
  std::string s = "dtype=" + std::to_string(static_cast<int>(TensorTypeToDataType(dtype_)));
  s += ", shape=[";
  for (std::size_t i = 0; i < shape_.Rank(); ++i) {
    if (i != 0) {
      s += ",";
    }
    s += shape_[i].ToString();
  }
  s += "]";
  return s;
}

TensorComparison Compare(const OptimTensor &a, const OptimTensor &b) {
  // Track which side carries information the other does not. If both
  // sides have unique information and none of it conflicts, the
  // descriptors can be merged (kMerge). If only one side has extra
  // information, it is strictly more (or less) precise. If neither
  // side has extra information, the descriptors are equivalent and we
  // also report kMerge — merging is a no-op in that case.
  bool a_more = false;
  bool b_more = false;

  const TensorProto::DataType ad = TensorTypeToDataType(a.Dtype());
  const TensorProto::DataType bd = TensorTypeToDataType(b.Dtype());
  if (ad != TensorProto::DataType::UNDEFINED && bd != TensorProto::DataType::UNDEFINED &&
      ad != bd) {
    return TensorComparison::kConflict;
  }
  if (ad != TensorProto::DataType::UNDEFINED && bd == TensorProto::DataType::UNDEFINED) {
    a_more = true;
  } else if (ad == TensorProto::DataType::UNDEFINED && bd != TensorProto::DataType::UNDEFINED) {
    b_more = true;
  }

  if (a.Shape().Rank() != b.Shape().Rank()) {
    return TensorComparison::kConflict;
  }
  for (std::size_t i = 0; i < a.Shape().Rank(); ++i) {
    const OptimDim &da = a.Shape()[i];
    const OptimDim &db = b.Shape()[i];
    if (da.IsInt() && db.IsInt()) {
      if (da.AsInt() != db.AsInt()) {
        return TensorComparison::kConflict;
      }
    } else if (da.IsInt() && !db.IsInt()) {
      a_more = true;
    } else if (!da.IsInt() && db.IsInt()) {
      b_more = true;
    }
    // Symbolic vs. symbolic: treated as equally precise even when the
    // expressions differ — neither side dominates.
  }

  if (a_more && !b_more) {
    return TensorComparison::kMorePrecise;
  }
  if (b_more && !a_more) {
    return TensorComparison::kLessPrecise;
  }
  return TensorComparison::kMerge;
}

} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
