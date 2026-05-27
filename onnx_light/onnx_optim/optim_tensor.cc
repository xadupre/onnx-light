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

namespace {

// Compares two :cpp:class:`OptimDim` values and updates the running
// precision accumulators. Sets ``conflict`` when the dimensions hold
// incompatible information (two different concrete integers or two
// different symbolic expressions). When one side is a concrete integer
// and the other a symbolic expression, the concrete side is considered
// more precise.
void CmpDimAccumulate(const OptimDim &a, const OptimDim &b, bool &lhs_more, bool &rhs_more,
                      bool &conflict) noexcept {
  if (a.IsInt() && b.IsInt()) {
    if (a.AsInt() != b.AsInt()) {
      conflict = true;
    }
    return;
  }
  if (a.IsExpr() && b.IsExpr()) {
    if (a.AsExpr() != b.AsExpr()) {
      conflict = true;
    }
    return;
  }
  if (a.IsInt()) {
    lhs_more = true;
  } else {
    rhs_more = true;
  }
}

// Compares two :cpp:class:`OptimShape` values (already known to have
// the same rank) and updates the running precision accumulators.
void CmpShapeAccumulate(const OptimShape &a, const OptimShape &b, bool &lhs_more, bool &rhs_more,
                        bool &conflict) noexcept {
  for (std::size_t i = 0; i < a.Rank(); ++i) {
    CmpDimAccumulate(a[i], b[i], lhs_more, rhs_more, conflict);
  }
}

} // namespace

OptimCmpResult OptimTensor::Cmp(const OptimTensor &other) const noexcept {
  bool lhs_more = false;
  bool rhs_more = false;
  bool conflict = false;

  // Element type.
  const bool lhs_dt_known = dtype_ != TensorType::kUndefined;
  const bool rhs_dt_known = other.dtype_ != TensorType::kUndefined;
  if (lhs_dt_known && rhs_dt_known) {
    if (dtype_ != other.dtype_) {
      conflict = true;
    }
  } else if (lhs_dt_known) {
    lhs_more = true;
  } else if (rhs_dt_known) {
    rhs_more = true;
  }

  // Shape: ranks must match; otherwise the descriptors are incompatible.
  if (shape_.Rank() != other.shape_.Rank()) {
    conflict = true;
  } else {
    CmpShapeAccumulate(shape_, other.shape_, lhs_more, rhs_more, conflict);
  }

  // Value-as-shape annotation.
  if (value_as_shape_.has_value() && other.value_as_shape_.has_value()) {
    const OptimShape &a = *value_as_shape_;
    const OptimShape &b = *other.value_as_shape_;
    if (a.Rank() != b.Rank()) {
      conflict = true;
    } else {
      CmpShapeAccumulate(a, b, lhs_more, rhs_more, conflict);
    }
  } else if (value_as_shape_.has_value()) {
    lhs_more = true;
  } else if (other.value_as_shape_.has_value()) {
    rhs_more = true;
  }

  // Data-pointer presence. Two distinct non-null pointers carry no
  // precision signal because the contents are not inspected.
  if (data_ != nullptr && other.data_ == nullptr) {
    lhs_more = true;
  } else if (data_ == nullptr && other.data_ != nullptr) {
    rhs_more = true;
  }

  if (conflict) {
    return OptimCmpResult::kConflict;
  }
  if (lhs_more && rhs_more) {
    return OptimCmpResult::kComplementary;
  }
  if (rhs_more) {
    return OptimCmpResult::kLessPrecise;
  }
  return OptimCmpResult::kMorePrecise;
}

} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
