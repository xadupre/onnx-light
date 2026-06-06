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

#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {

Device MakeGPUDevice(int index) {
  if (index < 0 || index > kMaxGPUIndex) {
    throw std::out_of_range("MakeGPUDevice index out of range");
  }
  return static_cast<Device>(static_cast<int32_t>(Device::kGPU0) + index);
}

std::string DeviceName(Device d) {
  switch (d) {
  case Device::kUndefined:
    return "Undefined";
  case Device::kCPU:
    return "CPU";
  default:
    break;
  }
  if (IsGPU(d)) {
    std::string out = "GPU";
    out.append(std::to_string(GPUIndex(d)));
    return out;
  }
  return "Unknown";
}

Device DeviceFromName(const std::string &name) {
  if (name == "CPU") {
    return Device::kCPU;
  }
  if (name == "Undefined") {
    return Device::kUndefined;
  }
  static constexpr const char *kGPUPrefix = "GPU";
  static constexpr std::size_t kGPUPrefixLen = 3;
  if (name.size() > kGPUPrefixLen && name.compare(0, kGPUPrefixLen, kGPUPrefix) == 0) {
    // The substring must be a non-empty sequence of decimal digits;
    // reject leading '+'/'-' or any other character to keep the
    // mapping unambiguous (e.g. "GPU+1" or "GPU 1" must not parse).
    for (std::size_t i = kGPUPrefixLen; i < name.size(); ++i) {
      if (name[i] < '0' || name[i] > '9') {
        return Device::kUndefined;
      }
    }
    try {
      const std::size_t index = std::stoul(name.substr(kGPUPrefixLen));
      if (index <= static_cast<std::size_t>(kMaxGPUIndex)) {
        return MakeGPUDevice(static_cast<int>(index));
      }
    } catch (const std::exception &) {
      // Fall through to kUndefined on overflow or parse error.
    }
  }
  return Device::kUndefined;
}

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
 * Returns the string representation of an ``OptimDim``: either the
 * decimal form of the integer value or the symbolic expression string.
 */
std::string OptimDim::ToString() const {
  if (IsInt()) {
    return std::to_string(AsInt());
  }
  return AsExpr();
}

/**
 * Renders the shape as a comma-separated list enclosed in square
 * brackets, e.g. ``"[2,3,N]"``. A rank-0 shape returns ``"[]"``.
 */
std::string OptimShape::ToString() const {
  std::string out;
  out.reserve(2 + dims_.size() * 3);
  out.push_back('[');
  for (std::size_t i = 0; i < dims_.size(); ++i) {
    if (i > 0) {
      out.push_back(',');
    }
    out.append(dims_[i].ToString());
  }
  out.push_back(']');
  return out;
}

namespace {

const char *TensorTypeName(TensorType t) {
  switch (t) {
  case TensorType::kUndefined:
    return "Undefined";
  case TensorType::kBool:
    return "Bool";
  case TensorType::kString:
    return "String";
  case TensorType::kUint8:
    return "Uint8";
  case TensorType::kUint16:
    return "Uint16";
  case TensorType::kUint32:
    return "Uint32";
  case TensorType::kUint64:
    return "Uint64";
  case TensorType::kInt8:
    return "Int8";
  case TensorType::kInt16:
    return "Int16";
  case TensorType::kInt32:
    return "Int32";
  case TensorType::kInt64:
    return "Int64";
  case TensorType::kFloat16:
    return "Float16";
  case TensorType::kBfloat16:
    return "Bfloat16";
  case TensorType::kFloat:
    return "Float";
  case TensorType::kDouble:
    return "Double";
  case TensorType::kComplex64:
    return "Complex64";
  case TensorType::kComplex128:
    return "Complex128";
  case TensorType::kFloat8e4m3fn:
    return "Float8e4m3fn";
  case TensorType::kFloat8e4m3fnuz:
    return "Float8e4m3fnuz";
  case TensorType::kFloat8e5m2:
    return "Float8e5m2";
  case TensorType::kFloat8e5m2fnuz:
    return "Float8e5m2fnuz";
  case TensorType::kFloat8e8m0:
    return "Float8e8m0";
  case TensorType::kFloat4e2m1:
    return "Float4e2m1";
  case TensorType::kUint4:
    return "Uint4";
  case TensorType::kInt4:
    return "Int4";
  case TensorType::kUint2:
    return "Uint2";
  case TensorType::kInt2:
    return "Int2";
  default:
    return "Unknown";
  }
}

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

/**
 * Renders the tensor as ``"OptimTensor(dtype=<name>, shape=<shape>"``
 * optionally followed by ``", value_as_shape=<shape>"`` when a
 * value-as-shape annotation is set and ``", data=<ptr>"`` when the
 * tensor references a non-null buffer.
 */
std::string OptimTensor::ToString() const {
  std::ostringstream oss;
  oss << "OptimTensor(dtype=" << TensorTypeName(dtype_) << ", shape=" << shape_.ToString();
  if (device_ != Device::kUndefined) {
    oss << ", device=" << DeviceName(device_);
  }
  if (value_as_shape_.has_value()) {
    oss << ", value_as_shape=" << value_as_shape_->ToString();
  }
  if (min_.has_value()) {
    oss << ", min=" << *min_;
  }
  if (max_.has_value()) {
    oss << ", max=" << *max_;
  }
  if (data_ != nullptr) {
    oss << ", data=" << data_;
  }
  oss << ")";
  return oss.str();
}

void OptimTensor::SetMinMax(double min, double max) {
  if (min > max) {
    throw std::invalid_argument("OptimTensor::SetMinMax requires min <= max");
  }
  min_ = min;
  max_ = max;
}

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

  // Device.
  const bool lhs_dev_known = device_ != Device::kUndefined;
  const bool rhs_dev_known = other.device_ != Device::kUndefined;
  if (lhs_dev_known && rhs_dev_known) {
    if (device_ != other.device_) {
      conflict = true;
    }
  } else if (lhs_dev_known) {
    lhs_more = true;
  } else if (rhs_dev_known) {
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

  // Min bound: a known bound is more precise than an absent one;
  // between two known bounds, the higher value is the tighter (more
  // precise) lower bound.
  if (min_.has_value() && other.min_.has_value()) {
    if (*min_ > *other.min_) {
      lhs_more = true;
    } else if (*min_ < *other.min_) {
      rhs_more = true;
    }
  } else if (min_.has_value()) {
    lhs_more = true;
  } else if (other.min_.has_value()) {
    rhs_more = true;
  }

  // Max bound: a known bound is more precise than an absent one;
  // between two known bounds, the lower value is the tighter (more
  // precise) upper bound.
  if (max_.has_value() && other.max_.has_value()) {
    if (*max_ < *other.max_) {
      lhs_more = true;
    } else if (*max_ > *other.max_) {
      rhs_more = true;
    }
  } else if (max_.has_value()) {
    lhs_more = true;
  } else if (other.max_.has_value()) {
    rhs_more = true;
  }

  // Cross-side interval disjointedness: when both sides expose enough
  // bounds to prove the intervals do not overlap, the descriptors
  // contradict each other.
  if (min_.has_value() && other.max_.has_value() && *min_ > *other.max_) {
    conflict = true;
  }
  if (other.min_.has_value() && max_.has_value() && *other.min_ > *max_) {
    conflict = true;
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

namespace {

// Builds an OptimShape from a TensorShapeProto, preserving symbolic
// dimensions: ``dim_value`` becomes a concrete int dim, ``dim_param``
// becomes a symbolic dim with the same name, and an unset dim becomes
// a fresh ``"?"`` placeholder. Mirrors the historical helper that
// lived in shape_inference.cc.
OptimShape ShapeFromTensorShapeProto(const TensorShapeProto &sp) {
  OptimShape shape;
  for (int i = 0; i < sp.dim().size(); ++i) {
    const TensorShapeProto::Dimension &d = sp.dim()[i];
    if (d.has_dim_value()) {
      shape.PushBack(OptimDim(static_cast<int64_t>(d.dim_value())));
    } else if (d.has_dim_param()) {
      shape.PushBack(OptimDim(std::string(d.dim_param().data(), d.dim_param().size())));
    } else {
      shape.PushBack(OptimDim(std::string("?")));
    }
  }
  return shape;
}

// Returns the index of the metadata entry whose key matches ``key``,
// or -1 when none is present.
int FindMetadataIndex(const ValueInfoProto &vi, const char *key) {
  for (int i = 0; i < vi.metadata_props().size(); ++i) {
    if (vi.metadata_props()[i].key().as_string() == key) {
      return i;
    }
  }
  return -1;
}

int FindDeviceMetadataIndex(const ValueInfoProto &vi) {
  return FindMetadataIndex(vi, kValueInfoDeviceMetadataKey);
}

// Removes the metadata entry at ``idx`` from ``vi`` in place using a
// swap-and-pop. ``idx`` must reference a valid entry.
void RemoveMetadataAt(ValueInfoProto &vi, int idx) {
  std::vector<StringStringEntryProto> &storage = vi.metadata_props().mutable_values();
  const std::size_t last = storage.size() - 1;
  const std::size_t i = static_cast<std::size_t>(idx);
  if (i != last) {
    const std::string key = storage[last].key().as_string();
    const std::string value = storage[last].value().as_string();
    storage[i].set_key(key);
    storage[i].set_value(value);
  }
  storage.pop_back();
}

// Writes/updates/removes a numeric metadata entry on ``vi``: when
// ``value`` is present, the entry keyed by ``key`` is updated in place
// or appended; when ``value`` is absent any pre-existing entry is
// removed. Numeric values are serialised with ``std::to_string`` which
// is locale-independent and round-trips through ``std::stod``.
void SetOrRemoveNumericMetadata(ValueInfoProto &vi, const char *key,
                                const std::optional<double> &value) {
  const int idx = FindMetadataIndex(vi, key);
  if (value.has_value()) {
    StringStringEntryProto *entry = idx >= 0
                                        ? vi.mutable_metadata_props(static_cast<std::size_t>(idx))
                                        : vi.add_metadata_props();
    entry->set_key(key);
    entry->set_value(std::to_string(*value));
  } else if (idx >= 0) {
    RemoveMetadataAt(vi, idx);
  }
}

// Reads an ``std::optional<double>`` from the metadata entry keyed by
// ``key`` on ``vi``. Returns an absent optional when the entry is
// missing or unparsable.
std::optional<double> ReadNumericMetadata(const ValueInfoProto &vi, const char *key) {
  const int idx = FindMetadataIndex(vi, key);
  if (idx < 0) {
    return std::nullopt;
  }
  try {
    std::size_t consumed = 0;
    const std::string value = vi.metadata_props()[idx].value().as_string();
    const double parsed = std::stod(value, &consumed);
    if (consumed != value.size()) {
      return std::nullopt;
    }
    return parsed;
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

} // namespace

bool OptimTensorFromValueInfo(const ValueInfoProto &vi, OptimTensor &out) {
  if (!vi.has_type() || !vi.type().has_tensor_type()) {
    return false;
  }
  const TypeProto::Tensor &tt = vi.type().tensor_type();
  const TensorType dtype = DataTypeToTensorType(tt.elem_type());
  OptimShape shape;
  if (tt.has_shape()) {
    shape = ShapeFromTensorShapeProto(tt.shape());
  }
  out = OptimTensor(nullptr, dtype, std::move(shape));
  const int idx = FindDeviceMetadataIndex(vi);
  if (idx >= 0) {
    const Device device = DeviceFromName(vi.metadata_props()[idx].value().as_string());
    if (device != Device::kUndefined) {
      out.SetDevice(device);
    }
  }
  const std::optional<double> min_v = ReadNumericMetadata(vi, kValueInfoMinMetadataKey);
  if (min_v.has_value()) {
    out.SetMin(*min_v);
  }
  const std::optional<double> max_v = ReadNumericMetadata(vi, kValueInfoMaxMetadataKey);
  if (max_v.has_value()) {
    out.SetMax(*max_v);
  }
  return true;
}

bool OptimTensorFromTensorProto(const TensorProto &tp, OptimTensor &out) {
  const TensorType dtype = DataTypeToTensorType(tp.data_type());
  if (dtype == TensorType::kUndefined) {
    return false;
  }
  OptimTensor tensor(nullptr, dtype, ShapeFromTensorProtoDims(tp));

  // Determine the element count from the dims; used both to gate the
  // value-as-shape heuristic and to validate decoded payload sizes.
  int64_t count = 1;
  bool count_known = true;
  for (std::size_t i = 0; i < tensor.Shape().Rank(); ++i) {
    const OptimDim &d = tensor.Shape()[i];
    if (!d.IsInt() || d.AsInt() < 0) {
      count_known = false;
      break;
    }
    count *= d.AsInt();
  }

  // Record the actual value bounds when the payload is readable. This
  // covers both integer and floating-point element types; tensors whose
  // data is absent (no typed field, no raw_data) or in an unsupported
  // dtype are left unannotated.
  if (count_known && count > 0) {
    if (IsIntegerTensorType(dtype)) {
      std::vector<int64_t> values;
      if (ReadIntegerValues(tp, values) && static_cast<int64_t>(values.size()) == count) {
        const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
        tensor.SetMinMax(static_cast<double>(*min_it), static_cast<double>(*max_it));
      }
    } else if (dtype == TensorType::kFloat || dtype == TensorType::kDouble) {
      std::vector<double> values;
      if (ReadFloatingValues(tp, values) && static_cast<int64_t>(values.size()) == count) {
        const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
        tensor.SetMinMax(*min_it, *max_it);
      }
    }
  }

  // Stamp small 1-D / 0-D integer tensors with a ValueAsShape annotation
  // so that downstream ops (such as Reshape) can see the actual target
  // shape values without re-reading the proto.
  if (IsIntegerTensorType(dtype) && tensor.Shape().Rank() <= 1 && count_known && count >= 0 &&
      count < kOptimValueAsShapeMaxElements) {
    std::vector<int64_t> values;
    if (ReadIntegerValues(tp, values) && static_cast<int64_t>(values.size()) == count) {
      OptimShape value_shape;
      for (int64_t v : values) {
        value_shape.PushBack(OptimDim(v));
      }
      tensor.SetValueAsShape(std::move(value_shape));
    }
  }

  out = std::move(tensor);
  return true;
}

bool OptimTensorToValueInfo(const OptimTensor &tensor, ValueInfoProto &vi) {
  const TensorProto::DataType dtype = TensorTypeToDataType(tensor.Dtype());
  if (dtype == TensorProto::DataType::UNDEFINED) {
    return false;
  }
  // Reset any pre-existing type/shape information so it is replaced
  // wholesale by the inferred descriptor.
  vi.clear_type();
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(static_cast<int>(dtype));
  TensorShapeProto *sp = tt->add_shape();
  for (std::size_t i = 0; i < tensor.Shape().Rank(); ++i) {
    const OptimDim &d = tensor.Shape()[i];
    TensorShapeProto::Dimension *dim = sp->add_dim();
    if (d.IsInt()) {
      dim->set_dim_value(d.AsInt());
    } else {
      dim->set_dim_param(d.AsExpr());
    }
  }
  // Round-trip the device through metadata_props. When the device is
  // known, update an existing entry in place or append a new one;
  // when undefined, drop any existing entry so the wire form does not
  // grow stale information.
  const int idx = FindDeviceMetadataIndex(vi);
  if (tensor.GetDevice() != Device::kUndefined) {
    StringStringEntryProto *entry = idx >= 0
                                        ? vi.mutable_metadata_props(static_cast<std::size_t>(idx))
                                        : vi.add_metadata_props();
    entry->set_key(kValueInfoDeviceMetadataKey);
    entry->set_value(DeviceName(tensor.GetDevice()));
  } else if (idx >= 0) {
    RemoveMetadataAt(vi, idx);
  }
  // Round-trip the optional ``min``/``max`` bounds the same way. An
  // absent bound removes the corresponding metadata entry (if any).
  SetOrRemoveNumericMetadata(vi, kValueInfoMinMetadataKey,
                             tensor.HasMin() ? std::optional<double>(tensor.Min())
                                             : std::optional<double>());
  SetOrRemoveNumericMetadata(vi, kValueInfoMaxMetadataKey,
                             tensor.HasMax() ? std::optional<double>(tensor.Max())
                                             : std::optional<double>());
  return true;
}

} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
