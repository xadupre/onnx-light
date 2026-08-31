// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "onnx_light_helpers.h"

// IMPORTANT: onnx_light_helpers.h must be included first so that
// ONNX_LIGHT_NAMESPACE is defined before tensor_type.h declares its namespace
// block. tensor_type.h intentionally does not include onnx_light_helpers.h
// itself (see onnx_op/test_onnx_op_*.cc for the #undef trick that relies on
// this).
#include "onnx_proto/onnx.h"
#include "onnx_proto/type_helper.h"

/**
 * @file sym_tensor.h
 * @brief Lightweight, non-owning tensor description used by ONNX
 *        graph optimisation passes.
 *
 * The ``core::symbolic`` namespace exposes three small value types,
 * consumed by the ``onnx_shapes`` shape-inference library:
 *
 *   - :cpp:class:`SymDim` — a single shape dimension, either a
 *     concrete ``int64_t`` or a symbolic string expression.
 *   - :cpp:class:`SymShape` — an ordered, bounded-rank collection of
 *     :cpp:class:`SymDim`.
 *   - :cpp:class:`SymTensor` — a non-owning view over a contiguous
 *     buffer carrying a :cpp:type:`TensorType`, an
 *     :cpp:class:`SymShape`, and an optional shape annotation when
 *     the tensor itself represents a shape (e.g. the ``shape`` input of
 *     ``Reshape``).
 *
 * These types are intentionally header-only friendly and never
 * allocate the tensor data they describe; callers are responsible for
 * the lifetime of the underlying buffer.
 */

namespace ONNX_LIGHT_NAMESPACE::core {
namespace shapes {
// Forward declaration: SymShape::FitsConcreteShape records the concrete
// values it binds to symbolic dimensions in a ShapesContext (defined in
// onnx_core/shapes/shapes_context.h, part of the same lib_onnx_core).
class ShapesContext;
} // namespace shapes
namespace symbolic {

/// Reuse the TensorType enumeration defined by ``onnx_core`` so that
/// ``onnx_shapes`` is fully aligned with the rest of the operator stack.
using TensorType = ONNX_LIGHT_NAMESPACE::onnx_proto::TensorType;

/**
 * Maps a ``TensorProto::DataType`` to the matching :cpp:type:`TensorType`
 * enumerator used by :cpp:class:`SymTensor`. Returns
 * :cpp:enumerator:`TensorType::kUndefined` for any data type that is not
 * representable in the ``onnx_shapes`` stack (e.g. ``UNDEFINED``).
 */
constexpr TensorType DataTypeToTensorType(TensorProto::DataType dtype) {
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
  case TensorProto::DataType::FLOAT6E2M3:
    return TensorType::kFloat6e2m3;
  case TensorProto::DataType::FLOAT6E3M2:
    return TensorType::kFloat6e3m2;
  default:
    return TensorType::kUndefined;
  }
}

/**
 * Maps a :cpp:type:`TensorType` enumerator back to the matching
 * ``TensorProto::DataType``. This is the inverse of
 * :cpp:func:`DataTypeToTensorType` and is used to write inferred
 * element types into ``ValueInfoProto`` / ``TypeProto::Tensor``.
 *
 * Only the scalar tensor types are supported (the sequence- and
 * optional-typed enumerators do not have a single matching scalar
 * ``DataType``). For sequence/optional types or for
 * :cpp:enumerator:`TensorType::kUndefined`, the function returns
 * ``TensorProto::DataType::UNDEFINED``.
 */
constexpr TensorProto::DataType TensorTypeToDataType(TensorType t) {
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
  case TensorType::kFloat6e2m3:
    return TensorProto::DataType::FLOAT6E2M3;
  case TensorType::kFloat6e3m2:
    return TensorProto::DataType::FLOAT6E3M2;
  default:
    return TensorProto::DataType::UNDEFINED;
  }
}

/**
 * Returns whether a tensor element type can be interpreted as shape
 * dimensions for ``ValueAsShape``.
 *
 * Accepted integer types are signed/unsigned 8, 16, 32 and 64-bit
 * integers.
 *
 * @param t Tensor element type to evaluate.
 * @return ``true`` if ``t`` is one of the supported integer types.
 */
constexpr bool IsIntegerTensorType(TensorType t) {
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

/**
 * A single shape dimension that is either a concrete non-negative integer or
 * a symbolic expression represented as a string. ``SymDim`` is used by
 * ``SymShape`` to describe both fully-known and partially-symbolic shapes.
 */
class SymDim {
public:
  /// Default constructs a zero-valued integer dimension.
  SymDim() : value_(static_cast<int64_t>(0)) {}

  /// Constructs an integer-valued dimension.
  SymDim(int64_t value) : value_(value) {}

  /// Constructs a symbolic dimension expressed as a string expression.
  SymDim(std::string expr) : value_(std::move(expr)) {}

  /// Convenience overload for C string literals.
  SymDim(const char *expr) : value_(std::string(expr)) {}

  /// Returns ``true`` when the dimension holds a concrete integer value.
  bool IsInt() const noexcept { return std::holds_alternative<int64_t>(value_); }

  /// Returns ``true`` when the dimension holds a symbolic expression string.
  bool IsExpr() const noexcept { return std::holds_alternative<std::string>(value_); }

  /// Returns the integer value. Throws ``std::bad_variant_access`` if the
  /// dimension is not an integer.
  int64_t AsInt() const { return std::get<int64_t>(value_); }

  /// Returns the symbolic expression. Throws ``std::bad_variant_access`` if
  /// the dimension is not a string expression.
  const std::string &AsExpr() const { return std::get<std::string>(value_); }

  /// Equality compares both the alternative and the underlying value.
  bool operator==(const SymDim &other) const noexcept { return value_ == other.value_; }
  bool operator!=(const SymDim &other) const noexcept { return !(*this == other); }

  /// Returns a human-readable representation of the dimension: the integer
  /// value for concrete dimensions, or the symbolic expression for string
  /// dimensions.
  std::string ToString() const;

private:
  std::variant<int64_t, std::string> value_;
};

/// Maximum number of dimensions an ``SymShape`` can describe inline. Shapes
/// in practice rarely exceed this rank, so storing the dimensions in a small
/// container keeps the structure compact and cache-friendly.
inline constexpr std::size_t kMaxOptimRank = 16;

/**
 * A short, value-typed shape composed of ``SymDim`` entries. The rank is
 * bounded by ``kMaxOptimRank`` so that the structure fits comfortably on the
 * stack. Symbolic and concrete dimensions can be mixed freely.
 */
class SymShape {
public:
  using value_type = SymDim;

  SymShape() = default;

  /// Constructs a shape from an initializer list of dimensions.
  SymShape(std::initializer_list<SymDim> dims);

  /// Constructs a shape from any iterable container of ``SymDim``.
  explicit SymShape(const std::vector<SymDim> &dims);

  /// Number of dimensions.
  std::size_t Rank() const noexcept { return dims_.size(); }

  /// ``true`` when the shape contains no dimensions (scalar / rank-0).
  bool Empty() const noexcept { return dims_.empty(); }

  /// Access a dimension by index. Throws ``std::out_of_range`` if invalid.
  const SymDim &operator[](std::size_t i) const { return dims_.at(i); }
  SymDim &operator[](std::size_t i) { return dims_.at(i); }

  /// Appends a new dimension. Throws ``std::length_error`` when the maximum
  /// rank ``kMaxOptimRank`` would be exceeded.
  void PushBack(SymDim dim);

  /// Returns ``true`` when every dimension is a concrete integer.
  bool IsFullyKnown() const noexcept;

  /// Computes the product of all integer dimensions. Returns ``1`` for a
  /// rank-0 (empty) shape, matching the standard scalar-element-count
  /// semantic. Throws ``std::runtime_error`` if any dimension is symbolic.
  int64_t NumElements() const;

  /// Returns ``true`` when the concrete shape described by ``rank`` /
  /// ``shape`` is compatible with this (possibly symbolic) shape.
  ///
  /// The concrete shape is a contiguous array of ``rank`` dimensions
  /// pointed to by ``shape``. Compatibility requires that ``rank`` equals
  /// :cpp:func:`Rank`, and, dimension by dimension: a concrete integer
  /// dimension must match exactly; a symbolic dimension is resolved against
  /// ``bindings`` (a :cpp:class:`core::shapes::ShapesContext` binding each
  /// symbolic expression to the integer value it first resolved to), so an
  /// inconsistent reuse of the same expression makes the shape incompatible;
  /// and a fully-unknown dimension (an empty expression) imposes no
  /// constraint. Newly resolved symbolic expressions are bound in
  /// ``bindings``.
  ///
  /// Returns:
  ///   ``true`` when the concrete shape fits, ``false`` otherwise.
  bool FitsConcreteShape(std::size_t rank, const int64_t *shape,
                         core::shapes::ShapesContext &bindings) const;

  /// Equality compares the dimensions element-wise.
  bool operator==(const SymShape &other) const noexcept { return dims_ == other.dims_; }
  bool operator!=(const SymShape &other) const noexcept { return !(*this == other); }

  /// Read-only access to the underlying dimensions.
  const std::vector<SymDim> &Dims() const noexcept { return dims_; }

  /// Returns a human-readable representation of the shape as a
  /// comma-separated list of dimensions enclosed in square brackets,
  /// e.g. ``"[2,3,N]"``. A rank-0 shape returns ``"[]"``.
  std::string ToString() const;

private:
  std::vector<SymDim> dims_;
};

/**
 * Builds an :cpp:class:`SymShape` from ``tensor_proto.dims()``.
 *
 * ONNX stores tensor dimensions as non-negative 64-bit integers in
 * ``TensorProto::dims``; this helper converts every dimension into an
 * :cpp:class:`SymDim` and appends them in order.
 *
 * @param tensor_proto Tensor whose ``dims`` field is converted.
 * @return Converted shape with the same rank and dimension values.
 */
SymShape ShapeFromTensorProtoDims(const TensorProto &tensor_proto);

/**
 * Result of comparing two :cpp:class:`SymTensor` descriptors with
 * :cpp:func:`SymTensor::Cmp`. The four outcomes describe the relative
 * precision of the two descriptors when they are interpreted as
 * statements about the same logical tensor.
 *
 * - :cpp:enumerator:`SymCmpResult::kConflict` — the two descriptors
 *   carry contradictory information (e.g. different known element types,
 *   different ranks, or incompatible concrete dimension values) and
 *   cannot both be true at the same time.
 * - :cpp:enumerator:`SymCmpResult::kMorePrecise` — ``*this`` is at
 *   least as precise as ``other`` on every field and strictly more
 *   precise on at least one (or both are equivalent — the equal case
 *   is reported as ``kMorePrecise``).
 * - :cpp:enumerator:`SymCmpResult::kLessPrecise` — ``other`` is
 *   strictly more precise than ``*this``.
 * - :cpp:enumerator:`SymCmpResult::kComplementary` — neither
 *   descriptor dominates: each one carries some information the other
 *   one is missing, but the two are mutually compatible.
 */
enum class SymCmpResult {
  kConflict,
  kMorePrecise,
  kLessPrecise,
  kComplementary,
};

/**
 * Maximum GPU index supported by :cpp:enum:`Device`. The enumeration
 * exposes one enumerator per GPU from ``kGPU0`` up to
 * ``kGPU<kMaxGPUIndex>`` (inclusive).
 */
inline constexpr int kMaxGPUIndex = 8191;

/**
 * Logical device on which an :cpp:class:`SymTensor` resides.
 *
 * The enumeration is intentionally compact: ``kUndefined`` denotes the
 * "no information" state (the default), ``kCPU`` denotes the host CPU,
 * and the contiguous range ``[kGPU0, kGPU<kMaxGPUIndex>]`` enumerates
 * up to ``kMaxGPUIndex + 1`` distinct GPU devices. The numeric value of
 * ``kGPU<i>`` is ``static_cast<int32_t>(kGPU0) + i``; the helpers
 * :cpp:func:`MakeGPUDevice`, :cpp:func:`IsGPU` and
 * :cpp:func:`GPUIndex` should be preferred over direct casts.
 */
enum class Device : int32_t {
  kUndefined = -2,
  kCPU = -1,
  /// First GPU device. ``kGPU<i>`` corresponds to
  /// ``static_cast<Device>(static_cast<int32_t>(kGPU0) + i)`` for
  /// ``0 <= i <= kMaxGPUIndex``.
  kGPU0 = 0,
  /// Last addressable GPU device (index ``kMaxGPUIndex``).
  kGPU8191 = kGPU0 + kMaxGPUIndex,
};

/**
 * Returns the :cpp:enumerator:`Device::kGPU0` + ``index`` enumerator.
 *
 * @param index GPU index in the inclusive range ``[0, kMaxGPUIndex]``.
 * @throws std::out_of_range if ``index`` is outside the supported range.
 */
Device MakeGPUDevice(int index);

/// Returns ``true`` when ``d`` is one of the ``kGPU0``..``kGPU8191``
/// enumerators.
constexpr bool IsGPU(Device d) noexcept {
  return static_cast<int32_t>(d) >= static_cast<int32_t>(Device::kGPU0) &&
         static_cast<int32_t>(d) <= static_cast<int32_t>(Device::kGPU8191);
}

/// Returns the GPU index of ``d`` (``0`` for ``kGPU0``,
/// ``kMaxGPUIndex`` for ``kGPU8191``) or ``-1`` when ``d`` is not a
/// GPU device.
constexpr int GPUIndex(Device d) noexcept {
  return IsGPU(d) ? static_cast<int>(static_cast<int32_t>(d) - static_cast<int32_t>(Device::kGPU0))
                  : -1;
}

/// Returns a human-readable name for ``d``: ``"Undefined"``, ``"CPU"``,
/// ``"GPU<i>"`` for GPU devices, or ``"Unknown"`` for any out-of-range
/// value.
std::string DeviceName(Device d);

/// Returns the suffix that encodes ``d`` into a dispatch-table key
/// identifier. The default host devices (:cpp:enumerator:`Device::kCPU` and
/// :cpp:enumerator:`Device::kUndefined`) return the empty string so that a
/// key keeps its plain ``"<domain>:<op_type>"`` form; any other device
/// returns ``":<device>"`` (a leading ``':'`` followed by the integer value
/// of the enumerator) to disambiguate. Shared by the kernel and peak-memory
/// dispatch tables so both key device-specific entries identically.
std::string DeviceKeySuffix(Device d);

/**
 * Parses a device name back into a :cpp:enum:`Device` enumerator.
 *
 * Recognises the exact strings produced by :cpp:func:`DeviceName`:
 * ``"Undefined"``, ``"CPU"`` and ``"GPU<i>"`` where ``i`` is a
 * decimal integer in ``[0, kMaxGPUIndex]``. Any other input —
 * including the empty string, ``"Unknown"``, a different case, a
 * leading sign or whitespace, or an out-of-range GPU index — yields
 * :cpp:enumerator:`Device::kUndefined`.
 */
Device DeviceFromName(const std::string &name);

/**
 * A non-owning view over a contiguous tensor buffer. ``SymTensor`` never
 * allocates: the caller is responsible for the lifetime of the underlying
 * memory referenced by ``data``. The shape may contain symbolic dimensions
 * which is useful for representing intermediate values during optimisation
 * passes where the concrete shape is not yet known.
 */
class SymTensor {
public:
  /// Constructs an empty (null) tensor.
  SymTensor() = default;

  /**
   * Constructs an ``SymTensor`` referencing an external buffer.
   *
   * @param data Pointer to the first element of the externally-owned buffer.
   *             May be ``nullptr`` only when ``shape`` is empty or all
   *             integer dimensions are zero.
   * @param dtype Element type of the data referenced by ``data``.
   * @param shape Shape describing the layout of the data.
   */
  SymTensor(void *data, TensorType dtype, SymShape shape)
      : data_(data), dtype_(dtype), shape_(std::move(shape)) {}

  /// Pointer to the externally-owned buffer. The view itself is ``const``
  /// but the buffer it references is not, mirroring ``std::span`` semantics.
  void *Data() const noexcept { return data_; }

  /// Element type of the referenced buffer.
  TensorType Dtype() const noexcept { return dtype_; }

  /// Logical device on which the buffer resides. Defaults to
  /// :cpp:enumerator:`Device::kUndefined` ("no information").
  Device GetDevice() const noexcept { return device_; }

  /// Sets the logical device on which the buffer resides.
  void SetDevice(Device device) noexcept { device_ = device; }

  /// ``true`` when a lower bound on the tensor's values is known.
  bool HasMin() const noexcept { return min_.has_value(); }

  /// ``true`` when an upper bound on the tensor's values is known.
  bool HasMax() const noexcept { return max_.has_value(); }

  /// Lower bound on the tensor's values. Throws
  /// ``std::bad_optional_access`` when :cpp:func:`HasMin` is ``false``.
  double Min() const { return min_.value(); }

  /// Upper bound on the tensor's values. Throws
  /// ``std::bad_optional_access`` when :cpp:func:`HasMax` is ``false``.
  double Max() const { return max_.value(); }

  /// Sets the lower bound on the tensor's values.
  void SetMin(double value) noexcept { min_ = value; }

  /// Sets the upper bound on the tensor's values.
  void SetMax(double value) noexcept { max_ = value; }

  /// Sets both bounds in a single call. ``min`` must be ``<= max``;
  /// otherwise ``std::invalid_argument`` is thrown.
  void SetMinMax(double min, double max);

  /// Clears the stored lower bound.
  void ClearMin() noexcept { min_.reset(); }

  /// Clears the stored upper bound.
  void ClearMax() noexcept { max_.reset(); }

  /// Clears both bounds.
  void ClearMinMax() noexcept {
    min_.reset();
    max_.reset();
  }

  /// Returns ``true`` when the tensor is known to be a "null constant",
  /// i.e. when both bounds are known and equal to zero. This is the
  /// canonical condition exploited by optimisation passes that need to
  /// detect all-zero tensors without inspecting the buffer.
  bool IsNullConstant() const noexcept {
    return min_.has_value() && max_.has_value() && *min_ == 0.0 && *max_ == 0.0;
  }

  /// Shape of the tensor (may contain symbolic dimensions).
  const SymShape &Shape() const noexcept { return shape_; }
  SymShape &Shape() noexcept { return shape_; }

  /// ``true`` when the tensor has no associated data pointer.
  bool IsNull() const noexcept { return data_ == nullptr; }

  /**
   * Tags the tensor as carrying a shape value (e.g. the ``shape`` input of a
   * ``Reshape`` node) and stores that shape. An empty ``SymShape`` is
   * permitted: it denotes a rank-0 / scalar shape value.
   */
  void SetValueAsShape(SymShape shape) { value_as_shape_ = std::move(shape); }

  /// Clears the value-as-shape annotation so that ``HasValueAsShape`` becomes
  /// ``false`` again.
  void ClearValueAsShape() noexcept { value_as_shape_.reset(); }

  /// ``true`` when the tensor's value is interpreted as a shape. An empty
  /// stored shape still returns ``true`` — use ``ValueAsShape().Empty()`` to
  /// distinguish the empty case.
  bool HasValueAsShape() const noexcept { return value_as_shape_.has_value(); }

  /// Returns the shape value carried by this tensor. Throws
  /// ``std::bad_optional_access`` if ``HasValueAsShape()`` is ``false``.
  const SymShape &ValueAsShape() const { return value_as_shape_.value(); }
  SymShape &ValueAsShape() { return value_as_shape_.value(); }

  /// Equality compares the data pointer, dtype, device, shape, the
  /// optional value-as-shape annotation, and the optional ``min``/``max``
  /// value bounds. Because :cpp:class:`SymTensor` is a non-owning
  /// view, two tensors are considered equal only when they refer to the
  /// same external buffer.
  bool operator==(const SymTensor &other) const noexcept {
    return data_ == other.data_ && dtype_ == other.dtype_ && device_ == other.device_ &&
           shape_ == other.shape_ && value_as_shape_ == other.value_as_shape_ &&
           min_ == other.min_ && max_ == other.max_;
  }
  bool operator!=(const SymTensor &other) const noexcept { return !(*this == other); }

  /// Returns a human-readable representation of the tensor of the form
  /// ``"SymTensor(dtype=<name>, shape=<shape>[, device=<name>][, value_as_shape=<shape>][,
  /// min=<v>][, max=<v>][, data=<ptr>])"``. The ``device`` component is omitted when the
  /// device is :cpp:enumerator:`Device::kUndefined`. The ``data`` component is
  /// omitted when the tensor holds no buffer. The ``value_as_shape``
  /// component is omitted when no shape annotation is attached. The
  /// ``min`` and ``max`` components are each omitted when the
  /// corresponding bound is not set. The ``<name>`` is the unqualified
  /// ``TensorType`` enumerator name (e.g. ``"Float"``, ``"Int64"``,
  /// ``"Undefined"``).
  std::string ToString() const;
  /**
   * Compares the information carried by ``*this`` and ``other`` and reports
   * which descriptor is more precise (see :cpp:enum:`SymCmpResult`).
   *
   * The comparison covers, in order:
   *   - the element type (an unknown :cpp:enumerator:`TensorType::kUndefined`
   *     is treated as "no information"; two different known types yield
   *     :cpp:enumerator:`SymCmpResult::kConflict`);
   *   - the device (an unknown :cpp:enumerator:`Device::kUndefined` is
   *     treated as "no information"; two different known devices yield
   *     :cpp:enumerator:`SymCmpResult::kConflict`);
   *   - the shape rank (different ranks yield ``kConflict``);
   *   - each dimension (two different concrete integers or two different
   *     symbolic expressions yield ``kConflict``; an integer is considered
   *     more precise than a symbolic expression at the same position);
   *   - the optional :cpp:func:`ValueAsShape` annotation (handled with the
   *     same rules as the main shape; a present annotation is more precise
   *     than an absent one);
   *   - the optional ``min`` and ``max`` bounds: a known bound is more
   *     precise than an absent one; two known bounds that produce a
   *     tighter interval (higher ``min`` or lower ``max``) are more
   *     precise; intervals that are provably disjoint
   *     (``a.min > b.max`` or ``b.min > a.max``) yield ``kConflict``;
   *   - the data-pointer presence (a non-null pointer is more precise than
   *     a null one; two distinct non-null pointers carry no precision
   *     signal because :cpp:class:`SymTensor` is a non-owning view and
   *     the buffer contents are not inspected).
   *
   * When ``*this`` and ``other`` are equivalent on every field, the result
   * is :cpp:enumerator:`SymCmpResult::kMorePrecise` (``*this`` is at
   * least as precise as ``other``).
   */
  SymCmpResult Cmp(const SymTensor &other) const noexcept;

private:
  void *data_ = nullptr;
  TensorType dtype_ = TensorType::kUndefined;
  Device device_ = Device::kUndefined;
  SymShape shape_{};
  std::optional<SymShape> value_as_shape_{};
  std::optional<double> min_{};
  std::optional<double> max_{};
};

/// Well-known key used to round-trip the :cpp:func:`SymTensor::Min`
/// bound through the ``ValueInfoProto::metadata_props`` field.
inline constexpr const char *kValueInfoMinMetadataKey = "min";

/// Well-known key used to round-trip the :cpp:func:`SymTensor::Max`
/// bound through the ``ValueInfoProto::metadata_props`` field.
inline constexpr const char *kValueInfoMaxMetadataKey = "max";

/// Well-known key used to round-trip :cpp:enum:`Device` through the
/// ``ValueInfoProto::metadata_props`` field. Exposed so that callers
/// outside ``onnx_shapes`` can inspect or remove the entry.
inline constexpr const char *kValueInfoDeviceMetadataKey = "device";

/**
 * Populates ``out`` from a ``ValueInfoProto`` describing a tensor.
 *
 * The element type and (optional) shape are read from
 * ``vi.type().tensor_type()``. When ``vi.metadata_props()`` contains an
 * entry whose key matches :cpp:var:`kValueInfoDeviceMetadataKey`, its
 * value is parsed with :cpp:func:`DeviceFromName` and assigned to the
 * resulting tensor; otherwise the device is left as
 * :cpp:enumerator:`Device::kUndefined`.
 *
 * @param vi  ``ValueInfoProto`` to read from.
 * @param out Tensor to overwrite on success.
 * @return ``true`` when ``vi`` wraps a tensor type; ``false`` for
 *         sequence/map/optional/sparse types (which ``SymTensor``
 *         does not model), in which case ``out`` is left untouched.
 */
bool SymTensorFromValueInfo(const ValueInfoProto &vi, SymTensor &out);

/// Maximum element count of a small integer tensor for which
/// :cpp:func:`SymTensorFromTensorProto` (and friends) populate the
/// :cpp:func:`SymTensor::ValueAsShape` annotation. Tensors beyond
/// this threshold are not data-propagated (the dtype, shape and
/// ``min``/``max`` bounds are still recorded normally).
///
/// The limit is set to ``kMaxOptimRank + 1`` so that any 1-D integer
/// constant whose length fits within an :cpp:class:`SymShape` (i.e., up
/// to ``kMaxOptimRank`` elements) can have its values propagated. This
/// enables operators like ``Unsqueeze`` to infer fully-concrete output
/// shapes even when the ``axes`` input is a large initializer (e.g., rank
/// 15 or 16 models).
inline constexpr int64_t kOptimValueAsShapeMaxElements = static_cast<int64_t>(kMaxOptimRank) + 1;

/**
 * Populates ``out`` from a ``TensorProto`` (typically a graph initializer).
 *
 * The element type is read from ``tp.data_type()`` and the shape is built
 * from ``tp.dims()`` via :cpp:func:`ShapeFromTensorProtoDims` (every
 * dimension becomes a concrete integer :cpp:class:`SymDim`). The data
 * pointer is left null and the device is left
 * :cpp:enumerator:`Device::kUndefined`.
 *
 * When the tensor payload is readable, the function also annotates the
 * resulting descriptor with:
 *   - the ``min``/``max`` value bounds derived from the actual content.
 *     Both integer (INT8/16/32/64, UINT8/16/32/64) and floating-point
 *     (FLOAT, DOUBLE) element types are supported. The bounds are skipped
 *     for empty tensors and for tensors whose payload cannot be decoded
 *     (no typed field and no ``raw_data``, unsupported dtype, etc.);
 *   - the :cpp:func:`SymTensor::ValueAsShape` annotation when the
 *     element type is integer, the rank is at most one, and the element
 *     count is strictly less than :cpp:var:`kOptimValueAsShapeMaxElements`
 *     (``kMaxOptimRank + 1``, i.e., at most ``kMaxOptimRank`` elements).
 *     This covers every 1-D integer constant whose values fit inside an
 *     :cpp:class:`SymShape`, enabling ``Reshape``, ``Unsqueeze``, and
 *     similar operators to infer fully-concrete output shapes from
 *     large axes or target-shape initializers.
 *
 * @param tp  ``TensorProto`` to read from.
 * @param out Tensor to overwrite on success.
 * @return ``false`` (and leaves ``out`` untouched) when
 *         ``tp.data_type()`` is ``TensorProto::DataType::UNDEFINED``;
 *         ``true`` otherwise.
 */
bool SymTensorFromTensorProto(const TensorProto &tp, SymTensor &out);

/**
 * Writes the ``(dtype, shape, device)`` triple carried by ``tensor``
 * into ``vi``.
 *
 * Any pre-existing ``type`` on ``vi`` is overwritten so that the
 * inferred descriptor takes precedence. The device is encoded as a
 * ``metadata_props`` entry keyed by
 * :cpp:var:`kValueInfoDeviceMetadataKey`; if such an entry already
 * exists it is updated in place, and if the tensor's device is
 * :cpp:enumerator:`Device::kUndefined` the existing entry (when any)
 * is removed. The ``name`` and ``doc_string`` fields of ``vi`` are
 * never touched.
 *
 * @param tensor Source tensor.
 * @param vi     Destination ``ValueInfoProto``.
 * @return ``false`` (and leaves ``vi`` unchanged) when ``tensor`` has
 *         an undefined element type, since ``TensorProto::DataType``
 *         provides no meaningful encoding for it; ``true`` otherwise.
 */
bool SymTensorToValueInfo(const SymTensor &tensor, ValueInfoProto &vi);

} // namespace symbolic
} // namespace ONNX_LIGHT_NAMESPACE::core
