// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared helpers used by the per-kernel ``KernelBase::Run`` implementations
// (each concrete ``kernel::Op`` now defines its own ``Run`` alongside its
// ``operator()`` in the kernel's own translation unit). These helpers were
// previously file-local to ``kernel_dispatch_table.cc``; they are gathered
// here so every kernel source file can reuse them without duplication.
//
// The helpers live directly in ``onnx_kernels`` (not ``onnx_kernels::kernel``)
// so that unqualified type names such as ``Shape`` and ``Tensor`` resolve to
// the ``core::runtime`` types re-exported at ``onnx_kernels`` scope rather than
// to the like-named ``kernel::Shape`` kernel class.

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/simple_tensor.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels {
// Re-exports the runtime types moved to ``onnx_core::runtime`` so the helper
// bodies below can keep referring to them unqualified.
using namespace ::onnx_light::core::runtime;

// Prototype for an empty (zero-element) tensor; its shape/data are copied when
// building a concrete empty tensor of a requested element type.
inline const Tensor &EmptyTensorPrototype() {
  static const Tensor kEmptyTensor = [] {
    Tensor empty;
    empty.shape.push_back(0);
    return empty;
  }();
  return kEmptyTensor;
}

// Reads a required TENSOR-valued attribute from ``node`` and converts it
// to a :cpp:class:`Tensor`. Throws ``std::invalid_argument`` if the
// attribute is missing or not of type TENSOR.
inline Tensor GetRequiredAttributeTensor(const NodeProto &node, const std::string &name) {
  const AttributeProto *attr = FindAttribute(node, name);
  EXT_ENFORCE_INVALID(attr != nullptr, "RunNode: op '", node.op_type(), "' is missing '", name,
                      "' TENSOR attribute.");
  EXT_ENFORCE_INVALID(!(attr->type() != AttributeProto::AttributeType::TENSOR),
                      "RunNode: attribute '", name, "' of op '", node.op_type(),
                      "' must be a TENSOR.");
  return TensorFromProto(attr->t());
}

// Returns an empty (zero-element) tensor of ``data_type``.
inline Tensor MakeEmptyTensor(int32_t data_type) {
  const Tensor &empty = EmptyTensorPrototype();
  return Tensor("", data_type, empty.shape, empty.data);
}

// Same as :func:`GetRequiredAttributeTensor` but returns an empty (zero-
// element) tensor of ``fallback_dtype`` when the attribute is absent.
inline Tensor GetAttributeTensorOrEmpty(const NodeProto &node, const std::string &name,
                                        int32_t fallback_dtype) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return MakeEmptyTensor(fallback_dtype);
  }
  EXT_ENFORCE_INVALID(!(attr->type() != AttributeProto::AttributeType::TENSOR),
                      "RunNode: attribute '", name, "' of op '", node.op_type(),
                      "' must be a TENSOR.");
  return TensorFromProto(attr->t());
}

// Reads the ONNX ``seed`` attribute (declared as FLOAT in the schema for
// the random generator and Bernoulli/Multinomial ops) and converts it to
// the ``int64_t`` value expected by the kernel constructors. Returns
// ``-1`` (the shared ``kNoSeed`` sentinel) when the attribute is absent.
inline int64_t GetSeedAttr(const NodeProto &node) {
  if (FindAttribute(node, "seed") == nullptr) {
    return -1;
  }
  return static_cast<int64_t>(GetAttributeFloatOrDefault(node, "seed", 0.0f));
}

// Reads the ONNX ``dtype`` attribute (INT, default 0) shared by Bernoulli,
// Multinomial and the four Random* generator ops.
inline int32_t GetDtypeAttr(const NodeProto &node) {
  return static_cast<int32_t>(GetAttributeIntOrDefault(node, "dtype", 0));
}

// The normalization kernels (BatchNormalization, GroupNormalization,
// InstanceNormalization, LayerNormalization, RMSNormalization) share the
// ``epsilon`` attribute (default 1e-5f) and several share the ``axis``
// attribute (default -1).
inline float GetEpsilon(const NodeProto &node) {
  return GetAttributeFloatOrDefault(node, "epsilon", 1e-5f);
}

inline int64_t GetNormAxis(const NodeProto &node) {
  return GetAttributeIntOrDefault(node, "axis", -1);
}

inline void RequireInputRange(const NodeProto &node, int min_inputs, int max_inputs) {
  const int n = node.input_size();
  EXT_ENFORCE_INVALID(!(n < min_inputs || n > max_inputs), "RunNode: op '", node.op_type(),
                      "' expects ", min_inputs, " to ", max_inputs, " input(s), got ", n, ".");
}

inline void RequireOutputRange(const NodeProto &node, int min_outputs, int max_outputs) {
  const int n = node.output_size();
  EXT_ENFORCE_INVALID(!(n < min_outputs || n > max_outputs), "RunNode: op '", node.op_type(),
                      "' expects ", min_outputs, " to ", max_outputs, " output(s), got ", n, ".");
}

// Returns a zero-copy span view of the typed element data in ``t``.
// Throws ``std::invalid_argument`` if ``t.data_type`` does not match ``T``.
template <typename T> std::span<const T> TensorSpan(const Tensor &t) {
  const int64_t count = t.element_count();
  if (count == 0) {
    return {};
  }
  return {t.As<T>(), static_cast<size_t>(count)};
}

// Shared attributes consumed by SVMRegressor and SVMClassifier.
struct SVMCommonAttrs {
  std::string kernel_type;
  float gamma;
  float coef0;
  float degree;
  std::vector<float> support_vectors;
  std::vector<float> coefficients;
  std::vector<float> rho;
};

inline SVMCommonAttrs ParseSVMCommonAttrs(const NodeProto &node, const char *op_name) {
  SVMCommonAttrs a;
  a.kernel_type = GetAttributeStringOrDefault(node, "kernel_type", "LINEAR");
  const std::vector<float> kernel_params =
      GetAttributeFloatsOrDefault(node, "kernel_params", {0.0f, 0.0f, 0.0f});
  EXT_ENFORCE_INVALID(!(kernel_params.size() < 3), "RunNode: ", op_name,
                      " 'kernel_params' must have at least 3 floats.");
  a.gamma = kernel_params[0];
  a.coef0 = kernel_params[1];
  a.degree = kernel_params[2];
  a.support_vectors = GetAttributeFloatsOrDefault(node, "support_vectors", {});
  a.coefficients = GetAttributeFloatsOrDefault(node, "coefficients", {});
  a.rho = GetAttributeFloatsOrDefault(node, "rho", {});
  return a;
}

// Dispatches ``fn`` on the element type of ``x`` for SVM* ops. ``fn`` is invoked
// with a ``T*`` tag pointer (always null) so the caller can recover ``T`` via
// ``std::remove_pointer_t<decltype(tag)>``.
template <class Fn>
auto DispatchSVMByDataType(const Tensor &x, const char *op_name, Fn &&fn)
    -> decltype(fn(static_cast<float *>(nullptr))) {
  switch (x.data_type) {
  case static_cast<int32_t>(DataType::FLOAT):
    return fn(static_cast<float *>(nullptr));
  case static_cast<int32_t>(DataType::DOUBLE):
    return fn(static_cast<double *>(nullptr));
  case static_cast<int32_t>(DataType::INT64):
    return fn(static_cast<int64_t *>(nullptr));
  case static_cast<int32_t>(DataType::INT32):
    return fn(static_cast<int32_t *>(nullptr));
  default:
    EXT_THROW_INVALID("RunNode: ", op_name, " input 'X' must be FLOAT, DOUBLE, INT32 or INT64.");
  }
}

// Same dispatch as :func:`DispatchSVMByDataType` but used for the classic
// TreeEnsembleRegressor/TreeEnsembleClassifier ops, which accept the same
// set of input element types (FLOAT, DOUBLE, INT32, INT64) per the
// ``ai.onnx.ml`` schema.
template <class Fn>
auto DispatchTreeEnsembleClassicByDataType(const Tensor &x, const char *op_name, Fn &&fn)
    -> decltype(fn(static_cast<float *>(nullptr))) {
  return DispatchSVMByDataType(x, op_name, std::forward<Fn>(fn));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels
