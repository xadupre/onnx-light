// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

template <typename T> bool ImputerMatches(T value, T replaced) { return value == replaced; }

// Specialization for float: treat NaN as matching NaN.
template <> bool ImputerMatches<float>(float value, float replaced) {
  if (std::isnan(replaced)) {
    return std::isnan(value);
  }
  return value == replaced;
}

// Specialization for double: treat NaN as matching NaN.
template <> bool ImputerMatches<double>(double value, double replaced) {
  if (std::isnan(replaced)) {
    return std::isnan(value);
  }
  return value == replaced;
}

template <typename T> void ValidateInput(const Tensor &x) {
  EXT_ENFORCE_INVALID(x.data_type == TensorElementType<T>::value,
                      "kernel::Imputer input data_type does not match the requested element T.");
}

template <typename T>
void ValidateImputedValues(const std::vector<T> &imputed_values, int64_t last_dim) {
  EXT_ENFORCE_INVALID(!imputed_values.empty(),
                      "kernel::Imputer requires non-empty 'imputed_values'.");
  EXT_ENFORCE_INVALID(
      imputed_values.size() == 1u || static_cast<int64_t>(imputed_values.size()) == last_dim,
      "kernel::Imputer requires 'imputed_values' length to be 1 or to match the size of "
      "the last dimension of the input.");
}

int64_t LastDim(const onnx_kernels::Shape &shape) { return shape.empty() ? 1 : shape.back(); }

template <typename T>
void ApplyImputer(const Tensor &x, const std::vector<T> &imputed_values, T replaced_value, T *out) {
  const T *px = x.As<T>();
  const int64_t n = x.element_count();
  const int64_t stride = static_cast<int64_t>(imputed_values.size());
  if (stride == 1) {
    const T imputed = imputed_values[0];
    for (int64_t i = 0; i < n; ++i) {
      out[i] = ImputerMatches(px[i], replaced_value) ? imputed : px[i];
    }
  } else {
    for (int64_t i = 0; i < n; ++i) {
      const int64_t k = i % stride;
      out[i] = ImputerMatches(px[i], replaced_value) ? imputed_values[k] : px[i];
    }
  }
}

} // namespace

template <typename T>
Tensor Imputer::operator()(const Tensor &x, const std::vector<T> &imputed_values, T replaced_value,
                           RuntimeContext * /*rt*/) const {
  ValidateInput<T>(x);
  ValidateImputedValues<T>(imputed_values, LastDim(x.shape));
  const int64_t n = x.element_count();
  Tensor out = MakeOutputTensor(TensorElementType<T>::value, x.shape,
                                static_cast<size_t>(n) * sizeof(T), ctx_.allocator);
  ApplyImputer<T>(x, imputed_values, replaced_value, reinterpret_cast<T *>(out.mutable_bytes()));
  return out;
}

template <typename T>
void Imputer::operator()(const Tensor &x, const std::vector<T> &imputed_values, T replaced_value,
                         Tensor &output) const {
  ValidateInput<T>(x);
  ValidateImputedValues<T>(imputed_values, LastDim(x.shape));
  EXT_ENFORCE_INVALID(output.data_type == TensorElementType<T>::value,
                      "kernel::Imputer preallocated output dtype must match the input dtype.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Imputer preallocated output shape must match the input shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == static_cast<size_t>(x.element_count()) * sizeof(T),
                      "kernel::Imputer preallocated output buffer is incorrectly sized.");
  ApplyImputer<T>(x, imputed_values, replaced_value, output.As<T>());
}

// Explicit instantiations for the supported element types.
#define ONNX_LIGHT_INSTANTIATE_IMPUTER(T)                                                          \
  template Tensor Imputer::operator()(const Tensor &, const std::vector<T> &, T, RuntimeContext *) \
      const;                                                                                       \
  template void Imputer::operator()(const Tensor &, const std::vector<T> &, T, Tensor &) const

ONNX_LIGHT_INSTANTIATE_IMPUTER(float);
ONNX_LIGHT_INSTANTIATE_IMPUTER(double);
ONNX_LIGHT_INSTANTIATE_IMPUTER(int64_t);
ONNX_LIGHT_INSTANTIATE_IMPUTER(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_IMPUTER

void Imputer::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());

  // Per the ``ai.onnx.ml::Imputer`` schema, exactly one of
  // ``imputed_value_floats``/``replaced_value_float`` (for floating-point
  // inputs) or ``imputed_value_int64s``/``replaced_value_int64`` (for
  // integer inputs) must be defined. The runtime selects the
  // appropriate pair based on the input element type.
  const std::vector<float> imputed_value_floats =
      GetAttributeFloatsOrDefault(node, "imputed_value_floats", {});
  const std::vector<int64_t> imputed_value_int64s =
      GetAttributeIntsOrDefault(node, "imputed_value_int64s", {});
  const float replaced_value_float = GetAttributeFloatOrDefault(node, "replaced_value_float", 0.0f);
  const int64_t replaced_value_int64 =
      GetAttributeIntOrDefault(node, "replaced_value_int64", static_cast<int64_t>(0));

  onnx_kernels::kernel::Imputer imputer(rt.kernel_ctx());
  Tensor y = DispatchSVMByDataType(x, "Imputer", [&](auto *tag) {
    using T = std::remove_pointer_t<decltype(tag)>;
    (void)tag;
    if constexpr (std::is_floating_point_v<T>) {
      std::vector<T> imputed_values(imputed_value_floats.begin(), imputed_value_floats.end());
      return imputer.template operator()<T>(x, imputed_values,
                                            static_cast<T>(replaced_value_float));
    } else {
      std::vector<T> imputed_values;
      imputed_values.reserve(imputed_value_int64s.size());
      for (int64_t value : imputed_value_int64s) {
        imputed_values.push_back(static_cast<T>(value));
      }
      return imputer.template operator()<T>(x, imputed_values,
                                            static_cast<T>(replaced_value_int64));
    }
  });
  SetOutput(node, 0, std::move(y), rt.tensors());
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
