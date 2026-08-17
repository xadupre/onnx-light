// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include "onnx_light_helpers.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

template <typename T> int32_t InputDataType() noexcept { return TensorElementType<T>::value; }

template <typename T> void ValidateNumericInput(const Tensor &x, const std::vector<int64_t> &cats) {
  EXT_ENFORCE_INVALID(x.data_type == InputDataType<T>(),
                      "kernel::OneHotEncoder input data_type does not match the requested T.");
  EXT_ENFORCE_INVALID(!cats.empty(),
                      "kernel::OneHotEncoder requires at least one category in cats_int64s.");
}

void ValidateStringInput(const Tensor &x, const ParamStrings &cats) {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::STRING),
                      "kernel::OneHotEncoder expects a STRING input for string categories.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(x.string_data.size()) == x.element_count(),
                      "kernel::OneHotEncoder STRING input string_data size does not match shape.");
  EXT_ENFORCE_INVALID(!cats.empty(),
                      "kernel::OneHotEncoder requires at least one category in cats_strings.");
}

onnx_kernels::Shape OneHotShape(const onnx_kernels::Shape &input_shape, int64_t num_cats) {
  onnx_kernels::Shape out_shape = input_shape;
  out_shape.push_back(num_cats);
  return out_shape;
}

template <typename T>
void FillOneHotNumeric(const Tensor &x, const std::vector<int64_t> &cats, bool zeros, float *out) {
  const int64_t n = x.element_count();
  const int64_t k = static_cast<int64_t>(cats.size());
  const T *px = x.As<T>();
  std::memset(out, 0, static_cast<size_t>(n) * static_cast<size_t>(k) * sizeof(float));
  for (int64_t i = 0; i < n; ++i) {
    const int64_t value = static_cast<int64_t>(px[i]);
    bool matched = false;
    for (int64_t j = 0; j < k; ++j) {
      if (cats[static_cast<size_t>(j)] == value) {
        out[i * k + j] = 1.0f;
        matched = true;
        break;
      }
    }
    EXT_ENFORCE_INVALID(matched || zeros,
                        "kernel::OneHotEncoder: input value not found in cats_int64s and "
                        "zeros=false.");
  }
}

void FillOneHotString(const Tensor &x, const ParamStrings &cats, bool zeros, float *out) {
  const int64_t n = x.element_count();
  const int64_t k = static_cast<int64_t>(cats.size());
  const std::vector<std::string> &px = x.AsStrings();
  std::memset(out, 0, static_cast<size_t>(n) * static_cast<size_t>(k) * sizeof(float));
  for (int64_t i = 0; i < n; ++i) {
    bool matched = false;
    for (int64_t j = 0; j < k; ++j) {
      if (px[static_cast<size_t>(i)] == cats[static_cast<size_t>(j)]) {
        out[i * k + j] = 1.0f;
        matched = true;
        break;
      }
    }
    EXT_ENFORCE_INVALID(matched || zeros,
                        "kernel::OneHotEncoder: input value not found in cats_strings and "
                        "zeros=false.");
  }
}

void ValidatePreallocatedOutput(const Tensor &output, const onnx_kernels::Shape &expected_shape) {
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::OneHotEncoder preallocated output dtype must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::OneHotEncoder preallocated output shape does not match the expected "
                      "one-hot output shape.");
  int64_t expected_n = 1;
  for (int64_t d : expected_shape) {
    expected_n *= d;
  }
  EXT_ENFORCE_INVALID(output.size_bytes() == static_cast<size_t>(expected_n) * sizeof(float),
                      "kernel::OneHotEncoder preallocated output buffer is incorrectly sized.");
}

} // namespace

template <typename T>
Tensor OneHotEncoder::operator()(const Tensor &x, const std::vector<int64_t> &cats, bool zeros,
                                 RuntimeContext *rt) const {
  ValidateNumericInput<T>(x, cats);
  const onnx_kernels::Shape out_shape = OneHotShape(x.shape, static_cast<int64_t>(cats.size()));
  const int64_t total = x.element_count() * static_cast<int64_t>(cats.size());
  const size_t n_bytes = static_cast<size_t>(total) * sizeof(float);
  Tensor out =
      rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::FLOAT), out_shape, n_bytes)
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), out_shape, n_bytes,
                            ctx_.allocator);
  FillOneHotNumeric<T>(x, cats, zeros, reinterpret_cast<float *>(out.mutable_bytes()));
  return out;
}

Tensor OneHotEncoder::operator()(const Tensor &x, const ParamStrings &cats, bool zeros,
                                 RuntimeContext *rt) const {
  ValidateStringInput(x, cats);
  const onnx_kernels::Shape out_shape = OneHotShape(x.shape, static_cast<int64_t>(cats.size()));
  const int64_t total = x.element_count() * static_cast<int64_t>(cats.size());
  const size_t n_bytes = static_cast<size_t>(total) * sizeof(float);
  Tensor out =
      rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::FLOAT), out_shape, n_bytes)
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), out_shape, n_bytes,
                            ctx_.allocator);
  FillOneHotString(x, cats, zeros, reinterpret_cast<float *>(out.mutable_bytes()));
  return out;
}

template <typename T>
void OneHotEncoder::operator()(const Tensor &x, const std::vector<int64_t> &cats, bool zeros,
                               Tensor &output) const {
  ValidateNumericInput<T>(x, cats);
  const onnx_kernels::Shape expected_shape =
      OneHotShape(x.shape, static_cast<int64_t>(cats.size()));
  ValidatePreallocatedOutput(output, expected_shape);
  FillOneHotNumeric<T>(x, cats, zeros, output.AsFloat());
}

void OneHotEncoder::operator()(const Tensor &x, const ParamStrings &cats, bool zeros,
                               Tensor &output) const {
  ValidateStringInput(x, cats);
  const onnx_kernels::Shape expected_shape =
      OneHotShape(x.shape, static_cast<int64_t>(cats.size()));
  ValidatePreallocatedOutput(output, expected_shape);
  FillOneHotString(x, cats, zeros, output.AsFloat());
}

// Explicit instantiations for the supported numeric element types.
#define ONNX_LIGHT_INSTANTIATE_ONE_HOT_ENCODER(T)                                                  \
  template Tensor OneHotEncoder::operator()<T>(const Tensor &, const std::vector<int64_t> &, bool, \
                                               RuntimeContext *) const;                            \
  template void OneHotEncoder::operator()<T>(const Tensor &, const std::vector<int64_t> &, bool,   \
                                             Tensor &) const

ONNX_LIGHT_INSTANTIATE_ONE_HOT_ENCODER(int64_t);
ONNX_LIGHT_INSTANTIATE_ONE_HOT_ENCODER(int32_t);
ONNX_LIGHT_INSTANTIATE_ONE_HOT_ENCODER(float);
ONNX_LIGHT_INSTANTIATE_ONE_HOT_ENCODER(double);

#undef ONNX_LIGHT_INSTANTIATE_ONE_HOT_ENCODER

void OneHotEncoder::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());

  const AttributeProto *cats_int64s = FindAttribute(node, "cats_int64s");
  const AttributeProto *cats_strings = FindAttribute(node, "cats_strings");
  const int n_cats = (cats_int64s != nullptr) + (cats_strings != nullptr);
  EXT_ENFORCE_INVALID(n_cats == 1, "RunNode: OneHotEncoder requires exactly one of 'cats_int64s' "
                                   "or 'cats_strings' to be set.");

  // The ``zeros`` attribute defaults to 1 per the ai.onnx.ml schema.
  const bool zeros = GetAttributeIntOrDefault(node, "zeros", 1) != 0;

  onnx_kernels::kernel::OneHotEncoder one_hot(rt.kernel_ctx());
  Tensor y;
  if (cats_int64s != nullptr) {
    std::vector<int64_t> cats;
    cats.reserve(cats_int64s->ints().size());
    for (int64_t v : cats_int64s->ints()) {
      cats.push_back(v);
    }
    y = DispatchSVMByDataType(x, "OneHotEncoder", [&](auto *tag) {
      using T = std::remove_pointer_t<decltype(tag)>;
      (void)tag;
      return one_hot.template operator()<T>(x, cats, zeros, &rt);
    });
  } else {
    ParamStrings cats;
    cats.reserve(cats_strings->strings().size());
    for (size_t i = 0; i < cats_strings->strings().size(); ++i) {
      cats.push_back(cats_strings->strings()[i]);
    }
    EXT_ENFORCE_INVALID(!(x.data_type != static_cast<int32_t>(DataType::STRING)),
                        "RunNode: OneHotEncoder with 'cats_strings' requires input X "
                        "of element type STRING.");
    y = one_hot(x, cats, zeros, &rt);
  }
  SetOutput(node, 0, std::move(y), rt.tensors());
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
