// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernel_dispatch_table.h"

#include "onnx_kernels/kernels/generator/include_generator_kernels.h"
#include "onnx_kernels/kernels/image/include_image_kernels.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_kernels/kernels/object_detection/include_object_detection_kernels.h"
#include "onnx_kernels/kernels/quantization/include_quantization_kernels.h"
#include "onnx_kernels/kernels/reduction/include_reduction_kernels.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_kernels/kernels/training/include_training_kernels.h"
#include "onnx_kernels/node_helpers.h"
#include "onnx_kernels/simple_tensor.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

namespace {

using detail::FindAttribute;
using detail::GetAttributeFloatOrDefault;
using detail::GetAttributeFloatsOrDefault;
using detail::GetAttributeIntOrDefault;
using detail::GetAttributeIntsOrDefault;
using detail::GetAttributeStringOrDefault;
using detail::GetAttributeStringsOrDefault;
using detail::GetInput;
using detail::GetOptionalInput;
using detail::GetRequiredAttributeString;
using detail::RequireInputCount;
using detail::RequireMinInputCount;
using detail::RequireOutputCount;
using detail::SetOutput;

// ---------------------------------------------------------------------------
// Trampoline factories. Each helper returns a NodeKernelFn that:
//   * validates the node's input/output count,
//   * reads the typed inputs from ``rt.tensors()`` by name (and the
//     relevant attributes from ``node.attribute()``),
//   * constructs the kernel with ``rt.kernel_ctx()``,
//   * stores the produced output back in ``rt.tensors()`` by name.
//
// Centralising the boilerplate keeps the dispatch table compact and
// makes the per-operator entries below one-liners.
// ---------------------------------------------------------------------------

template <class KernelT> NodeKernelFn MakeUnaryTrampoline() {
  return [](const NodeProto &node, RuntimeContext &rt) {
    RequireInputCount(node, 1);
    RequireOutputCount(node, 1);
    const Tensor &x = GetInput(node, 0, rt.tensors());
    KernelT kernel(rt.kernel_ctx());
    SetOutput(node, 0, kernel(x), rt.tensors());
  };
}

template <class KernelT> NodeKernelFn MakeBinaryTrampoline() {
  return [](const NodeProto &node, RuntimeContext &rt) {
    RequireInputCount(node, 2);
    RequireOutputCount(node, 1);
    const Tensor &x = GetInput(node, 0, rt.tensors());
    const Tensor &y = GetInput(node, 1, rt.tensors());
    KernelT kernel(rt.kernel_ctx());
    SetOutput(node, 0, kernel(x, y), rt.tensors());
  };
}

// Wraps a kernel of the form
//   ``Tensor operator()(const Tensor&, const Tensor&)`` /
//   ``Tensor operator()(const Tensor&, const Tensor&, const Tensor&)``
// where the third input is optional (e.g. ``QuantizeLinear`` /
// ``DequantizeLinear`` zero-point).
template <class KernelT> NodeKernelFn MakeBinaryWithOptionalThirdTrampoline() {
  return [](const NodeProto &node, RuntimeContext &rt) {
    RequireMinInputCount(node, 2);
    if (node.input_size() > 3) {
      throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                  "' expects 2 or 3 inputs, got " +
                                  std::to_string(node.input_size()) + ".");
    }
    RequireOutputCount(node, 1);
    const Tensor &a = GetInput(node, 0, rt.tensors());
    const Tensor &b = GetInput(node, 1, rt.tensors());
    const Tensor *c = GetOptionalInput(node, 2, rt.tensors());
    KernelT kernel(rt.kernel_ctx());
    if (c != nullptr) {
      SetOutput(node, 0, kernel(a, b, *c), rt.tensors());
    } else {
      SetOutput(node, 0, kernel(a, b), rt.tensors());
    }
  };
}

template <class KernelT> NodeKernelFn MakeTernaryTrampoline() {
  return [](const NodeProto &node, RuntimeContext &rt) {
    RequireInputCount(node, 3);
    RequireOutputCount(node, 1);
    const Tensor &a = GetInput(node, 0, rt.tensors());
    const Tensor &b = GetInput(node, 1, rt.tensors());
    const Tensor &c = GetInput(node, 2, rt.tensors());
    KernelT kernel(rt.kernel_ctx());
    SetOutput(node, 0, kernel(a, b, c), rt.tensors());
  };
}

// Wraps a kernel of the form ``Tensor operator()(const std::vector<Tensor>&)``
// (the variadic element-wise reducers: ``Sum``, ``Max``, ``Min``, ``Mean``).
template <class KernelT> NodeKernelFn MakeVariadicTrampoline(int min_inputs = 1) {
  return [min_inputs](const NodeProto &node, RuntimeContext &rt) {
    RequireMinInputCount(node, min_inputs);
    RequireOutputCount(node, 1);
    std::vector<Tensor> inputs;
    inputs.reserve(node.input_size());
    for (int i = 0; i < node.input_size(); ++i) {
      inputs.push_back(GetInput(node, i, rt.tensors()));
    }
    KernelT kernel(rt.kernel_ctx());
    SetOutput(node, 0, kernel(inputs), rt.tensors());
  };
}

// Wraps a kernel of the form ``Tensor operator()(const Tensor&, float alpha)``.
template <class KernelT>
NodeKernelFn MakeUnaryAlphaTrampoline(const char *attr_name, float default_alpha) {
  const std::string name(attr_name);
  return [name, default_alpha](const NodeProto &node, RuntimeContext &rt) {
    RequireInputCount(node, 1);
    RequireOutputCount(node, 1);
    const Tensor &x = GetInput(node, 0, rt.tensors());
    const float alpha = GetAttributeFloatOrDefault(node, name, default_alpha);
    KernelT kernel(rt.kernel_ctx());
    SetOutput(node, 0, kernel(x, alpha), rt.tensors());
  };
}

// Wraps a kernel of the form ``Tensor operator()(const Tensor&, int64_t axis)``
// (``Softmax``, ``LogSoftmax``, ``Hardmax``). Opset 13+ defaults ``axis`` to
// ``-1``, which matches the kernel reference implementation.
template <class KernelT> NodeKernelFn MakeAxisTrampoline(int64_t default_axis = -1) {
  return [default_axis](const NodeProto &node, RuntimeContext &rt) {
    RequireInputCount(node, 1);
    RequireOutputCount(node, 1);
    const Tensor &x = GetInput(node, 0, rt.tensors());
    const int64_t axis = GetAttributeIntOrDefault(node, "axis", default_axis);
    KernelT kernel(rt.kernel_ctx());
    SetOutput(node, 0, kernel(x, axis), rt.tensors());
  };
}

// Wraps a kernel of the form ``Tensor operator()(const Tensor&, int32_t to)``
// where ``to`` is the required ONNX INT attribute naming the target data type
// (for example ``Cast`` and ``BitCast``).
template <class KernelT> NodeKernelFn MakeUnaryToTrampoline() {
  return [](const NodeProto &node, RuntimeContext &rt) {
    RequireInputCount(node, 1);
    RequireOutputCount(node, 1);
    const Tensor &x = GetInput(node, 0, rt.tensors());
    const int32_t to = static_cast<int32_t>(GetAttributeIntOrDefault(node, "to", -1));
    if (to < 0) {
      throw std::invalid_argument("RunNode: " + node.op_type().as_string() +
                                  " requires INT attribute 'to'.");
    }
    KernelT kernel(rt.kernel_ctx());
    SetOutput(node, 0, kernel(x, to), rt.tensors());
  };
}

// Creates trampolines for reduction kernels of the form:
//   ``operator()(data, keepdims, noop_with_empty_axes)``
//   ``operator()(data, axes, keepdims, noop_with_empty_axes)``
// where ``axes`` is either an optional second input (opset 13+/18+ depending
// on the operator) or an ``axes`` INTS attribute (older opsets).
template <class KernelT> NodeKernelFn MakeReduceTrampoline() {
  return [](const NodeProto &node, RuntimeContext &rt) {
    RequireMinInputCount(node, 1);
    if (node.input_size() > 2) {
      throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                  "' expects at most 2 inputs.");
    }
    RequireOutputCount(node, 1);
    const Tensor &data = GetInput(node, 0, rt.tensors());
    const bool keepdims = GetAttributeIntOrDefault(node, "keepdims", 1) != 0;
    const bool noop_with_empty_axes =
        GetAttributeIntOrDefault(node, "noop_with_empty_axes", 0) != 0;
    KernelT kernel(rt.kernel_ctx());

    const Tensor *axes_input = GetOptionalInput(node, 1, rt.tensors());
    if (axes_input != nullptr) {
      SetOutput(node, 0, kernel(data, *axes_input, keepdims, noop_with_empty_axes), rt.tensors());
      return;
    }

    const std::vector<int64_t> axes_attr = GetAttributeIntsOrDefault(node, "axes", {});
    if (!axes_attr.empty()) {
      const Tensor axes =
          Tensor::FromInt64("", {static_cast<int64_t>(axes_attr.size())}, axes_attr);
      SetOutput(node, 0, kernel(data, axes, keepdims, noop_with_empty_axes), rt.tensors());
      return;
    }

    SetOutput(node, 0, kernel(data, keepdims, noop_with_empty_axes), rt.tensors());
  };
}

// Trampoline for ``Squeeze`` / ``Unsqueeze``: both take 1 or 2 inputs where the
// optional second input (opset 13+) is a 1-D INT64 tensor of ``axes``. For the
// legacy opset (<13), ``axes`` is provided as an INTS attribute instead.
template <class KernelT> NodeKernelFn MakeSqueezeLikeTrampoline(const char *op_name) {
  return [op_name](const NodeProto &node, RuntimeContext &rt) {
    RequireMinInputCount(node, 1);
    if (node.input_size() > 2) {
      throw std::invalid_argument(std::string("RunNode: op '") + op_name +
                                  "' expects at most 2 inputs.");
    }
    RequireOutputCount(node, 1);
    const Tensor &data = GetInput(node, 0, rt.tensors());
    std::vector<int64_t> axes;
    const Tensor *axes_input = GetOptionalInput(node, 1, rt.tensors());
    if (axes_input != nullptr) {
      if (axes_input->data_type != static_cast<int32_t>(DataType::INT64) ||
          axes_input->shape.size() != 1) {
        throw std::invalid_argument(std::string("RunNode: ") + op_name +
                                    " 'axes' input must be a 1-D INT64 tensor.");
      }
      const int64_t n = axes_input->element_count();
      const int64_t *p = axes_input->AsInt64();
      axes.assign(p, p + n);
    } else {
      axes = GetAttributeIntsOrDefault(node, "axes", {});
    }
    KernelT k(rt.kernel_ctx());
    SetOutput(node, 0, k(data, axes), rt.tensors());
  };
}

template <class KernelT> NodeKernelFn MakeArgReduceTrampoline() {
  return [](const NodeProto &node, RuntimeContext &rt) {
    RequireInputCount(node, 1);
    RequireOutputCount(node, 1);
    const Tensor &data = GetInput(node, 0, rt.tensors());
    const int64_t axis = GetAttributeIntOrDefault(node, "axis", 0);
    const bool keepdims = GetAttributeIntOrDefault(node, "keepdims", 1) != 0;
    const bool select_last_index = GetAttributeIntOrDefault(node, "select_last_index", 0) != 0;
    KernelT kernel(rt.kernel_ctx());
    SetOutput(node, 0, kernel(data, axis, keepdims, select_last_index), rt.tensors());
  };
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
  if (kernel_params.size() < 3) {
    throw std::invalid_argument(std::string("RunNode: ") + op_name +
                                " 'kernel_params' must have at least 3 floats.");
  }
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
auto DispatchSVMByDataType(const Tensor &x, const char *op_name,
                           Fn &&fn) -> decltype(fn(static_cast<float *>(nullptr))) {
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
    throw std::invalid_argument(std::string("RunNode: ") + op_name +
                                " input 'X' must be FLOAT, DOUBLE, INT32 or INT64.");
  }
}

// Same dispatch as :func:`DispatchSVMByDataType` but used for the classic
// TreeEnsembleRegressor/TreeEnsembleClassifier ops, which accept the same
// set of input element types (FLOAT, DOUBLE, INT32, INT64) per the
// ``ai.onnx.ml`` schema.
template <class Fn>
auto DispatchTreeEnsembleClassicByDataType(const Tensor &x, const char *op_name,
                                           Fn &&fn) -> decltype(fn(static_cast<float *>(nullptr))) {
  return DispatchSVMByDataType(x, op_name, std::forward<Fn>(fn));
}

// Reads a required TENSOR-valued attribute from ``node`` and converts it
// to a :cpp:class:`Tensor`. Throws ``std::invalid_argument`` if the
// attribute is missing or not of type TENSOR.
inline Tensor GetRequiredAttributeTensor(const NodeProto &node, const std::string &name) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() + "' is missing '" +
                                name + "' TENSOR attribute.");
  }
  if (attr->type() != AttributeProto::AttributeType::TENSOR) {
    throw std::invalid_argument("RunNode: attribute '" + name + "' of op '" +
                                node.op_type().as_string() + "' must be a TENSOR.");
  }
  return TensorFromProto(attr->t());
}

// Same as :func:`GetRequiredAttributeTensor` but returns an empty (zero-
// element) tensor of ``fallback_dtype`` when the attribute is absent.
inline Tensor GetAttributeTensorOrEmpty(const NodeProto &node, const std::string &name,
                                        int32_t fallback_dtype) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return Tensor("", fallback_dtype, std::vector<int64_t>{0}, std::vector<uint8_t>{});
  }
  if (attr->type() != AttributeProto::AttributeType::TENSOR) {
    throw std::invalid_argument("RunNode: attribute '" + name + "' of op '" +
                                node.op_type().as_string() + "' must be a TENSOR.");
  }
  return TensorFromProto(attr->t());
}

// Copies the typed contents of ``t`` into a ``std::vector<T>``. Throws
// ``std::invalid_argument`` if ``t.data_type`` does not match ``T``.
template <typename T> std::vector<T> TensorToVector(const Tensor &t) {
  const int64_t count = t.element_count();
  std::vector<T> out(static_cast<size_t>(count));
  if (count > 0) {
    const T *src = t.As<T>();
    std::copy(src, src + count, out.begin());
  }
  return out;
}

} // namespace

const std::unordered_map<std::string, NodeKernelFn> &KernelDispatchTable() {
  static const std::unordered_map<std::string, NodeKernelFn> table = {
      {"ai.onnx:Abs", MakeUnaryTrampoline<kernel::Abs>()},
      {"ai.onnx:Acos", MakeUnaryTrampoline<kernel::Acos>()},
      {"ai.onnx:Acosh", MakeUnaryTrampoline<kernel::Acosh>()},
      {"ai.onnx:Add", MakeBinaryTrampoline<kernel::Add>()},
      {"ai.onnx:AffineGrid",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &theta = GetInput(node, 0, rt.tensors());
         const Tensor &size = GetInput(node, 1, rt.tensors());
         kernel::AffineGrid::Attributes affine_grid_attrs;
         affine_grid_attrs.align_corners = GetAttributeIntOrDefault(node, "align_corners", 0);
         kernel::AffineGrid affine_grid_kernel(rt.kernel_ctx());
         SetOutput(node, 0, affine_grid_kernel(theta, size, affine_grid_attrs), rt.tensors());
       }},
      {"ai.onnx:And", MakeBinaryTrampoline<kernel::And>()},
      {"ai.onnx:ArgMax", MakeArgReduceTrampoline<kernel::ArgMax>()},
      {"ai.onnx:ArgMin", MakeArgReduceTrampoline<kernel::ArgMin>()},
      {"ai.onnx:Asin", MakeUnaryTrampoline<kernel::Asin>()},
      {"ai.onnx:Asinh", MakeUnaryTrampoline<kernel::Asinh>()},
      {"ai.onnx:Atan", MakeUnaryTrampoline<kernel::Atan>()},
      {"ai.onnx:Atanh", MakeUnaryTrampoline<kernel::Atanh>()},
      {"ai.onnx:Attention",
       [](const NodeProto &node, RuntimeContext &rt) {
         if (node.input_size() < 3 || node.input_size() > 6) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects between 3 and 6 input(s), got " +
                                       std::to_string(node.input_size()) + ".");
         }
         if (node.output_size() < 1 || node.output_size() > 4) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects between 1 and 4 output(s), got " +
                                       std::to_string(node.output_size()) + ".");
         }
         const Tensor &q = GetInput(node, 0, rt.tensors());
         const Tensor &k = GetInput(node, 1, rt.tensors());
         const Tensor &v = GetInput(node, 2, rt.tensors());
         const Tensor *attn_mask = GetOptionalInput(node, 3, rt.tensors());
         const Tensor *past_key = GetOptionalInput(node, 4, rt.tensors());
         const Tensor *past_value = GetOptionalInput(node, 5, rt.tensors());

         kernel::Attention::Attributes attrs;
         if (FindAttribute(node, "scale") != nullptr) {
           attrs.has_scale = true;
           attrs.scale = GetAttributeFloatOrDefault(node, "scale", 0.0f);
         }
         attrs.is_causal = GetAttributeIntOrDefault(node, "is_causal", 0) != 0;
         attrs.softcap = GetAttributeFloatOrDefault(node, "softcap", 0.0f);
         attrs.qk_matmul_output_mode =
             static_cast<int>(GetAttributeIntOrDefault(node, "qk_matmul_output_mode", 0));
         attrs.q_num_heads = GetAttributeIntOrDefault(node, "q_num_heads", 0);
         attrs.kv_num_heads = GetAttributeIntOrDefault(node, "kv_num_heads", 0);

         kernel::Attention kernel(rt.kernel_ctx());
         kernel::Attention::Result result = kernel(q, k, v, attrs, attn_mask, past_key, past_value);
         SetOutput(node, 0, std::move(result.Y), rt.tensors());

         auto set_optional_output = [&node, &rt](int index, Tensor output) {
           if (index >= node.output_size()) {
             return;
           }
           const std::string name = node.output(index).as_string();
           if (name.empty()) {
             return;
           }
           output.name = name;
           rt.tensors()[name] = std::move(output);
         };
         set_optional_output(1, std::move(result.present_key));
         set_optional_output(2, std::move(result.present_value));
         set_optional_output(3, std::move(result.qk_matmul_output));
       }},
      {"ai.onnx:BitShift",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &y = GetInput(node, 1, rt.tensors());
         const std::string direction = GetRequiredAttributeString(node, "direction");
         kernel::BitShift::Direction dir;
         if (direction == "LEFT") {
           dir = kernel::BitShift::Direction::kLeft;
         } else if (direction == "RIGHT") {
           dir = kernel::BitShift::Direction::kRight;
         } else {
           throw std::invalid_argument(
               "RunNode: BitShift 'direction' must be 'LEFT' or 'RIGHT', got '" + direction + "'.");
         }
         kernel::BitShift k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, y, dir), rt.tensors());
       }},
      {"ai.onnx:BitCast", MakeUnaryToTrampoline<kernel::BitCast>()},
      {"ai.onnx:BitwiseAnd", MakeBinaryTrampoline<kernel::BitwiseAnd>()},
      {"ai.onnx:BitwiseNot", MakeUnaryTrampoline<kernel::BitwiseNot>()},
      {"ai.onnx:BitwiseOr", MakeBinaryTrampoline<kernel::BitwiseOr>()},
      {"ai.onnx:BitwiseXor", MakeBinaryTrampoline<kernel::BitwiseXor>()},
      {"ai.onnx:CausalConvWithState",
       [](const NodeProto &node, RuntimeContext &rt) {
         if (node.input_size() < 2 || node.input_size() > 4) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects between 2 and 4 input(s), got " +
                                       std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 2);
         const Tensor &input = GetInput(node, 0, rt.tensors());
         const Tensor &weight = GetInput(node, 1, rt.tensors());
         const Tensor *bias = GetOptionalInput(node, 2, rt.tensors());
         const Tensor *past_state = GetOptionalInput(node, 3, rt.tensors());

         kernel::CausalConvWithState::Attributes attrs;
         attrs.activation = GetAttributeStringOrDefault(node, "activation", "none");

         kernel::CausalConvWithState kernel(rt.kernel_ctx());
         auto [output, present_state] =
             kernel(input, weight, bias != nullptr ? *bias : Tensor{},
                    past_state != nullptr ? *past_state : Tensor{}, attrs);
         SetOutput(node, 0, std::move(output), rt.tensors());
         SetOutput(node, 1, std::move(present_state), rt.tensors());
       }},
      {"ai.onnx:Cast", MakeUnaryToTrampoline<kernel::Cast>()},
      {"ai.onnx:Ceil", MakeUnaryTrampoline<kernel::Ceil>()},
      {"ai.onnx:Celu", MakeUnaryAlphaTrampoline<kernel::Celu>("alpha", 1.0f)},
      {"ai.onnx:Clip",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor *min = GetOptionalInput(node, 1, rt.tensors());
         const Tensor *max = GetOptionalInput(node, 2, rt.tensors());
         kernel::Clip k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, min, max), rt.tensors());
       }},
      {"ai.onnx:Conv",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 2);
         if (node.input_size() > 3) {
           throw std::invalid_argument("RunNode: op 'Conv' expects at most 3 inputs, got " +
                                       std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &w = GetInput(node, 1, rt.tensors());
         const Tensor *b = GetOptionalInput(node, 2, rt.tensors());
         kernel::Conv::Attributes attrs;
         attrs.kernel_shape = GetAttributeIntsOrDefault(node, "kernel_shape", {});
         attrs.strides = GetAttributeIntsOrDefault(node, "strides", {});
         attrs.pads = GetAttributeIntsOrDefault(node, "pads", {});
         attrs.dilations = GetAttributeIntsOrDefault(node, "dilations", {});
         attrs.group = GetAttributeIntOrDefault(node, "group", 1);
         attrs.auto_pad = GetAttributeStringOrDefault(node, "auto_pad", "NOTSET");
         kernel::Conv k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, w, b != nullptr ? *b : Tensor{}, attrs), rt.tensors());
       }},
      {"ai.onnx:ConvInteger",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 2);
         if (node.input_size() > 4) {
           throw std::invalid_argument("RunNode: op 'ConvInteger' expects at most 4 inputs, got " +
                                       std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &w = GetInput(node, 1, rt.tensors());
         const Tensor *x_zp = GetOptionalInput(node, 2, rt.tensors());
         const Tensor *w_zp = GetOptionalInput(node, 3, rt.tensors());
         kernel::ConvInteger::Attributes attrs;
         attrs.kernel_shape = GetAttributeIntsOrDefault(node, "kernel_shape", {});
         attrs.strides = GetAttributeIntsOrDefault(node, "strides", {});
         attrs.pads = GetAttributeIntsOrDefault(node, "pads", {});
         attrs.dilations = GetAttributeIntsOrDefault(node, "dilations", {});
         attrs.group = GetAttributeIntOrDefault(node, "group", 1);
         attrs.auto_pad = GetAttributeStringOrDefault(node, "auto_pad", "NOTSET");
         kernel::ConvInteger k(rt.kernel_ctx());
         SetOutput(node, 0,
                   k(x, w, x_zp != nullptr ? *x_zp : Tensor{},
                     w_zp != nullptr ? *w_zp : Tensor{}, attrs),
                   rt.tensors());
       }},
      {"ai.onnx:Cos", MakeUnaryTrampoline<kernel::Cos>()},
      {"ai.onnx:Cosh", MakeUnaryTrampoline<kernel::Cosh>()},
      {"ai.onnx:DeformConv",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 3);
         if (node.input_size() > 5) {
           throw std::invalid_argument("RunNode: op 'DeformConv' expects at most 5 inputs, got " +
                                       std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &w = GetInput(node, 1, rt.tensors());
         const Tensor &offset = GetInput(node, 2, rt.tensors());
         const Tensor *b = GetOptionalInput(node, 3, rt.tensors());
         const Tensor *mask = GetOptionalInput(node, 4, rt.tensors());
         kernel::DeformConv::Attributes attrs;
         attrs.kernel_shape = GetAttributeIntsOrDefault(node, "kernel_shape", {});
         attrs.strides = GetAttributeIntsOrDefault(node, "strides", {});
         attrs.pads = GetAttributeIntsOrDefault(node, "pads", {});
         attrs.dilations = GetAttributeIntsOrDefault(node, "dilations", {});
         attrs.group = GetAttributeIntOrDefault(node, "group", 1);
         attrs.offset_group = GetAttributeIntOrDefault(node, "offset_group", 1);
         kernel::DeformConv k(rt.kernel_ctx());
         SetOutput(node, 0,
                   k(x, w, offset, b != nullptr ? *b : Tensor{},
                     mask != nullptr ? *mask : Tensor{}, attrs),
                   rt.tensors());
       }},
      {"ai.onnx:Det", MakeUnaryTrampoline<kernel::Det>()},
      {"ai.onnx:DequantizeLinear", MakeBinaryWithOptionalThirdTrampoline<kernel::DequantizeLinear>()},
      {"ai.onnx:DFT",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 1);
         if (node.input_size() > 3) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects at most 3 inputs.");
         }
         RequireOutputCount(node, 1);
         const Tensor &input = GetInput(node, 0, rt.tensors());
         const Tensor *dft_length = GetOptionalInput(node, 1, rt.tensors());
         const bool inverse = GetAttributeIntOrDefault(node, "inverse", 0) != 0;
         const bool onesided = GetAttributeIntOrDefault(node, "onesided", 0) != 0;
         int64_t axis = 1;
         const int64_t opset_version = rt.kernel_ctx().opset.version;
         if (opset_version >= 20) {
           const Tensor *axis_tensor = GetOptionalInput(node, 2, rt.tensors());
           if (axis_tensor != nullptr) {
             if (axis_tensor->element_count() != 1) {
               throw std::invalid_argument(
                   "RunNode: DFT 'axis' input must be a scalar tensor (or a 1-D "
                   "tensor with a single element).");
             }
             switch (axis_tensor->data_type) {
             case DataType::INT64:
               axis = axis_tensor->AsInt64()[0];
               break;
             case DataType::INT32:
               axis = static_cast<int64_t>(axis_tensor->AsInt32()[0]);
               break;
             default:
               throw std::invalid_argument(
                   "RunNode: DFT 'axis' input must be INT32 or INT64.");
             }
           }
         } else {
           axis = GetAttributeIntOrDefault(node, "axis", 1);
         }
         kernel::DFT k(rt.kernel_ctx());
         SetOutput(node, 0, k(input, dft_length, axis, onesided, inverse), rt.tensors());
       }},
      {"ai.onnx:Div", MakeBinaryTrampoline<kernel::Div>()},
      {"ai.onnx:DynamicQuantizeLinear",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 3);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         kernel::DynamicQuantizeLinear k(rt.kernel_ctx());
         auto out = k(x);
         SetOutput(node, 0, std::move(std::get<0>(out)), rt.tensors());
         SetOutput(node, 1, std::move(std::get<1>(out)), rt.tensors());
         SetOutput(node, 2, std::move(std::get<2>(out)), rt.tensors());
       }},
      {"ai.onnx:Einsum",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 1);
         RequireOutputCount(node, 1);
         std::vector<Tensor> inputs;
         inputs.reserve(node.input_size());
         for (int i = 0; i < node.input_size(); ++i) {
           inputs.push_back(GetInput(node, i, rt.tensors()));
         }
         const std::string equation = GetRequiredAttributeString(node, "equation");
         kernel::Einsum k(rt.kernel_ctx());
         SetOutput(node, 0, k(inputs, equation), rt.tensors());
       }},
      {"ai.onnx:Elu", MakeUnaryAlphaTrampoline<kernel::Elu>("alpha", 1.0f)},
      {"ai.onnx:Equal", MakeBinaryTrampoline<kernel::Equal>()},
      {"ai.onnx:Erf", MakeUnaryTrampoline<kernel::Erf>()},
      {"ai.onnx:Exp", MakeUnaryTrampoline<kernel::Exp>()},
      {"ai.onnx:EyeLike",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const int64_t k = GetAttributeIntOrDefault(node, "k", 0);
         const int64_t dtype = GetAttributeIntOrDefault(node, "dtype", 0);
         kernel::EyeLike kernel(rt.kernel_ctx());
         SetOutput(node, 0, kernel(x, k, static_cast<int32_t>(dtype)), rt.tensors());
       }},
      {"ai.onnx:Floor", MakeUnaryTrampoline<kernel::Floor>()},
      {"ai.onnx:Gelu",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const std::string approximate = GetAttributeStringOrDefault(node, "approximate", "none");
         kernel::Gelu k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, approximate), rt.tensors());
       }},
      {"ai.onnx:Gemm",
       [](const NodeProto &node, RuntimeContext &rt) {
         if (node.input_size() < 2 || node.input_size() > 3) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects between 2 and 3 input(s), got " +
                                       std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 1);
         const Tensor &a = GetInput(node, 0, rt.tensors());
         const Tensor &b = GetInput(node, 1, rt.tensors());
         const Tensor *c = GetOptionalInput(node, 2, rt.tensors());
         const float alpha = GetAttributeFloatOrDefault(node, "alpha", 1.0f);
         const float beta = GetAttributeFloatOrDefault(node, "beta", 1.0f);
         const int64_t transA = GetAttributeIntOrDefault(node, "transA", 0);
         const int64_t transB = GetAttributeIntOrDefault(node, "transB", 0);
         kernel::Gemm k(rt.kernel_ctx());
         SetOutput(node, 0, k(a, b, c, alpha, beta, transA, transB), rt.tensors());
       }},
      {"ai.onnx:Greater", MakeBinaryTrampoline<kernel::Greater>()},
      {"ai.onnx:GreaterOrEqual", MakeBinaryTrampoline<kernel::GreaterOrEqual>()},
      {"ai.onnx:HardSigmoid",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const float alpha = GetAttributeFloatOrDefault(node, "alpha", 0.2f);
         const float beta = GetAttributeFloatOrDefault(node, "beta", 0.5f);
         kernel::HardSigmoid k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, alpha, beta), rt.tensors());
       }},
      {"ai.onnx:HardSwish", MakeUnaryTrampoline<kernel::HardSwish>()},
      {"ai.onnx:Hardmax", MakeAxisTrampoline<kernel::Hardmax>()},
      {"ai.onnx:ImageDecoder",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &encoded_stream = GetInput(node, 0, rt.tensors());
         const std::string pixel_format =
             GetAttributeStringOrDefault(node, "pixel_format", "RGB");
         kernel::ImageDecoder image_decoder_kernel(rt.kernel_ctx());
         SetOutput(node, 0, image_decoder_kernel(encoded_stream, pixel_format), rt.tensors());
       }},
      {"ai.onnx:IsInf",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const int64_t detect_positive = GetAttributeIntOrDefault(node, "detect_positive", 1);
         const int64_t detect_negative = GetAttributeIntOrDefault(node, "detect_negative", 1);
         kernel::IsInf k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, detect_positive, detect_negative), rt.tensors());
       }},
      {"ai.onnx:IsNaN", MakeUnaryTrampoline<kernel::IsNaN>()},
      {"ai.onnx:LeakyRelu", MakeUnaryAlphaTrampoline<kernel::LeakyRelu>("alpha", 0.01f)},
      {"ai.onnx:Less", MakeBinaryTrampoline<kernel::Less>()},
      {"ai.onnx:LessOrEqual", MakeBinaryTrampoline<kernel::LessOrEqual>()},
      {"ai.onnx:Log", MakeUnaryTrampoline<kernel::Log>()},
      {"ai.onnx:LogSoftmax", MakeAxisTrampoline<kernel::LogSoftmax>()},
      {"ai.onnx:MatMul", MakeBinaryTrampoline<kernel::MatMul>()},
      {"ai.onnx:Max", MakeVariadicTrampoline<kernel::Max>()},
      {"ai.onnx:Mean", MakeVariadicTrampoline<kernel::Mean>()},
      {"ai.onnx:Min", MakeVariadicTrampoline<kernel::Min>()},
      {"ai.onnx:Mish", MakeUnaryTrampoline<kernel::Mish>()},
      {"ai.onnx:Mod",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &y = GetInput(node, 1, rt.tensors());
         const int64_t fmod = GetAttributeIntOrDefault(node, "fmod", 0);
         kernel::Mod k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, y, fmod), rt.tensors());
       }},
      {"ai.onnx:Mul", MakeBinaryTrampoline<kernel::Mul>()},
      {"ai.onnx:Neg", MakeUnaryTrampoline<kernel::Neg>()},
      {"ai.onnx:NonMaxSuppression",
       [](const NodeProto &node, RuntimeContext &rt) {
         if (node.input_size() < 2 || node.input_size() > 5) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects between 2 and 5 inputs, got " +
                                       std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 1);
         const Tensor &boxes = GetInput(node, 0, rt.tensors());
         const Tensor &scores = GetInput(node, 1, rt.tensors());
         const Tensor *max_output_boxes_per_class = GetOptionalInput(node, 2, rt.tensors());
         const Tensor *iou_threshold = GetOptionalInput(node, 3, rt.tensors());
         const Tensor *score_threshold = GetOptionalInput(node, 4, rt.tensors());
         // For ONNX NonMaxSuppression (opset 10+), this runtime path uses the
         // single schema attribute center_point_box (default 0).
         kernel::NonMaxSuppression::Attributes attrs;
         attrs.center_point_box = GetAttributeIntOrDefault(node, "center_point_box", 0);
         kernel::NonMaxSuppression k(rt.kernel_ctx());
         SetOutput(node, 0,
                   k(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
                     attrs),
                   rt.tensors());
       }},
      {"ai.onnx:Not", MakeUnaryTrampoline<kernel::Not>()},
      {"ai.onnx:Or", MakeBinaryTrampoline<kernel::Or>()},
      {"ai.onnx:Pow", MakeBinaryTrampoline<kernel::Pow>()},
      {"ai.onnx:PRelu", MakeBinaryTrampoline<kernel::PRelu>()},
      {"ai.onnx:QuantizeLinear", MakeBinaryWithOptionalThirdTrampoline<kernel::QuantizeLinear>()},
      {"ai.onnx:Reciprocal", MakeUnaryTrampoline<kernel::Reciprocal>()},
      {"ai.onnx:ReduceL1", MakeReduceTrampoline<kernel::ReduceL1>()},
      {"ai.onnx:ReduceL2", MakeReduceTrampoline<kernel::ReduceL2>()},
      {"ai.onnx:ReduceLogSum", MakeReduceTrampoline<kernel::ReduceLogSum>()},
      {"ai.onnx:ReduceLogSumExp", MakeReduceTrampoline<kernel::ReduceLogSumExp>()},
      {"ai.onnx:ReduceMax", MakeReduceTrampoline<kernel::ReduceMax>()},
      {"ai.onnx:ReduceMean", MakeReduceTrampoline<kernel::ReduceMean>()},
      {"ai.onnx:ReduceMin", MakeReduceTrampoline<kernel::ReduceMin>()},
      {"ai.onnx:ReduceProd", MakeReduceTrampoline<kernel::ReduceProd>()},
      {"ai.onnx:ReduceSum", MakeReduceTrampoline<kernel::ReduceSum>()},
      {"ai.onnx:ReduceSumSquare", MakeReduceTrampoline<kernel::ReduceSumSquare>()},
      {"ai.onnx:Relu", MakeUnaryTrampoline<kernel::Relu>()},
      {"ai.onnx:Reshape",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &data = GetInput(node, 0, rt.tensors());
         const Tensor &shape = GetInput(node, 1, rt.tensors());
         const int64_t allowzero = GetAttributeIntOrDefault(node, "allowzero", 0);
         kernel::Reshape k(rt.kernel_ctx());
         SetOutput(node, 0, k(data, shape, allowzero), rt.tensors());
       }},
      {"ai.onnx:Round", MakeUnaryTrampoline<kernel::Round>()},
      {"ai.onnx:Selu",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const float alpha = GetAttributeFloatOrDefault(node, "alpha", 1.67326319217681884765625f);
         const float gamma = GetAttributeFloatOrDefault(node, "gamma", 1.05070102214813232421875f);
         kernel::Selu k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, alpha, gamma), rt.tensors());
       }},
      {"ai.onnx:Shape",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &data = GetInput(node, 0, rt.tensors());
         kernel::Shape::Attributes shape_attrs;
         shape_attrs.start = GetAttributeIntOrDefault(node, "start", 0);
         const AttributeProto *end_attr = FindAttribute(node, "end");
         if (end_attr != nullptr) {
           shape_attrs.end = end_attr->i();
         }
         kernel::Shape shape_kernel(rt.kernel_ctx());
         SetOutput(node, 0, shape_kernel(data, shape_attrs), rt.tensors());
       }},
      {"ai.onnx:Shrink",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const float bias = GetAttributeFloatOrDefault(node, "bias", 0.0f);
         const float lambd = GetAttributeFloatOrDefault(node, "lambd", 0.5f);
         kernel::Shrink k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, bias, lambd), rt.tensors());
       }},
      {"ai.onnx:Sigmoid", MakeUnaryTrampoline<kernel::Sigmoid>()},
      {"ai.onnx:Sign", MakeUnaryTrampoline<kernel::Sign>()},
      {"ai.onnx:Sin", MakeUnaryTrampoline<kernel::Sin>()},
      {"ai.onnx:Sinh", MakeUnaryTrampoline<kernel::Sinh>()},
      {"ai.onnx:Softmax", MakeAxisTrampoline<kernel::Softmax>()},
      {"ai.onnx:Softplus", MakeUnaryTrampoline<kernel::Softplus>()},
      {"ai.onnx:Softsign", MakeUnaryTrampoline<kernel::Softsign>()},
      {"ai.onnx:Sqrt", MakeUnaryTrampoline<kernel::Sqrt>()},
      {"ai.onnx:Squeeze", MakeSqueezeLikeTrampoline<kernel::Squeeze>("Squeeze")},
      {"ai.onnx:Sub", MakeBinaryTrampoline<kernel::Sub>()},
      {"ai.onnx:Sum", MakeVariadicTrampoline<kernel::Sum>()},
      {"ai.onnx:Swish", MakeUnaryAlphaTrampoline<kernel::Swish>("alpha", 1.0f)},
      {"ai.onnx:Tan", MakeUnaryTrampoline<kernel::Tan>()},
      {"ai.onnx:Tanh", MakeUnaryTrampoline<kernel::Tanh>()},
      {"ai.onnx:ThresholdedRelu", MakeUnaryAlphaTrampoline<kernel::ThresholdedRelu>("alpha", 1.0f)},
      {"ai.onnx:Unsqueeze", MakeSqueezeLikeTrampoline<kernel::Unsqueeze>("Unsqueeze")},
      {"ai.onnx:Where", MakeTernaryTrampoline<kernel::Where>()},
      {"ai.onnx:Xor", MakeBinaryTrampoline<kernel::Xor>()},

      // ai.onnx.preview.training 
      {"ai.onnx.preview.training:Adagrad",
       [](const NodeProto &node, RuntimeContext &rt) {
         if (node.input_size() < 5 || (node.input_size() - 2) % 3 != 0) {
           throw std::invalid_argument(
               "RunNode: op 'Adagrad' expects 2 + 3*N inputs (got " +
               std::to_string(node.input_size()) + ").");
         }
         const int64_t n = (node.input_size() - 2) / 3;
         if (node.output_size() != 2 * n) {
           throw std::invalid_argument("RunNode: op 'Adagrad' expects 2*N outputs (got " +
                                       std::to_string(node.output_size()) + " for N=" +
                                       std::to_string(n) + ").");
         }
         const Tensor &R = GetInput(node, 0, rt.tensors());
         const Tensor &T = GetInput(node, 1, rt.tensors());
         std::vector<Tensor> Xs, Gs, Hs;
         Xs.reserve(n);
         Gs.reserve(n);
         Hs.reserve(n);
         for (int64_t i = 0; i < n; ++i) {
           Xs.push_back(GetInput(node, static_cast<int>(2 + i), rt.tensors()));
           Gs.push_back(GetInput(node, static_cast<int>(2 + n + i), rt.tensors()));
           Hs.push_back(GetInput(node, static_cast<int>(2 + 2 * n + i), rt.tensors()));
         }
         const float epsilon = GetAttributeFloatOrDefault(node, "epsilon", 0.0f);
         const float decay_factor = GetAttributeFloatOrDefault(node, "decay_factor", 0.0f);
         const float norm_coefficient =
             GetAttributeFloatOrDefault(node, "norm_coefficient", 0.0f);
         kernel::Adagrad k(rt.kernel_ctx());
         std::vector<Tensor> outs =
             k(R, T, Xs, Gs, Hs, epsilon, decay_factor, norm_coefficient);
         for (int64_t i = 0; i < 2 * n; ++i) {
           SetOutput(node, static_cast<int>(i), std::move(outs[static_cast<size_t>(i)]),
                     rt.tensors());
         }
       }},
      {"ai.onnx.preview.training:Adam",
       [](const NodeProto &node, RuntimeContext &rt) {
         if (node.input_size() < 6 || (node.input_size() - 2) % 4 != 0) {
           throw std::invalid_argument("RunNode: op 'Adam' expects 2 + 4*N inputs (got " +
                                       std::to_string(node.input_size()) + ").");
         }
         const int64_t n = (node.input_size() - 2) / 4;
         if (node.output_size() != 3 * n) {
           throw std::invalid_argument("RunNode: op 'Adam' expects 3*N outputs (got " +
                                       std::to_string(node.output_size()) + " for N=" +
                                       std::to_string(n) + ").");
         }
         const Tensor &R = GetInput(node, 0, rt.tensors());
         const Tensor &T = GetInput(node, 1, rt.tensors());
         std::vector<Tensor> Xs, Gs, Vs, Hs;
         Xs.reserve(n);
         Gs.reserve(n);
         Vs.reserve(n);
         Hs.reserve(n);
         for (int64_t i = 0; i < n; ++i) {
           Xs.push_back(GetInput(node, static_cast<int>(2 + i), rt.tensors()));
           Gs.push_back(GetInput(node, static_cast<int>(2 + n + i), rt.tensors()));
           Vs.push_back(GetInput(node, static_cast<int>(2 + 2 * n + i), rt.tensors()));
           Hs.push_back(GetInput(node, static_cast<int>(2 + 3 * n + i), rt.tensors()));
         }
         const float alpha = GetAttributeFloatOrDefault(node, "alpha", 0.9f);
         const float beta = GetAttributeFloatOrDefault(node, "beta", 0.999f);
         const float epsilon = GetAttributeFloatOrDefault(node, "epsilon", 1e-6f);
         const float norm_coefficient =
             GetAttributeFloatOrDefault(node, "norm_coefficient", 0.0f);
         const float norm_coefficient_post =
             GetAttributeFloatOrDefault(node, "norm_coefficient_post", 0.0f);
         kernel::Adam k(rt.kernel_ctx());
         std::vector<Tensor> outs = k(R, T, Xs, Gs, Vs, Hs, alpha, beta, epsilon,
                                      norm_coefficient, norm_coefficient_post);
         for (int64_t i = 0; i < 3 * n; ++i) {
           SetOutput(node, static_cast<int>(i), std::move(outs[static_cast<size_t>(i)]),
                     rt.tensors());
         }
       }},
      {"ai.onnx.preview.training:Momentum",
       [](const NodeProto &node, RuntimeContext &rt) {
         if (node.input_size() < 5 || (node.input_size() - 2) % 3 != 0) {
           throw std::invalid_argument(
               "RunNode: op 'Momentum' expects 2 + 3*N inputs (got " +
               std::to_string(node.input_size()) + ").");
         }
         const int64_t n = (node.input_size() - 2) / 3;
         if (node.output_size() != 2 * n) {
           throw std::invalid_argument("RunNode: op 'Momentum' expects 2*N outputs (got " +
                                       std::to_string(node.output_size()) + " for N=" +
                                       std::to_string(n) + ").");
         }
         const Tensor &R = GetInput(node, 0, rt.tensors());
         const Tensor &T = GetInput(node, 1, rt.tensors());
         std::vector<Tensor> Xs, Gs, Vs;
         Xs.reserve(n);
         Gs.reserve(n);
         Vs.reserve(n);
         for (int64_t i = 0; i < n; ++i) {
           Xs.push_back(GetInput(node, static_cast<int>(2 + i), rt.tensors()));
           Gs.push_back(GetInput(node, static_cast<int>(2 + n + i), rt.tensors()));
           Vs.push_back(GetInput(node, static_cast<int>(2 + 2 * n + i), rt.tensors()));
         }
         const float alpha = GetAttributeFloatOrDefault(node, "alpha", 0.0f);
         const float beta = GetAttributeFloatOrDefault(node, "beta", 0.0f);
         const float norm_coefficient =
             GetAttributeFloatOrDefault(node, "norm_coefficient", 0.0f);
         const std::string mode_str = GetAttributeStringOrDefault(node, "mode", "standard");
         kernel::Momentum::Mode mode;
         if (mode_str == "standard") {
           mode = kernel::Momentum::Mode::kStandard;
         } else if (mode_str == "nesterov") {
           mode = kernel::Momentum::Mode::kNesterov;
         } else {
           throw std::invalid_argument(
               "RunNode: Momentum 'mode' must be 'standard' or 'nesterov', got '" + mode_str +
               "'.");
         }
         kernel::Momentum k(rt.kernel_ctx());
         std::vector<Tensor> outs = k(R, T, Xs, Gs, Vs, alpha, beta, norm_coefficient, mode);
         for (int64_t i = 0; i < 2 * n; ++i) {
           SetOutput(node, static_cast<int>(i), std::move(outs[static_cast<size_t>(i)]),
                     rt.tensors());
         }
       }},

      // ai.onnx.ml
      {"ai.onnx.ml:SVMRegressor",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const SVMCommonAttrs a = ParseSVMCommonAttrs(node, "SVMRegressor");
         kernel::SVMRegressor svm(rt.kernel_ctx());
         Tensor y = DispatchSVMByDataType(x, "SVMRegressor", [&](auto *tag) {
           using T = std::remove_pointer_t<decltype(tag)>;
           (void)tag;
           return svm.template operator()<T>(x, a.support_vectors, a.coefficients, a.rho,
                                             a.kernel_type.c_str(), a.gamma, a.coef0, a.degree);
         });
         SetOutput(node, 0, std::move(y), rt.tensors());
       }},
      {"ai.onnx.ml:SVMClassifier",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 2);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const SVMCommonAttrs a = ParseSVMCommonAttrs(node, "SVMClassifier");
         const std::vector<int64_t> vectors_per_class =
             GetAttributeIntsOrDefault(node, "vectors_per_class", {});
         const std::vector<int64_t> classlabels_ints =
             GetAttributeIntsOrDefault(node, "classlabels_ints", {});
         const std::vector<std::string> classlabels_strings =
             GetAttributeStringsOrDefault(node, "classlabels_strings", {});
         const bool use_strings = !classlabels_strings.empty();
         const bool has_ints = !classlabels_ints.empty();
         if (use_strings == has_ints) {
           throw std::invalid_argument(
               "RunNode: SVMClassifier requires exactly one of 'classlabels_ints' or "
               "'classlabels_strings' to be set.");
         }
         kernel::SVMClassifier svm(rt.kernel_ctx());
         std::pair<Tensor, Tensor> yz =
             DispatchSVMByDataType(x, "SVMClassifier", [&](auto *tag) {
               using T = std::remove_pointer_t<decltype(tag)>;
               (void)tag;
               return use_strings
                          ? svm.template operator()<T>(x, a.support_vectors, a.coefficients, a.rho,
                                                       vectors_per_class, classlabels_strings,
                                                       a.kernel_type.c_str(), a.gamma, a.coef0,
                                                       a.degree)
                          : svm.template operator()<T>(x, a.support_vectors, a.coefficients, a.rho,
                                                       vectors_per_class, classlabels_ints,
                                                       a.kernel_type.c_str(), a.gamma, a.coef0,
                                                       a.degree);
             });
         SetOutput(node, 0, std::move(yz.first), rt.tensors());
         SetOutput(node, 1, std::move(yz.second), rt.tensors());
       }},
      {"ai.onnx.ml:TreeEnsembleRegressor",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const std::vector<int64_t> nodes_treeids =
             GetAttributeIntsOrDefault(node, "nodes_treeids", {});
         const std::vector<int64_t> nodes_nodeids =
             GetAttributeIntsOrDefault(node, "nodes_nodeids", {});
         const std::vector<int64_t> nodes_featureids =
             GetAttributeIntsOrDefault(node, "nodes_featureids", {});
         const std::vector<float> nodes_values =
             GetAttributeFloatsOrDefault(node, "nodes_values", {});
         const std::vector<std::string> nodes_modes =
             GetAttributeStringsOrDefault(node, "nodes_modes", {});
         const std::vector<int64_t> nodes_truenodeids =
             GetAttributeIntsOrDefault(node, "nodes_truenodeids", {});
         const std::vector<int64_t> nodes_falsenodeids =
             GetAttributeIntsOrDefault(node, "nodes_falsenodeids", {});
         const std::vector<int64_t> nodes_missing =
             GetAttributeIntsOrDefault(node, "nodes_missing_value_tracks_true", {});
         const std::vector<int64_t> target_treeids =
             GetAttributeIntsOrDefault(node, "target_treeids", {});
         const std::vector<int64_t> target_nodeids =
             GetAttributeIntsOrDefault(node, "target_nodeids", {});
         const std::vector<int64_t> target_ids =
             GetAttributeIntsOrDefault(node, "target_ids", {});
         const std::vector<float> target_weights =
             GetAttributeFloatsOrDefault(node, "target_weights", {});
         const int64_t n_targets = GetAttributeIntOrDefault(node, "n_targets", 1);
         const std::string aggregate_function =
             GetAttributeStringOrDefault(node, "aggregate_function", "SUM");
         const std::string post_transform =
             GetAttributeStringOrDefault(node, "post_transform", "NONE");
         const std::vector<float> base_values =
             GetAttributeFloatsOrDefault(node, "base_values", {});
         kernel::TreeEnsembleRegressor reg(rt.kernel_ctx());
         Tensor y =
             DispatchTreeEnsembleClassicByDataType(x, "TreeEnsembleRegressor", [&](auto *tag) {
               using T = std::remove_pointer_t<decltype(tag)>;
               (void)tag;
               return reg.template operator()<T>(
                   x, nodes_treeids, nodes_nodeids, nodes_featureids, nodes_values, nodes_modes,
                   nodes_truenodeids, nodes_falsenodeids, nodes_missing, target_treeids,
                   target_nodeids, target_ids, target_weights, n_targets, aggregate_function,
                   post_transform, base_values);
             });
         SetOutput(node, 0, std::move(y), rt.tensors());
       }},
      {"ai.onnx.ml:TreeEnsembleClassifier",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 2);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const std::vector<int64_t> nodes_treeids =
             GetAttributeIntsOrDefault(node, "nodes_treeids", {});
         const std::vector<int64_t> nodes_nodeids =
             GetAttributeIntsOrDefault(node, "nodes_nodeids", {});
         const std::vector<int64_t> nodes_featureids =
             GetAttributeIntsOrDefault(node, "nodes_featureids", {});
         const std::vector<float> nodes_values =
             GetAttributeFloatsOrDefault(node, "nodes_values", {});
         const std::vector<std::string> nodes_modes =
             GetAttributeStringsOrDefault(node, "nodes_modes", {});
         const std::vector<int64_t> nodes_truenodeids =
             GetAttributeIntsOrDefault(node, "nodes_truenodeids", {});
         const std::vector<int64_t> nodes_falsenodeids =
             GetAttributeIntsOrDefault(node, "nodes_falsenodeids", {});
         const std::vector<int64_t> nodes_missing =
             GetAttributeIntsOrDefault(node, "nodes_missing_value_tracks_true", {});
         const std::vector<int64_t> class_treeids =
             GetAttributeIntsOrDefault(node, "class_treeids", {});
         const std::vector<int64_t> class_nodeids =
             GetAttributeIntsOrDefault(node, "class_nodeids", {});
         const std::vector<int64_t> class_ids =
             GetAttributeIntsOrDefault(node, "class_ids", {});
         const std::vector<float> class_weights =
             GetAttributeFloatsOrDefault(node, "class_weights", {});
         const std::vector<int64_t> classlabels_int64s =
             GetAttributeIntsOrDefault(node, "classlabels_int64s", {});
         const std::vector<std::string> classlabels_strings =
             GetAttributeStringsOrDefault(node, "classlabels_strings", {});
         const std::vector<float> base_values =
             GetAttributeFloatsOrDefault(node, "base_values", {});
         const std::string post_transform =
             GetAttributeStringOrDefault(node, "post_transform", "NONE");
         const bool use_strings = !classlabels_strings.empty();
         const bool has_ints = !classlabels_int64s.empty();
         if (use_strings == has_ints) {
           throw std::invalid_argument(
               "RunNode: TreeEnsembleClassifier requires exactly one of "
               "'classlabels_int64s' or 'classlabels_strings' to be set.");
         }
         kernel::TreeEnsembleClassifier cls(rt.kernel_ctx());
         std::pair<Tensor, Tensor> yz = DispatchTreeEnsembleClassicByDataType(
             x, "TreeEnsembleClassifier", [&](auto *tag) {
               using T = std::remove_pointer_t<decltype(tag)>;
               (void)tag;
               return use_strings
                          ? cls.template operator()<T>(
                                x, nodes_treeids, nodes_nodeids, nodes_featureids, nodes_values,
                                nodes_modes, nodes_truenodeids, nodes_falsenodeids, nodes_missing,
                                class_treeids, class_nodeids, class_ids, class_weights,
                                classlabels_strings, base_values, post_transform)
                          : cls.template operator()<T>(
                                x, nodes_treeids, nodes_nodeids, nodes_featureids, nodes_values,
                                nodes_modes, nodes_truenodeids, nodes_falsenodeids, nodes_missing,
                                class_treeids, class_nodeids, class_ids, class_weights,
                                classlabels_int64s, base_values, post_transform);
             });
         SetOutput(node, 0, std::move(yz.first), rt.tensors());
         SetOutput(node, 1, std::move(yz.second), rt.tensors());
       }},
      {"ai.onnx.ml:TreeEnsemble",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const std::vector<int64_t> tree_roots =
             GetAttributeIntsOrDefault(node, "tree_roots", {});
         const std::vector<int64_t> nodes_featureids =
             GetAttributeIntsOrDefault(node, "nodes_featureids", {});
         const std::vector<int64_t> nodes_truenodeids =
             GetAttributeIntsOrDefault(node, "nodes_truenodeids", {});
         const std::vector<int64_t> nodes_falsenodeids =
             GetAttributeIntsOrDefault(node, "nodes_falsenodeids", {});
         const std::vector<int64_t> nodes_trueleafs =
             GetAttributeIntsOrDefault(node, "nodes_trueleafs", {});
         const std::vector<int64_t> nodes_falseleafs =
             GetAttributeIntsOrDefault(node, "nodes_falseleafs", {});
         const std::vector<int64_t> nodes_missing =
             GetAttributeIntsOrDefault(node, "nodes_missing_value_tracks_true", {});
         const std::vector<int64_t> leaf_targetids =
             GetAttributeIntsOrDefault(node, "leaf_targetids", {});
         const int64_t n_targets = GetAttributeIntOrDefault(node, "n_targets", 1);
         const int64_t aggregate_function =
             GetAttributeIntOrDefault(node, "aggregate_function", 1);
         const int64_t post_transform = GetAttributeIntOrDefault(node, "post_transform", 0);
         const Tensor nodes_splits = GetRequiredAttributeTensor(node, "nodes_splits");
         const Tensor leaf_weights = GetRequiredAttributeTensor(node, "leaf_weights");
         const Tensor nodes_modes_t = GetRequiredAttributeTensor(node, "nodes_modes");
         const Tensor membership_values =
             GetAttributeTensorOrEmpty(node, "membership_values", x.data_type);
         if (nodes_modes_t.data_type != static_cast<int32_t>(DataType::UINT8)) {
           throw std::invalid_argument(
               "RunNode: TreeEnsemble attribute 'nodes_modes' must be a UINT8 tensor.");
         }
         if (nodes_splits.data_type != x.data_type ||
             leaf_weights.data_type != x.data_type) {
           throw std::invalid_argument(
               "RunNode: TreeEnsemble attributes 'nodes_splits' and 'leaf_weights' must "
               "have the same element type as input 'X'.");
         }
         if (membership_values.element_count() > 0 &&
             membership_values.data_type != x.data_type) {
           throw std::invalid_argument(
               "RunNode: TreeEnsemble attribute 'membership_values' must have the same "
               "element type as input 'X'.");
         }
         const std::vector<uint8_t> nodes_modes_vec = TensorToVector<uint8_t>(nodes_modes_t);
         kernel::TreeEnsemble tree_ens(rt.kernel_ctx());
         Tensor y;
         switch (x.data_type) {
         case static_cast<int32_t>(DataType::FLOAT): {
           const std::vector<float> splits = TensorToVector<float>(nodes_splits);
           const std::vector<float> leaves = TensorToVector<float>(leaf_weights);
           const std::vector<float> members =
               membership_values.element_count() > 0
                   ? TensorToVector<float>(membership_values)
                   : std::vector<float>{};
           y = tree_ens.operator()<float>(x, tree_roots, nodes_featureids, splits, nodes_modes_vec,
                                          nodes_truenodeids, nodes_falsenodeids, nodes_trueleafs,
                                          nodes_falseleafs, nodes_missing, leaf_targetids, leaves,
                                          members, n_targets, aggregate_function, post_transform);
           break;
         }
         case static_cast<int32_t>(DataType::DOUBLE): {
           const std::vector<double> splits = TensorToVector<double>(nodes_splits);
           const std::vector<double> leaves = TensorToVector<double>(leaf_weights);
           const std::vector<double> members =
               membership_values.element_count() > 0
                   ? TensorToVector<double>(membership_values)
                   : std::vector<double>{};
           y = tree_ens.operator()<double>(x, tree_roots, nodes_featureids, splits, nodes_modes_vec,
                                           nodes_truenodeids, nodes_falsenodeids, nodes_trueleafs,
                                           nodes_falseleafs, nodes_missing, leaf_targetids, leaves,
                                           members, n_targets, aggregate_function, post_transform);
           break;
         }
         default:
           throw std::invalid_argument(
               "RunNode: TreeEnsemble input 'X' must be FLOAT or DOUBLE.");
         }
         SetOutput(node, 0, std::move(y), rt.tensors());
       }},
  };
  return table;
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
