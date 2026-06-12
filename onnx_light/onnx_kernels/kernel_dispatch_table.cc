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
#include "onnx_kernels/kernels/optional/include_optional_kernels.h"
#include "onnx_kernels/kernels/preview/include_preview_kernels.h"
#include "onnx_kernels/kernels/quantization/include_quantization_kernels.h"
#include "onnx_kernels/kernels/reduction/include_reduction_kernels.h"
#include "onnx_kernels/kernels/sequence/include_sequence_kernels.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_kernels/kernels/text/include_text_kernels.h"
#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_kernels/kernels/training/include_training_kernels.h"
#include "onnx_kernels/node_helpers.h"
#include "onnx_kernels/run_nodes.h"
#include "onnx_kernels/simple_tensor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
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
using detail::GetInputSequence;
using detail::GetOptionalInput;
using detail::GetRequiredAttributeInt;
using detail::GetRequiredAttributeString;
using detail::RequireInputCount;
using detail::RequireMinInputCount;
using detail::RequireOutputCount;
using detail::SetOutput;
using detail::SetOutputSequence;

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
    SetOutput(node, 0, kernel(x), rt);
  };
}

template <class KernelT> NodeKernelFn MakeBinaryTrampoline() {
  return [](const NodeProto &node, RuntimeContext &rt) {
    RequireInputCount(node, 2);
    RequireOutputCount(node, 1);
    const Tensor &x = GetInput(node, 0, rt.tensors());
    const Tensor &y = GetInput(node, 1, rt.tensors());
    KernelT kernel(rt.kernel_ctx());
    SetOutput(node, 0, kernel(x, y), rt);
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
      SetOutput(node, 0, kernel(a, b, *c), rt);
    } else {
      SetOutput(node, 0, kernel(a, b), rt);
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
    SetOutput(node, 0, kernel(a, b, c), rt);
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
    SetOutput(node, 0, kernel(inputs), rt);
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
    SetOutput(node, 0, kernel(x, alpha), rt);
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
    SetOutput(node, 0, kernel(x, axis), rt);
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
    SetOutput(node, 0, kernel(x, to), rt);
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
      SetOutput(node, 0, kernel(data, *axes_input, keepdims, noop_with_empty_axes), rt);
      return;
    }

    const std::vector<int64_t> axes_attr = GetAttributeIntsOrDefault(node, "axes", {});
    if (!axes_attr.empty()) {
      const Tensor axes =
          Tensor::FromInt64("", {static_cast<int64_t>(axes_attr.size())}, axes_attr);
      SetOutput(node, 0, kernel(data, axes, keepdims, noop_with_empty_axes), rt);
      return;
    }

    SetOutput(node, 0, kernel(data, keepdims, noop_with_empty_axes), rt);
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
    SetOutput(node, 0, k(data, axes), rt);
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
    SetOutput(node, 0, kernel(data, axis, keepdims, select_last_index), rt);
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

// Wraps a generator kernel of the form
// ``Tensor operator()(const std::vector<int64_t>&, float, float, int64_t, int32_t)``
// (``RandomNormal``, ``RandomUniform``): no inputs, ``shape`` INTS attribute
// and two FLOAT attributes (``mean``/``scale`` or ``low``/``high``).
template <class KernelT>
NodeKernelFn MakeRandomGenTrampoline(const char *attr_a, float default_a, const char *attr_b,
                                     float default_b) {
  const std::string name_a(attr_a);
  const std::string name_b(attr_b);
  return [name_a, default_a, name_b, default_b](const NodeProto &node, RuntimeContext &rt) {
    RequireInputCount(node, 0);
    RequireOutputCount(node, 1);
    const std::vector<int64_t> shape = GetAttributeIntsOrDefault(node, "shape", {});
    const float a = GetAttributeFloatOrDefault(node, name_a, default_a);
    const float b = GetAttributeFloatOrDefault(node, name_b, default_b);
    KernelT kernel(rt.kernel_ctx());
    SetOutput(node, 0, kernel(shape, a, b, GetSeedAttr(node), GetDtypeAttr(node)), rt);
  };
}

// Wraps a generator kernel of the form
// ``Tensor operator()(const Tensor&, float, float, int64_t, int32_t)``
// (``RandomNormalLike``, ``RandomUniformLike``): one input and two FLOAT
// attributes (``mean``/``scale`` or ``low``/``high``).
template <class KernelT>
NodeKernelFn MakeRandomLikeTrampoline(const char *attr_a, float default_a, const char *attr_b,
                                      float default_b) {
  const std::string name_a(attr_a);
  const std::string name_b(attr_b);
  return [name_a, default_a, name_b, default_b](const NodeProto &node, RuntimeContext &rt) {
    RequireInputCount(node, 1);
    RequireOutputCount(node, 1);
    const Tensor &input = GetInput(node, 0, rt.tensors());
    const float a = GetAttributeFloatOrDefault(node, name_a, default_a);
    const float b = GetAttributeFloatOrDefault(node, name_b, default_b);
    KernelT kernel(rt.kernel_ctx());
    SetOutput(node, 0, kernel(input, a, b, GetSeedAttr(node), GetDtypeAttr(node)), rt);
  };
}

// Wraps a window-generation kernel of the form
// ``Tensor operator()(const Tensor& size, bool periodic)`` (``BlackmanWindow``,
// ``HannWindow``, ``HammingWindow``): one INT scalar input, optional ``periodic``
// (default 1) and ``output_datatype`` (default FLOAT=1) attributes. Only
// ``output_datatype == FLOAT`` is supported because the underlying kernels
// always produce FLOAT outputs.
template <class KernelT> NodeKernelFn MakeWindowTrampoline(const char *op_name) {
  const std::string name(op_name);
  return [name](const NodeProto &node, RuntimeContext &rt) {
    RequireInputCount(node, 1);
    RequireOutputCount(node, 1);
    const Tensor &size = GetInput(node, 0, rt.tensors());
    const int64_t output_datatype =
        GetAttributeIntOrDefault(node, "output_datatype", static_cast<int64_t>(DataType::FLOAT));
    if (output_datatype != static_cast<int64_t>(DataType::FLOAT)) {
      throw std::invalid_argument("RunNode: op '" + name +
                                  "' only supports output_datatype=FLOAT.");
    }
    const bool periodic = GetAttributeIntOrDefault(node, "periodic", 1) != 0;
    KernelT kernel(rt.kernel_ctx());
    SetOutput(node, 0, kernel(size, periodic), rt.tensors());
  };
}

// Trampoline for cumulative reduction ops (CumSum, CumProd) which take an
// input tensor and an ``axis`` scalar input plus the boolean ``exclusive``
// and ``reverse`` int attributes.
template <class KernelT> NodeKernelFn MakeCumulativeTrampoline() {
  return [](const NodeProto &node, RuntimeContext &rt) {
    RequireInputCount(node, 2);
    RequireOutputCount(node, 1);
    const Tensor &x = GetInput(node, 0, rt.tensors());
    const Tensor &axis = GetInput(node, 1, rt.tensors());
    const bool exclusive = GetAttributeIntOrDefault(node, "exclusive", 0) != 0;
    const bool reverse = GetAttributeIntOrDefault(node, "reverse", 0) != 0;
    KernelT kernel(rt.kernel_ctx());
    SetOutput(node, 0, kernel(x, axis, exclusive, reverse), rt);
  };
}

// ---------------------------------------------------------------------------
// Normalization op helpers
// ---------------------------------------------------------------------------
// The normalization kernels (BatchNormalization, GroupNormalization,
// InstanceNormalization, LayerNormalization, RMSNormalization) share the
// ``epsilon`` attribute (default 1e-5f). Several also share the ``axis``
// attribute (default -1) and the (X, scale, [bias]) input pattern. The
// helpers below centralise that boilerplate so each per-op runner stays
// focused on its kernel-specific glue.

inline float GetEpsilon(const NodeProto &node) {
  return GetAttributeFloatOrDefault(node, "epsilon", 1e-5f);
}

inline int64_t GetNormAxis(const NodeProto &node) {
  return GetAttributeIntOrDefault(node, "axis", -1);
}

inline void RequireInputRange(const NodeProto &node, int min_inputs, int max_inputs) {
  const int n = node.input_size();
  if (n < min_inputs || n > max_inputs) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() + "' expects " +
                                std::to_string(min_inputs) + " to " + std::to_string(max_inputs) +
                                " input(s), got " + std::to_string(n) + ".");
  }
}

inline void RequireOutputRange(const NodeProto &node, int min_outputs, int max_outputs) {
  const int n = node.output_size();
  if (n < min_outputs || n > max_outputs) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() + "' expects " +
                                std::to_string(min_outputs) + " to " + std::to_string(max_outputs) +
                                " output(s), got " + std::to_string(n) + ".");
  }
}

void RunBatchNormalization(const NodeProto &node, RuntimeContext &rt) {
  RequireInputCount(node, 5);
  RequireOutputRange(node, 1, 3);
  if (GetAttributeIntOrDefault(node, "training_mode", 0) != 0) {
    throw std::invalid_argument(
        "RunNode: op 'BatchNormalization' training_mode=1 is not supported.");
  }
  if (node.output_size() != 1) {
    throw std::invalid_argument("RunNode: op 'BatchNormalization' only supports a single output "
                                "(running_mean / running_var require training_mode=1).");
  }
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &scale = GetInput(node, 1, rt.tensors());
  const Tensor &bias = GetInput(node, 2, rt.tensors());
  const Tensor &input_mean = GetInput(node, 3, rt.tensors());
  const Tensor &input_var = GetInput(node, 4, rt.tensors());
  kernel::BatchNormalization k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, scale, bias, input_mean, input_var, GetEpsilon(node)), rt.tensors());
}

void RunGroupNormalization(const NodeProto &node, RuntimeContext &rt) {
  RequireInputCount(node, 3);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &scale = GetInput(node, 1, rt.tensors());
  const Tensor &bias = GetInput(node, 2, rt.tensors());
  const int64_t num_groups = GetAttributeIntOrDefault(node, "num_groups", 0);
  kernel::GroupNormalization k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, scale, bias, num_groups, GetEpsilon(node)), rt.tensors());
}

void RunInstanceNormalization(const NodeProto &node, RuntimeContext &rt) {
  RequireInputCount(node, 3);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &scale = GetInput(node, 1, rt.tensors());
  const Tensor &bias = GetInput(node, 2, rt.tensors());
  kernel::InstanceNormalization k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, scale, bias, GetEpsilon(node)), rt.tensors());
}

void RunLayerNormalization(const NodeProto &node, RuntimeContext &rt) {
  RequireInputRange(node, 2, 3);
  RequireOutputRange(node, 1, 3);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &scale = GetInput(node, 1, rt.tensors());
  const Tensor *b = GetOptionalInput(node, 2, rt.tensors());
  kernel::LayerNormalization k(rt.kernel_ctx());
  auto [y, mean, inv_std_dev] =
      k(x, scale, b != nullptr ? *b : Tensor{}, GetNormAxis(node), GetEpsilon(node));
  SetOutput(node, 0, std::move(y), rt.tensors());
  if (node.output_size() >= 2) {
    SetOutput(node, 1, std::move(mean), rt.tensors());
  }
  if (node.output_size() >= 3) {
    SetOutput(node, 2, std::move(inv_std_dev), rt.tensors());
  }
}

void RunRMSNormalization(const NodeProto &node, RuntimeContext &rt) {
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &scale = GetInput(node, 1, rt.tensors());
  kernel::RMSNormalization k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, scale, GetNormAxis(node), GetEpsilon(node)), rt.tensors());
}

// Shared spatial pooling attributes consumed by AveragePool / MaxPool
// (and other pool-style ops). The defaults mirror the ONNX schema.
struct PoolCommonAttrs {
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> dilations;
  bool ceil_mode;
  std::string auto_pad;
};

inline PoolCommonAttrs ParsePoolCommonAttrs(const NodeProto &node) {
  PoolCommonAttrs a;
  a.kernel_shape = GetAttributeIntsOrDefault(node, "kernel_shape", {});
  a.strides = GetAttributeIntsOrDefault(node, "strides", {});
  a.pads = GetAttributeIntsOrDefault(node, "pads", {});
  a.dilations = GetAttributeIntsOrDefault(node, "dilations", {});
  a.ceil_mode = GetAttributeIntOrDefault(node, "ceil_mode", 0) != 0;
  a.auto_pad = GetAttributeStringOrDefault(node, "auto_pad", "NOTSET");
  return a;
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
         SetOutput(node, 0, affine_grid_kernel(theta, size, affine_grid_attrs), rt);
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
         SetOutput(node, 0, std::move(result.Y), rt);

         auto set_optional_output = [&node, &rt](int index, Tensor output) {
           if (index >= node.output_size()) {
             return;
           }
           const std::string name = node.output(index).as_string();
           if (name.empty()) {
             return;
           }
           output.name = name;
           rt.Put(name, std::move(output), TensorEventKind::kIntermediate);
         };
         set_optional_output(1, std::move(result.present_key));
         set_optional_output(2, std::move(result.present_value));
         set_optional_output(3, std::move(result.qk_matmul_output));
       }},
      {"ai.onnx:AveragePool",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const PoolCommonAttrs a = ParsePoolCommonAttrs(node);
         const bool count_include_pad =
             GetAttributeIntOrDefault(node, "count_include_pad", 0) != 0;
         kernel::AveragePool k(rt.kernel_ctx());
         SetOutput(node, 0,
                   k(x, a.kernel_shape, a.strides, a.pads, a.ceil_mode, count_include_pad,
                     a.dilations, a.auto_pad),
                   rt.tensors());
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
         SetOutput(node, 0, k(x, y, dir), rt);
       }},
      {"ai.onnx:BitCast", MakeUnaryToTrampoline<kernel::BitCast>()},
      {"ai.onnx:BitwiseAnd", MakeBinaryTrampoline<kernel::BitwiseAnd>()},
      {"ai.onnx:BitwiseNot", MakeUnaryTrampoline<kernel::BitwiseNot>()},
      {"ai.onnx:BitwiseOr", MakeBinaryTrampoline<kernel::BitwiseOr>()},
      {"ai.onnx:BitwiseXor", MakeBinaryTrampoline<kernel::BitwiseXor>()},
      {"ai.onnx:BatchNormalization", RunBatchNormalization},
      {"ai.onnx:Bernoulli",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &input = GetInput(node, 0, rt.tensors());
         kernel::Bernoulli kernel(rt.kernel_ctx());
         SetOutput(node, 0, kernel(input, GetSeedAttr(node), GetDtypeAttr(node)), rt);
       }},
      {"ai.onnx:BlackmanWindow", MakeWindowTrampoline<kernel::BlackmanWindow>("BlackmanWindow")},
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
         SetOutput(node, 0, std::move(output), rt);
         SetOutput(node, 1, std::move(present_state), rt);
       }},
      {"ai.onnx:Cast", MakeUnaryToTrampoline<kernel::Cast>()},
      {"ai.onnx:CastLike",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &target_type = GetInput(node, 1, rt.tensors());
         kernel::CastLike k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, target_type), rt);
       }},
      {"ai.onnx:Ceil", MakeUnaryTrampoline<kernel::Ceil>()},
      {"ai.onnx:Celu", MakeUnaryAlphaTrampoline<kernel::Celu>("alpha", 1.0f)},
      {"ai.onnx:CenterCropPad",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &input_data = GetInput(node, 0, rt.tensors());
         const Tensor &shape = GetInput(node, 1, rt.tensors());
         kernel::CenterCropPad::Attributes attrs;
         const AttributeProto *axes_attr = FindAttribute(node, "axes");
         if (axes_attr != nullptr) {
           attrs.axes = GetAttributeIntsOrDefault(node, "axes", {});
           attrs.axes_present = true;
         }
         kernel::CenterCropPad k(rt.kernel_ctx());
         SetOutput(node, 0, k(input_data, shape, attrs), rt);
       }},
      {"ai.onnx:Constant",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireOutputCount(node, 1);
         kernel::Constant k(rt.kernel_ctx());
         Tensor y;
         if (FindAttribute(node, "value") != nullptr) {
           y = k(GetRequiredAttributeTensor(node, "value"));
         } else if (FindAttribute(node, "value_float") != nullptr) {
           const float v = GetAttributeFloatOrDefault(node, "value_float", 0.0f);
           y = Tensor::FromFloat("", /*shape=*/{}, {v});
         } else if (FindAttribute(node, "value_floats") != nullptr) {
           const std::vector<float> vs =
               GetAttributeFloatsOrDefault(node, "value_floats", {});
           y = Tensor::FromFloat("", {static_cast<int64_t>(vs.size())}, vs);
         } else if (FindAttribute(node, "value_int") != nullptr) {
           const int64_t v = GetAttributeIntOrDefault(node, "value_int", 0);
           y = Tensor::FromInt64("", /*shape=*/{}, {v});
         } else if (FindAttribute(node, "value_ints") != nullptr) {
           const std::vector<int64_t> vs =
               GetAttributeIntsOrDefault(node, "value_ints", {});
           y = Tensor::FromInt64("", {static_cast<int64_t>(vs.size())}, vs);
         } else if (FindAttribute(node, "value_string") != nullptr) {
           const std::string v = GetAttributeStringOrDefault(node, "value_string", "");
           y = Tensor::FromStrings("", /*shape=*/{}, {v});
         } else if (FindAttribute(node, "value_strings") != nullptr) {
           const std::vector<std::string> vs =
               GetAttributeStringsOrDefault(node, "value_strings", {});
           y = Tensor::FromStrings("", {static_cast<int64_t>(vs.size())}, vs);
         } else {
           throw std::invalid_argument(
               "RunNode: op 'Constant' requires one of: value, value_float, "
               "value_floats, value_int, value_ints, value_string, value_strings.");
         }
         SetOutput(node, 0, std::move(y), rt.tensors());
       }},
      {"ai.onnx:ConstantOfShape",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &shape = GetInput(node, 0, rt.tensors());
         Tensor value;
         if (FindAttribute(node, "value") != nullptr) {
           value = GetRequiredAttributeTensor(node, "value");
         }
         kernel::ConstantOfShape k(rt.kernel_ctx());
         SetOutput(node, 0, k(shape, value), rt);
       }},
      {"ai.onnx:Clip",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor *min = GetOptionalInput(node, 1, rt.tensors());
         const Tensor *max = GetOptionalInput(node, 2, rt.tensors());
         kernel::Clip k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, min, max), rt);
       }},
      {"ai.onnx:Col2Im",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 3);
         RequireOutputCount(node, 1);
         const Tensor &input = GetInput(node, 0, rt.tensors());
         const Tensor &image_shape = GetInput(node, 1, rt.tensors());
         const Tensor &block_shape = GetInput(node, 2, rt.tensors());
         kernel::Col2Im::Attributes attrs;
         attrs.dilations = GetAttributeIntsOrDefault(node, "dilations", {});
         attrs.pads = GetAttributeIntsOrDefault(node, "pads", {});
         attrs.strides = GetAttributeIntsOrDefault(node, "strides", {});
         kernel::Col2Im k(rt.kernel_ctx());
         SetOutput(node, 0, k(input, image_shape, block_shape, attrs), rt);
       }},
      {"ai.onnx:Compress",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &input = GetInput(node, 0, rt.tensors());
         const Tensor &condition = GetInput(node, 1, rt.tensors());
         const AttributeProto *axis_attr = FindAttribute(node, "axis");
         std::optional<int64_t> axis;
         if (axis_attr != nullptr) {
           axis = axis_attr->i();
         }
         kernel::Compress k(rt.kernel_ctx());
         SetOutput(node, 0, k(input, condition, axis), rt);
       }},
      {"ai.onnx:Concat",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 1);
         RequireOutputCount(node, 1);
         std::vector<Tensor> inputs;
         inputs.reserve(node.input_size());
         for (int i = 0; i < node.input_size(); ++i) {
           inputs.push_back(GetInput(node, i, rt.tensors()));
         }
         const AttributeProto *axis_attr = FindAttribute(node, "axis");
         if (axis_attr == nullptr) {
           throw std::invalid_argument(
               "RunNode: op 'Concat' is missing required attribute 'axis'.");
         }
         const int64_t axis = axis_attr->i();
         kernel::Concat k(rt.kernel_ctx());
         SetOutput(node, 0, k(inputs, axis), rt);
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
         SetOutput(node, 0, k(x, w, b != nullptr ? *b : Tensor{}, attrs), rt);
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
      {"ai.onnx:ConvTranspose",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 2);
         if (node.input_size() > 3) {
           throw std::invalid_argument("RunNode: op 'ConvTranspose' expects at most 3 inputs, got " +
                                       std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &w = GetInput(node, 1, rt.tensors());
         const Tensor *b = GetOptionalInput(node, 2, rt.tensors());
         kernel::ConvTranspose::Attributes attrs;
         attrs.kernel_shape = GetAttributeIntsOrDefault(node, "kernel_shape", {});
         attrs.strides = GetAttributeIntsOrDefault(node, "strides", {});
         attrs.pads = GetAttributeIntsOrDefault(node, "pads", {});
         attrs.dilations = GetAttributeIntsOrDefault(node, "dilations", {});
         attrs.output_padding = GetAttributeIntsOrDefault(node, "output_padding", {});
         attrs.output_shape = GetAttributeIntsOrDefault(node, "output_shape", {});
         attrs.group = GetAttributeIntOrDefault(node, "group", 1);
         attrs.auto_pad = GetAttributeStringOrDefault(node, "auto_pad", "NOTSET");
         kernel::ConvTranspose k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, w, b != nullptr ? *b : Tensor{}, attrs), rt);
       }},
      {"ai.onnx:Cos", MakeUnaryTrampoline<kernel::Cos>()},
      {"ai.onnx:Cosh", MakeUnaryTrampoline<kernel::Cosh>()},
      {"ai.onnx:CumSum", MakeCumulativeTrampoline<kernel::CumSum>()},
      {"ai.onnx:CumProd", MakeCumulativeTrampoline<kernel::CumProd>()},
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
      {"ai.onnx:DepthToSpace",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &input = GetInput(node, 0, rt.tensors());
         kernel::DepthToSpace::Attributes attrs;
         const AttributeProto *blocksize_attr = FindAttribute(node, "blocksize");
         if (blocksize_attr == nullptr) {
           throw std::invalid_argument("RunNode: DepthToSpace requires attribute 'blocksize'.");
         }
         if (blocksize_attr->type() != AttributeProto::AttributeType::INT) {
           throw std::invalid_argument("RunNode: DepthToSpace attribute 'blocksize' must be INT.");
         }
         attrs.blocksize = blocksize_attr->i();
         attrs.mode = GetAttributeStringOrDefault(node, "mode", "DCR");
         kernel::DepthToSpace kernel(rt.kernel_ctx());
         SetOutput(node, 0, kernel(input, attrs), rt);
       }},
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
         SetOutput(node, 0, k(input, dft_length, axis, onesided, inverse), rt);
       }},
      {"ai.onnx:Div", MakeBinaryTrampoline<kernel::Div>()},
      {"ai.onnx:Dropout",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputRange(node, 1, 3);
         RequireOutputRange(node, 1, 2);
         const Tensor &data = GetInput(node, 0, rt.tensors());

         // ``ratio``: from input[1] (scalar T1) when present, else from the
         // pre-opset-12 ``ratio`` attribute (FLOAT, default 0.5).
         float ratio = GetAttributeFloatOrDefault(node, "ratio", 0.5f);
         const Tensor *ratio_input = GetOptionalInput(node, 1, rt.tensors());
         if (ratio_input != nullptr) {
           if (ratio_input->element_count() != 1) {
             throw std::invalid_argument(
                 "RunNode: op 'Dropout' input 'ratio' must be a scalar tensor.");
           }
           switch (ratio_input->data_type) {
           case static_cast<int32_t>(DataType::FLOAT):
             ratio = ratio_input->AsFloat()[0];
             break;
           case static_cast<int32_t>(DataType::DOUBLE):
             ratio = static_cast<float>(ratio_input->AsDouble()[0]);
             break;
           default:
             throw std::invalid_argument(
                 "RunNode: op 'Dropout' input 'ratio' must be FLOAT or DOUBLE.");
           }
         }

         // ``training_mode``: from input[2] (scalar BOOL) when present,
         // otherwise defaults to false (inference behaviour).
         bool training_mode = false;
         const Tensor *training_input = GetOptionalInput(node, 2, rt.tensors());
         if (training_input != nullptr) {
           if (training_input->element_count() != 1) {
             throw std::invalid_argument(
                 "RunNode: op 'Dropout' input 'training_mode' must be a scalar tensor.");
           }
           if (training_input->data_type != static_cast<int32_t>(DataType::BOOL)) {
             throw std::invalid_argument(
                 "RunNode: op 'Dropout' input 'training_mode' must be BOOL.");
           }
           training_mode = training_input->AsBool()[0] != 0;
         }

         const int64_t seed = GetAttributeIntOrDefault(node, "seed", kernel::Dropout::kNoSeed);

         kernel::Dropout k(rt.kernel_ctx());
         if (node.output_size() == 2) {
           auto out = k(data, ratio, training_mode, seed);
           SetOutput(node, 0, std::move(out.first), rt);
           SetOutput(node, 1, std::move(out.second), rt);
         } else {
           Tensor mask("", static_cast<int32_t>(DataType::BOOL), data.shape,
                       std::vector<uint8_t>(static_cast<std::size_t>(data.element_count()), 1));
           SetOutput(node, 0, k(data, ratio, training_mode, mask, seed), rt);
         }
       }},
      {"ai.onnx:DynamicQuantizeLinear",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 3);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         kernel::DynamicQuantizeLinear k(rt.kernel_ctx());
         auto out = k(x);
         SetOutput(node, 0, std::move(std::get<0>(out)), rt);
         SetOutput(node, 1, std::move(std::get<1>(out)), rt);
         SetOutput(node, 2, std::move(std::get<2>(out)), rt);
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
         SetOutput(node, 0, k(inputs, equation), rt);
       }},
      {"ai.onnx:Elu", MakeUnaryAlphaTrampoline<kernel::Elu>("alpha", 1.0f)},
      {"ai.onnx:Equal", MakeBinaryTrampoline<kernel::Equal>()},
      {"ai.onnx:Erf", MakeUnaryTrampoline<kernel::Erf>()},
      {"ai.onnx:Exp", MakeUnaryTrampoline<kernel::Exp>()},
      {"ai.onnx:Expand",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &input = GetInput(node, 0, rt.tensors());
         const Tensor &shape = GetInput(node, 1, rt.tensors());
         kernel::Expand k(rt.kernel_ctx());
         SetOutput(node, 0, k(input, shape), rt);
       }},
      {"ai.onnx:EyeLike",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const int64_t k = GetAttributeIntOrDefault(node, "k", 0);
         const int64_t dtype = GetAttributeIntOrDefault(node, "dtype", 0);
         kernel::EyeLike kernel(rt.kernel_ctx());
         SetOutput(node, 0, kernel(x, k, static_cast<int32_t>(dtype)), rt);
       }},
      {"ai.onnx:Flatten", MakeAxisTrampoline<kernel::Flatten>(1)},
      {"ai.onnx:Floor", MakeUnaryTrampoline<kernel::Floor>()},
      {"ai.onnx:Gather",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &data = GetInput(node, 0, rt.tensors());
         const Tensor &indices = GetInput(node, 1, rt.tensors());
         const int64_t axis = GetAttributeIntOrDefault(node, "axis", 0);
         kernel::Gather k(rt.kernel_ctx());
         SetOutput(node, 0, k(data, indices, axis), rt);
       }},
      {"ai.onnx:GatherElements",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &data = GetInput(node, 0, rt.tensors());
         const Tensor &indices = GetInput(node, 1, rt.tensors());
         const int64_t axis = GetAttributeIntOrDefault(node, "axis", 0);
         kernel::GatherElements k(rt.kernel_ctx());
         SetOutput(node, 0, k(data, indices, axis), rt);
       }},
      {"ai.onnx:GatherND",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &data = GetInput(node, 0, rt.tensors());
         const Tensor &indices = GetInput(node, 1, rt.tensors());
         const int64_t batch_dims = GetAttributeIntOrDefault(node, "batch_dims", 0);
         kernel::GatherND k(rt.kernel_ctx());
         SetOutput(node, 0, k(data, indices, batch_dims), rt);
       }},
      {"ai.onnx:Gelu",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const std::string approximate = GetAttributeStringOrDefault(node, "approximate", "none");
         kernel::Gelu k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, approximate), rt);
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
         SetOutput(node, 0, k(a, b, c, alpha, beta, transA, transB), rt);
       }},
      {"ai.onnx:GlobalAveragePool",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         kernel::GlobalAveragePool k(rt.kernel_ctx());
         SetOutput(node, 0, k(x), rt.tensors());
       }},
      {"ai.onnx:GlobalLpPool",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const int64_t p = GetAttributeIntOrDefault(node, "p", 2);
         kernel::GlobalLpPool k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, p), rt.tensors());
       }},
      {"ai.onnx:GlobalMaxPool",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         kernel::GlobalMaxPool k(rt.kernel_ctx());
         SetOutput(node, 0, k(x), rt.tensors());
       }},
      {"ai.onnx:Greater", MakeBinaryTrampoline<kernel::Greater>()},
      {"ai.onnx:GreaterOrEqual", MakeBinaryTrampoline<kernel::GreaterOrEqual>()},
      {"ai.onnx:GridSample",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &grid = GetInput(node, 1, rt.tensors());
         kernel::GridSample::Attributes attrs;
         attrs.mode = GetAttributeStringOrDefault(node, "mode", attrs.mode);
         attrs.padding_mode =
             GetAttributeStringOrDefault(node, "padding_mode", attrs.padding_mode);
         attrs.align_corners =
             GetAttributeIntOrDefault(node, "align_corners", attrs.align_corners);
         kernel::GridSample k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, grid, attrs), rt);
       }},
      {"ai.onnx:GroupNormalization", RunGroupNormalization},
      {"ai.onnx:GRU",
       [](const NodeProto &node, RuntimeContext &rt) {
         if (node.input_size() < 3 || node.input_size() > 6) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects between 3 and 6 input(s), got " +
                                       std::to_string(node.input_size()) + ".");
         }
         if (node.output_size() < 1 || node.output_size() > 2) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects 1 or 2 output(s), got " +
                                       std::to_string(node.output_size()) + ".");
         }

         // Unsupported attributes: only the default ``forward`` direction
         // with the default ``Sigmoid``/``Tanh`` activations and no
         // ``clip`` are implemented; ``layout=0`` and ``layout=1`` are
         // both supported.
         const std::string direction =
             GetAttributeStringOrDefault(node, "direction", "forward");
         if (direction != "forward") {
           throw std::invalid_argument(
               "RunNode: op 'GRU' only supports direction='forward', got '" + direction + "'.");
         }
         if (FindAttribute(node, "activations") != nullptr) {
           throw std::invalid_argument(
               "RunNode: op 'GRU' does not support the 'activations' attribute.");
         }
         if (FindAttribute(node, "activation_alpha") != nullptr ||
             FindAttribute(node, "activation_beta") != nullptr) {
           throw std::invalid_argument(
               "RunNode: op 'GRU' does not support 'activation_alpha'/'activation_beta'.");
         }
         if (FindAttribute(node, "clip") != nullptr) {
           throw std::invalid_argument("RunNode: op 'GRU' does not support the 'clip' attribute.");
         }
         const int64_t layout = GetAttributeIntOrDefault(node, "layout", 0);

         // ``sequence_lens`` (input #4) is not supported: it requires
         // per-batch sequence handling that the FLOAT kernel does not
         // implement.
         const Tensor *sequence_lens = GetOptionalInput(node, 4, rt.tensors());
         if (sequence_lens != nullptr) {
           throw std::invalid_argument(
               "RunNode: op 'GRU' does not support the optional 'sequence_lens' input.");
         }

         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &w = GetInput(node, 1, rt.tensors());
         const Tensor &r = GetInput(node, 2, rt.tensors());
         const Tensor *b = GetOptionalInput(node, 3, rt.tensors());
         const Tensor *initial_h = GetOptionalInput(node, 5, rt.tensors());

         const int64_t linear_before_reset =
             GetAttributeIntOrDefault(node, "linear_before_reset", 0);

         kernel::GRU kernel(rt.kernel_ctx());
         auto [y, y_h] = kernel(x, w, r, b != nullptr ? *b : Tensor{},
                                initial_h != nullptr ? *initial_h : Tensor{},
                                linear_before_reset, layout);

         auto set_optional_output = [&node, &rt](int index, Tensor output) {
           if (index >= node.output_size()) {
             return;
           }
           const std::string name = node.output(index).as_string();
           if (name.empty()) {
             return;
           }
           output.name = name;
           rt.Put(name, std::move(output), TensorEventKind::kIntermediate);
         };
         set_optional_output(0, std::move(y));
         set_optional_output(1, std::move(y_h));
       }},
      {"ai.onnx:HardSigmoid",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const float alpha = GetAttributeFloatOrDefault(node, "alpha", 0.2f);
         const float beta = GetAttributeFloatOrDefault(node, "beta", 0.5f);
         kernel::HardSigmoid k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, alpha, beta), rt);
       }},
      {"ai.onnx:HardSwish", MakeUnaryTrampoline<kernel::HardSwish>()},
      {"ai.onnx:Hardmax", MakeAxisTrampoline<kernel::Hardmax>()},
      {"ai.onnx:HammingWindow", MakeWindowTrampoline<kernel::HammingWindow>("HammingWindow")},
      {"ai.onnx:HannWindow", MakeWindowTrampoline<kernel::HannWindow>("HannWindow")},
      {"ai.onnx:Identity", MakeUnaryTrampoline<kernel::Identity>()},
      {"ai.onnx:ImageDecoder",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &encoded_stream = GetInput(node, 0, rt.tensors());
         const std::string pixel_format =
             GetAttributeStringOrDefault(node, "pixel_format", "RGB");
         kernel::ImageDecoder image_decoder_kernel(rt.kernel_ctx());
         SetOutput(node, 0, image_decoder_kernel(encoded_stream, pixel_format), rt);
       }},
      {"ai.onnx:InstanceNormalization", RunInstanceNormalization},
      {"ai.onnx:IsInf",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const int64_t detect_positive = GetAttributeIntOrDefault(node, "detect_positive", 1);
         const int64_t detect_negative = GetAttributeIntOrDefault(node, "detect_negative", 1);
         kernel::IsInf k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, detect_positive, detect_negative), rt);
       }},
      {"ai.onnx:IsNaN", MakeUnaryTrampoline<kernel::IsNaN>()},
      {"ai.onnx:LayerNormalization", RunLayerNormalization},
      {"ai.onnx:LeakyRelu", MakeUnaryAlphaTrampoline<kernel::LeakyRelu>("alpha", 0.01f)},
      {"ai.onnx:Less", MakeBinaryTrampoline<kernel::Less>()},
      {"ai.onnx:LessOrEqual", MakeBinaryTrampoline<kernel::LessOrEqual>()},
      {"ai.onnx:LinearAttention",
       [](const NodeProto &node, RuntimeContext &rt) {
         if (node.input_size() < 3 || node.input_size() > 6) {
           throw std::invalid_argument("RunNode: op 'LinearAttention' expects between 3 and 6 "
                                       "input(s), got " +
                                       std::to_string(node.input_size()) + ".");
         }
         if (node.output_size() < 1 || node.output_size() > 2) {
           throw std::invalid_argument("RunNode: op 'LinearAttention' expects 1 or 2 output(s), "
                                       "got " +
                                       std::to_string(node.output_size()) + ".");
         }
         const Tensor &query = GetInput(node, 0, rt.tensors());
         const Tensor &key = GetInput(node, 1, rt.tensors());
         const Tensor &value = GetInput(node, 2, rt.tensors());
         const Tensor *past_state = GetOptionalInput(node, 3, rt.tensors());
         const Tensor *decay = GetOptionalInput(node, 4, rt.tensors());
         const Tensor *beta = GetOptionalInput(node, 5, rt.tensors());

         kernel::LinearAttention::Attributes attrs;
         attrs.update_rule = GetAttributeStringOrDefault(node, "update_rule", "gated_delta");
         if (FindAttribute(node, "scale") != nullptr) {
           attrs.has_scale = true;
           attrs.scale = GetAttributeFloatOrDefault(node, "scale", 0.0f);
         }
         attrs.q_num_heads = GetAttributeIntOrDefault(node, "q_num_heads", 0);
         attrs.kv_num_heads = GetAttributeIntOrDefault(node, "kv_num_heads", 0);
         attrs.chunk_size = GetAttributeIntOrDefault(node, "chunk_size", 64);

         kernel::LinearAttention k(rt.kernel_ctx());
         kernel::LinearAttention::Result result =
             k(query, key, value, attrs, past_state, decay, beta);
         SetOutput(node, 0, std::move(result.output), rt.tensors());

         if (node.output_size() >= 2) {
           const std::string present_name = node.output(1).as_string();
           if (!present_name.empty()) {
             result.present_state.name = present_name;
             rt.tensors()[present_name] = std::move(result.present_state);
           }
         }
       }},
      {"ai.onnx:Log", MakeUnaryTrampoline<kernel::Log>()},
      {"ai.onnx:LogSoftmax", MakeAxisTrampoline<kernel::LogSoftmax>()},
      {"ai.onnx:LSTM",
       [](const NodeProto &node, RuntimeContext &rt) {
         if (node.input_size() < 3 || node.input_size() > 8) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects between 3 and 8 input(s), got " +
                                       std::to_string(node.input_size()) + ".");
         }
         if (node.output_size() < 1 || node.output_size() > 3) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects between 1 and 3 output(s), got " +
                                       std::to_string(node.output_size()) + ".");
         }

         // Unsupported attributes: only the default ``forward`` direction
         // with the default ``Sigmoid``/``Tanh``/``Tanh`` activations, no
         // ``clip``, ``input_forget == 0``, and ``layout == 0`` are
         // implemented.
         const std::string direction =
             GetAttributeStringOrDefault(node, "direction", "forward");
         if (direction != "forward") {
           throw std::invalid_argument(
               "RunNode: op 'LSTM' only supports direction='forward', got '" + direction + "'.");
         }
         if (FindAttribute(node, "activations") != nullptr) {
           throw std::invalid_argument(
               "RunNode: op 'LSTM' does not support the 'activations' attribute.");
         }
         if (FindAttribute(node, "activation_alpha") != nullptr ||
             FindAttribute(node, "activation_beta") != nullptr) {
           throw std::invalid_argument(
               "RunNode: op 'LSTM' does not support 'activation_alpha'/'activation_beta'.");
         }
         if (FindAttribute(node, "clip") != nullptr) {
           throw std::invalid_argument("RunNode: op 'LSTM' does not support the 'clip' attribute.");
         }
         if (GetAttributeIntOrDefault(node, "input_forget", 0) != 0) {
           throw std::invalid_argument("RunNode: op 'LSTM' only supports input_forget=0.");
         }
         const int64_t layout = GetAttributeIntOrDefault(node, "layout", 0);

         // ``sequence_lens`` (input #4) is not supported: it requires
         // per-batch sequence handling that the FLOAT kernel does not
         // implement.
         const Tensor *sequence_lens = GetOptionalInput(node, 4, rt.tensors());
         if (sequence_lens != nullptr) {
           throw std::invalid_argument(
               "RunNode: op 'LSTM' does not support the optional 'sequence_lens' input.");
         }

         // The current kernel only produces (Y, Y_h); the optional third
         // output ``Y_c`` (final cell state) is not implemented.
         if (node.output_size() >= 3 && !node.output(2).as_string().empty()) {
           throw std::invalid_argument(
               "RunNode: op 'LSTM' does not support the optional third output 'Y_c'.");
         }

         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &w = GetInput(node, 1, rt.tensors());
         const Tensor &r = GetInput(node, 2, rt.tensors());
         const Tensor *b = GetOptionalInput(node, 3, rt.tensors());
         const Tensor *initial_h = GetOptionalInput(node, 5, rt.tensors());
         const Tensor *initial_c = GetOptionalInput(node, 6, rt.tensors());
         const Tensor *p = GetOptionalInput(node, 7, rt.tensors());

         kernel::LSTM kernel(rt.kernel_ctx());
         auto [y, y_h] = kernel(x, w, r, b != nullptr ? *b : Tensor{},
                                initial_h != nullptr ? *initial_h : Tensor{},
                                initial_c != nullptr ? *initial_c : Tensor{},
                                p != nullptr ? *p : Tensor{}, layout);

         auto set_optional_output = [&node, &rt](int index, Tensor output) {
           if (index >= node.output_size()) {
             return;
           }
           const std::string name = node.output(index).as_string();
           if (name.empty()) {
             return;
           }
           output.name = name;
           rt.Put(name, std::move(output), TensorEventKind::kIntermediate);
         };
         set_optional_output(0, std::move(y));
         set_optional_output(1, std::move(y_h));
       }},
      {"ai.onnx:LRN",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const int64_t size = GetRequiredAttributeInt(node, "size");
         const float alpha = GetAttributeFloatOrDefault(node, "alpha", 0.0001f);
         const float beta = GetAttributeFloatOrDefault(node, "beta", 0.75f);
         const float bias = GetAttributeFloatOrDefault(node, "bias", 1.0f);
         kernel::LRN k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, size, alpha, beta, bias), rt.tensors());
       }},
      {"ai.onnx:LpNormalization",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const int64_t axis = GetAttributeIntOrDefault(node, "axis", -1);
         const int64_t p = GetAttributeIntOrDefault(node, "p", 2);
         kernel::LpNormalization k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, axis, p), rt.tensors());
       }},
      {"ai.onnx:LpPool",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const PoolCommonAttrs a = ParsePoolCommonAttrs(node);
         const int64_t p = GetAttributeIntOrDefault(node, "p", 2);
         kernel::LpPool k(rt.kernel_ctx());
         SetOutput(node, 0,
                   k(x, a.kernel_shape, a.strides, a.pads, p, a.ceil_mode, a.dilations, a.auto_pad),
                   rt.tensors());
       }},
      {"ai.onnx:MatMul", MakeBinaryTrampoline<kernel::MatMul>()},
      {"ai.onnx:MatMulInteger",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 2);
         if (node.input_size() > 4) {
           throw std::invalid_argument("RunNode: op 'MatMulInteger' expects at most 4 inputs, got " +
                                       std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 1);
         const Tensor &a = GetInput(node, 0, rt.tensors());
         const Tensor &b = GetInput(node, 1, rt.tensors());
         const Tensor *a_zp = GetOptionalInput(node, 2, rt.tensors());
         const Tensor *b_zp = GetOptionalInput(node, 3, rt.tensors());
         kernel::MatMulInteger k(rt.kernel_ctx());
         SetOutput(node, 0,
                   k(a, b, a_zp != nullptr ? *a_zp : Tensor{},
                     b_zp != nullptr ? *b_zp : Tensor{}),
                   rt.tensors());
       }},
      {"ai.onnx:Max", MakeVariadicTrampoline<kernel::Max>()},
      {"ai.onnx:MaxPool",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         if (node.output_size() < 1 || node.output_size() > 2) {
           throw std::invalid_argument("RunNode: op 'MaxPool' expects 1 or 2 output(s), got " +
                                       std::to_string(node.output_size()) + ".");
         }
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const PoolCommonAttrs a = ParsePoolCommonAttrs(node);
         const int64_t storage_order = GetAttributeIntOrDefault(node, "storage_order", 0);
         kernel::MaxPool k(rt.kernel_ctx());
         const bool need_indices =
             node.output_size() == 2 && !node.output(1).as_string().empty();
         if (need_indices) {
           auto result = k.WithIndices(x, a.kernel_shape, a.strides, a.pads, a.ceil_mode,
                                       a.dilations, storage_order, a.auto_pad);
           SetOutput(node, 0, std::move(result.first), rt.tensors());
           const std::string indices_name = node.output(1).as_string();
           result.second.name = indices_name;
           rt.tensors()[indices_name] = std::move(result.second);
         } else {
           SetOutput(node, 0,
                     k(x, a.kernel_shape, a.strides, a.pads, a.ceil_mode, a.dilations,
                       storage_order, a.auto_pad),
                     rt.tensors());
         }
       }},
      {"ai.onnx:MaxRoiPool",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &rois = GetInput(node, 1, rt.tensors());
         kernel::MaxRoiPool::Attributes attrs;
         attrs.pooled_shape = GetAttributeIntsOrDefault(node, "pooled_shape", {});
         attrs.spatial_scale = GetAttributeFloatOrDefault(node, "spatial_scale", 1.0f);
         kernel::MaxRoiPool k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, rois, attrs), rt.tensors());
       }},
      {"ai.onnx:MaxUnpool",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputRange(node, 2, 3);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &indices = GetInput(node, 1, rt.tensors());
         const std::vector<int64_t> kernel_shape =
             GetAttributeIntsOrDefault(node, "kernel_shape", {});
         const std::vector<int64_t> strides = GetAttributeIntsOrDefault(node, "strides", {});
         const std::vector<int64_t> pads = GetAttributeIntsOrDefault(node, "pads", {});
         kernel::MaxUnpool k(rt.kernel_ctx());
         const Tensor *output_shape = GetOptionalInput(node, 2, rt.tensors());
         if (output_shape != nullptr) {
           SetOutput(node, 0, k(x, indices, *output_shape, kernel_shape, strides, pads),
                     rt.tensors());
         } else {
           SetOutput(node, 0, k(x, indices, kernel_shape, strides, pads), rt.tensors());
         }
       }},
      {"ai.onnx:Mean", MakeVariadicTrampoline<kernel::Mean>()},
      {"ai.onnx:MeanVarianceNormalization",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const std::vector<int64_t> axes = GetAttributeIntsOrDefault(node, "axes", {0, 2, 3});
         kernel::MeanVarianceNormalization k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, axes), rt.tensors());
       }},
      {"ai.onnx:MelWeightMatrix",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 5);
         RequireOutputCount(node, 1);
         const Tensor &num_mel_bins = GetInput(node, 0, rt.tensors());
         const Tensor &dft_length = GetInput(node, 1, rt.tensors());
         const Tensor &sample_rate = GetInput(node, 2, rt.tensors());
         const Tensor &lower_edge_hertz = GetInput(node, 3, rt.tensors());
         const Tensor &upper_edge_hertz = GetInput(node, 4, rt.tensors());
         const DataType output_dtype = static_cast<DataType>(GetAttributeIntOrDefault(
             node, "output_datatype", static_cast<int64_t>(DataType::FLOAT)));
         kernel::MelWeightMatrix k(rt.kernel_ctx());
         SetOutput(node, 0,
                   k(num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz,
                     output_dtype),
                   rt.tensors());
       }},
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
         SetOutput(node, 0, k(x, y, fmod), rt);
       }},
      {"ai.onnx:Mul", MakeBinaryTrampoline<kernel::Mul>()},
      {"ai.onnx:Multinomial",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &input = GetInput(node, 0, rt.tensors());
         const int64_t sample_size = GetAttributeIntOrDefault(node, "sample_size", 1);
         kernel::Multinomial kernel(rt.kernel_ctx());
         SetOutput(node, 0, kernel(input, sample_size, GetSeedAttr(node), GetDtypeAttr(node)), rt);
       }},
      {"ai.onnx:Neg", MakeUnaryTrampoline<kernel::Neg>()},
      {"ai.onnx:NegativeLogLikelihoodLoss",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputRange(node, 2, 3);
         RequireOutputCount(node, 1);
         const Tensor &input = GetInput(node, 0, rt.tensors());
         const Tensor &target = GetInput(node, 1, rt.tensors());
         const Tensor *weight = GetOptionalInput(node, 2, rt.tensors());
         const std::string reduction = GetAttributeStringOrDefault(node, "reduction", "mean");
         const bool has_ignore_index = FindAttribute(node, "ignore_index") != nullptr;
         const int64_t ignore_index = GetAttributeIntOrDefault(node, "ignore_index", 0);
         kernel::NegativeLogLikelihoodLoss k(rt.kernel_ctx());
         SetOutput(node, 0, k(input, target, weight, reduction, has_ignore_index, ignore_index),
                   rt);
       }},
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
      {"ai.onnx:NonZero", MakeUnaryTrampoline<kernel::NonZero>()},
      {"ai.onnx:Not", MakeUnaryTrampoline<kernel::Not>()},
      {"ai.onnx:OneHot",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 3);
         RequireOutputCount(node, 1);
         const Tensor &indices = GetInput(node, 0, rt.tensors());
         const Tensor &depth = GetInput(node, 1, rt.tensors());
         const Tensor &values = GetInput(node, 2, rt.tensors());
         kernel::OneHot::Attributes attrs;
         attrs.axis = GetAttributeIntOrDefault(node, "axis", -1);
         kernel::OneHot k(rt.kernel_ctx());
         SetOutput(node, 0, k(indices, depth, values, attrs), rt.tensors());
       }},
      {"ai.onnx:Or", MakeBinaryTrampoline<kernel::Or>()},
      // ai.onnx Optional / OptionalGetElement / OptionalHasElement
      // (since opset 15; opset 18 widens the supported input types).
      // The runtime models Optional<Tensor> / Optional<Sequence> as a
      // simple passthrough — see ``kernels/optional/`` for details.
      {"ai.onnx:Optional",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const std::string input_name = node.input(0).as_string();
         if (rt.HasSequence(input_name)) {
           // Sequence-typed input: passthrough into the optional-of-sequence
           // output. The Optional kernel itself has no sequence overload
           // because the runtime ``Sequence`` already models the value, so
           // we copy the input sequence into the output slot directly.
           SetOutputSequence(node, 0, rt.GetSequence(input_name), rt);
         } else {
           const Tensor &input = GetInput(node, 0, rt.tensors());
           kernel::Optional k(rt.kernel_ctx());
           SetOutput(node, 0, k(input), rt);
         }
       }},
      {"ai.onnx:OptionalGetElement",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const std::string input_name = node.input(0).as_string();
         kernel::OptionalGetElement k(rt.kernel_ctx());
         if (rt.HasSequence(input_name)) {
           const Sequence &input_seq = GetInputSequence(node, 0, rt);
           SetOutputSequence(node, 0, k(input_seq), rt);
         } else {
           const Tensor &input = GetInput(node, 0, rt.tensors());
           SetOutput(node, 0, k(input), rt);
         }
       }},
      {"ai.onnx:OptionalHasElement",
       [](const NodeProto &node, RuntimeContext &rt) {
         if (node.input_size() > 1) {
           throw std::invalid_argument(
               "RunNode: op 'OptionalHasElement' expects 0 or 1 inputs, got " +
               std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 1);
         kernel::OptionalHasElement k(rt.kernel_ctx());
         if (node.input_size() == 0 || node.input(0).as_string().empty()) {
           // Opset 18 omitted-input flavour: scalar ``false``.
           SetOutput(node, 0, k(), rt);
           return;
         }
         const std::string input_name = node.input(0).as_string();
         if (rt.HasSequence(input_name)) {
           const Sequence &input_seq = GetInputSequence(node, 0, rt);
           SetOutput(node, 0, k(input_seq), rt);
         } else {
           const Tensor &input = GetInput(node, 0, rt.tensors());
           SetOutput(node, 0, k(input), rt);
         }
       }},
      {"ai.onnx:Pad",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 1);
         if (node.input_size() > 4) {
           throw std::invalid_argument("RunNode: op 'Pad' expects at most 4 inputs.");
         }
         RequireOutputCount(node, 1);
         const Tensor &data = GetInput(node, 0, rt.tensors());
         const std::string mode = GetAttributeStringOrDefault(node, "mode", "constant");
         kernel::Pad k(rt.kernel_ctx());

         // Opset 11+: ``pads`` is the second input.
         if (node.input_size() >= 2) {
           const Tensor &pads = GetInput(node, 1, rt.tensors());
           const Tensor *constant_value = GetOptionalInput(node, 2, rt.tensors());
           const Tensor *axes = GetOptionalInput(node, 3, rt.tensors());
           SetOutput(node, 0, k(data, pads, constant_value, axes, mode), rt);
           return;
         }

         // Legacy opset (<11): ``pads`` is an INTS attribute and ``value`` is a
         // FLOAT attribute (default 0).
         const std::vector<int64_t> pads_attr = GetAttributeIntsOrDefault(node, "pads", {});
         const Tensor pads =
             Tensor::FromInt64("", {static_cast<int64_t>(pads_attr.size())}, pads_attr);
         const float value = GetAttributeFloatOrDefault(node, "value", 0.0f);
         if (data.data_type == static_cast<int32_t>(DataType::FLOAT)) {
           const Tensor cv = Tensor::FromFloat("", /*shape=*/{}, {value});
           SetOutput(node, 0, k(data, pads, &cv, /*axes=*/nullptr, mode), rt);
         } else {
           // For non-float dtypes the legacy form's float ``value`` attribute is
           // ill-defined; fall back to a zero-initialized constant.
           SetOutput(node, 0, k(data, pads, /*constant_value=*/nullptr, /*axes=*/nullptr, mode),
                     rt);
         }
       }},
      {"ai.onnx:Pow", MakeBinaryTrampoline<kernel::Pow>()},
      {"ai.onnx:PRelu", MakeBinaryTrampoline<kernel::PRelu>()},
      {"ai.onnx:QLinearConv",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 8);
         if (node.input_size() > 9) {
           throw std::invalid_argument("RunNode: op 'QLinearConv' expects at most 9 inputs, got " +
                                       std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &x_scale = GetInput(node, 1, rt.tensors());
         const Tensor &x_zero_point = GetInput(node, 2, rt.tensors());
         const Tensor &w = GetInput(node, 3, rt.tensors());
         const Tensor &w_scale = GetInput(node, 4, rt.tensors());
         const Tensor &w_zero_point = GetInput(node, 5, rt.tensors());
         const Tensor &y_scale = GetInput(node, 6, rt.tensors());
         const Tensor &y_zero_point = GetInput(node, 7, rt.tensors());
         const Tensor *b = GetOptionalInput(node, 8, rt.tensors());
         kernel::QLinearConv::Attributes attrs;
         attrs.kernel_shape = GetAttributeIntsOrDefault(node, "kernel_shape", {});
         attrs.strides = GetAttributeIntsOrDefault(node, "strides", {});
         attrs.pads = GetAttributeIntsOrDefault(node, "pads", {});
         attrs.dilations = GetAttributeIntsOrDefault(node, "dilations", {});
         attrs.group = GetAttributeIntOrDefault(node, "group", 1);
         attrs.auto_pad = GetAttributeStringOrDefault(node, "auto_pad", "NOTSET");
         kernel::QLinearConv k(rt.kernel_ctx());
         SetOutput(node, 0,
                   k(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point,
                     b != nullptr ? *b : Tensor{}, attrs),
                   rt);
       }},
      {"ai.onnx:QLinearMatMul",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 8);
         RequireOutputCount(node, 1);
         const Tensor &a = GetInput(node, 0, rt.tensors());
         const Tensor &a_scale = GetInput(node, 1, rt.tensors());
         const Tensor &a_zero_point = GetInput(node, 2, rt.tensors());
         const Tensor &b = GetInput(node, 3, rt.tensors());
         const Tensor &b_scale = GetInput(node, 4, rt.tensors());
         const Tensor &b_zero_point = GetInput(node, 5, rt.tensors());
         const Tensor &y_scale = GetInput(node, 6, rt.tensors());
         const Tensor &y_zero_point = GetInput(node, 7, rt.tensors());
         kernel::QLinearMatMul k(rt.kernel_ctx());
         SetOutput(node, 0,
                   k(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point),
                   rt);
       }},
      {"ai.onnx:QuantizeLinear", MakeBinaryWithOptionalThirdTrampoline<kernel::QuantizeLinear>()},
      {"ai.onnx:RandomNormal",
       MakeRandomGenTrampoline<kernel::RandomNormal>("mean", 0.0f, "scale", 1.0f)},
      {"ai.onnx:RandomNormalLike",
       MakeRandomLikeTrampoline<kernel::RandomNormalLike>("mean", 0.0f, "scale", 1.0f)},
      {"ai.onnx:RandomUniform",
       MakeRandomGenTrampoline<kernel::RandomUniform>("low", 0.0f, "high", 1.0f)},
      {"ai.onnx:RandomUniformLike",
       MakeRandomLikeTrampoline<kernel::RandomUniformLike>("low", 0.0f, "high", 1.0f)},
      {"ai.onnx:Range",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 3);
         RequireOutputCount(node, 1);
         const Tensor &start = GetInput(node, 0, rt.tensors());
         const Tensor &limit = GetInput(node, 1, rt.tensors());
         const Tensor &delta = GetInput(node, 2, rt.tensors());
         kernel::Range k(rt.kernel_ctx());
         SetOutput(node, 0, k(start, limit, delta), rt);
       }},
      {"ai.onnx:RMSNormalization", RunRMSNormalization},
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
      {"ai.onnx:RegexFullMatch",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const std::string pattern = GetAttributeStringOrDefault(node, "pattern", "");
         kernel::RegexFullMatch k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, pattern), rt);
       }},
      {"ai.onnx:Relu", MakeUnaryTrampoline<kernel::Relu>()},
      {"ai.onnx:Reshape",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &data = GetInput(node, 0, rt.tensors());
         const Tensor &shape = GetInput(node, 1, rt.tensors());
         const int64_t allowzero = GetAttributeIntOrDefault(node, "allowzero", 0);
         kernel::Reshape k(rt.kernel_ctx());
         SetOutput(node, 0, k(data, shape, allowzero), rt);
       }},
      {"ai.onnx:Resize",
       [](const NodeProto &node, RuntimeContext &rt) {
         if (node.input_size() < 1 || node.input_size() > 4) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects between 1 and 4 input(s), got " +
                                       std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor *roi = GetOptionalInput(node, 1, rt.tensors());
         const Tensor *scales = GetOptionalInput(node, 2, rt.tensors());
         const Tensor *sizes = GetOptionalInput(node, 3, rt.tensors());
         if ((scales == nullptr) == (sizes == nullptr)) {
           throw std::invalid_argument(
               "RunNode: op 'Resize' requires exactly one of 'scales' or 'sizes' to be "
               "provided.");
         }

         kernel::Resize::Attributes attrs;
         attrs.mode = GetAttributeStringOrDefault(node, "mode", attrs.mode);
         attrs.coordinate_transformation_mode = GetAttributeStringOrDefault(
             node, "coordinate_transformation_mode", attrs.coordinate_transformation_mode);
         attrs.nearest_mode =
             GetAttributeStringOrDefault(node, "nearest_mode", attrs.nearest_mode);
         attrs.axes = GetAttributeIntsOrDefault(node, "axes", attrs.axes);
         attrs.keep_aspect_ratio_policy = GetAttributeStringOrDefault(
             node, "keep_aspect_ratio_policy", attrs.keep_aspect_ratio_policy);
         attrs.cubic_coeff_a =
             GetAttributeFloatOrDefault(node, "cubic_coeff_a", attrs.cubic_coeff_a);
         attrs.exclude_outside =
             GetAttributeIntOrDefault(node, "exclude_outside", attrs.exclude_outside);
         attrs.extrapolation_value = GetAttributeFloatOrDefault(
             node, "extrapolation_value", attrs.extrapolation_value);
         if (roi != nullptr) {
           if (roi->data_type != DataType::FLOAT) {
             throw std::invalid_argument(
                 "RunNode: op 'Resize' 'roi' input must be a FLOAT tensor.");
           }
           if (roi->shape.size() != 1) {
             throw std::invalid_argument("RunNode: op 'Resize' 'roi' input must be 1-D.");
           }
           const int64_t n = roi->shape[0];
           attrs.roi.assign(static_cast<std::size_t>(n), 0.0f);
           if (n > 0) {
             std::memcpy(attrs.roi.data(), roi->bytes(),
                         static_cast<std::size_t>(n) * sizeof(float));
           }
         }

         kernel::Resize k(rt.kernel_ctx());
         if (scales != nullptr) {
           SetOutput(node, 0, k(x, *scales, attrs), rt);
         } else {
           SetOutput(node, 0, k.ResizeSizes(x, *sizes, attrs), rt);
         }
       }},
      {"ai.onnx:ReverseSequence",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &input = GetInput(node, 0, rt.tensors());
         const Tensor &sequence_lens = GetInput(node, 1, rt.tensors());
         kernel::ReverseSequence::Attributes attrs;
         attrs.time_axis = GetAttributeIntOrDefault(node, "time_axis", attrs.time_axis);
         attrs.batch_axis = GetAttributeIntOrDefault(node, "batch_axis", attrs.batch_axis);
         kernel::ReverseSequence k(rt.kernel_ctx());
         SetOutput(node, 0, k(input, sequence_lens, attrs), rt);
       }},
      {"ai.onnx:RoiAlign",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 3);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &rois = GetInput(node, 1, rt.tensors());
         const Tensor &batch_indices = GetInput(node, 2, rt.tensors());
         kernel::RoiAlign::Attributes attrs;
         attrs.mode = GetAttributeStringOrDefault(node, "mode", attrs.mode);
         attrs.output_height = GetAttributeIntOrDefault(node, "output_height", attrs.output_height);
         attrs.output_width = GetAttributeIntOrDefault(node, "output_width", attrs.output_width);
         attrs.sampling_ratio =
             GetAttributeIntOrDefault(node, "sampling_ratio", attrs.sampling_ratio);
         attrs.spatial_scale =
             GetAttributeFloatOrDefault(node, "spatial_scale", attrs.spatial_scale);
         // Opset 10 has no ``coordinate_transformation_mode`` attribute and
         // behaves like ``output_half_pixel``; opset 16+ defaults to
         // ``half_pixel``.
         const std::string default_ctm =
             rt.kernel_ctx().opset.version < 16 ? "output_half_pixel" : "half_pixel";
         attrs.coordinate_transformation_mode =
             GetAttributeStringOrDefault(node, "coordinate_transformation_mode", default_ctm);
         kernel::RoiAlign k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, rois, batch_indices, attrs), rt);
       }},
      {"ai.onnx:Round", MakeUnaryTrampoline<kernel::Round>()},
      {"ai.onnx:RNN",
       [](const NodeProto &node, RuntimeContext &rt) {
         if (node.input_size() < 3 || node.input_size() > 6) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects between 3 and 6 input(s), got " +
                                       std::to_string(node.input_size()) + ".");
         }
         if (node.output_size() < 1 || node.output_size() > 2) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects 1 or 2 output(s), got " +
                                       std::to_string(node.output_size()) + ".");
         }

         // Unsupported attributes: only the default ``forward`` direction
         // with the default ``Tanh`` activation and no ``clip`` are
         // implemented; ``layout=0`` and ``layout=1`` are both supported.
         const std::string direction =
             GetAttributeStringOrDefault(node, "direction", "forward");
         if (direction != "forward") {
           throw std::invalid_argument(
               "RunNode: op 'RNN' only supports direction='forward', got '" + direction + "'.");
         }
         if (const AttributeProto *activations = FindAttribute(node, "activations");
             activations != nullptr) {
           const std::vector<std::string> values =
               GetAttributeStringsOrDefault(node, "activations", {});
           if (values.size() != 1 || values[0] != "Tanh") {
             throw std::invalid_argument(
                 "RunNode: op 'RNN' only supports the default activations=['Tanh'].");
           }
         }
         if (FindAttribute(node, "activation_alpha") != nullptr ||
             FindAttribute(node, "activation_beta") != nullptr) {
           throw std::invalid_argument(
               "RunNode: op 'RNN' does not support 'activation_alpha'/'activation_beta'.");
         }
         if (FindAttribute(node, "clip") != nullptr) {
           throw std::invalid_argument("RunNode: op 'RNN' does not support the 'clip' attribute.");
         }
         const int64_t layout = GetAttributeIntOrDefault(node, "layout", 0);

         // ``sequence_lens`` (input #4) is not supported: it requires
         // per-batch sequence handling that the FLOAT kernel does not
         // implement.
         const Tensor *sequence_lens = GetOptionalInput(node, 4, rt.tensors());
         if (sequence_lens != nullptr) {
           throw std::invalid_argument(
               "RunNode: op 'RNN' does not support the optional 'sequence_lens' input.");
         }

         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &w = GetInput(node, 1, rt.tensors());
         const Tensor &r = GetInput(node, 2, rt.tensors());
         const Tensor *b = GetOptionalInput(node, 3, rt.tensors());
         const Tensor *initial_h = GetOptionalInput(node, 5, rt.tensors());

         kernel::RNN kernel(rt.kernel_ctx());
         auto [y, y_h] = kernel(x, w, r, b != nullptr ? *b : Tensor{},
                                initial_h != nullptr ? *initial_h : Tensor{}, layout);

         auto set_optional_output = [&node, &rt](int index, Tensor output) {
           if (index >= node.output_size()) {
             return;
           }
           const std::string name = node.output(index).as_string();
           if (name.empty()) {
             return;
           }
           output.name = name;
           rt.Put(name, std::move(output), TensorEventKind::kIntermediate);
         };
         set_optional_output(0, std::move(y));
         set_optional_output(1, std::move(y_h));
       }},
      {"ai.onnx:RotaryEmbedding",
       [](const NodeProto &node, RuntimeContext &rt) {
         if (node.input_size() < 3 || node.input_size() > 4) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects between 3 and 4 input(s), got " +
                                       std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &cos_cache = GetInput(node, 1, rt.tensors());
         const Tensor &sin_cache = GetInput(node, 2, rt.tensors());
         const Tensor *position_ids = GetOptionalInput(node, 3, rt.tensors());

         kernel::RotaryEmbedding::Attributes attrs;
         attrs.interleaved = GetAttributeIntOrDefault(node, "interleaved", 0) != 0;
         attrs.rotary_embedding_dim =
             GetAttributeIntOrDefault(node, "rotary_embedding_dim", 0);
         attrs.num_heads = GetAttributeIntOrDefault(node, "num_heads", 0);

         kernel::RotaryEmbedding kernel(rt.kernel_ctx());
         const Tensor empty;
         const Tensor &pos = (position_ids != nullptr) ? *position_ids : empty;
         SetOutput(node, 0, kernel(x, cos_cache, sin_cache, pos, attrs), rt.tensors());
       }},
      {"ai.onnx:ScatterElements",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 3);
         RequireOutputCount(node, 1);
         const Tensor &data = GetInput(node, 0, rt.tensors());
         const Tensor &indices = GetInput(node, 1, rt.tensors());
         const Tensor &updates = GetInput(node, 2, rt.tensors());
         kernel::ScatterElements::Attributes attrs;
         attrs.axis = GetAttributeIntOrDefault(node, "axis", 0);
         attrs.reduction = GetAttributeStringOrDefault(node, "reduction", "none");
         kernel::ScatterElements k(rt.kernel_ctx());
         SetOutput(node, 0, k(data, indices, updates, attrs), rt);
       }},
      {"ai.onnx:ScatterND",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 3);
         RequireOutputCount(node, 1);
         const Tensor &data = GetInput(node, 0, rt.tensors());
         const Tensor &indices = GetInput(node, 1, rt.tensors());
         const Tensor &updates = GetInput(node, 2, rt.tensors());
         kernel::ScatterND::Attributes attrs;
         attrs.reduction = GetAttributeStringOrDefault(node, "reduction", "none");
         kernel::ScatterND k(rt.kernel_ctx());
         SetOutput(node, 0, k(data, indices, updates, attrs), rt);
       }},
      {"ai.onnx:Selu",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const float alpha = GetAttributeFloatOrDefault(node, "alpha", 1.67326319217681884765625f);
         const float gamma = GetAttributeFloatOrDefault(node, "gamma", 1.05070102214813232421875f);
         kernel::Selu k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, alpha, gamma), rt);
       }},

      // Sequence operators (ai.onnx). Sequence-typed graph edges are
      // carried in :cpp:func:`RuntimeContext::sequences`; tensor-typed
      // inputs/outputs continue to flow through ``rt.tensors()``.
      {"ai.onnx:ConcatFromSequence",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Sequence &input_sequence = GetInputSequence(node, 0, rt);
         const AttributeProto *axis_attr = FindAttribute(node, "axis");
         if (axis_attr == nullptr) {
           throw std::invalid_argument(
               "RunNode: op 'ConcatFromSequence' is missing required attribute 'axis'.");
         }
         const int64_t axis = axis_attr->i();
         const int64_t new_axis = GetAttributeIntOrDefault(node, "new_axis", 0);
         kernel::ConcatFromSequence k(rt.kernel_ctx());
         SetOutput(node, 0, k(input_sequence.values, axis, new_axis), rt);
       }},
      {"ai.onnx:SequenceAt",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Sequence &input_sequence = GetInputSequence(node, 0, rt);
         const Tensor &position = GetInput(node, 1, rt.tensors());
         kernel::SequenceAt k(rt.kernel_ctx());
         SetOutput(node, 0, k(input_sequence, position), rt);
       }},
      {"ai.onnx:SequenceConstruct",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 1);
         RequireOutputCount(node, 1);
         std::vector<Tensor> inputs;
         inputs.reserve(node.input_size());
         for (int i = 0; i < node.input_size(); ++i) {
           inputs.push_back(GetInput(node, i, rt.tensors()));
         }
         kernel::SequenceConstruct k(rt.kernel_ctx());
         SetOutputSequence(node, 0, k.AsSequence(inputs), rt);
       }},
      {"ai.onnx:SequenceEmpty",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 0);
         RequireOutputCount(node, 1);
         const int64_t dtype = GetAttributeIntOrDefault(node, "dtype", 0);
         kernel::SequenceEmpty k(rt.kernel_ctx());
         SetOutputSequence(node, 0, k(static_cast<int32_t>(dtype)), rt);
       }},
      {"ai.onnx:SequenceErase",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 1);
         if (node.input_size() > 2) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects 1 or 2 inputs, got " +
                                       std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 1);
         const Sequence &input_sequence = GetInputSequence(node, 0, rt);
         const Tensor *position = GetOptionalInput(node, 1, rt.tensors());
         kernel::SequenceErase k(rt.kernel_ctx());
         SetOutputSequence(node, 0, k(input_sequence, position), rt);
       }},
      {"ai.onnx:SequenceInsert",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 2);
         if (node.input_size() > 3) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects 2 or 3 inputs, got " +
                                       std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 1);
         const Sequence &input_sequence = GetInputSequence(node, 0, rt);
         const Tensor &tensor = GetInput(node, 1, rt.tensors());
         const Tensor *position = GetOptionalInput(node, 2, rt.tensors());
         kernel::SequenceInsert k(rt.kernel_ctx());
         SetOutputSequence(node, 0, k(input_sequence, tensor, position), rt);
       }},
      {"ai.onnx:SequenceLength",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Sequence &input_sequence = GetInputSequence(node, 0, rt);
         kernel::SequenceLength k(rt.kernel_ctx());
         SetOutput(node, 0, k(input_sequence), rt);
       }},
      {"ai.onnx:Split",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 1);
         if (node.input_size() > 2) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects 1 or 2 inputs, got " +
                                       std::to_string(node.input_size()) + ".");
         }
         if (node.output_size() < 1) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects at least 1 output, got 0.");
         }
         const Tensor &input = GetInput(node, 0, rt.tensors());
         const int64_t axis = GetAttributeIntOrDefault(node, "axis", 0);

         // Resolve ``split``: from the optional 2nd input (opset >= 13), from the
         // legacy ``split`` attribute (opset <= 12), or unspecified.
         std::vector<int64_t> split;
         const Tensor *split_input = GetOptionalInput(node, 1, rt.tensors());
         if (split_input != nullptr) {
           split = TensorToVector<int64_t>(*split_input);
         } else {
           split = GetAttributeIntsOrDefault(node, "split", {});
         }

         // ``num_outputs`` (opset >= 18) defaults to the number of outputs of the
         // node when neither ``split`` nor the attribute is provided.
         int64_t num_outputs = GetAttributeIntOrDefault(node, "num_outputs", 0);
         if (split.empty() && num_outputs <= 0) {
           num_outputs = static_cast<int64_t>(node.output_size());
         }

         kernel::Split k(rt.kernel_ctx());
         std::vector<Tensor> outputs = k(input, axis, split, num_outputs);
         if (static_cast<int>(outputs.size()) != node.output_size()) {
           throw std::invalid_argument(
               "RunNode: op 'Split' produced " + std::to_string(outputs.size()) +
               " outputs but node declares " + std::to_string(node.output_size()) + ".");
         }
         for (int i = 0; i < node.output_size(); ++i) {
           SetOutput(node, i, std::move(outputs[static_cast<size_t>(i)]), rt);
         }
       }},
      {"ai.onnx:SplitToSequence",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 1);
         if (node.input_size() > 2) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects 1 or 2 inputs, got " +
                                       std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 1);
         const Tensor &input = GetInput(node, 0, rt.tensors());
         const Tensor *split = GetOptionalInput(node, 1, rt.tensors());
         const int64_t axis = GetAttributeIntOrDefault(node, "axis", 0);
         const int64_t keepdims = GetAttributeIntOrDefault(node, "keepdims", 1);
         kernel::SplitToSequence k(rt.kernel_ctx());
         SetOutputSequence(node, 0, k(input, split, axis, keepdims), rt);
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
         SetOutput(node, 0, shape_kernel(data, shape_attrs), rt);
       }},
      {"ai.onnx:Shrink",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const float bias = GetAttributeFloatOrDefault(node, "bias", 0.0f);
         const float lambd = GetAttributeFloatOrDefault(node, "lambd", 0.5f);
         kernel::Shrink k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, bias, lambd), rt);
       }},
      {"ai.onnx:Sigmoid", MakeUnaryTrampoline<kernel::Sigmoid>()},
      {"ai.onnx:Sign", MakeUnaryTrampoline<kernel::Sign>()},
      {"ai.onnx:Sin", MakeUnaryTrampoline<kernel::Sin>()},
      {"ai.onnx:Sinh", MakeUnaryTrampoline<kernel::Sinh>()},
      {"ai.onnx:Size",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &data = GetInput(node, 0, rt.tensors());
         kernel::Size k(rt.kernel_ctx());
         SetOutput(node, 0, k(data), rt);
       }},
      {"ai.onnx:Slice",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 3);
         if (node.input_size() > 5) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects between 3 and 5 input(s), got " +
                                       std::to_string(node.input_size()) + ".");
         }
         RequireOutputCount(node, 1);
         const Tensor &data = GetInput(node, 0, rt.tensors());
         const Tensor &starts = GetInput(node, 1, rt.tensors());
         const Tensor &ends = GetInput(node, 2, rt.tensors());
         const Tensor *axes = GetOptionalInput(node, 3, rt.tensors());
         const Tensor *steps = GetOptionalInput(node, 4, rt.tensors());
         kernel::Slice k(rt.kernel_ctx());
         SetOutput(node, 0, k(data, starts, ends, axes, steps), rt);
       }},
      {"ai.onnx:Softmax", MakeAxisTrampoline<kernel::Softmax>()},
      {"ai.onnx:SoftmaxCrossEntropyLoss",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputRange(node, 2, 3);
         RequireOutputRange(node, 1, 2);
         const Tensor &scores = GetInput(node, 0, rt.tensors());
         const Tensor &labels = GetInput(node, 1, rt.tensors());
         const Tensor *weights = GetOptionalInput(node, 2, rt.tensors());
         const std::string reduction = GetAttributeStringOrDefault(node, "reduction", "mean");
         const bool has_ignore_index = FindAttribute(node, "ignore_index") != nullptr;
         const int64_t ignore_index = GetAttributeIntOrDefault(node, "ignore_index", 0);
         kernel::SoftmaxCrossEntropyLoss k(rt.kernel_ctx());
         auto [loss, log_prob] =
             k(scores, labels, weights, reduction, has_ignore_index, ignore_index);
         SetOutput(node, 0, std::move(loss), rt);
         if (node.output_size() >= 2) {
           SetOutput(node, 1, std::move(log_prob), rt);
         }
       }},
      {"ai.onnx:Softplus", MakeUnaryTrampoline<kernel::Softplus>()},
      {"ai.onnx:Softsign", MakeUnaryTrampoline<kernel::Softsign>()},
      {"ai.onnx:Sqrt", MakeUnaryTrampoline<kernel::Sqrt>()},
      {"ai.onnx:Squeeze", MakeSqueezeLikeTrampoline<kernel::Squeeze>("Squeeze")},
      {"ai.onnx:STFT",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 2);
         if (node.input_size() > 4) {
           throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                       "' expects at most 4 inputs.");
         }
         RequireOutputCount(node, 1);
         const Tensor &signal = GetInput(node, 0, rt.tensors());
         const Tensor &frame_step = GetInput(node, 1, rt.tensors());
         const Tensor *window = GetOptionalInput(node, 2, rt.tensors());
         const Tensor *frame_length = GetOptionalInput(node, 3, rt.tensors());
         const bool onesided = GetAttributeIntOrDefault(node, "onesided", 1) != 0;
         kernel::STFT k(rt.kernel_ctx());
         SetOutput(node, 0, k(signal, frame_step, window, frame_length, onesided), rt.tensors());
       }},
      {"ai.onnx:StringNormalizer",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const std::string case_change_action_attr =
             GetAttributeStringOrDefault(node, "case_change_action", "NONE");
         const bool is_case_sensitive =
             GetAttributeIntOrDefault(node, "is_case_sensitive", 0) != 0;
         const std::vector<std::string> stopwords =
             GetAttributeStringsOrDefault(node, "stopwords", {});
         kernel::StringNormalizer k(rt.kernel_ctx());
         SetOutput(node, 0,
                   k(x, kernel::StringNormalizer::ParseCaseChangeAction(case_change_action_attr),
                     is_case_sensitive, stopwords),
                   rt);
       }},
      {"ai.onnx:StringSplit",
       [](const NodeProto &node, RuntimeContext &rt) {
        RequireInputCount(node, 1);
        RequireOutputCount(node, 2);
        const Tensor &x = GetInput(node, 0, rt.tensors());
        const std::string delimiter = GetAttributeStringOrDefault(node, "delimiter", "");
        const int64_t maxsplit = GetAttributeIntOrDefault(node, "maxsplit", -1);
        kernel::StringSplit k(rt.kernel_ctx());
        auto out = k(x, delimiter, maxsplit);
        SetOutput(node, 0, std::move(out.first), rt);
        SetOutput(node, 1, std::move(out.second), rt);
       }},
      {"ai.onnx:Sub", MakeBinaryTrampoline<kernel::Sub>()},
      {"ai.onnx:Sum", MakeVariadicTrampoline<kernel::Sum>()},
      {"ai.onnx:Swish", MakeUnaryAlphaTrampoline<kernel::Swish>("alpha", 1.0f)},
      {"ai.onnx:Tan", MakeUnaryTrampoline<kernel::Tan>()},
      {"ai.onnx:Tanh", MakeUnaryTrampoline<kernel::Tanh>()},
      {"ai.onnx:TensorScatter",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputRange(node, 2, 3);
         RequireOutputCount(node, 1);
         const Tensor &past_cache = GetInput(node, 0, rt.tensors());
         const Tensor &update = GetInput(node, 1, rt.tensors());
         const Tensor *write_indices = GetOptionalInput(node, 2, rt.tensors());
         kernel::TensorScatter::Attributes attrs;
         attrs.axis = GetAttributeIntOrDefault(node, "axis", -2);
         attrs.mode = GetAttributeStringOrDefault(node, "mode", "linear");
         kernel::TensorScatter kernel(rt.kernel_ctx());
         SetOutput(node, 0, kernel(past_cache, update, write_indices, attrs), rt);
       }},
      {"ai.onnx:TfIdfVectorizer",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const std::string mode_attr = GetRequiredAttributeString(node, "mode");
         const int64_t min_gram_length = GetAttributeIntOrDefault(node, "min_gram_length", 1);
         const int64_t max_gram_length = GetAttributeIntOrDefault(node, "max_gram_length", 1);
         const int64_t max_skip_count = GetAttributeIntOrDefault(node, "max_skip_count", 0);
         const std::vector<int64_t> ngram_counts =
             GetAttributeIntsOrDefault(node, "ngram_counts", {});
         const std::vector<int64_t> ngram_indexes =
             GetAttributeIntsOrDefault(node, "ngram_indexes", {});
         const std::vector<int64_t> pool_int64s =
             GetAttributeIntsOrDefault(node, "pool_int64s", {});
         const std::vector<std::string> pool_strings =
             GetAttributeStringsOrDefault(node, "pool_strings", {});
         const std::vector<float> weights = GetAttributeFloatsOrDefault(node, "weights", {});
         kernel::TfIdfVectorizer k(rt.kernel_ctx());
         SetOutput(node, 0,
                   k(x, kernel::TfIdfVectorizer::ParseMode(mode_attr), min_gram_length,
                     max_gram_length, max_skip_count, ngram_counts, ngram_indexes, pool_int64s,
                     pool_strings, weights),
                   rt);
       }},
      {"ai.onnx:ThresholdedRelu", MakeUnaryAlphaTrampoline<kernel::ThresholdedRelu>("alpha", 1.0f)},
      {"ai.onnx:TopK",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireOutputCount(node, 2);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const int64_t axis = GetAttributeIntOrDefault(node, "axis", -1);
         const bool largest = GetAttributeIntOrDefault(node, "largest", 1) != 0;
         const bool sorted = GetAttributeIntOrDefault(node, "sorted", 1) != 0;

         // ``k``: from input[1] (1-D INT64 tensor of size 1) for opset >= 10,
         // otherwise from the pre-opset-10 ``k`` INT attribute (required).
         int64_t k = 0;
         const int64_t opset_version = rt.kernel_ctx().opset.version;
         if (opset_version >= 10) {
           RequireInputCount(node, 2);
           const Tensor &k_tensor = GetInput(node, 1, rt.tensors());
           if (k_tensor.element_count() != 1) {
             throw std::invalid_argument(
                 "RunNode: op 'TopK' input 'K' must be a 1-D tensor with a single element.");
           }
           if (k_tensor.data_type != static_cast<int32_t>(DataType::INT64)) {
             throw std::invalid_argument("RunNode: op 'TopK' input 'K' must be INT64.");
           }
           k = k_tensor.AsInt64()[0];
         } else {
           RequireInputCount(node, 1);
           const AttributeProto *k_attr = FindAttribute(node, "k");
           if (k_attr == nullptr) {
             throw std::invalid_argument(
                 "RunNode: op 'TopK' requires the 'k' attribute for opset < 10.");
           }
           k = k_attr->i();
         }

         kernel::TopK kernel(rt.kernel_ctx());
         auto out = kernel(x, k, axis, largest, sorted);
         SetOutput(node, 0, std::move(out.first), rt);
         SetOutput(node, 1, std::move(out.second), rt);
       }},
      {"ai.onnx:Transpose",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &data = GetInput(node, 0, rt.tensors());
         const std::vector<int64_t> perm = GetAttributeIntsOrDefault(node, "perm", {});
         kernel::Transpose k(rt.kernel_ctx());
         SetOutput(node, 0, k(data, perm), rt);
       }},
      {"ai.onnx:Trilu",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputRange(node, 1, 2);
         RequireOutputCount(node, 1);
         const Tensor &input = GetInput(node, 0, rt.tensors());
         const Tensor *k = GetOptionalInput(node, 1, rt.tensors());
         kernel::Trilu::Attributes attrs;
         attrs.upper = GetAttributeIntOrDefault(node, "upper", 1);
         kernel::Trilu kernel(rt.kernel_ctx());
         SetOutput(node, 0, kernel(input, k, attrs), rt);
       }},
      {"ai.onnx:Unique",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputRange(node, 1, 4);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         kernel::Unique::Attributes attrs;
         attrs.sorted = GetAttributeIntOrDefault(node, "sorted", 1) != 0;
         const AttributeProto *axis_attr = FindAttribute(node, "axis");
         if (axis_attr != nullptr) {
           attrs.axis = axis_attr->i();
         }
         kernel::Unique k(rt.kernel_ctx());
         auto out = k(x, attrs);
         SetOutput(node, 0, std::move(out.y), rt);
         if (node.output_size() >= 2) {
           SetOutput(node, 1, std::move(out.indices), rt);
         }
         if (node.output_size() >= 3) {
           SetOutput(node, 2, std::move(out.inverse_indices), rt);
         }
         if (node.output_size() >= 4) {
           SetOutput(node, 3, std::move(out.counts), rt);
         }
       }},
      {"ai.onnx:Unsqueeze", MakeSqueezeLikeTrampoline<kernel::Unsqueeze>("Unsqueeze")},
      {"ai.onnx:Upsample",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &scales = GetInput(node, 1, rt.tensors());
         kernel::Upsample::Attributes attrs;
         attrs.mode = GetAttributeStringOrDefault(node, "mode", attrs.mode);
         kernel::Upsample k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, scales, attrs), rt);
       }},
      {"ai.onnx:Where", MakeTernaryTrampoline<kernel::Where>()},
      {"ai.onnx:Xor", MakeBinaryTrampoline<kernel::Xor>()},

      // ai.onnx.preview
      {"ai.onnx.preview:FlexAttention",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 3);
         RequireOutputCount(node, 1);
         const Tensor &Q = GetInput(node, 0, rt.tensors());
         const Tensor &K = GetInput(node, 1, rt.tensors());
         const Tensor &V = GetInput(node, 2, rt.tensors());

         // Resolve the scale once: use the explicit attribute if present, otherwise
         // fall back to 1/sqrt(head_size) — matching the kernel's own default.
         const float scale =
             FindAttribute(node, "scale") != nullptr
                 ? GetAttributeFloatOrDefault(node, "scale", 1.0f)
                 : 1.0f / std::sqrt(static_cast<float>(Q.shape[3]));

         kernel::FlexAttention flex(rt.kernel_ctx());
         Tensor Y;
         kernel::FlexAttention::ScoreModFn score_mod_fn;
         kernel::FlexAttention::ProbModFn prob_mod_fn;
         const AttributeProto *score_mod_attr = FindAttribute(node, "score_mod");
         if (score_mod_attr != nullptr) {
           const GraphProto &score_mod_graph = score_mod_attr->ref_g();
           if (score_mod_graph.input().empty()) {
             throw std::invalid_argument(
                 "RunNode: 'score_mod' subgraph must declare at least one input.");
           }
           const std::string in_name = score_mod_graph.input()[0].name().as_string();
           score_mod_fn = [&score_mod_graph, in_name, &rt](Tensor &scores) {
             auto outputs = RunSubgraph(score_mod_graph, {{in_name, scores}}, rt);
             if (!outputs.empty()) {
               scores = std::move(outputs[0]);
             }
           };
         }
         const AttributeProto *prob_mod_attr = FindAttribute(node, "prob_mod");
         if (prob_mod_attr != nullptr) {
           const GraphProto &prob_mod_graph = prob_mod_attr->ref_g();
           if (prob_mod_graph.input().empty()) {
             throw std::invalid_argument(
                 "RunNode: 'prob_mod' subgraph must declare at least one input.");
           }
           const std::string in_name = prob_mod_graph.input()[0].name().as_string();
           prob_mod_fn = [&prob_mod_graph, in_name, &rt](Tensor &probs) {
             auto outputs = RunSubgraph(prob_mod_graph, {{in_name, probs}}, rt);
             if (!outputs.empty()) {
               probs = std::move(outputs[0]);
             }
           };
         }
         if (score_mod_fn || prob_mod_fn) {
           Y = flex(Q, K, V, scale, score_mod_fn, prob_mod_fn);
         } else {
           Y = flex(Q, K, V, scale);
         }
         SetOutput(node, 0, std::move(Y), rt.tensors());
       }},

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
      {"ai.onnx.ml:CastMap",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         // CastMap's map(int64, T) input is represented at runtime as two
         // tensors in the RuntimeContext named "<input>_keys" (INT64) and
         // "<input>_values" (FLOAT or STRING), where <input> is the node's
         // formal input name (e.g. "x" -> "x_keys" and "x_values").
         const std::string map_input = node.input(0).as_string();
         const std::string keys_name = map_input + "_keys";
         const std::string values_name = map_input + "_values";
         auto keys_it = rt.tensors().find(keys_name);
         auto values_it = rt.tensors().find(values_name);
         if (keys_it == rt.tensors().end()) {
           throw std::invalid_argument("RunNode: CastMap map input '" + map_input +
                                       "' requires tensor '" + keys_name + "' (INT64 keys).");
         }
         if (values_it == rt.tensors().end()) {
           throw std::invalid_argument("RunNode: CastMap map input '" + map_input +
                                       "' requires tensor '" + values_name +
                                       "' (FLOAT or STRING values).");
         }
         const Tensor &x_keys = keys_it->second;
         const Tensor &x_values = values_it->second;
         if (x_keys.data_type != static_cast<int32_t>(DataType::INT64)) {
           throw std::invalid_argument("RunNode: CastMap '" + keys_name +
                                       "' must be an INT64 tensor.");
         }
         const std::vector<int64_t> keys = TensorToVector<int64_t>(x_keys);
         const std::string cast_to = GetAttributeStringOrDefault(node, "cast_to", "TO_FLOAT");
         const std::string map_form = GetAttributeStringOrDefault(node, "map_form", "DENSE");
         const int64_t max_map = GetAttributeIntOrDefault(node, "max_map", 0);
         if (cast_to != "TO_FLOAT" && cast_to != "TO_INT64" && cast_to != "TO_STRING") {
           throw std::invalid_argument(
               "RunNode: CastMap attribute 'cast_to' must be 'TO_FLOAT', 'TO_INT64', or "
               "'TO_STRING'.");
         }
         kernel::CastMap cast_map(rt.kernel_ctx());
         Tensor y;
         switch (x_values.data_type) {
         case static_cast<int32_t>(DataType::FLOAT): {
           const std::vector<float> values = TensorToVector<float>(x_values);
           if (cast_to == "TO_FLOAT") {
             y = cast_map.operator()<float, float>(keys, values, cast_to, map_form, max_map);
           } else if (cast_to == "TO_INT64") {
             y = cast_map.operator()<float, int64_t>(keys, values, cast_to, map_form, max_map);
           } else {
             y = cast_map.operator()<float, std::string>(keys, values, cast_to, map_form, max_map);
           }
           break;
         }
         case static_cast<int32_t>(DataType::STRING): {
           const std::vector<std::string> &values = x_values.AsStrings();
           if (cast_to == "TO_FLOAT") {
             y = cast_map.operator()<std::string, float>(keys, values, cast_to, map_form, max_map);
           } else if (cast_to == "TO_INT64") {
             y = cast_map.operator()<std::string, int64_t>(keys, values, cast_to, map_form,
                                                            max_map);
           } else {
             y = cast_map.operator()<std::string, std::string>(keys, values, cast_to, map_form,
                                                                max_map);
           }
           break;
         }
         default:
           throw std::invalid_argument("RunNode: CastMap '" + values_name +
                                       "' must be a FLOAT or STRING tensor.");
         }
         SetOutput(node, 0, std::move(y), rt);
       }},
      {"ai.onnx.ml:DictVectorizer",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         // DictVectorizer's map(K, V) input is represented at runtime as two
         // tensors in the RuntimeContext named "<input>_keys" (INT64 or
         // STRING) and "<input>_values" (INT64, FLOAT, DOUBLE, or STRING),
         // where <input> is the node's formal input name (e.g. "x" ->
         // "x_keys" and "x_values"). Mirrors the CastMap convention above.
         const std::string map_input = node.input(0).as_string();
         const std::string keys_name = map_input + "_keys";
         const std::string values_name = map_input + "_values";
         auto keys_it = rt.tensors().find(keys_name);
         auto values_it = rt.tensors().find(values_name);
         if (keys_it == rt.tensors().end()) {
           throw std::invalid_argument("RunNode: DictVectorizer map input '" + map_input +
                                       "' requires tensor '" + keys_name + "' (INT64 or STRING keys).");
         }
         if (values_it == rt.tensors().end()) {
           throw std::invalid_argument("RunNode: DictVectorizer map input '" + map_input +
                                       "' requires tensor '" + values_name +
                                       "' (INT64, FLOAT, DOUBLE, or STRING values).");
         }
         const Tensor &x_keys = keys_it->second;
         const Tensor &x_values = values_it->second;
         const AttributeProto *str_vocab = FindAttribute(node, "string_vocabulary");
         const AttributeProto *int_vocab = FindAttribute(node, "int64_vocabulary");
         const bool has_str = str_vocab != nullptr && str_vocab->strings_size() > 0;
         const bool has_int = int_vocab != nullptr && int_vocab->ints_size() > 0;
         if (has_str == has_int) {
           throw std::invalid_argument(
               "RunNode: DictVectorizer requires exactly one of 'string_vocabulary' or "
               "'int64_vocabulary' to be specified and non-empty.");
         }
         kernel::DictVectorizer dict(rt.kernel_ctx());
         Tensor y;
         if (has_str) {
           if (x_keys.data_type != static_cast<int32_t>(DataType::STRING)) {
             throw std::invalid_argument("RunNode: DictVectorizer '" + keys_name +
                                         "' must be a STRING tensor when 'string_vocabulary' is set.");
           }
           const std::vector<std::string> &keys = x_keys.AsStrings();
           std::vector<std::string> vocab;
           vocab.reserve(str_vocab->strings_size());
           for (size_t i = 0; i < str_vocab->strings_size(); ++i) {
             vocab.emplace_back(str_vocab->strings(i).as_string());
           }
           switch (x_values.data_type) {
           case static_cast<int32_t>(DataType::INT64): {
             const std::vector<int64_t> values = TensorToVector<int64_t>(x_values);
             y = dict.operator()<std::string, int64_t>(keys, values, vocab);
             break;
           }
           case static_cast<int32_t>(DataType::FLOAT): {
             const std::vector<float> values = TensorToVector<float>(x_values);
             y = dict.operator()<std::string, float>(keys, values, vocab);
             break;
           }
           case static_cast<int32_t>(DataType::DOUBLE): {
             const std::vector<double> values = TensorToVector<double>(x_values);
             y = dict.operator()<std::string, double>(keys, values, vocab);
             break;
           }
           default:
             throw std::invalid_argument("RunNode: DictVectorizer '" + values_name +
                                         "' must be INT64, FLOAT, or DOUBLE when "
                                         "'string_vocabulary' is set.");
           }
         } else {
           if (x_keys.data_type != static_cast<int32_t>(DataType::INT64)) {
             throw std::invalid_argument("RunNode: DictVectorizer '" + keys_name +
                                         "' must be an INT64 tensor when 'int64_vocabulary' is set.");
           }
           const std::vector<int64_t> keys = TensorToVector<int64_t>(x_keys);
           std::vector<int64_t> vocab;
           vocab.reserve(int_vocab->ints_size());
           for (size_t i = 0; i < int_vocab->ints_size(); ++i) {
             vocab.push_back(int_vocab->ints(i));
           }
           switch (x_values.data_type) {
           case static_cast<int32_t>(DataType::FLOAT): {
             const std::vector<float> values = TensorToVector<float>(x_values);
             y = dict.operator()<int64_t, float>(keys, values, vocab);
             break;
           }
           case static_cast<int32_t>(DataType::DOUBLE): {
             const std::vector<double> values = TensorToVector<double>(x_values);
             y = dict.operator()<int64_t, double>(keys, values, vocab);
             break;
           }
           case static_cast<int32_t>(DataType::STRING): {
             const std::vector<std::string> &values = x_values.AsStrings();
             y = dict.operator()<int64_t, std::string>(keys, values, vocab);
             break;
           }
           default:
             throw std::invalid_argument("RunNode: DictVectorizer '" + values_name +
                                         "' must be FLOAT, DOUBLE, or STRING when "
                                         "'int64_vocabulary' is set.");
           }
         }
         SetOutput(node, 0, std::move(y), rt);
       }},
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
         SetOutput(node, 0, std::move(y), rt);
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
         SetOutput(node, 0, std::move(yz.first), rt);
         SetOutput(node, 1, std::move(yz.second), rt);
       }},
      {"ai.onnx.ml:LinearRegressor",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const std::vector<float> coefficients =
             GetAttributeFloatsOrDefault(node, "coefficients", {});
         const std::vector<float> intercepts =
             GetAttributeFloatsOrDefault(node, "intercepts", {});
         const int64_t targets = GetAttributeIntOrDefault(node, "targets", 1);
         const std::string post_transform =
             GetAttributeStringOrDefault(node, "post_transform", "NONE");
         kernel::LinearRegressor reg(rt.kernel_ctx());
         Tensor y = DispatchSVMByDataType(x, "LinearRegressor", [&](auto *tag) {
           using T = std::remove_pointer_t<decltype(tag)>;
           (void)tag;
           return reg.template operator()<T>(x, coefficients, intercepts, targets, post_transform);
         });
         SetOutput(node, 0, std::move(y), rt);
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
         SetOutput(node, 0, std::move(y), rt);
       }},
      {"ai.onnx.ml:LinearClassifier",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 2);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const std::vector<float> coefficients =
             GetAttributeFloatsOrDefault(node, "coefficients", {});
         const std::vector<float> intercepts =
             GetAttributeFloatsOrDefault(node, "intercepts", {});
         const std::string post_transform =
             GetAttributeStringOrDefault(node, "post_transform", "NONE");
         const std::vector<int64_t> classlabels_ints =
             GetAttributeIntsOrDefault(node, "classlabels_ints", {});
         const std::vector<std::string> classlabels_strings =
             GetAttributeStringsOrDefault(node, "classlabels_strings", {});
         const bool use_strings = !classlabels_strings.empty();
         const bool has_ints = !classlabels_ints.empty();
         if (use_strings == has_ints) {
           throw std::invalid_argument(
               "RunNode: LinearClassifier requires exactly one of 'classlabels_ints' or "
               "'classlabels_strings' to be set.");
         }
         kernel::LinearClassifier cls(rt.kernel_ctx());
         std::pair<Tensor, Tensor> yz =
             DispatchSVMByDataType(x, "LinearClassifier", [&](auto *tag) {
               using T = std::remove_pointer_t<decltype(tag)>;
               (void)tag;
               return use_strings ? cls.template operator()<T>(x, coefficients, intercepts,
                                                               classlabels_strings, post_transform)
                                  : cls.template operator()<T>(x, coefficients, intercepts,
                                                               classlabels_ints, post_transform);
             });
         SetOutput(node, 0, std::move(yz.first), rt);
         SetOutput(node, 1, std::move(yz.second), rt);
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
         SetOutput(node, 0, std::move(yz.first), rt);
         SetOutput(node, 1, std::move(yz.second), rt);
       }},
      {"ai.onnx.ml:Binarizer",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const float threshold = GetAttributeFloatOrDefault(node, "threshold", 0.0f);
         kernel::Binarizer binarizer(rt.kernel_ctx());
         Tensor y = DispatchSVMByDataType(x, "Binarizer", [&](auto *tag) {
           using T = std::remove_pointer_t<decltype(tag)>;
           (void)tag;
           return binarizer.template operator()<T>(x, static_cast<T>(threshold));
         });
         SetOutput(node, 0, std::move(y), rt.tensors());
       }},
      {"ai.onnx.ml:Normalizer",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const std::string norm = GetAttributeStringOrDefault(node, "norm", "MAX");
         kernel::Normalizer normalizer(rt.kernel_ctx());
         Tensor y = DispatchSVMByDataType(x, "Normalizer", [&](auto *tag) {
           using T = std::remove_pointer_t<decltype(tag)>;
           (void)tag;
           return normalizer.template operator()<T>(x, norm);
         });
         SetOutput(node, 0, std::move(y), rt.tensors());
       }},
      {"ai.onnx.ml:Scaler",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const std::vector<float> offset = GetAttributeFloatsOrDefault(node, "offset", {});
         const std::vector<float> scale = GetAttributeFloatsOrDefault(node, "scale", {});
         kernel::Scaler scaler(rt.kernel_ctx());
         Tensor y = DispatchSVMByDataType(x, "Scaler", [&](auto *tag) {
           using T = std::remove_pointer_t<decltype(tag)>;
           (void)tag;
           return scaler.template operator()<T>(x, offset, scale);
         });
         SetOutput(node, 0, std::move(y), rt.tensors());
       }},
      {"ai.onnx.ml:ArrayFeatureExtractor",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 2);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const Tensor &y = GetInput(node, 1, rt.tensors());
         kernel::ArrayFeatureExtractor afe(rt.kernel_ctx());
         Tensor z = DispatchSVMByDataType(x, "ArrayFeatureExtractor", [&](auto *tag) {
           using T = std::remove_pointer_t<decltype(tag)>;
           (void)tag;
           return afe.template operator()<T>(x, y);
         });
         SetOutput(node, 0, std::move(z), rt.tensors());
       }},
      {"ai.onnx.ml:Imputer",
       [](const NodeProto &node, RuntimeContext &rt) {
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
         const float replaced_value_float =
             GetAttributeFloatOrDefault(node, "replaced_value_float", 0.0f);
         const int64_t replaced_value_int64 =
             GetAttributeIntOrDefault(node, "replaced_value_int64", static_cast<int64_t>(0));

         kernel::Imputer imputer(rt.kernel_ctx());
         Tensor y = DispatchSVMByDataType(x, "Imputer", [&](auto *tag) {
           using T = std::remove_pointer_t<decltype(tag)>;
           (void)tag;
           if constexpr (std::is_floating_point_v<T>) {
             std::vector<T> imputed_values(imputed_value_floats.begin(),
                                           imputed_value_floats.end());
             return imputer.template operator()<T>(x, imputed_values,
                                                   static_cast<T>(replaced_value_float));
           } else {
             std::vector<T> imputed_values(imputed_value_int64s.begin(),
                                           imputed_value_int64s.end());
             return imputer.template operator()<T>(x, imputed_values,
                                                   static_cast<T>(replaced_value_int64));
           }
         });
         SetOutput(node, 0, std::move(y), rt.tensors());
       }},
      {"ai.onnx.ml:CategoryMapper",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());

         // ``cats_strings`` and ``cats_int64s`` are both required per the
         // ``ai.onnx.ml::CategoryMapper`` schema and must have the same length.
         const std::vector<std::string> cats_strings =
             GetAttributeStringsOrDefault(node, "cats_strings", {});
         const std::vector<int64_t> cats_int64s =
             GetAttributeIntsOrDefault(node, "cats_int64s", {});
         const std::string default_string =
             GetAttributeStringOrDefault(node, "default_string", std::string("_Unused"));
         const int64_t default_int64 =
             GetAttributeIntOrDefault(node, "default_int64", static_cast<int64_t>(-1));

         kernel::CategoryMapper category_mapper(rt.kernel_ctx());
         Tensor y;
         switch (x.data_type) {
         case static_cast<int32_t>(DataType::STRING):
           y = category_mapper.operator()<std::string, int64_t>(x, cats_strings, cats_int64s,
                                                                default_int64);
           break;
         case static_cast<int32_t>(DataType::INT64):
           y = category_mapper.operator()<int64_t, std::string>(x, cats_strings, cats_int64s,
                                                                default_string);
           break;
         default:
           throw std::invalid_argument(
               "RunNode: CategoryMapper input X must have element type STRING or INT64.");
         }
         SetOutput(node, 0, std::move(y), rt.tensors());
       }},
      {"ai.onnx.ml:LabelEncoder",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());

         // Identify the key source (exactly one of keys_int64s, keys_floats,
         // keys_strings, keys_tensor must be present per the ONNX spec).
         const AttributeProto *keys_int64s = FindAttribute(node, "keys_int64s");
         const AttributeProto *keys_floats = FindAttribute(node, "keys_floats");
         const AttributeProto *keys_strings = FindAttribute(node, "keys_strings");
         const AttributeProto *keys_tensor = FindAttribute(node, "keys_tensor");
         const int n_keys = (keys_int64s != nullptr) + (keys_floats != nullptr) +
                            (keys_strings != nullptr) + (keys_tensor != nullptr);
         if (n_keys != 1) {
           throw std::invalid_argument(
               "RunNode: LabelEncoder requires exactly one of 'keys_int64s', "
               "'keys_floats', 'keys_strings' or 'keys_tensor' to be set.");
         }

         // Identify the value source.
         const AttributeProto *values_int64s = FindAttribute(node, "values_int64s");
         const AttributeProto *values_floats = FindAttribute(node, "values_floats");
         const AttributeProto *values_strings = FindAttribute(node, "values_strings");
         const AttributeProto *values_tensor = FindAttribute(node, "values_tensor");
         const int n_values = (values_int64s != nullptr) + (values_floats != nullptr) +
                              (values_strings != nullptr) + (values_tensor != nullptr);
         if (n_values != 1) {
           throw std::invalid_argument(
               "RunNode: LabelEncoder requires exactly one of 'values_int64s', "
               "'values_floats', 'values_strings' or 'values_tensor' to be set.");
         }

         // Resolve KeyT.
         enum class KeyKind { Int64, Float, String };
         KeyKind key_kind;
         std::vector<int64_t> keys_i64;
         std::vector<float> keys_f32;
         std::vector<std::string> keys_str;
         if (keys_int64s != nullptr) {
           key_kind = KeyKind::Int64;
           for (int64_t v : keys_int64s->ints()) {
             keys_i64.push_back(v);
           }
         } else if (keys_floats != nullptr) {
           key_kind = KeyKind::Float;
           for (float v : keys_floats->floats()) {
             keys_f32.push_back(v);
           }
         } else if (keys_strings != nullptr) {
           key_kind = KeyKind::String;
           for (const auto &v : keys_strings->strings()) {
             keys_str.push_back(v.as_string());
           }
         } else {
           const Tensor kt = TensorFromProto(keys_tensor->t());
           switch (kt.data_type) {
           case static_cast<int32_t>(DataType::INT64):
             key_kind = KeyKind::Int64;
             keys_i64 = TensorToVector<int64_t>(kt);
             break;
           case static_cast<int32_t>(DataType::FLOAT):
             key_kind = KeyKind::Float;
             keys_f32 = TensorToVector<float>(kt);
             break;
           case static_cast<int32_t>(DataType::STRING):
             key_kind = KeyKind::String;
             keys_str = kt.AsStrings();
             break;
           default:
             throw std::invalid_argument(
                 "RunNode: LabelEncoder 'keys_tensor' must have element type "
                 "INT64, FLOAT or STRING.");
           }
         }

         // Resolve ValueT and look up the (optional) default attribute.
         enum class ValueKind { Int64, Float, Int16 };
         ValueKind value_kind;
         std::vector<int64_t> values_i64;
         std::vector<float> values_f32;
         std::vector<int16_t> values_i16;
         if (values_int64s != nullptr) {
           value_kind = ValueKind::Int64;
           for (int64_t v : values_int64s->ints()) {
             values_i64.push_back(v);
           }
         } else if (values_floats != nullptr) {
           value_kind = ValueKind::Float;
           for (float v : values_floats->floats()) {
             values_f32.push_back(v);
           }
         } else if (values_strings != nullptr) {
           throw std::invalid_argument(
               "RunNode: LabelEncoder with 'values_strings' is not supported "
               "by this kernel registration.");
         } else {
           const Tensor vt = TensorFromProto(values_tensor->t());
           switch (vt.data_type) {
           case static_cast<int32_t>(DataType::INT64):
             value_kind = ValueKind::Int64;
             values_i64 = TensorToVector<int64_t>(vt);
             break;
           case static_cast<int32_t>(DataType::FLOAT):
             value_kind = ValueKind::Float;
             values_f32 = TensorToVector<float>(vt);
             break;
           case static_cast<int32_t>(DataType::INT16):
             value_kind = ValueKind::Int16;
             values_i16 = TensorToVector<int16_t>(vt);
             break;
           default:
             throw std::invalid_argument(
                 "RunNode: LabelEncoder 'values_tensor' must have element "
                 "type INT64, FLOAT or INT16.");
           }
         }

         int64_t default_i64 = -1;
         float default_f32 = 0.0f;
         int16_t default_i16 = -1;
         const AttributeProto *default_int64 = FindAttribute(node, "default_int64");
         const AttributeProto *default_float = FindAttribute(node, "default_float");
         const AttributeProto *default_tensor_attr = FindAttribute(node, "default_tensor");
         if (default_int64 != nullptr) {
           default_i64 = default_int64->i();
         }
         if (default_float != nullptr) {
           default_f32 = default_float->f();
         }
         if (default_tensor_attr != nullptr) {
           const Tensor dt = TensorFromProto(default_tensor_attr->t());
           if (dt.element_count() != 1) {
             throw std::invalid_argument(
                 "RunNode: LabelEncoder 'default_tensor' must contain exactly one element.");
           }
           switch (dt.data_type) {
           case static_cast<int32_t>(DataType::INT64):
             default_i64 = dt.AsInt64()[0];
             break;
           case static_cast<int32_t>(DataType::FLOAT):
             default_f32 = dt.AsFloat()[0];
             break;
           case static_cast<int32_t>(DataType::INT16):
             default_i16 = dt.AsInt16()[0];
             break;
           default:
             throw std::invalid_argument(
                 "RunNode: LabelEncoder 'default_tensor' must have element "
                 "type INT64, FLOAT or INT16.");
           }
         }

         kernel::LabelEncoder label_encoder(rt.kernel_ctx());
         Tensor out;
         if (key_kind == KeyKind::Int64 && value_kind == ValueKind::Int64) {
           out = label_encoder.operator()<int64_t, int64_t>(x, keys_i64, values_i64, default_i64);
         } else if (key_kind == KeyKind::Int64 && value_kind == ValueKind::Float) {
           out = label_encoder.operator()<int64_t, float>(x, keys_i64, values_f32, default_f32);
         } else if (key_kind == KeyKind::Float && value_kind == ValueKind::Int64) {
           out = label_encoder.operator()<float, int64_t>(x, keys_f32, values_i64, default_i64);
         } else if (key_kind == KeyKind::Float && value_kind == ValueKind::Float) {
           out = label_encoder.operator()<float, float>(x, keys_f32, values_f32, default_f32);
         } else if (key_kind == KeyKind::String && value_kind == ValueKind::Int64) {
           out =
               label_encoder.operator()<std::string, int64_t>(x, keys_str, values_i64, default_i64);
         } else if (key_kind == KeyKind::String && value_kind == ValueKind::Int16) {
           out =
               label_encoder.operator()<std::string, int16_t>(x, keys_str, values_i16, default_i16);
         } else {
           throw std::invalid_argument(
               "RunNode: LabelEncoder key/value type combination is not supported.");
         }
         SetOutput(node, 0, std::move(out), rt.tensors());
       }},
      {"ai.onnx.ml:OneHotEncoder",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());

         const AttributeProto *cats_int64s = FindAttribute(node, "cats_int64s");
         const AttributeProto *cats_strings = FindAttribute(node, "cats_strings");
         const int n_cats = (cats_int64s != nullptr) + (cats_strings != nullptr);
         if (n_cats != 1) {
           throw std::invalid_argument(
               "RunNode: OneHotEncoder requires exactly one of 'cats_int64s' "
               "or 'cats_strings' to be set.");
         }

         // The ``zeros`` attribute defaults to 1 per the ai.onnx.ml schema.
         const bool zeros = GetAttributeIntOrDefault(node, "zeros", 1) != 0;

         kernel::OneHotEncoder one_hot(rt.kernel_ctx());
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
             return one_hot.template operator()<T>(x, cats, zeros);
           });
         } else {
           std::vector<std::string> cats;
           cats.reserve(cats_strings->strings().size());
           for (size_t i = 0; i < cats_strings->strings().size(); ++i) {
             cats.push_back(cats_strings->strings()[i].as_string());
           }
           if (x.data_type != static_cast<int32_t>(DataType::STRING)) {
             throw std::invalid_argument(
                 "RunNode: OneHotEncoder with 'cats_strings' requires input X "
                 "of element type STRING.");
           }
           y = one_hot(x, cats, zeros);
         }
         SetOutput(node, 0, std::move(y), rt.tensors());
       }},
      {"ai.onnx.ml:FeatureVectorizer",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireMinInputCount(node, 1);
         RequireOutputCount(node, 1);
         std::vector<Tensor> inputs;
         inputs.reserve(node.input_size());
         for (int i = 0; i < node.input_size(); ++i) {
           inputs.push_back(GetInput(node, i, rt.tensors()));
         }
         const std::vector<int64_t> inputdimensions =
             GetAttributeIntsOrDefault(node, "inputdimensions", {});
         kernel::FeatureVectorizer fv(rt.kernel_ctx());
         SetOutput(node, 0, fv(inputs, inputdimensions), rt.tensors());
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
         SetOutput(node, 0, std::move(y), rt);
       }},
  };
  return table;
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
