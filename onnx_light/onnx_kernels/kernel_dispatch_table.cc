// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernel_dispatch_table.h"

#include "onnx_kernels/kernels/generator/include_generator_kernels.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/kernels/reduction/include_reduction_kernels.h"
#include "onnx_kernels/node_helpers.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

namespace {

using detail::GetAttributeFloatOrDefault;
using detail::GetAttributeIntOrDefault;
using detail::GetAttributeIntsOrDefault;
using detail::GetAttributeStringOrDefault;
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

// Wraps reduction kernels of the form:
//   ``operator()(data, keepdims, noop_with_empty_axes)``
//   ``operator()(data, axes, keepdims, noop_with_empty_axes)``
// where ``axes`` is either an optional second input (opset 13+/18+ depending
// on the operator) or an ``axes`` INTS attribute (older opsets).
template <class KernelT> NodeKernelFn MakeReduceTrampoline() {
  return [](const NodeProto &node, RuntimeContext &rt) {
    RequireMinInputCount(node, 1);
    if (node.input_size() > 2) {
      throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() +
                                  "' expects 1 or 2 inputs.");
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

} // namespace

const std::unordered_map<std::string, NodeKernelFn> &KernelDispatchTable() {
  static const std::unordered_map<std::string, NodeKernelFn> table = {
      // -----------------------------------------------------------------
      // Element-wise unary math (no attributes).
      // -----------------------------------------------------------------
      {"ai.onnx:Abs", MakeUnaryTrampoline<kernel::Abs>()},
      {"ai.onnx:Acos", MakeUnaryTrampoline<kernel::Acos>()},
      {"ai.onnx:Acosh", MakeUnaryTrampoline<kernel::Acosh>()},
      {"ai.onnx:Asin", MakeUnaryTrampoline<kernel::Asin>()},
      {"ai.onnx:Asinh", MakeUnaryTrampoline<kernel::Asinh>()},
      {"ai.onnx:Atan", MakeUnaryTrampoline<kernel::Atan>()},
      {"ai.onnx:Atanh", MakeUnaryTrampoline<kernel::Atanh>()},
      {"ai.onnx:Ceil", MakeUnaryTrampoline<kernel::Ceil>()},
      {"ai.onnx:Cos", MakeUnaryTrampoline<kernel::Cos>()},
      {"ai.onnx:Cosh", MakeUnaryTrampoline<kernel::Cosh>()},
      {"ai.onnx:Det", MakeUnaryTrampoline<kernel::Det>()},
      {"ai.onnx:Erf", MakeUnaryTrampoline<kernel::Erf>()},
      {"ai.onnx:Exp", MakeUnaryTrampoline<kernel::Exp>()},
      {"ai.onnx:Floor", MakeUnaryTrampoline<kernel::Floor>()},
      {"ai.onnx:HardSwish", MakeUnaryTrampoline<kernel::HardSwish>()},
      {"ai.onnx:Log", MakeUnaryTrampoline<kernel::Log>()},
      {"ai.onnx:Mish", MakeUnaryTrampoline<kernel::Mish>()},
      {"ai.onnx:Neg", MakeUnaryTrampoline<kernel::Neg>()},
      {"ai.onnx:Reciprocal", MakeUnaryTrampoline<kernel::Reciprocal>()},
      {"ai.onnx:Relu", MakeUnaryTrampoline<kernel::Relu>()},
      {"ai.onnx:Round", MakeUnaryTrampoline<kernel::Round>()},
      {"ai.onnx:Sigmoid", MakeUnaryTrampoline<kernel::Sigmoid>()},
      {"ai.onnx:Sign", MakeUnaryTrampoline<kernel::Sign>()},
      {"ai.onnx:Sin", MakeUnaryTrampoline<kernel::Sin>()},
      {"ai.onnx:Sinh", MakeUnaryTrampoline<kernel::Sinh>()},
      {"ai.onnx:Softplus", MakeUnaryTrampoline<kernel::Softplus>()},
      {"ai.onnx:Softsign", MakeUnaryTrampoline<kernel::Softsign>()},
      {"ai.onnx:Sqrt", MakeUnaryTrampoline<kernel::Sqrt>()},
      {"ai.onnx:Tan", MakeUnaryTrampoline<kernel::Tan>()},
      {"ai.onnx:Tanh", MakeUnaryTrampoline<kernel::Tanh>()},

      // -----------------------------------------------------------------
      // Element-wise binary math (no attributes; NumPy-style broadcasting).
      // -----------------------------------------------------------------
      {"ai.onnx:Add", MakeBinaryTrampoline<kernel::Add>()},
      {"ai.onnx:Div", MakeBinaryTrampoline<kernel::Div>()},
      {"ai.onnx:MatMul", MakeBinaryTrampoline<kernel::MatMul>()},
      {"ai.onnx:Mul", MakeBinaryTrampoline<kernel::Mul>()},
      {"ai.onnx:PRelu", MakeBinaryTrampoline<kernel::PRelu>()},
      {"ai.onnx:Pow", MakeBinaryTrampoline<kernel::Pow>()},
      {"ai.onnx:Sub", MakeBinaryTrampoline<kernel::Sub>()},

      // -----------------------------------------------------------------
      // Variadic element-wise reducers.
      // -----------------------------------------------------------------
      {"ai.onnx:Sum", MakeVariadicTrampoline<kernel::Sum>()},
      {"ai.onnx:Max", MakeVariadicTrampoline<kernel::Max>()},
      {"ai.onnx:Min", MakeVariadicTrampoline<kernel::Min>()},
      {"ai.onnx:Mean", MakeVariadicTrampoline<kernel::Mean>()},

      // -----------------------------------------------------------------
      // Reduction kernels.
      // -----------------------------------------------------------------
      {"ai.onnx:ArgMax", MakeArgReduceTrampoline<kernel::ArgMax>()},
      {"ai.onnx:ArgMin", MakeArgReduceTrampoline<kernel::ArgMin>()},
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

      // -----------------------------------------------------------------
      // Element-wise unary math with a single scalar attribute.
      // -----------------------------------------------------------------
      // ``alpha`` defaults match the ONNX schema:
      //   Elu: 1.0    Celu: 1.0    LeakyRelu: 0.01
      //   ThresholdedRelu: 1.0    Swish: 1.0
      {"ai.onnx:Celu", MakeUnaryAlphaTrampoline<kernel::Celu>("alpha", 1.0f)},
      {"ai.onnx:Elu", MakeUnaryAlphaTrampoline<kernel::Elu>("alpha", 1.0f)},
      {"ai.onnx:LeakyRelu", MakeUnaryAlphaTrampoline<kernel::LeakyRelu>("alpha", 0.01f)},
      {"ai.onnx:Swish", MakeUnaryAlphaTrampoline<kernel::Swish>("alpha", 1.0f)},
      {"ai.onnx:ThresholdedRelu", MakeUnaryAlphaTrampoline<kernel::ThresholdedRelu>("alpha", 1.0f)},

      // -----------------------------------------------------------------
      // Softmax-family: ``axis`` attribute (default -1 in opset 13+).
      // -----------------------------------------------------------------
      {"ai.onnx:Hardmax", MakeAxisTrampoline<kernel::Hardmax>()},
      {"ai.onnx:LogSoftmax", MakeAxisTrampoline<kernel::LogSoftmax>()},
      {"ai.onnx:Softmax", MakeAxisTrampoline<kernel::Softmax>()},

      // -----------------------------------------------------------------
      // Unary kernels with two scalar attributes.
      // -----------------------------------------------------------------
      // ``HardSigmoid``: alpha=0.2, beta=0.5 (ONNX schema defaults).
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
      // ``Selu``: ONNX defaults match the kernel's own defaults.
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
      // ``Shrink``: bias=0.0, lambd=0.5 (ONNX schema defaults).
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

      // -----------------------------------------------------------------
      // Unary kernels with a single string attribute.
      // -----------------------------------------------------------------
      // ``Gelu``: ``approximate`` is "none" by default.
      {"ai.onnx:Gelu",
       [](const NodeProto &node, RuntimeContext &rt) {
         RequireInputCount(node, 1);
         RequireOutputCount(node, 1);
         const Tensor &x = GetInput(node, 0, rt.tensors());
         const std::string approximate = GetAttributeStringOrDefault(node, "approximate", "none");
         kernel::Gelu k(rt.kernel_ctx());
         SetOutput(node, 0, k(x, approximate), rt.tensors());
       }},

      // -----------------------------------------------------------------
      // Binary kernels with attributes.
      // -----------------------------------------------------------------
      // ``Mod``: ``fmod`` is 0 (NumPy-style integer modulo) by default.
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

      // -----------------------------------------------------------------
      // Multi-input ``Clip`` (1-3 inputs since opset 11). ``min`` and
      // ``max`` are optional and may be absent or wired with an empty
      // input name.
      // -----------------------------------------------------------------
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

      // -----------------------------------------------------------------
      // Element-wise logical / bitwise operators (no attributes).
      // -----------------------------------------------------------------
      // Binary.
      {"ai.onnx:And", MakeBinaryTrampoline<kernel::And>()},
      {"ai.onnx:BitwiseAnd", MakeBinaryTrampoline<kernel::BitwiseAnd>()},
      {"ai.onnx:BitwiseOr", MakeBinaryTrampoline<kernel::BitwiseOr>()},
      {"ai.onnx:BitwiseXor", MakeBinaryTrampoline<kernel::BitwiseXor>()},
      {"ai.onnx:Equal", MakeBinaryTrampoline<kernel::Equal>()},
      {"ai.onnx:Greater", MakeBinaryTrampoline<kernel::Greater>()},
      {"ai.onnx:GreaterOrEqual", MakeBinaryTrampoline<kernel::GreaterOrEqual>()},
      {"ai.onnx:Less", MakeBinaryTrampoline<kernel::Less>()},
      {"ai.onnx:LessOrEqual", MakeBinaryTrampoline<kernel::LessOrEqual>()},
      {"ai.onnx:Or", MakeBinaryTrampoline<kernel::Or>()},
      {"ai.onnx:Xor", MakeBinaryTrampoline<kernel::Xor>()},
      // Unary.
      {"ai.onnx:BitwiseNot", MakeUnaryTrampoline<kernel::BitwiseNot>()},
      {"ai.onnx:IsNaN", MakeUnaryTrampoline<kernel::IsNaN>()},
      {"ai.onnx:Not", MakeUnaryTrampoline<kernel::Not>()},
      // Ternary.
      {"ai.onnx:Where", MakeTernaryTrampoline<kernel::Where>()},

      // -----------------------------------------------------------------
      // ``IsInf``: ``detect_positive`` / ``detect_negative`` default to 1.
      // -----------------------------------------------------------------
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

      // -----------------------------------------------------------------
      // Generator kernels.
      // -----------------------------------------------------------------
      // ``EyeLike``: ``k`` defaults to 0; ``dtype`` is optional and, when
      // absent, the output element type matches the input (encoded as 0
      // by ``kernel::EyeLike``).
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

      // -----------------------------------------------------------------
      // ``BitShift``: ``direction`` is a required STRING attribute
      // (``"LEFT"`` or ``"RIGHT"``).
      // -----------------------------------------------------------------
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
  };
  return table;
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
