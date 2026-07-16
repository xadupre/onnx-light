// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_gradient/gradient.h"
#include "onnx_proto/onnx.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;

void AddOnnxPyGradient(nb::module_ &m);

NB_MODULE(_onnxpygradient, m) {
  m.doc() = "onnx_gradient bindings: reverse-mode automatic differentiation for ONNX graphs";
  AddOnnxPyGradient(m);
}

void AddOnnxPyGradient(nb::module_ &m) {
  m.def(
      "gradient_of_nodes",
      [](const std::vector<NodeProto> &nodes, const std::vector<std::string> &inputs,
         const std::vector<TensorProto> &initializers, const std::vector<std::string> &xs,
         const std::string &y, const std::vector<std::string> &zs) -> FunctionProto {
        return onnx_gradient::GradientOfNodes(nodes, inputs, initializers, xs, y, zs);
      },
      nb::arg("nodes"), nb::arg("inputs"), nb::arg("initializers"), nb::arg("xs"), nb::arg("y"),
      nb::arg("zs"),
      R"doc(
Compute the gradient FunctionProto from a list of ONNX nodes.

Performs reverse-mode automatic differentiation over *nodes* and returns a
FunctionProto that computes the partial derivatives of *y* with respect to
each variable in *xs*.

The returned FunctionProto has:

* inputs  – *xs* values followed by *zs* values, then ``"dy"`` (the incoming
  gradient of *y*; pass ``ones_like(y)`` for a scalar loss).
* outputs – one gradient tensor per element of *xs*, named ``"grad_<x>"``.

Supported forward operators: MatMul, Gemm, Add, Sub, Mul, Div, Neg, Identity,
Relu, ReduceSum, ReduceMean, Reshape, Transpose, Sigmoid, Tanh.

Parameters
----------
nodes : list[NodeProto]
    The forward computation nodes in topological order.
inputs : list[str]
    Names of all graph inputs.  This parameter is accepted for API symmetry
    with the C++ ``GradientOfNodes`` signature; it is not used by the
    current autodiff algorithm but is reserved for future use (e.g.
    distinguishing graph inputs from initializers during gradient pruning).
initializers : list[TensorProto]
    Constant tensors embedded in the forward graph.
xs : list[str]
    Variable names to differentiate with respect to.
y : str
    The output tensor name whose gradient is computed.
zs : list[str]
    Additional non-differentiable input variable names.

Returns
-------
FunctionProto
    The gradient computation encoded as a FunctionProto.

Raises
------
ValueError
    If *xs* is empty, *y* is empty, or *y* is not produced by any node.
RuntimeError
    If an unsupported op_type is encountered on the path from inputs to *y*.
)doc");

  m.def(
      "gradient_of_function",
      [](const FunctionProto &function, const std::vector<std::string> &xs, const std::string &y,
         const std::vector<std::string> &zs) -> FunctionProto {
        return onnx_gradient::GradientOfFunction(function, xs, y, zs);
      },
      nb::arg("function"), nb::arg("xs"), nb::arg("y"), nb::arg("zs"),
      R"doc(
Compute the gradient FunctionProto from an existing FunctionProto.

The *function* is expected to take its initializers as regular inputs (i.e.
model parameters are part of the function's input list rather than embedded
graph initializers). This is the standard pattern when expressing a model as a
pure function for training.

Parameters
----------
function : FunctionProto
    The forward computation as a FunctionProto.
xs : list[str]
    Variable names (among *function* inputs) to differentiate with respect to.
y : str
    The output tensor name whose gradient is computed.
zs : list[str]
    Additional non-differentiable input variable names.

Returns
-------
FunctionProto
    The gradient computation encoded as a FunctionProto.

Raises
------
ValueError
    If *xs* is empty, *y* is empty, or *y* is not produced by any node.
RuntimeError
    If an unsupported op_type is encountered on the path from inputs to *y*.
)doc");
}
