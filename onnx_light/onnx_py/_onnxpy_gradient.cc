// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/onnx_gradient/gradient.h"
#include "onnx_extensions/onnx_gradient/gradient/grad_dispatcher.h"
#include "onnx_proto/onnx.h"
#include <algorithm>
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
  // Expose GradRegistry as an opaque class so Python code can create a
  // customised copy of the default registry and pass it to the gradient
  // functions.
  nb::class_<onnx_gradient::GradRegistry>(
      m, "GradRegistry", "Maps (domain, op_type) pairs to backward gradient functions.")
      .def(nb::init<>(), "Creates an empty registry.")
      .def_static(
          "default",
          // Returns a value copy so that Python callers get an independent mutable registry
          // that can be customised without affecting the C++ static default.
          []() -> onnx_gradient::GradRegistry { return onnx_gradient::DefaultGradRegistry(); },
          "Returns a new independent copy of the built-in gradient registry.  "
          "Modifications to the returned registry do not affect the built-in defaults.")
      .def(
          "op_types",
          [](const onnx_gradient::GradRegistry &self,
             const std::string &domain) -> std::vector<std::string> {
            std::vector<std::string> result;
            for (const auto &entry : self) {
              if (entry.first.first == domain) {
                result.push_back(entry.first.second);
              }
            }
            std::sort(result.begin(), result.end());
            return result;
          },
          nb::arg("domain") = "",
          R"doc(
Returns a sorted list of op_type names registered for *domain*.

Parameters
----------
domain : str
    Operator domain (default: ``""`` for standard ONNX ops).

Returns
-------
list[str]
    Sorted op_type names registered for *domain*.
)doc");

  m.def(
      "register_gradient_function",
      [](const std::string &domain, const std::string &op_type, nb::callable fn,
         onnx_gradient::GradRegistry &registry) {
        onnx_gradient::RegisterGradientFunction(
            domain, op_type,
            // Capture fn by value so the GradFn closure keeps the Python callable alive
            // beyond the scope of register_gradient_function.
            [fn](const NodeProto &node, const std::string &output_grad,
                 std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                 FunctionProto &func) -> bool {
              nb::gil_scoped_acquire gil;
              return nb::cast<bool>(fn(node, output_grad, grad_accum, counter, func));
            },
            registry);
      },
      nb::arg("domain"), nb::arg("op_type"), nb::arg("fn"), nb::arg("registry"),
      R"doc(
Registers a custom backward function for (*domain*, *op_type*) in *registry*.

The callable *fn* must have the signature::

    fn(node: NodeProto,
       output_grad: str,
       grad_accum: dict[str, str],
       counter: int,
       func: FunctionProto) -> bool

Parameters
----------
domain : str
    The operator domain (e.g. ``""`` for standard ONNX, ``"com.example"`` for custom ops).
op_type : str
    The ONNX operator type name (e.g. ``"MyCustomOp"``).
fn : callable
    The backward function implementing the gradient rule.
registry : GradRegistry
    The registry to insert into (modified in place).
)doc");

  m.def(
      "gradient_of_nodes",
      [](const std::vector<NodeProto> &nodes, const std::vector<std::string> &inputs,
         const std::vector<TensorProto> &initializers, const std::vector<std::string> &xs,
         const std::string &y, const std::vector<std::string> &zs,
         const onnx_gradient::GradRegistry *registry) -> FunctionProto {
        return onnx_gradient::GradientOfNodes(
            nodes, inputs, initializers, xs, y, zs,
            registry != nullptr ? *registry : onnx_gradient::DefaultGradRegistry());
      },
      nb::arg("nodes"), nb::arg("inputs"), nb::arg("initializers"), nb::arg("xs"), nb::arg("y"),
      nb::arg("zs"), nb::arg("registry").none() = nb::none(),
      R"doc(
Compute the gradient FunctionProto from a list of ONNX nodes.

Performs reverse-mode automatic differentiation over *nodes* and returns a
FunctionProto that computes the partial derivatives of *y* with respect to
each variable in *xs*.

The returned FunctionProto has:

* inputs  – *xs* values followed by *zs* values, then ``"dy"`` (the incoming
  gradient of *y*; pass ``ones_like(y)`` for a scalar loss).
* outputs – one gradient tensor per element of *xs*, named ``"grad_<x>"``.

Parameters
----------
nodes : list[NodeProto]
    The forward computation nodes in topological order.
inputs : list[str]
    Names of all graph inputs.  This parameter is accepted for API symmetry;
    it is currently unused but reserved for future use (e.g. distinguishing
    graph inputs from initializers during gradient pruning).
initializers : list[TensorProto]
    Constant tensors embedded in the forward graph.
xs : list[str]
    Variable names to differentiate with respect to.
y : str
    The output tensor name whose gradient is computed.
zs : list[str]
    Additional non-differentiable input variable names.
registry : GradRegistry | None
    Operator-to-GradFn map.  ``None`` uses the built-in default registry.
    Pass a customised copy (created via ``GradRegistry.default()`` then
    modified with ``register_gradient_function``) to support custom operators.

Returns
-------
FunctionProto
    The gradient computation encoded as a FunctionProto.

Raises
------
ValueError
    If *xs* is empty, *y* is empty, or *y* is not produced by any node.
RuntimeError
    If an op_type is not found in the registry on the path from inputs to *y*.
)doc");

  m.def(
      "gradient_of_function",
      [](const FunctionProto &function, const std::vector<std::string> &xs, const std::string &y,
         const std::vector<std::string> &zs,
         const onnx_gradient::GradRegistry *registry) -> FunctionProto {
        return onnx_gradient::GradientOfFunction(
            function, xs, y, zs,
            registry != nullptr ? *registry : onnx_gradient::DefaultGradRegistry());
      },
      nb::arg("function"), nb::arg("xs"), nb::arg("y"), nb::arg("zs"),
      nb::arg("registry").none() = nb::none(),
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
registry : GradRegistry | None
    Operator-to-GradFn map.  ``None`` uses the built-in default registry.

Returns
-------
FunctionProto
    The gradient computation encoded as a FunctionProto.

Raises
------
ValueError
    If *xs* is empty, *y* is empty, or *y* is not produced by any node.
RuntimeError
    If an op_type is not found in the registry on the path from inputs to *y*.
)doc");
}
