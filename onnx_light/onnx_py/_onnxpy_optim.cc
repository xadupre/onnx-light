#include "onnx_optim/expressions.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_inference.h"
#include "onnx_optim/shapes/shapes_context.h"
#include <nanobind/operators.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>
#include <vector>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;

void AddOnnxPyExpressions(nb::module_ &m);
void AddOnnxPyShapeInference(nb::module_ &m);

NB_MODULE(_onnxpyoptim, m) {
  m.doc() = "onnx optim bindings from python: symbolic dimension expressions and "
            "shape inference helpers (operating on the same proto format).";
  AddOnnxPyExpressions(m);
  AddOnnxPyShapeInference(m);
}

void AddOnnxPyExpressions(nb::module_ &m) {
  // -----------------------------------------------------------------------
  // Submodule `expressions`
  // Symbolic dimension-expression utilities (simplify, evaluate, rename).
  // -----------------------------------------------------------------------
  {
    namespace expr = ::onnx_light::onnx_optim::expressions;

    auto expressions_mod = m.def_submodule("expressions");
    expressions_mod.doc() =
        "Symbolic dimension expression utilities: simplify, evaluate, and rename expressions.";

    // simplify_expression(str | int) -> str | int
    expressions_mod.def(
        "simplify_expression",
        [](nb::object arg) -> nb::object {
          if (nb::isinstance<nb::int_>(arg)) {
            int64_t v = nb::cast<int64_t>(arg);
            auto r = expr::simplify_expression(v);
            return std::holds_alternative<int64_t>(r) ? nb::cast(std::get<int64_t>(r))
                                                      : nb::cast(std::get<std::string>(r));
          }
          std::string s = nb::cast<std::string>(arg);
          auto r = expr::simplify_expression(s);
          return std::holds_alternative<int64_t>(r) ? nb::cast(std::get<int64_t>(r))
                                                    : nb::cast(std::get<std::string>(r));
        },
        nb::arg("expr"),
        "Simplifies a symbolic or integer expression. Returns int when fully numeric, "
        "otherwise a simplified string.");

    // simplify_two_expressions(expr1, expr2) -> dict[str, int]
    expressions_mod.def(
        "simplify_two_expressions",
        [](const std::string &e1, const std::string &e2) {
          return expr::simplify_two_expressions(e1, e2);
        },
        nb::arg("expr1"), nb::arg("expr2"),
        "Returns the non-zero coefficient map of (expr1) - (expr2).");

    // evaluate_expression(expr, context) -> int
    expressions_mod.def(
        "evaluate_expression",
        [](const std::string &e, const std::unordered_map<std::string, int64_t> &ctx) {
          return expr::evaluate_expression(e, ctx);
        },
        nb::arg("expression"), nb::arg("context"),
        "Evaluates an expression given variable assignments. Returns an integer.");

    // parse_expression_tokens(expr) -> set[str]
    expressions_mod.def(
        "parse_expression_tokens",
        [](const std::string &e) { return expr::parse_expression_tokens(e); }, nb::arg("expr"),
        "Returns the set of variable names referenced in the expression.");

    // rename_expression(expr, mapping) -> str
    expressions_mod.def(
        "rename_expression",
        [](const std::string &e, const std::unordered_map<std::string, std::string> &m) {
          return expr::rename_expression(e, m);
        },
        nb::arg("expr"), nb::arg("mapping"),
        "Renames variables in an expression using the provided mapping.");

    // rename_dynamic_expression(expression, replacements) -> str
    expressions_mod.def(
        "rename_dynamic_expression",
        [](const std::string &e, const std::unordered_map<std::string, std::string> &r) {
          return expr::rename_dynamic_expression(e, r);
        },
        nb::arg("expression"), nb::arg("replacements"),
        "Renames variables and applies Max->xor conversion and simplification.");

    // dim_add / dim_sub / dim_mul / dim_div / dim_mod / dim_max / dim_min
    // Each accepts int | str for both arguments and returns int | str.
    auto to_dim = [](nb::object arg) -> expr::DimType {
      if (nb::isinstance<nb::int_>(arg))
        return nb::cast<int64_t>(arg);
      return nb::cast<std::string>(arg);
    };
    auto from_dim = [](const expr::DimType &d) -> nb::object {
      if (std::holds_alternative<int64_t>(d))
        return nb::cast(std::get<int64_t>(d));
      return nb::cast(std::get<std::string>(d));
    };

    expressions_mod.def(
        "dim_add",
        [to_dim, from_dim](nb::object a, nb::object b) {
          return from_dim(expr::dim_add(to_dim(a), to_dim(b)));
        },
        nb::arg("a"), nb::arg("b"), "Adds two dimensions.");

    expressions_mod.def(
        "dim_sub",
        [to_dim, from_dim](nb::object a, nb::object b) {
          return from_dim(expr::dim_sub(to_dim(a), to_dim(b)));
        },
        nb::arg("a"), nb::arg("b"), "Subtracts dimension b from a.");

    expressions_mod.def(
        "dim_mul",
        [to_dim, from_dim](nb::object a, nb::object b) {
          return from_dim(expr::dim_mul(to_dim(a), to_dim(b)));
        },
        nb::arg("a"), nb::arg("b"), "Multiplies two dimensions.");

    expressions_mod.def(
        "dim_multi_mul",
        [to_dim, from_dim](nb::list args) {
          std::vector<expr::DimType> dims;
          dims.reserve(nb::len(args));
          for (auto item : args)
            dims.push_back(to_dim(nb::cast<nb::object>(item)));
          return from_dim(expr::dim_multi_mul(dims));
        },
        nb::arg("args"), "Multiplies a sequence of dimensions.");

    expressions_mod.def(
        "dim_div",
        [to_dim, from_dim](nb::object a, nb::object b) {
          return from_dim(expr::dim_div(to_dim(a), to_dim(b)));
        },
        nb::arg("a"), nb::arg("b"), "Floor-divides dimension a by b.");

    expressions_mod.def(
        "dim_mod",
        [to_dim, from_dim](nb::object a, nb::object b) {
          return from_dim(expr::dim_mod(to_dim(a), to_dim(b)));
        },
        nb::arg("a"), nb::arg("b"), "Computes a modulo b.");

    expressions_mod.def(
        "dim_max",
        [to_dim, from_dim](nb::object a, nb::object b) {
          return from_dim(expr::dim_max(to_dim(a), to_dim(b)));
        },
        nb::arg("a"), nb::arg("b"), "Returns the maximum of two dimensions.");

    expressions_mod.def(
        "dim_min",
        [to_dim, from_dim](nb::object a, nb::object b) {
          return from_dim(expr::dim_min(to_dim(a), to_dim(b)));
        },
        nb::arg("a"), nb::arg("b"), "Returns the minimum of two dimensions.");
  }

  // -----------------------------------------------------------------------
  // Submodule `shape_inference`
  // (full set of bindings provided by AddOnnxPyShapeInference)
  // -----------------------------------------------------------------------
}

void AddOnnxPyShapeInference(nb::module_ &m) {
  namespace onnx_shapes = ::onnx_light::onnx_optim::shapes;
  using ::onnx_light::onnx_optim::DataTypeToTensorType;
  using ::onnx_light::onnx_optim::OptimDim;
  using ::onnx_light::onnx_optim::OptimShape;
  using ::onnx_light::onnx_optim::OptimTensor;
  using ::onnx_light::onnx_optim::TensorType;
  using ::onnx_light::onnx_optim::TensorTypeToDataType;

  auto shape_mod = m.def_submodule("shape_inference");
  shape_mod.doc() = "Shape-inference bindings backed by ``onnx_optim``: exposes ``ShapesContext``, "
                    "``ComputeShapeNode`` and the related ``ComputeShape{Graph,Model}`` / "
                    "``ApplyInferredShapesTo{Graph,Model}`` helpers, together with the value "
                    "types (``OptimDim``, ``OptimShape``, ``OptimTensor``) used to describe "
                    "tensor descriptors stored in the context.";

  // Convert an OptimDim to a Python object (int when concrete, str otherwise).
  auto dim_to_object = [](const OptimDim &d) -> nb::object {
    if (d.IsInt())
      return nb::cast(d.AsInt());
    return nb::cast(d.AsExpr());
  };

  // Convert a Python object (int | str) to an OptimDim.
  auto object_to_dim = [](nb::handle h) -> OptimDim {
    if (nb::isinstance<nb::int_>(h))
      return OptimDim(nb::cast<int64_t>(h));
    return OptimDim(nb::cast<std::string>(h));
  };

  // Convert an iterable of int|str into an OptimShape.
  auto iterable_to_shape = [object_to_dim](nb::handle dims) -> OptimShape {
    OptimShape shape;
    nb::iterator it = nb::iter(dims);
    nb::iterator end = nb::iterator::sentinel();
    for (; it != end; ++it)
      shape.PushBack(object_to_dim(*it));
    return shape;
  };

  // Convert an OptimShape to a Python list of int|str.
  auto shape_to_list = [dim_to_object](const OptimShape &s) -> nb::list {
    nb::list out;
    for (const auto &d : s.Dims())
      out.append(dim_to_object(d));
    return out;
  };

  // -----------------------------------------------------------------------
  // OptimDim
  // -----------------------------------------------------------------------
  nb::class_<OptimDim>(shape_mod, "OptimDim",
                       "A single shape dimension that is either a concrete integer or a "
                       "symbolic string expression.")
      .def(nb::init<>())
      .def(nb::init<int64_t>(), nb::arg("value"))
      .def(nb::init<std::string>(), nb::arg("expr"))
      .def("is_int", &OptimDim::IsInt,
           "Returns True when the dimension holds a concrete integer value.")
      .def("is_expr", &OptimDim::IsExpr,
           "Returns True when the dimension holds a symbolic string expression.")
      .def("as_int", &OptimDim::AsInt,
           "Returns the integer value. Raises if the dimension is symbolic.")
      .def(
          "as_expr", [](const OptimDim &d) -> std::string { return d.AsExpr(); },
          "Returns the symbolic expression. Raises if the dimension is an integer.")
      .def(
          "value", [dim_to_object](const OptimDim &d) -> nb::object { return dim_to_object(d); },
          "Returns the underlying value as either ``int`` or ``str``.")
      .def("__str__", &OptimDim::ToString)
      .def("__repr__",
           [](const OptimDim &d) {
             if (d.IsInt()) {
               return std::string("OptimDim(") + std::to_string(d.AsInt()) + ")";
             }
             return std::string("OptimDim('") + d.AsExpr() + "')";
           })
      .def(nb::self == nb::self)
      .def(nb::self != nb::self);

  // -----------------------------------------------------------------------
  // OptimShape
  // -----------------------------------------------------------------------
  nb::class_<OptimShape>(
      shape_mod, "OptimShape",
      "Ordered, bounded-rank collection of OptimDim entries describing a tensor shape.")
      .def(nb::init<>())
      .def(
          "__init__",
          [iterable_to_shape](OptimShape *self, nb::handle dims) {
            new (self) OptimShape(iterable_to_shape(dims));
          },
          nb::arg("dims"), "Constructs a shape from an iterable of ``int`` or ``str`` dimensions.")
      .def("rank", &OptimShape::Rank, "Number of dimensions.")
      .def("empty", &OptimShape::Empty, "True when the shape is rank-0.")
      .def("is_fully_known", &OptimShape::IsFullyKnown,
           "True when every dimension is a concrete integer.")
      .def(
          "dims", [shape_to_list](const OptimShape &s) -> nb::list { return shape_to_list(s); },
          "Returns the dimensions as a list of ``int`` or ``str``.")
      .def("__len__", &OptimShape::Rank)
      .def(
          "__getitem__",
          [dim_to_object](const OptimShape &s, std::size_t i) -> nb::object {
            return dim_to_object(s[i]);
          },
          nb::arg("i"))
      .def("__iter__", [shape_to_list](const OptimShape &s) { return nb::iter(shape_to_list(s)); })
      .def("__str__", &OptimShape::ToString)
      .def("__repr__",
           [](const OptimShape &s) {
             std::string out = "OptimShape([";
             for (std::size_t i = 0; i < s.Rank(); ++i) {
               if (i > 0) {
                 out += ", ";
               }
               if (s[i].IsInt()) {
                 out += std::to_string(s[i].AsInt());
               } else {
                 out += "'";
                 out += s[i].AsExpr();
                 out += "'";
               }
             }
             out += "])";
             return out;
           })
      .def(nb::self == nb::self)
      .def(nb::self != nb::self);

  // -----------------------------------------------------------------------
  // OptimTensor
  // -----------------------------------------------------------------------
  nb::class_<OptimTensor>(
      shape_mod, "OptimTensor",
      "Lightweight (non-owning) tensor descriptor with an element type and an "
      "OptimShape, optionally annotated with a value-as-shape and value bounds. "
      "The Python binding never references a buffer; only the descriptor metadata "
      "is carried.")
      .def(nb::init<>())
      .def(
          "__init__",
          [iterable_to_shape](OptimTensor *self, int dtype, nb::handle dims) {
            TensorType t = DataTypeToTensorType(static_cast<TensorProto::DataType>(dtype));
            new (self) OptimTensor(nullptr, t, iterable_to_shape(dims));
          },
          nb::arg("dtype"), nb::arg("shape"),
          "Constructs an OptimTensor from a ``TensorProto.DataType`` integer and an "
          "iterable of ``int`` or ``str`` dimensions.")
      .def_prop_ro(
          "dtype",
          [](const OptimTensor &t) -> int {
            return static_cast<int>(TensorTypeToDataType(t.Dtype()));
          },
          "Element type as a ``TensorProto.DataType`` integer.")
      .def_prop_ro(
          "shape", [](const OptimTensor &t) -> OptimShape { return t.Shape(); },
          "Shape (a copy of the underlying OptimShape).")
      .def("is_null", &OptimTensor::IsNull, "True when no data buffer is attached.")
      .def("has_min", &OptimTensor::HasMin)
      .def("has_max", &OptimTensor::HasMax)
      .def("min", &OptimTensor::Min)
      .def("max", &OptimTensor::Max)
      .def("set_min", &OptimTensor::SetMin, nb::arg("value"))
      .def("set_max", &OptimTensor::SetMax, nb::arg("value"))
      .def("clear_min", &OptimTensor::ClearMin)
      .def("clear_max", &OptimTensor::ClearMax)
      .def("has_value_as_shape", &OptimTensor::HasValueAsShape,
           "True when the tensor's value is interpreted as a shape.")
      .def(
          "value_as_shape", [](const OptimTensor &t) -> OptimShape { return t.ValueAsShape(); },
          "Returns the value-as-shape annotation. Raises if not set.")
      .def(
          "set_value_as_shape",
          [iterable_to_shape](OptimTensor &t, nb::handle dims) {
            t.SetValueAsShape(iterable_to_shape(dims));
          },
          nb::arg("shape"),
          "Tags the tensor as carrying a shape value (e.g. the ``shape`` input of "
          "``Reshape``) and stores that shape.")
      .def("clear_value_as_shape", &OptimTensor::ClearValueAsShape)
      .def("__str__", &OptimTensor::ToString)
      .def("__repr__", &OptimTensor::ToString)
      .def(nb::self == nb::self)
      .def(nb::self != nb::self);

  // -----------------------------------------------------------------------
  // ShapesContext
  // -----------------------------------------------------------------------
  nb::class_<onnx_shapes::ShapesContext>(
      shape_mod, "ShapesContext",
      "In/out container shared by the per-operator ``ComputeShape*`` shape-inference "
      "functions. Holds a ``name -> OptimTensor`` map, a ``name -> OptimSequence`` map "
      "and a ``domain -> opset_version`` map mirroring ``opset_import``.")
      .def(nb::init<>())
      // Tensor descriptors
      .def(
          "set",
          [](onnx_shapes::ShapesContext &c, const std::string &name, OptimTensor t) {
            c.Set(name, std::move(t));
          },
          nb::arg("name"), nb::arg("tensor"),
          "Inserts or replaces the tensor descriptor stored under ``name``.")
      .def("has", &onnx_shapes::ShapesContext::Has, nb::arg("name"),
           "True when a tensor descriptor is stored under ``name``.")
      .def(
          "get",
          [](const onnx_shapes::ShapesContext &c, const std::string &name) -> OptimTensor {
            return c.Get(name);
          },
          nb::arg("name"),
          "Returns (a copy of) the tensor descriptor stored under ``name``. "
          "Raises KeyError when no such entry exists.")
      .def("size", &onnx_shapes::ShapesContext::Size,
           "Number of tensor descriptors currently stored.")
      .def("empty", &onnx_shapes::ShapesContext::Empty, "True when no entries are stored.")
      .def("clear", &onnx_shapes::ShapesContext::Clear,
           "Removes every entry (tensors, sequences and opset versions).")
      .def(
          "names",
          [](const onnx_shapes::ShapesContext &c) -> nb::list {
            nb::list out;
            for (const auto &kv : c.Tensors())
              out.append(kv.first);
            return out;
          },
          "Returns the list of names of stored tensor descriptors.")
      // Sequence descriptors
      .def("has_sequence", &onnx_shapes::ShapesContext::HasSequence, nb::arg("name"),
           "True when a sequence descriptor is stored under ``name``.")
      .def("sequences_size", &onnx_shapes::ShapesContext::SequencesSize,
           "Number of sequence descriptors currently stored.")
      .def(
          "sequence_names",
          [](const onnx_shapes::ShapesContext &c) -> nb::list {
            nb::list out;
            for (const auto &kv : c.Sequences())
              out.append(kv.first);
            return out;
          },
          "Returns the list of names of stored sequence descriptors.")
      // Opset versions
      .def("set_opset_version", &onnx_shapes::ShapesContext::SetOpsetVersion, nb::arg("domain"),
           nb::arg("opset_version"),
           "Records the opset version for ``domain``. An empty domain is normalised "
           "to ``ai.onnx``.")
      .def("has_opset_version", &onnx_shapes::ShapesContext::HasOpsetVersion, nb::arg("domain"),
           "True when an opset version has been recorded for ``domain``.")
      .def("opset_version", &onnx_shapes::ShapesContext::OpsetVersion, nb::arg("domain"),
           "Returns the recorded opset version of ``domain``, or "
           "``kUnknownOpsetVersion`` (-1) when none was recorded.")
      .def(
          "opsets",
          [](const onnx_shapes::ShapesContext &c) -> std::unordered_map<std::string, int> {
            return c.Opsets();
          },
          "Returns a copy of the ``domain -> opset_version`` map.");

  shape_mod.attr("kUnknownOpsetVersion") = onnx_shapes::kUnknownOpsetVersion;
  shape_mod.attr("kOnnxDomain") = onnx_shapes::kOnnxDomain;

  // -----------------------------------------------------------------------
  // Free functions
  // -----------------------------------------------------------------------
  shape_mod.def(
      "compute_shape_node",
      [](onnx_shapes::ShapesContext &ctx, const NodeProto &node) {
        onnx_shapes::ComputeShapeNode(ctx, node);
      },
      nb::arg("ctx"), nb::arg("node"),
      "Dispatches a single ``NodeProto`` to the matching per-operator "
      "``ComputeShape*`` function and stores the resulting output tensor "
      "descriptors in ``ctx``. The node's input descriptors must already be "
      "present in ``ctx``.");

  shape_mod.def(
      "check_inputs_available",
      [](const onnx_shapes::ShapesContext &ctx, const NodeProto &node) {
        onnx_shapes::CheckInputsAvailable(ctx, node);
      },
      nb::arg("ctx"), nb::arg("node"),
      "Raises ``ValueError`` if any non-empty input name declared by ``node`` is "
      "missing from ``ctx``.");

  shape_mod.def(
      "check_outputs_not_available",
      [](const onnx_shapes::ShapesContext &ctx, const NodeProto &node) {
        onnx_shapes::CheckOutputsNotAvailable(ctx, node);
      },
      nb::arg("ctx"), nb::arg("node"),
      "Raises ``ValueError`` if any non-empty output name declared by ``node`` "
      "already has an entry in ``ctx``.");

  shape_mod.def(
      "compute_shape_graph",
      [](onnx_shapes::ShapesContext &ctx, const GraphProto &graph) {
        onnx_shapes::ComputeShapeGraph(ctx, graph);
      },
      nb::arg("ctx"), nb::arg("graph"),
      "Seeds ``ctx`` from the initializers and inputs of ``graph`` and then runs "
      "``compute_shape_node`` on every node in topological order.");

  shape_mod.def(
      "compute_shape_model",
      [](onnx_shapes::ShapesContext &ctx, const ModelProto &model) {
        onnx_shapes::ComputeShapeModel(ctx, model);
      },
      nb::arg("ctx"), nb::arg("model"),
      "Records every ``(domain, version)`` pair from ``model.opset_import`` in "
      "``ctx`` and delegates to ``compute_shape_graph``.");

  shape_mod.def(
      "apply_inferred_shapes_to_graph",
      [](const onnx_shapes::ShapesContext &ctx, GraphProto &graph) {
        onnx_shapes::ApplyInferredShapesToGraph(ctx, graph);
      },
      nb::arg("ctx"), nb::arg("graph"),
      "Writes the shape and element-type descriptors stored in ``ctx`` back into "
      "``graph.output`` and ``graph.value_info``.");

  shape_mod.def(
      "apply_inferred_shapes_to_model",
      [](const onnx_shapes::ShapesContext &ctx, ModelProto &model) {
        onnx_shapes::ApplyInferredShapesToModel(ctx, model);
      },
      nb::arg("ctx"), nb::arg("model"),
      "Writes the shape and element-type descriptors stored in ``ctx`` back into "
      "``model.graph``.");

  shape_mod.def(
      "infer_shapes_model", [](ModelProto &model) { onnx_shapes::InferShapesModel(model); },
      nb::arg("model"),
      "Runs shape inference on ``model`` and writes the inferred element types and shapes "
      "back into ``model.graph.output`` and ``model.graph.value_info``. The ModelProto is "
      "mutated in place.");
}
