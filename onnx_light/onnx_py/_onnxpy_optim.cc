#include "onnx_optim/annotations/inplace_reuse.h"
#include "onnx_optim/annotations/value_tags.h"
#include "onnx_optim/expressions.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_inference.h"
#include "onnx_optim/shapes/shapes_context.h"
#include <algorithm>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>
#include <sstream>
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

    // CompareResult — outcome of comparing two symbolic expressions.
    nb::enum_<expr::CompareResult>(
        expressions_mod, "CompareResult", nb::is_arithmetic(),
        "Outcome of :func:`compare_expressions`, assuming every symbolic token is "
        "positive or null.")
        .value("Smaller", expr::CompareResult::Smaller,
               "The first expression is always strictly smaller than the second.")
        .value("Equal", expr::CompareResult::Equal, "The two expressions are always equal.")
        .value("Greater", expr::CompareResult::Greater,
               "The first expression is always strictly greater than the second.")
        .value("Unknown", expr::CompareResult::Unknown,
               "The relationship cannot be determined for all non-negative token values.");

    // ExpressionComparison — result + simplified difference of compare_expressions.
    nb::class_<expr::ExpressionComparison>(
        expressions_mod, "ExpressionComparison",
        "Result of :func:`compare_expressions`. Holds the :class:`CompareResult` "
        "``result`` together with the simplified ``difference`` (expr2) - (expr1), "
        "an ``int`` when numeric and a ``str`` otherwise.")
        .def_ro("result", &expr::ExpressionComparison::result,
                ":class:`CompareResult` describing how ``expr1`` compares to ``expr2``.")
        .def_prop_ro(
            "difference",
            [](const expr::ExpressionComparison &c) -> nb::object {
              return std::holds_alternative<int64_t>(c.difference)
                         ? nb::cast(std::get<int64_t>(c.difference))
                         : nb::cast(std::get<std::string>(c.difference));
            },
            "Simplified value of (expr2) - (expr1); ``int`` when numeric, ``str`` otherwise.")
        .def("__repr__", [](const expr::ExpressionComparison &c) {
          const char *name = "Unknown";
          switch (c.result) {
          case expr::CompareResult::Smaller:
            name = "Smaller";
            break;
          case expr::CompareResult::Equal:
            name = "Equal";
            break;
          case expr::CompareResult::Greater:
            name = "Greater";
            break;
          case expr::CompareResult::Unknown:
            name = "Unknown";
            break;
          }
          std::string diff = std::holds_alternative<int64_t>(c.difference)
                                 ? std::to_string(std::get<int64_t>(c.difference))
                                 : "'" + std::get<std::string>(c.difference) + "'";
          return std::string("ExpressionComparison(result=CompareResult.") + name +
                 ", difference=" + diff + ")";
        });

    // compare_expressions(expr1, expr2) -> ExpressionComparison
    expressions_mod.def(
        "compare_expressions",
        [](const std::string &e1, const std::string &e2) {
          return expr::compare_expressions(e1, e2);
        },
        nb::arg("expr1"), nb::arg("expr2"),
        "Compares expr1 to expr2 assuming all tokens are positive or null. Returns an "
        ":class:`ExpressionComparison` whose ``result`` is a :class:`CompareResult` and whose "
        "``difference`` is the simplified value of (expr2) - (expr1).");

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
        "dim_exact_div",
        [to_dim, from_dim](nb::object a, nb::object b) {
          return from_dim(expr::dim_exact_div(to_dim(a), to_dim(b)));
        },
        nb::arg("a"), nb::arg("b"),
        "Exactly divides dimension a by b, asserting the division has no remainder. "
        "Unlike floor division (//), exact division (/: ) commutes with multiplication: "
        "c*(a/:b) == (c*a)/:b, allowing the simplifier to cancel common factors more "
        "aggressively.");

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
      .def(nb::self != nb::self)
      .def(
          "__eq__",
          [](const OptimTensor &t, const ValueInfoProto &vi) {
            OptimTensor vi_tensor;
            return ::onnx_light::onnx_optim::OptimTensorFromValueInfo(vi, vi_tensor) &&
                   t == vi_tensor;
          },
          nb::arg("other"),
          "Compares this descriptor against a ``ValueInfoProto`` by converting the "
          "value-info tensor type/shape into an ``OptimTensor`` and checking "
          "descriptor equality.");

  // -----------------------------------------------------------------------
  // ShapeEventAction — enum classifying a ShapeEvent record.
  // Mirrors :cpp:enum:`onnx_optim::shapes::ShapeEventAction`.
  // -----------------------------------------------------------------------
  nb::enum_<onnx_shapes::ShapeEventAction>(shape_mod, "ShapeEventAction", nb::is_arithmetic(),
                                           "Action kind recorded in a :class:`ShapeEvent`. "
                                           "``kAdd`` / ``kReplace`` mark tensor-descriptor "
                                           "mutations; ``kComputeNode`` marks the dispatch of "
                                           "a single shape-inference node; ``kConstraint`` / "
                                           "``kConstraintMax`` mark newly inserted symbolic "
                                           "dimension constraints.")
      .value("kAdd", onnx_shapes::ShapeEventAction::kAdd,
             "A new tensor descriptor was added to the context.")
      .value("kReplace", onnx_shapes::ShapeEventAction::kReplace,
             "An existing tensor descriptor was replaced.")
      .value("kComputeNode", onnx_shapes::ShapeEventAction::kComputeNode,
             "Shape inference was dispatched for a node.")
      .value("kConstraint", onnx_shapes::ShapeEventAction::kConstraint,
             "A new equality constraint between two symbolic dimensions was inserted.")
      .value("kConstraintMax", onnx_shapes::ShapeEventAction::kConstraintMax,
             "A new less-than-or-equal constraint between two symbolic dimensions was inserted.");

  // -----------------------------------------------------------------------
  // ShapeEvent — append-only log entry for a single shape-inference event.
  // Mirrors :cpp:class:`onnx_optim::shapes::ShapeEvent`; ``shape`` is exposed
  // as a list of per-dimension strings so symbolic dims are preserved.
  // -----------------------------------------------------------------------
  nb::class_<onnx_shapes::ShapeEvent>(
      shape_mod, "ShapeEvent",
      "One entry of the :meth:`ShapesContext.events` log. ``add`` / ``replace`` "
      "events describe a tensor descriptor mutation performed through "
      "``ShapesContext.set`` and carry the descriptor ``name``, ``data_type`` "
      "(a ``TensorProto.DataType`` integer) and ``shape`` (a list of per-dimension "
      "strings, preserving symbolic dims). ``compute_node`` events summarise the "
      "shape-inference dispatch of a single node and carry ``op_domain``, "
      "``op_type`` and ``inputs`` instead. ``constraint`` / ``constraint_max`` "
      "events record a newly inserted symbolic-dimension constraint and carry its "
      "two operands in ``inputs``.")
      .def_prop_ro(
          "action", [](const onnx_shapes::ShapeEvent &ev) { return ev.action; },
          ":class:`ShapeEventAction` member describing the event kind: "
          "``kAdd``, ``kReplace``, ``kComputeNode``, ``kConstraint`` or "
          "``kConstraintMax``.")
      .def_ro("name", &onnx_shapes::ShapeEvent::name,
              "Value name targeted by the mutation. Empty for ``compute_node`` / "
              "``constraint`` / ``constraint_max`` events.")
      .def_ro("data_type", &onnx_shapes::ShapeEvent::data_type,
              "``TensorProto.DataType`` integer of the descriptor, or ``UNDEFINED`` (0) "
              "for ``compute_node`` / ``constraint`` / ``constraint_max`` events.")
      .def_ro("shape", &onnx_shapes::ShapeEvent::shape,
              "Descriptor shape as a list of per-dimension strings (decimal integers for "
              "concrete dims, symbolic expressions otherwise). Empty for ``compute_node`` / "
              "``constraint`` / ``constraint_max`` events.")
      .def_ro("op_domain", &onnx_shapes::ShapeEvent::op_domain,
              "For ``compute_node`` events: normalised ONNX op domain of the dispatched "
              "node (default domain reported as ``\"ai.onnx\"``). Empty otherwise.")
      .def_ro("op_type", &onnx_shapes::ShapeEvent::op_type,
              "For ``compute_node`` events: ONNX ``op_type`` of the dispatched node. "
              "Empty otherwise.")
      .def_ro("inputs", &onnx_shapes::ShapeEvent::inputs,
              "For ``compute_node`` events: ordered list of input names consumed by the "
              "node (matching ``NodeProto.input``). For ``constraint`` / ``constraint_max`` "
              "events: the two constraint operands. Empty otherwise.")
      .def_ro("node_index", &onnx_shapes::ShapeEvent::node_index,
              "Index of the node this event is associated with: ``-1`` for graph inputs, "
              "``-2`` for initializers, and the position (``>= 0``) of the producing / "
              "dispatched node otherwise (``-1`` when no producing node is known).")
      .def_ro("subgraph_node_index", &onnx_shapes::ShapeEvent::subgraph_node_index,
              "Index of the control-flow node in the parent graph whose attribute subgraph "
              "produced this event. ``-1`` for top-level-graph events.")
      .def_ro("subgraph_attr_name", &onnx_shapes::ShapeEvent::subgraph_attr_name,
              "Attribute name of the subgraph within the owning control-flow node "
              "(``\"body\"``, ``\"then_branch\"``, ``\"else_branch\"``, etc.). "
              "Empty for top-level-graph events.")
      .def(
          "as_dict",
          [](const onnx_shapes::ShapeEvent &ev) {
            nb::dict d;
            d["action"] = std::string(onnx_shapes::ShapeEventActionName(ev.action));
            d["name"] = ev.name;
            d["data_type"] = ev.data_type;
            d["shape"] = ev.shape;
            d["op_domain"] = ev.op_domain;
            d["op_type"] = ev.op_type;
            d["inputs"] = ev.inputs;
            d["node_index"] = ev.node_index;
            d["subgraph_node_index"] = ev.subgraph_node_index;
            d["subgraph_attr_name"] = ev.subgraph_attr_name;
            return d;
          },
          "Returns the event fields as a plain Python ``dict`` (trivially "
          "renderable as a table, serialisable, etc.).")
      .def("__repr__", [](const onnx_shapes::ShapeEvent &ev) {
        return std::string("ShapeEvent(action='") + onnx_shapes::ShapeEventActionName(ev.action) +
               "', name='" + ev.name + "', op_type='" + ev.op_type +
               "', data_type=" + std::to_string(ev.data_type) +
               ", node_index=" + std::to_string(ev.node_index) +
               ", subgraph_node_index=" + std::to_string(ev.subgraph_node_index) +
               ", subgraph_attr_name='" + ev.subgraph_attr_name + "')";
      });

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
      .def(
          "has",
          [](const onnx_shapes::ShapesContext &c, const std::string &name) { return c.Has(name); },
          nb::arg("name"), "True when a tensor descriptor is stored under ``name``.")
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
      // Event logging (opt-in; mirrors ``RuntimeContext.events``).
      .def_prop_rw(
          "events_enabled", [](const onnx_shapes::ShapesContext &c) { return c.events_enabled(); },
          [](onnx_shapes::ShapesContext &c, bool v) { c.set_events_enabled(v); },
          "When ``True``, ``set`` records ``add`` / ``replace`` events, "
          "``compute_shape_node`` records a ``compute_node`` event per dispatched "
          "node, and ``add_constraint`` / ``add_less_equal_constraint`` record "
          "``constraint`` / ``constraint_max`` events. Default is ``False`` for "
          "maximum throughput; enable only when tracing shape inference.")
      .def(
          "events",
          [](const onnx_shapes::ShapesContext &c) -> nb::list {
            nb::list out;
            for (const auto &ev : c.Events())
              out.append(nb::cast(ev));
            return out;
          },
          "Returns the append-only shape-inference event log as a list of "
          ":class:`ShapeEvent` instances. Empty unless ``events_enabled`` was set "
          "before running shape inference.")
      .def(
          "clear_events", [](onnx_shapes::ShapesContext &c) { c.ClearEvents(); },
          "Empties the event log without otherwise touching the context.")
      .def(
          "__repr__",
          [](const onnx_shapes::ShapesContext &c) {
            std::vector<std::string> tensor_names;
            tensor_names.reserve(c.Tensors().size());
            for (const auto &kv : c.Tensors())
              tensor_names.push_back(kv.first);
            std::sort(tensor_names.begin(), tensor_names.end());

            std::vector<std::string> sequence_names;
            sequence_names.reserve(c.Sequences().size());
            for (const auto &kv : c.Sequences())
              sequence_names.push_back(kv.first);
            std::sort(sequence_names.begin(), sequence_names.end());

            std::vector<std::string> opset_domains;
            opset_domains.reserve(c.Opsets().size());
            for (const auto &kv : c.Opsets())
              opset_domains.push_back(kv.first);
            std::sort(opset_domains.begin(), opset_domains.end());

            std::ostringstream os;
            os << "ShapesContext(tensors=[";
            for (size_t i = 0; i < tensor_names.size(); ++i) {
              if (i > 0)
                os << ", ";
              os << "'" << tensor_names[i] << "'";
            }
            os << "], sequences=[";
            for (size_t i = 0; i < sequence_names.size(); ++i) {
              if (i > 0)
                os << ", ";
              os << "'" << sequence_names[i] << "'";
            }
            os << "], opsets={";
            for (size_t i = 0; i < opset_domains.size(); ++i) {
              if (i > 0)
                os << ", ";
              const auto &domain = opset_domains[i];
              os << "'" << domain << "': " << c.OpsetVersion(domain);
            }
            os << "})";
            return os.str();
          },
          "Returns a deterministic representation of tensor names, sequence names "
          "and opset versions stored in this context.")
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
      .def(
          "has_sequence",
          [](const onnx_shapes::ShapesContext &c, const std::string &name) {
            return c.HasSequence(name);
          },
          nb::arg("name"), "True when a sequence descriptor is stored under ``name``.")
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
      // Child contexts retained for control-flow subgraphs.
      .def(
          "has_subgraph_context",
          [](const onnx_shapes::ShapesContext &c, int64_t node_index,
             const std::string &attr_name) { return c.HasSubgraphContext(node_index, attr_name); },
          nb::arg("node_index"), nb::arg("attr_name"),
          "True when a child context was retained for the subgraph ``attr_name`` of the "
          "control-flow node at ``node_index``.")
      .def(
          "subgraph_context",
          [](const onnx_shapes::ShapesContext &c, int64_t node_index,
             const std::string &attr_name) -> const onnx_shapes::ShapesContext & {
            return c.GetSubgraphContext(node_index, attr_name);
          },
          nb::arg("node_index"), nb::arg("attr_name"), nb::rv_policy::reference_internal,
          "Returns the child :class:`ShapesContext` retained for the subgraph ``attr_name`` of "
          "the control-flow node at ``node_index``. Raises ``IndexError`` if absent.")
      .def("subgraph_contexts_size", &onnx_shapes::ShapesContext::SubgraphContextsSize,
           "Number of retained child contexts.")
      .def(
          "subgraph_context_keys",
          [](const onnx_shapes::ShapesContext &c) -> nb::list {
            nb::list out;
            for (const auto &kv : c.SubgraphContexts())
              out.append(nb::make_tuple(kv.first.first, kv.first.second));
            return out;
          },
          "Returns retained child-context keys as ``(node_index, attr_name)`` tuples.")
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
          "Returns a copy of the ``domain -> opset_version`` map.")
      // Custom shape-inference callbacks.
      .def(
          "set_custom_shape_inference_function",
          [](onnx_shapes::ShapesContext &c, const std::string &domain, const std::string &op_type,
             nb::callable fn) {
            c.SetCustomShapeInferenceFunction(
                domain, op_type,
                [py_fn = std::move(fn)](onnx_shapes::ShapesContext &ctx, const NodeProto &node) {
                  nb::gil_scoped_acquire lock;
                  py_fn(nb::cast(&ctx, nb::rv_policy::reference),
                        nb::cast(&node, nb::rv_policy::reference));
                });
          },
          nb::arg("domain"), nb::arg("op_type"), nb::arg("fn"),
          "Registers a Python callback for ``(domain, op_type)``. The callback receives "
          "``(ctx, node)`` and can populate output descriptors in ``ctx``. An empty "
          "``domain`` is normalised to ``ai.onnx``.")
      .def(
          "has_custom_shape_inference_function",
          [](const onnx_shapes::ShapesContext &c, const std::string &domain,
             const std::string &op_type) {
            return c.GetCustomShapeInferenceFunction(domain, op_type) != nullptr;
          },
          nb::arg("domain"), nb::arg("op_type"),
          "True when a custom callback is registered for ``(domain, op_type)``.")
      .def(
          "custom_shape_inference_keys",
          [](const onnx_shapes::ShapesContext &c) -> nb::list {
            nb::list out;
            for (const auto &kv : c.CustomShapeInferenceFunctions())
              out.append(kv.first);
            return out;
          },
          "Returns registered custom callback keys as ``\"<domain>:<op_type>\"``.")
      // Symbolic-dimension equality constraints.
      .def(
          "add_constraint",
          [](onnx_shapes::ShapesContext &c, const std::string &a, const std::string &b) {
            return c.AddConstraint(a, b);
          },
          nb::arg("a"), nb::arg("b"),
          "Records an equality constraint between two symbolic dimension names. "
          "The pair is canonicalised so ``(a, b)`` and ``(b, a)`` are stored only "
          "once, and ``a == a`` is dropped. Returns True when a new constraint was "
          "inserted, False otherwise.")
      .def(
          "has_constraint",
          [](const onnx_shapes::ShapesContext &c, const std::string &a, const std::string &b) {
            return c.HasConstraint(a, b);
          },
          nb::arg("a"), nb::arg("b"),
          "True when an equality constraint between ``a`` and ``b`` was recorded "
          "(canonical order is applied before lookup; ``a == a`` always returns True).")
      .def("constraints_size", &onnx_shapes::ShapesContext::ConstraintsSize,
           "Number of recorded equality constraints.")
      .def(
          "constraints",
          [](const onnx_shapes::ShapesContext &c) -> nb::list {
            nb::list out;
            for (const auto &p : c.Constraints()) {
              out.append(nb::make_tuple(p.first, p.second));
            }
            return out;
          },
          "Returns the list of recorded equality constraints as ``(lhs, rhs)`` "
          "tuples with ``lhs < rhs``.")
      // Symbolic-dimension upper-bound (less-or-equal) constraints.
      .def(
          "add_less_equal_constraint",
          [](onnx_shapes::ShapesContext &c, const std::string &lhs, const std::string &rhs) {
            return c.AddLessEqualConstraint(lhs, rhs);
          },
          nb::arg("lhs"), nb::arg("rhs"),
          "Records that the symbolic dimension ``lhs`` is less than or equal to "
          "the dimension expression ``rhs``. ``lhs == rhs`` and empty operands "
          "are dropped. Returns True when a new constraint was inserted, False "
          "otherwise.")
      .def(
          "has_less_equal_constraint",
          [](const onnx_shapes::ShapesContext &c, const std::string &lhs, const std::string &rhs) {
            return c.HasLessEqualConstraint(lhs, rhs);
          },
          nb::arg("lhs"), nb::arg("rhs"),
          "True when a ``lhs <= rhs`` constraint was recorded "
          "(``lhs == rhs`` always returns True).")
      .def("less_equal_constraints_size", &onnx_shapes::ShapesContext::LessEqualConstraintsSize,
           "Number of recorded ``<=`` constraints.")
      .def(
          "less_equal_constraints",
          [](const onnx_shapes::ShapesContext &c) -> nb::list {
            nb::list out;
            for (const auto &p : c.LessEqualConstraints()) {
              out.append(nb::make_tuple(p.first, p.second));
            }
            return out;
          },
          "Returns the list of recorded ``<=`` constraints as ordered "
          "``(lhs, rhs)`` tuples meaning ``lhs <= rhs``.")
      // Shape-inference drivers (also exposed as module-level free functions).
      .def(
          "compute_shape_node",
          [](onnx_shapes::ShapesContext &c, const NodeProto &node) { c.ComputeShapeNode(node); },
          nb::arg("node"),
          "Dispatches a single ``NodeProto`` to the matching per-operator "
          "``ComputeShape*`` function and stores the resulting output tensor "
          "descriptors in ``self``. The node's input descriptors must already be "
          "present in ``self``.")
      .def(
          "check_inputs_available",
          [](const onnx_shapes::ShapesContext &c, const NodeProto &node) {
            c.CheckInputsAvailable(node);
          },
          nb::arg("node"),
          "Raises ``ValueError`` if any non-empty input name declared by ``node`` is "
          "missing from ``self``.")
      .def(
          "check_outputs_not_available",
          [](const onnx_shapes::ShapesContext &c, const NodeProto &node) {
            c.CheckOutputsNotAvailable(node);
          },
          nb::arg("node"),
          "Raises ``ValueError`` if any non-empty output name declared by ``node`` "
          "already has an entry in ``self``.")
      .def(
          "compute_shape_graph",
          [](onnx_shapes::ShapesContext &c, const GraphProto &graph) {
            c.ComputeShapeGraph(graph);
          },
          nb::arg("graph"),
          "Seeds ``self`` from the initializers and inputs of ``graph`` and then runs "
          "``compute_shape_node`` on every node in topological order.")
      .def(
          "compute_shape_model",
          [](onnx_shapes::ShapesContext &c, const ModelProto &model,
             bool prefill_with_value_info_output) {
            c.ComputeShapeModel(model, prefill_with_value_info_output);
          },
          nb::arg("model"), nb::arg("prefill_with_value_info_output") = false,
          "Records every ``(domain, version)`` pair from ``model.opset_import`` in "
          "``self`` and delegates to ``compute_shape_graph``. When "
          "``prefill_with_value_info_output`` is true, tensor descriptors from "
          "``model.graph.value_info`` and ``model.graph.output`` are added as "
          "anchors and preferred when there is a non-conflicting choice at the end.")
      .def(
          "apply_inferred_shapes_to_graph",
          [](const onnx_shapes::ShapesContext &c, GraphProto &graph) {
            c.ApplyInferredShapesToGraph(graph);
          },
          nb::arg("graph"),
          "Writes the shape and element-type descriptors stored in ``self`` back into "
          "``graph.output`` and ``graph.value_info``.")
      .def(
          "apply_inferred_shapes_to_model",
          [](const onnx_shapes::ShapesContext &c, ModelProto &model) {
            c.ApplyInferredShapesToModel(model);
          },
          nb::arg("model"),
          "Writes the shape and element-type descriptors stored in ``self`` back into "
          "``model.graph``.");

  shape_mod.attr("kUnknownOpsetVersion") = onnx_shapes::kUnknownOpsetVersion;
  shape_mod.attr("kOnnxDomain") = onnx_shapes::kOnnxDomain;

  // -----------------------------------------------------------------------
  // Free functions
  // -----------------------------------------------------------------------
  shape_mod.def(
      "compute_shape_node",
      [](onnx_shapes::ShapesContext &ctx, const NodeProto &node) { ctx.ComputeShapeNode(node); },
      nb::arg("ctx"), nb::arg("node"),
      "Dispatches a single ``NodeProto`` to the matching per-operator "
      "``ComputeShape*`` function and stores the resulting output tensor "
      "descriptors in ``ctx``. The node's input descriptors must already be "
      "present in ``ctx``.");

  shape_mod.def(
      "check_inputs_available",
      [](const onnx_shapes::ShapesContext &ctx, const NodeProto &node) {
        ctx.CheckInputsAvailable(node);
      },
      nb::arg("ctx"), nb::arg("node"),
      "Raises ``ValueError`` if any non-empty input name declared by ``node`` is "
      "missing from ``ctx``.");

  shape_mod.def(
      "check_outputs_not_available",
      [](const onnx_shapes::ShapesContext &ctx, const NodeProto &node) {
        ctx.CheckOutputsNotAvailable(node);
      },
      nb::arg("ctx"), nb::arg("node"),
      "Raises ``ValueError`` if any non-empty output name declared by ``node`` "
      "already has an entry in ``ctx``.");

  shape_mod.def(
      "compute_shape_graph",
      [](onnx_shapes::ShapesContext &ctx, const GraphProto &graph) {
        ctx.ComputeShapeGraph(graph);
      },
      nb::arg("ctx"), nb::arg("graph"),
      "Seeds ``ctx`` from the initializers and inputs of ``graph`` and then runs "
      "``compute_shape_node`` on every node in topological order.");

  shape_mod.def(
      "compute_shape_model",
      [](onnx_shapes::ShapesContext &ctx, const ModelProto &model,
         bool prefill_with_value_info_output) {
        ctx.ComputeShapeModel(model, prefill_with_value_info_output);
      },
      nb::arg("ctx"), nb::arg("model"), nb::arg("prefill_with_value_info_output") = false,
      "Records every ``(domain, version)`` pair from ``model.opset_import`` in "
      "``ctx`` and delegates to ``compute_shape_graph``. When "
      "``prefill_with_value_info_output`` is true, tensor descriptors from "
      "``model.graph.value_info`` and ``model.graph.output`` are added as "
      "anchors and preferred when there is a non-conflicting choice at the end.");

  shape_mod.def(
      "apply_inferred_shapes_to_graph",
      [](const onnx_shapes::ShapesContext &ctx, GraphProto &graph) {
        ctx.ApplyInferredShapesToGraph(graph);
      },
      nb::arg("ctx"), nb::arg("graph"),
      "Writes the shape and element-type descriptors stored in ``ctx`` back into "
      "``graph.output`` and ``graph.value_info``.");

  shape_mod.def(
      "apply_inferred_shapes_to_model",
      [](const onnx_shapes::ShapesContext &ctx, ModelProto &model) {
        ctx.ApplyInferredShapesToModel(model);
      },
      nb::arg("ctx"), nb::arg("model"),
      "Writes the shape and element-type descriptors stored in ``ctx`` back into "
      "``model.graph``.");

  shape_mod.def(
      "infer_shapes_model",
      [](ModelProto &model, bool prefill_with_value_info_output) {
        onnx_shapes::InferShapesModel(model, prefill_with_value_info_output);
      },
      nb::arg("model"), nb::arg("prefill_with_value_info_output") = false,
      "Runs shape inference on ``model`` and writes the inferred element types and shapes "
      "back into ``model.graph.output`` and ``model.graph.value_info``. The ModelProto is "
      "mutated in place. When ``prefill_with_value_info_output`` is true, existing "
      "``value_info``/``output`` tensor descriptors are used as anchors.");

  // -----------------------------------------------------------------------
  // In-place reuse analysis
  // -----------------------------------------------------------------------
  shape_mod.attr("INPLACE_REUSE_METADATA_KEY") = onnx_shapes::kInPlaceReuseMetadataKey;
  shape_mod.attr("RELEASE_AFTER_METADATA_KEY") = onnx_shapes::kReleaseAfterMetadataKey;
  shape_mod.attr("VALUE_TAG_METADATA_KEY") = onnx_shapes::kValueTagMetadataKey;
  shape_mod.attr("VALUE_TAGS_METADATA_KEY") = onnx_shapes::kValueTagsMetadataKey;
  shape_mod.attr("NODE_TAG_METADATA_KEY") = onnx_shapes::kNodeTagMetadataKey;

  nb::enum_<onnx_shapes::InPlaceReuseKind>(
      shape_mod, "InPlaceReuseKind", nb::is_arithmetic(),
      "Classifies how the reused input buffer compares in size with the output: "
      "``kEqual`` when the input and output share the same element type and shape "
      "(same byte size, the preferred reuse); ``kGreater`` when the input buffer is "
      "strictly larger in bytes than the output.")
      .value("kEqual", onnx_shapes::InPlaceReuseKind::kEqual,
             "The input and output have the same byte size.")
      .value("kGreater", onnx_shapes::InPlaceReuseKind::kGreater,
             "The input buffer is strictly larger in bytes than the output.");

  nb::class_<onnx_shapes::InPlaceReuse>(
      shape_mod, "InPlaceReuse",
      "Represents one in-place reuse opportunity for a node: the output at ``output_index`` "
      "reuses the buffer of the input at ``input_index`` (both indices into the node's "
      "``output``/``input`` lists). ``kind`` records whether the input buffer has the same "
      "size as the output (``kEqual``) or is strictly larger (``kGreater``).")
      .def(nb::init<>())
      .def_rw("output_index", &onnx_shapes::InPlaceReuse::output_index)
      .def_rw("input_index", &onnx_shapes::InPlaceReuse::input_index)
      .def_rw("kind", &onnx_shapes::InPlaceReuse::kind)
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def("__repr__", [](const onnx_shapes::InPlaceReuse &r) {
        std::ostringstream os;
        const char *kind = r.kind == onnx_shapes::InPlaceReuseKind::kEqual ? "kEqual" : "kGreater";
        os << "InPlaceReuse(output_index=" << r.output_index << ", input_index=" << r.input_index
           << ", kind=" << kind << ")";
        return os.str();
      });

  nb::class_<onnx_shapes::ComputeContext>(
      shape_mod, "ComputeContext",
      "Holds the in-place reuse opportunities computed for a graph, mirroring the way "
      "``ShapesContext`` holds inferred descriptors. Populate it with "
      "``compute_inplace_reuse_graph`` (consuming a ``ShapesContext``), then read the result "
      "through ``reuse`` / ``node_reuse`` or persist it with ``write_to_metadata``.")
      .def(nb::init<>())
      .def(
          "compute_inplace_reuse_graph",
          [](onnx_shapes::ComputeContext &self, const GraphProto &graph,
             const onnx_shapes::ShapesContext &ctx, bool allow_input_overwrite) {
            self.ComputeInPlaceReuseGraph(graph, ctx, allow_input_overwrite);
          },
          nb::arg("graph"), nb::arg("ctx"), nb::arg("allow_input_overwrite") = false,
          "Guesses, for every node of ``graph``, which outputs reuse which input buffers in "
          "place, using the shapes already inferred into ``ctx``, and stores the result in this "
          "context (replacing any previous result).\n\n"
          "By default declared graph inputs are never overwritten in place; set "
          "``allow_input_overwrite=True`` to let an input be reused like an intermediate.")
      .def_prop_ro(
          "reuse", [](const onnx_shapes::ComputeContext &self) { return self.Reuse(); },
          "The per-node reuse opportunities as a list (one entry per node, same order as "
          "``graph.node``); each entry is a list of :class:`InPlaceReuse`.")
      .def(
          "node_reuse",
          [](const onnx_shapes::ComputeContext &self, std::size_t node_index) {
            return self.NodeReuse(node_index);
          },
          nb::arg("node_index"),
          "Returns the list of :class:`InPlaceReuse` opportunities discovered for the node at "
          "``node_index``. Raises ``IndexError`` when ``node_index`` is out of bounds.")
      .def(
          "write_to_metadata",
          [](const onnx_shapes::ComputeContext &self, GraphProto &graph) {
            self.WriteToMetadata(graph);
          },
          nb::arg("graph"),
          "Records the computed opportunities into each node's ``metadata_props`` under the keys "
          "``onnx_light.inplace_reuse`` and ``onnx_light.release_after``. The ``GraphProto`` is "
          "mutated in place and must be the same graph passed to "
          "``compute_inplace_reuse_graph``.")
      .def("clear", &onnx_shapes::ComputeContext::Clear, "Empties the stored result.")
      .def("__len__", [](const onnx_shapes::ComputeContext &self) { return self.Size(); });

  shape_mod.def(
      "compute_inplace_reuse",
      [](const onnx_shapes::ShapesContext &ctx, const GraphProto &graph,
         bool allow_input_overwrite) {
        return onnx_shapes::ComputeInPlaceReuse(graph, ctx, allow_input_overwrite);
      },
      nb::arg("ctx"), nb::arg("graph"), nb::arg("allow_input_overwrite") = false,
      "Guesses, for every node of ``graph``, which outputs reuse which input buffers in "
      "place, using the shapes already inferred into ``ctx``.\n\n"
      "By default declared graph inputs are never overwritten in place; set "
      "``allow_input_overwrite=True`` to let an input be reused like an intermediate.\n\n"
      "Returns: a list with one entry per node (same order as ``graph.node``); each entry is a "
      "list of :class:`InPlaceReuse`. The analysis is purely structural (matching element type, "
      "shape and value lifetime) and does not check whether a given kernel actually supports "
      "in-place execution.");

  shape_mod.def(
      "write_inplace_reuse_to_metadata",
      [](const onnx_shapes::ShapesContext &ctx, GraphProto &graph) {
        onnx_shapes::WriteInPlaceReuseToMetadata(graph, ctx);
      },
      nb::arg("ctx"), nb::arg("graph"),
      "Computes the in-place reuse opportunities for ``graph`` (see "
      ":func:`compute_inplace_reuse`) and records them into each node's ``metadata_props`` "
      "under the keys ``onnx_light.inplace_reuse`` and ``onnx_light.release_after``. "
      "The ``GraphProto`` is mutated in place.\n\n"
      "For every node with at least one opportunity, a single metadata entry is added (or "
      "updated in place if the key already exists) whose value lists the opportunities as "
      "``output_index:input_index:kind`` triplets separated by ``;`` (``kind`` being ``equal`` "
      "or ``greater``). For every node with releasable last-use inputs, one metadata entry is "
      "added under ``onnx_light.release_after`` as a ``;``-separated name list.");

  auto copy_node_list = [](nb::list nodes) {
    std::vector<NodeProto> copied;
    copied.reserve(nodes.size());
    for (nb::handle h : nodes) {
      copied.push_back(nb::cast<const NodeProto &>(h));
    }
    return copied;
  };

  shape_mod.def(
      "infer_value_and_node_tags",
      [](const GraphProto &graph) {
        const auto inferred = onnx_shapes::InferValueAndNodeTags(graph);
        return nb::make_tuple(inferred.first, inferred.second);
      },
      nb::arg("graph"),
      "Infers semantic ``shape``/``axes``/``weight`` tags for values and nodes in ``graph``.");
  shape_mod.def(
      "infer_value_and_node_tags",
      [](const FunctionProto &function) {
        const auto inferred = onnx_shapes::InferValueAndNodeTags(function);
        return nb::make_tuple(inferred.first, inferred.second);
      },
      nb::arg("function"),
      "Infers semantic ``shape``/``axes``/``weight`` tags for values and nodes in ``function``.");
  shape_mod.def(
      "infer_value_and_node_tags",
      [copy_node_list](nb::list nodes) {
        const auto inferred = onnx_shapes::InferValueAndNodeTags(copy_node_list(nodes));
        return nb::make_tuple(inferred.first, inferred.second);
      },
      nb::arg("nodes"), "Infers semantic ``shape``/``axes``/``weight`` tags for a node list.");

  shape_mod.def(
      "write_value_and_node_tags_to_metadata",
      [](GraphProto &graph) { onnx_shapes::WriteValueAndNodeTagsToMetadata(graph); },
      nb::arg("graph"), "Writes inferred value/node tags into graph metadata.");
  shape_mod.def(
      "write_value_and_node_tags_to_metadata",
      [](FunctionProto &function) { onnx_shapes::WriteValueAndNodeTagsToMetadata(function); },
      nb::arg("function"), "Writes inferred value/node tags into function metadata.");
  shape_mod.def(
      "write_value_and_node_tags_to_metadata",
      [](ModelProto &model) { onnx_shapes::WriteValueAndNodeTagsToMetadata(model); },
      nb::arg("model"), "Writes inferred value/node tags into ``model.graph`` metadata.");
  shape_mod.def(
      "write_value_and_node_tags_to_metadata",
      [copy_node_list](nb::list nodes) {
        const auto inferred = onnx_shapes::InferValueAndNodeTags(copy_node_list(nodes));
        const auto &node_tags = inferred.second;
        std::size_t i = 0;
        for (nb::handle h : nodes) {
          if (i < node_tags.size() && !node_tags[i].empty()) {
            nb::cast<NodeProto &>(h).add_metadata(onnx_shapes::kNodeTagMetadataKey, node_tags[i]);
          }
          ++i;
        }
      },
      nb::arg("nodes"), "Writes inferred node tags for a node list.");
}
