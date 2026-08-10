#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/compute/constant_info.h"
#include "onnx_core/compute/inplace_reuse.h"
#include "onnx_core/compute/peak_memory.h"
#include "onnx_core/compute/value_tags.h"
#include "onnx_core/expressions/expressions.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/shapes/shape_inference.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_sequence.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_proto/onnx_helper.h"
#include <algorithm>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <sstream>
#include <vector>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;

namespace {

constexpr const char *InPlaceReuseKindName(core::compute::InPlaceReuseKind kind) {
  switch (kind) {
  case core::compute::InPlaceReuseKind::kEqual:
    return "equal";
  case core::compute::InPlaceReuseKind::kGreater:
    return "greater";
  }
  return "unknown";
}

constexpr const char *InPlaceReuseKindEnumName(core::compute::InPlaceReuseKind kind) {
  switch (kind) {
  case core::compute::InPlaceReuseKind::kEqual:
    return "kEqual";
  case core::compute::InPlaceReuseKind::kGreater:
    return "kGreater";
  }
  return "kUnknown";
}

} // namespace

void AddOnnxPyExpressions(nb::module_ &m);
void AddOnnxPyShapeInference(nb::module_ &m);
void AddOnnxPyBuilder(nb::module_ &m);

NB_MODULE(_onnxpycore, m) {
  m.doc() = "onnx core bindings from python: symbolic dimension expressions and "
            "shape inference helpers (operating on the same proto format).";
  AddOnnxPyExpressions(m);
  AddOnnxPyShapeInference(m);
  AddOnnxPyBuilder(m);
}

void AddOnnxPyExpressions(nb::module_ &m) {
  // -----------------------------------------------------------------------
  // Submodule `expressions`
  // Symbolic dimension-expression utilities (simplify, evaluate, rename).
  // -----------------------------------------------------------------------
  {
    namespace expr = ::onnx_light::core::expressions;

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

    // DimRange — inclusive [lower, upper] range for a dimension variable.
    nb::class_<expr::DimRange>(
        expressions_mod, "DimRange",
        "Inclusive ``[lower, upper]`` range for a dimension variable.\n\n"
        "Each bound is an ``int`` when concrete or a ``str`` when symbolic.\n\n"
        "When ``upper`` equals ``lower`` the variable is **exactly constrained** to that\n"
        "value (an equality constraint with no slack).  In that case there is no\n"
        "separate upper bound beyond the equality itself.\n\n"
        "When no finite upper bound can be derived, ``upper`` is set to\n"
        ":data:`~onnx_light.onnx_core.expressions.INFINITY` (the string ``'+inf'``),\n"
        "the reserved infinity sentinel.  Test for it with ``dr.upper == INFINITY``.\n"
        "No valid dimension-variable name may equal ``'+inf'``.\n\n"
        "Returned by :func:`dim_ranges_from_expressions`.")
        .def_prop_ro(
            "lower",
            [from_dim](const expr::DimRange &r) -> nb::object { return from_dim(r.lower); },
            "Inclusive lower bound; ``int`` when numeric, ``str`` when symbolic.")
        .def_prop_ro(
            "upper",
            [from_dim](const expr::DimRange &r) -> nb::object { return from_dim(r.upper); },
            "Inclusive upper bound; ``int`` when numeric, ``str`` when symbolic.\n\n"
            "Equals ``lower`` when the variable is exactly constrained by a direct\n"
            "equality (e.g. ``var == value``).  When ``upper`` differs from ``lower``\n"
            "(floor-division chain with divisor product > 1), it is a true finite upper\n"
            "bound.  When no finite upper bound is known, ``upper`` equals\n"
            ":data:`~onnx_light.onnx_core.expressions.INFINITY` (``'+inf'``).")
        .def("__repr__",
             [from_dim](const expr::DimRange &r) {
               auto lo = from_dim(r.lower);
               auto hi = from_dim(r.upper);
               auto to_s = [](nb::object o) -> std::string {
                 if (nb::isinstance<nb::int_>(o))
                   return std::to_string(nb::cast<int64_t>(o));
                 return "'" + nb::cast<std::string>(o) + "'";
               };
               return std::string("DimRange(lower=") + to_s(lo) + ", upper=" + to_s(hi) + ")";
             })
        .def(
            "__eq__",
            [from_dim](const expr::DimRange &r, nb::object other) -> bool {
              if (!nb::isinstance<expr::DimRange>(other))
                return false;
              const auto &o = nb::cast<const expr::DimRange &>(other);
              return r.lower == o.lower && r.upper == o.upper;
            },
            nb::arg("other"))
        .def("__hash__", [](const expr::DimRange &) -> int64_t {
          throw nb::type_error("unhashable type: 'DimRange'");
        });

    // INFINITY — sentinel DimType value for an unbounded upper bound.
    expressions_mod.attr("INFINITY") = from_dim(expr::kDimInfinity);

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

    // dim_ranges_from_expressions(equalities, tokens=[]) -> dict[str, DimRange]
    expressions_mod.def(
        "dim_ranges_from_expressions",
        [](const std::vector<std::pair<std::string, std::string>> &equalities,
           const std::vector<std::string> &tokens) {
          return expr::dim_ranges_from_expressions(equalities, tokens);
        },
        nb::arg("equalities"), nb::arg("tokens") = std::vector<std::string>{},
        "Infers dimension ranges from equality constraints.\n\n"
        "Each element of *equalities* is a ``(lhs, rhs)`` pair representing the equality\n"
        "``lhs == rhs``.  For each variable that appears as the leaf of a floor-division\n"
        "chain on one side, computes the tight range:\n\n"
        "* ``var == value`` → ``{var: DimRange(value, value)}`` — the variable is exactly\n"
        "  constrained to ``value``; ``upper == lower`` and there is no separate upper\n"
        "  bound beyond the equality itself.\n"
        "* ``var // d₁ // … // dₙ == value`` → ``{var: DimRange(P*value, P*value+P-1)}``\n"
        "  where P = d₁·…·dₙ — a proper range with ``upper > lower`` whenever P > 1.\n\n"
        "Variables that do not match any supported pattern are **absent** from the result\n"
        "(they are not represented as entries with ``upper == INFINITY``).\n\n"
        "When *tokens* is non-empty, only the listed variables are returned.\n\n"
        "Returns a ``dict[str, DimRange]`` mapping each variable to its inclusive range.");
  }

  // -----------------------------------------------------------------------
  // Submodule `shape_inference`
  // (full set of bindings provided by AddOnnxPyShapeInference)
  // -----------------------------------------------------------------------
}

void AddOnnxPyShapeInference(nb::module_ &m) {
  // `core::shapes::DispatchTable()` starts out empty (`onnx_core` must not
  // depend on `onnx_shapes`), so the built-in `onnx_shapes` shape functions
  // have to be registered explicitly before any of the bindings below can
  // resolve an operator. Idempotent and cheap if this module init function
  // ever ran more than once.
  ::onnx_light::onnx_shapes::RegisterShapeFunctions();
  ::onnx_light::onnx_shapes::RegisterPeakMemoryFunctions();

  namespace onnx_compute = ::onnx_light::core::compute;
  namespace expr = ::onnx_light::core::expressions;
  namespace onnx_shapes = ::onnx_light::core::shapes;
  using ::onnx_light::core::symbolic::DataTypeToTensorType;
  using ::onnx_light::core::symbolic::Device;
  using ::onnx_light::core::symbolic::SymDim;
  using ::onnx_light::core::symbolic::SymSequence;
  using ::onnx_light::core::symbolic::SymShape;
  using ::onnx_light::core::symbolic::SymTensor;
  using ::onnx_light::core::symbolic::TensorTypeToDataType;
  using ::onnx_light::onnx_proto::TensorType;

  auto shape_mod = m.def_submodule("shape_inference");
  shape_mod.doc() =
      "Shape-inference bindings backed by ``onnx_shapes``: exposes ``ShapesContext``, "
      "``ComputeShapeNode`` and the related ``ComputeShape{Graph,Model}`` / "
      "``ApplyInferredShapesTo{Graph,Model}`` helpers, together with the value "
      "types (``SymDim``, ``SymShape``, ``SymTensor``) used to describe "
      "tensor descriptors stored in the context.";

  // Convert an SymDim to a Python object (int when concrete, str otherwise).
  auto dim_to_object = [](const SymDim &d) -> nb::object {
    if (d.IsInt())
      return nb::cast(d.AsInt());
    return nb::cast(d.AsExpr());
  };

  // Convert a Python object (int | str) to an SymDim.
  auto object_to_dim = [](nb::handle h) -> SymDim {
    if (nb::isinstance<nb::int_>(h))
      return SymDim(nb::cast<int64_t>(h));
    return SymDim(nb::cast<std::string>(h));
  };

  // Convert an iterable of int|str into an SymShape.
  auto iterable_to_shape = [object_to_dim](nb::handle dims) -> SymShape {
    SymShape shape;
    nb::iterator it = nb::iter(dims);
    nb::iterator end = nb::iterator::sentinel();
    for (; it != end; ++it)
      shape.PushBack(object_to_dim(*it));
    return shape;
  };

  // Convert an SymShape to a Python list of int|str.
  auto shape_to_list = [dim_to_object](const SymShape &s) -> nb::list {
    nb::list out;
    for (const auto &d : s.Dims())
      out.append(dim_to_object(d));
    return out;
  };

  // -----------------------------------------------------------------------
  // SymDim
  // -----------------------------------------------------------------------
  nb::class_<SymDim>(shape_mod, "SymDim",
                     "A single shape dimension that is either a concrete integer or a "
                     "symbolic string expression.")
      .def(nb::init<>())
      .def(nb::init<int64_t>(), nb::arg("value"))
      .def(nb::init<std::string>(), nb::arg("expr"))
      .def("is_int", &SymDim::IsInt,
           "Returns True when the dimension holds a concrete integer value.")
      .def("is_expr", &SymDim::IsExpr,
           "Returns True when the dimension holds a symbolic string expression.")
      .def("as_int", &SymDim::AsInt,
           "Returns the integer value. Raises if the dimension is symbolic.")
      .def(
          "as_expr", [](const SymDim &d) -> std::string { return d.AsExpr(); },
          "Returns the symbolic expression. Raises if the dimension is an integer.")
      .def(
          "value", [dim_to_object](const SymDim &d) -> nb::object { return dim_to_object(d); },
          "Returns the underlying value as either ``int`` or ``str``.")
      .def("__str__", &SymDim::ToString)
      .def("__repr__",
           [](const SymDim &d) {
             if (d.IsInt()) {
               return std::string("SymDim(") + std::to_string(d.AsInt()) + ")";
             }
             return std::string("SymDim('") + d.AsExpr() + "')";
           })
      .def(nb::self == nb::self)
      .def(nb::self != nb::self);

  // -----------------------------------------------------------------------
  // SymShape
  // -----------------------------------------------------------------------
  nb::class_<SymShape>(
      shape_mod, "SymShape",
      "Ordered, bounded-rank collection of SymDim entries describing a tensor shape.")
      .def(nb::init<>())
      .def(
          "__init__",
          [iterable_to_shape](SymShape *self, nb::handle dims) {
            new (self) SymShape(iterable_to_shape(dims));
          },
          nb::arg("dims"), "Constructs a shape from an iterable of ``int`` or ``str`` dimensions.")
      .def("rank", &SymShape::Rank, "Number of dimensions.")
      .def("empty", &SymShape::Empty, "True when the shape is rank-0.")
      .def("is_fully_known", &SymShape::IsFullyKnown,
           "True when every dimension is a concrete integer.")
      .def(
          "dims", [shape_to_list](const SymShape &s) -> nb::list { return shape_to_list(s); },
          "Returns the dimensions as a list of ``int`` or ``str``.")
      .def("__len__", &SymShape::Rank)
      .def(
          "__getitem__",
          [dim_to_object](const SymShape &s, std::size_t i) -> nb::object {
            return dim_to_object(s[i]);
          },
          nb::arg("i"))
      .def("__iter__", [shape_to_list](const SymShape &s) { return nb::iter(shape_to_list(s)); })
      .def("__str__", &SymShape::ToString)
      .def("__repr__",
           [](const SymShape &s) {
             std::string out = "SymShape([";
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
  // SymTensor
  // -----------------------------------------------------------------------
  nb::class_<SymTensor>(
      shape_mod, "SymTensor",
      "Lightweight (non-owning) tensor descriptor with an element type and an "
      "SymShape, optionally annotated with a value-as-shape and value bounds. "
      "The Python binding never references a buffer; only the descriptor metadata "
      "is carried.")
      .def(nb::init<>())
      .def(
          "__init__",
          [iterable_to_shape](SymTensor *self, int dtype, nb::handle dims) {
            TensorType t = DataTypeToTensorType(static_cast<TensorProto::DataType>(dtype));
            new (self) SymTensor(nullptr, t, iterable_to_shape(dims));
          },
          nb::arg("dtype"), nb::arg("shape"),
          "Constructs an SymTensor from a ``TensorProto.DataType`` integer and an "
          "iterable of ``int`` or ``str`` dimensions.")
      .def_prop_ro(
          "dtype",
          [](const SymTensor &t) -> int {
            return static_cast<int>(TensorTypeToDataType(t.Dtype()));
          },
          "Element type as a ``TensorProto.DataType`` integer.")
      .def_prop_ro(
          "shape", [](const SymTensor &t) -> SymShape { return t.Shape(); },
          "Shape (a copy of the underlying SymShape).")
      .def("is_null", &SymTensor::IsNull, "True when no data buffer is attached.")
      .def("has_min", &SymTensor::HasMin)
      .def("has_max", &SymTensor::HasMax)
      .def("min", &SymTensor::Min)
      .def("max", &SymTensor::Max)
      .def("set_min", &SymTensor::SetMin, nb::arg("value"))
      .def("set_max", &SymTensor::SetMax, nb::arg("value"))
      .def("clear_min", &SymTensor::ClearMin)
      .def("clear_max", &SymTensor::ClearMax)
      .def("has_value_as_shape", &SymTensor::HasValueAsShape,
           "True when the tensor's value is interpreted as a shape.")
      .def(
          "value_as_shape", [](const SymTensor &t) -> SymShape { return t.ValueAsShape(); },
          "Returns the value-as-shape annotation. Raises if not set.")
      .def(
          "set_value_as_shape",
          [iterable_to_shape](SymTensor &t, nb::handle dims) {
            t.SetValueAsShape(iterable_to_shape(dims));
          },
          nb::arg("shape"),
          "Tags the tensor as carrying a shape value (e.g. the ``shape`` input of "
          "``Reshape``) and stores that shape.")
      .def("clear_value_as_shape", &SymTensor::ClearValueAsShape)
      .def("__str__", &SymTensor::ToString)
      .def("__repr__", &SymTensor::ToString)
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def(
          "__eq__",
          [](const SymTensor &t, const ValueInfoProto &vi) {
            SymTensor vi_tensor;
            return ::onnx_light::core::symbolic::SymTensorFromValueInfo(vi, vi_tensor) &&
                   t == vi_tensor;
          },
          nb::arg("other"),
          "Compares this descriptor against a ``ValueInfoProto`` by converting the "
          "value-info tensor type/shape into an ``SymTensor`` and checking "
          "descriptor equality.");

  // -----------------------------------------------------------------------
  // SymSequence
  // -----------------------------------------------------------------------
  nb::class_<SymSequence>(
      shape_mod, "SymSequence",
      "Lightweight (non-owning) descriptor for an ONNX tensor-sequence value: a "
      "common element type shared by every tensor in the sequence, an optional "
      "per-element ``SymShape`` list, and a (possibly symbolic) sequence length.")
      .def(nb::init<>())
      .def(
          "__init__",
          [iterable_to_shape](SymSequence *self, int elem_dtype, nb::handle elem_shapes) {
            TensorType t = DataTypeToTensorType(static_cast<TensorProto::DataType>(elem_dtype));
            std::vector<SymShape> shapes;
            nb::iterator it = nb::iter(elem_shapes);
            nb::iterator end = nb::iterator::sentinel();
            for (; it != end; ++it)
              shapes.push_back(iterable_to_shape(*it));
            new (self) SymSequence(t, std::move(shapes));
          },
          nb::arg("elem_dtype"), nb::arg("elem_shapes"),
          "Constructs a ``SymSequence`` from a ``TensorProto.DataType`` integer and an "
          "iterable of per-element shapes (each an iterable of ``int`` or ``str`` "
          "dimensions). The sequence length is set to the number of supplied shapes.")
      .def(
          "__init__",
          [](SymSequence *self, int elem_dtype, nb::handle length) {
            TensorType t = DataTypeToTensorType(static_cast<TensorProto::DataType>(elem_dtype));
            SymDim dim = nb::isinstance<nb::int_>(length) ? SymDim(nb::cast<int64_t>(length))
                                                          : SymDim(nb::cast<std::string>(length));
            new (self) SymSequence(t, std::move(dim));
          },
          nb::arg("elem_dtype"), nb::arg("length"),
          "Constructs a ``SymSequence`` from a ``TensorProto.DataType`` integer and a "
          "(possibly symbolic) length (``int`` or ``str``). No per-element shape is "
          "recorded.")
      .def_prop_ro(
          "elem_dtype",
          [](const SymSequence &s) -> int {
            return static_cast<int>(TensorTypeToDataType(s.ElemDtype()));
          },
          "Element type shared by every tensor in the sequence, as a "
          "``TensorProto.DataType`` integer.")
      .def_prop_ro(
          "elem_shapes",
          [shape_to_list](const SymSequence &s) -> nb::list {
            nb::list out;
            for (const auto &shape : s.ElemShapes())
              out.append(shape_to_list(shape));
            return out;
          },
          "Per-element shapes, as a list of shape lists. Empty when "
          "``has_elem_shapes`` is ``False``.")
      .def_prop_ro(
          "length",
          [dim_to_object](const SymSequence &s) -> nb::object { return dim_to_object(s.Length()); },
          "Sequence length (an ``int`` when concrete, a ``str`` symbolic expression "
          "otherwise).")
      .def("has_elem_dtype", &SymSequence::HasElemDtype,
           "True when an element dtype has been recorded for this sequence.")
      .def("has_elem_shapes", &SymSequence::HasElemShapes,
           "True when per-element shapes have been recorded for this sequence.")
      .def(nb::self == nb::self)
      .def(nb::self != nb::self);

  // -----------------------------------------------------------------------
  // ShapeEventAction — enum classifying a ShapeEvent record.
  // Mirrors :cpp:enum:`core::shapes::ShapeEventAction`.
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
  // Mirrors :cpp:class:`core::shapes::ShapeEvent`; ``shape`` is exposed
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
      "functions. Holds a ``name -> SymTensor`` map, a ``name -> SymSequence`` map "
      "and a ``domain -> opset_version`` map mirroring ``opset_import``.")
      .def(nb::init<>())
      // Tensor descriptors
      .def(
          "set",
          [](onnx_shapes::ShapesContext &c, const std::string &name, SymTensor t) {
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
          [](const onnx_shapes::ShapesContext &c, const std::string &name) -> SymTensor {
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
      .def("remove_custom_shape_inference_function",
           &onnx_shapes::ShapesContext::RemoveCustomShapeInferenceFunction, nb::arg("domain"),
           nb::arg("op_type"),
           "Removes a custom callback for ``(domain, op_type)``. "
           "An empty domain is normalised to ``ai.onnx``. Returns ``True`` "
           "when an entry was removed.")
      .def("clear_custom_shape_inference_functions",
           &onnx_shapes::ShapesContext::ClearCustomShapeInferenceFunctions,
           "Removes every registered custom callback.")
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
  // Peak-memory estimation
  // -----------------------------------------------------------------------
  nb::enum_<Device>(shape_mod, "Device", nb::is_arithmetic(),
                    "Logical device an operator executes on, passed as the first argument to a "
                    "peak-memory function. ``kUndefined`` is the \"no information\" default, "
                    "``kCPU`` is the host CPU and ``kGPU0`` is the first GPU device.")
      .value("kUndefined", Device::kUndefined, "No device information.")
      .value("kCPU", Device::kCPU, "The host CPU.")
      .value("kGPU0", Device::kGPU0, "The first GPU device.");

  shape_mod.def(
      "compute_peak_memory",
      [](const std::string &domain, const std::string &op_type, Device device,
         const std::vector<SymShape> &input_shapes) -> int64_t {
        return onnx_shapes::ComputePeakMemory(domain, op_type, device, input_shapes);
      },
      nb::arg("domain"), nb::arg("op_type"), nb::arg("device"), nb::arg("input_shapes"),
      "Returns the estimated peak scratch memory (in bytes) for ``(domain, op_type)`` executed "
      "on ``device`` with inputs of shape ``input_shapes``. Operators without a registered "
      "peak-memory function return ``0``. Built-in ``onnx_shapes`` operators (e.g. "
      "``Attention``) are registered from C++ via ``RegisterPeakMemoryFunctions``.");

  shape_mod.def(
      "peak_memory_dispatch_table_keys",
      []() -> nb::list {
        nb::list out;
        for (const auto &kv : onnx_shapes::PeakMemoryDispatchTable())
          out.append(kv.first);
        return out;
      },
      "Returns the ``\"<domain>:<op_type>\"`` keys currently registered in the peak-memory "
      "dispatch table.");

  // -----------------------------------------------------------------------
  // In-place reuse analysis
  // -----------------------------------------------------------------------
  shape_mod.attr("INPLACE_REUSE_METADATA_KEY") = onnx_compute::kInPlaceReuseMetadataKey;
  shape_mod.attr("RELEASE_AFTER_METADATA_KEY") = onnx_compute::kReleaseAfterMetadataKey;
  shape_mod.attr("RELEASE_AFTER_SHAPE_TAG_METADATA_KEY") =
      onnx_compute::kReleaseAfterShapeTagMetadataKey;
  shape_mod.attr("VALUE_TAG_METADATA_KEY") = onnx_compute::kValueTagMetadataKey;
  shape_mod.attr("VALUE_TAGS_METADATA_KEY") = onnx_compute::kValueTagsMetadataKey;
  shape_mod.attr("NODE_TAG_METADATA_KEY") = onnx_compute::kNodeTagMetadataKey;
  shape_mod.attr("CONSTANT_METADATA_KEY") = onnx_compute::kConstantMetadataKey;

  nb::enum_<onnx_compute::InPlaceReuseKind>(
      shape_mod, "InPlaceReuseKind", nb::is_arithmetic(),
      "Classifies how the reused input buffer compares in size with the output: "
      "``kEqual`` when the input and output buffers have the same byte size "
      "(e.g. a Transpose or same-total-size Reshape, the preferred reuse); ``kGreater`` when the "
      "input buffer is "
      "strictly larger in bytes than the output.")
      .value("kEqual", onnx_compute::InPlaceReuseKind::kEqual,
             "The input and output have the same byte size.")
      .value("kGreater", onnx_compute::InPlaceReuseKind::kGreater,
             "The input buffer is strictly larger in bytes than the output.");

  nb::class_<onnx_compute::InPlaceReuse>(
      shape_mod, "InPlaceReuse",
      "Represents one in-place reuse opportunity for a node: the output at ``output_index`` "
      "reuses the buffer of the input at ``input_index`` (both indices into the node's "
      "``output``/``input`` lists). ``kind`` records whether the input buffer has the same "
      "size as the output (``kEqual``) or is strictly larger (``kGreater``).")
      .def(nb::init<>())
      .def_rw("output_index", &onnx_compute::InPlaceReuse::output_index)
      .def_rw("input_index", &onnx_compute::InPlaceReuse::input_index)
      .def_rw("kind", &onnx_compute::InPlaceReuse::kind)
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def("__repr__", [](const onnx_compute::InPlaceReuse &r) {
        std::ostringstream os;
        const char *kind = r.kind == onnx_compute::InPlaceReuseKind::kEqual ? "kEqual" : "kGreater";
        os << "InPlaceReuse(output_index=" << r.output_index << ", input_index=" << r.input_index
           << ", kind=" << kind << ")";
        return os.str();
      });

  nb::enum_<onnx_compute::ComputeEventAction>(
      shape_mod, "ComputeEventAction", nb::is_arithmetic(),
      "Classifies decisions logged by :class:`ComputeContext` when "
      "``events_enabled`` is ``True``: ``kInPlace`` for in-place matches, "
      "``kRelease`` for releasable last-use values, and ``kReleaseShapeTag`` "
      "for released values tagged ``\"shape\"``.")
      .value("kInPlace", onnx_compute::ComputeEventAction::kInPlace,
             "One output was matched to one input for in-place reuse.")
      .value("kRelease", onnx_compute::ComputeEventAction::kRelease,
             "One value reached its last use and can be released.")
      .value("kReleaseShapeTag", onnx_compute::ComputeEventAction::kReleaseShapeTag,
             "One released value was also tagged ``\"shape\"``.");

  nb::class_<onnx_compute::ComputeEvent>(
      shape_mod, "ComputeEvent",
      "One entry in :meth:`ComputeContext.events`. ``inplace`` events carry "
      "``output_index`` / ``input_index`` / ``kind``. ``release`` and "
      "``release_shape_tag`` events carry ``name``.")
      .def_prop_ro(
          "action", [](const onnx_compute::ComputeEvent &ev) { return ev.action; },
          ":class:`ComputeEventAction` value describing the decision kind.")
      .def_prop_ro(
          "node_index", [](const onnx_compute::ComputeEvent &ev) { return ev.node_index; },
          "Node index where the decision was made.")
      .def_prop_ro(
          "name", [](const onnx_compute::ComputeEvent &ev) { return ev.name; },
          "Value name for ``release`` / ``release_shape_tag`` events.")
      .def_prop_ro(
          "output_index", [](const onnx_compute::ComputeEvent &ev) { return ev.output_index; },
          "Output index for ``inplace`` events, ``-1`` otherwise.")
      .def_prop_ro(
          "input_index", [](const onnx_compute::ComputeEvent &ev) { return ev.input_index; },
          "Input index for ``inplace`` events, ``-1`` otherwise.")
      .def_prop_ro(
          "kind", [](const onnx_compute::ComputeEvent &ev) { return ev.kind; },
          "Reuse kind for ``inplace`` events.")
      .def(
          "as_dict",
          [](const onnx_compute::ComputeEvent &ev) {
            nb::dict d;
            d["action"] = std::string(onnx_compute::ComputeEventActionName(ev.action));
            d["node_index"] = ev.node_index;
            d["name"] = ev.name;
            d["output_index"] = ev.output_index;
            d["input_index"] = ev.input_index;
            d["kind"] = std::string(InPlaceReuseKindName(ev.kind));
            return d;
          },
          "Returns this event as a plain ``dict``.")
      .def("__repr__", [](const onnx_compute::ComputeEvent &ev) {
        std::ostringstream os;
        os << "ComputeEvent(action='" << onnx_compute::ComputeEventActionName(ev.action)
           << "', node_index=" << ev.node_index << ", name='" << ev.name
           << "', output_index=" << ev.output_index << ", input_index=" << ev.input_index
           << ", kind=" << InPlaceReuseKindEnumName(ev.kind) << ")";
        return os.str();
      });

  shape_mod.attr("NODE_MEMORY_TOTAL_BYTES_KEY") = onnx_compute::kNodeMemoryTotalBytesKey;
  shape_mod.attr("NODE_MEMORY_ALREADY_ALLOCATED_BYTES_KEY") =
      onnx_compute::kNodeMemoryAlreadyAllocatedBytesKey;
  shape_mod.attr("NODE_MEMORY_OUTPUT_ALLOCATION_BYTES_KEY") =
      onnx_compute::kNodeMemoryOutputAllocationBytesKey;
  shape_mod.attr("NODE_MEMORY_INPUTS_KEY") = onnx_compute::kNodeMemoryInputsKey;
  shape_mod.attr("NODE_MEMORY_INITIALIZERS_KEY") = onnx_compute::kNodeMemoryInitializersKey;
  shape_mod.attr("NODE_MEMORY_INTERMEDIATES_KEY") = onnx_compute::kNodeMemoryIntermediatesKey;
  shape_mod.attr("NODE_MEMORY_OUTPUTS_KEY") = onnx_compute::kNodeMemoryOutputsKey;
  shape_mod.attr("NODE_PEAK_MEMORY_KEY") = onnx_compute::kNodePeakMemoryMetadataKey;

  auto with_node_list = [](nb::handle nodes, auto &&fn) {
    if (nb::isinstance<utils::RepeatedProtoField<NodeProto>>(nodes)) {
      return fn(nb::cast<utils::RepeatedProtoField<NodeProto> &>(nodes));
    }
    utils::RepeatedProtoField<NodeProto> copied;
    for (nb::handle h : nb::borrow<nb::iterable>(nodes)) {
      copied.push_back(nb::cast<const NodeProto &>(h));
    }
    return fn(copied);
  };

  nb::class_<onnx_compute::ComputeContext>(
      shape_mod, "ComputeContext",
      "Holds the in-place reuse opportunities computed for a graph, mirroring the way "
      "``ShapesContext`` holds inferred descriptors. Populate it with "
      "``compute_inplace_reuse_graph`` (consuming a ``ShapesContext``), then read the result "
      "through ``reuse`` / ``node_reuse`` / ``memory`` or persist it with "
      "``write_to_metadata``.")
      .def(nb::init<>())
      .def(
          "compute_value_and_node_tags",
          [](onnx_compute::ComputeContext &self, const GraphProto &graph, int verbose) {
            (void)verbose;
            const auto inferred = self.ComputeValueAndNodeTags(graph);
            return nb::make_tuple(inferred.first, inferred.second);
          },
          nb::arg("graph"), nb::arg("verbose") = 0,
          "Computes semantic ``shape``/``axes``/``weight``/``ambiguous`` tags for values and nodes "
          "in ``graph`` "
          "and stores the result in this context. ``verbose`` is currently accepted for API "
          "compatibility and has no effect.")
      .def(
          "compute_value_and_node_tags",
          [](onnx_compute::ComputeContext &self, const FunctionProto &function, int verbose) {
            (void)verbose;
            const auto inferred = self.ComputeValueAndNodeTags(function);
            return nb::make_tuple(inferred.first, inferred.second);
          },
          nb::arg("function"), nb::arg("verbose") = 0,
          "Computes semantic ``shape``/``axes``/``weight``/``ambiguous`` tags for values and nodes "
          "in "
          "``function`` and stores the result in this context. ``verbose`` is currently accepted "
          "for API compatibility and has no effect.")
      .def(
          "compute_value_and_node_tags",
          [with_node_list](onnx_compute::ComputeContext &self, nb::handle nodes, int verbose) {
            (void)verbose;
            return with_node_list(nodes, [&self](const auto &typed_nodes) {
              const auto inferred = self.ComputeValueAndNodeTags(typed_nodes);
              return nb::make_tuple(inferred.first, inferred.second);
            });
          },
          nb::arg("nodes"), nb::arg("verbose") = 0,
          "Computes semantic ``shape``/``axes``/``weight``/``ambiguous`` tags for a node list and "
          "stores the "
          "result in this context. ``verbose`` is currently accepted for API compatibility and has "
          "no effect.")
      .def_prop_ro(
          "value_tags", [](const onnx_compute::ComputeContext &self) { return self.ValueTags(); },
          "Returns the last value-tag map computed by :meth:`compute_value_and_node_tags`.")
      .def_prop_ro(
          "node_tags", [](const onnx_compute::ComputeContext &self) { return self.NodeTags(); },
          "Returns the last per-node tag list computed by :meth:`compute_value_and_node_tags`.")
      .def(
          "node_tag",
          [](const onnx_compute::ComputeContext &self, std::size_t node_index) {
            return self.NodeTag(node_index);
          },
          nb::arg("node_index"),
          "Returns the last tag computed for the node at ``node_index``. Raises ``IndexError`` "
          "when ``node_index`` is out of bounds.")
      .def("try_set_value_tag", &onnx_compute::ComputeContext::TrySetValueTag, nb::arg("name"),
           nb::arg("tag"),
           "Sets a value tag in the current context and returns whether the map changed. "
           "Returns ``false`` when ``name`` is empty, ``tag`` is invalid/empty, or the existing "
           "value already carries that tag.")
      .def("set_node_tag", &onnx_compute::ComputeContext::SetNodeTag, nb::arg("node_index"),
           nb::arg("tag"),
           "Sets a node tag in the current context and returns whether the list changed. "
           "Returns ``false`` when ``tag`` is invalid/empty or the node already carries that tag. "
           "Raises ``IndexError`` when ``node_index`` is out of bounds.")
      .def(
          "set_custom_value_tag_function",
          [](onnx_compute::ComputeContext &c, const std::string &domain, const std::string &op_type,
             nb::callable fn) {
            c.SetCustomValueTagFunction(
                domain, op_type,
                [py_fn = std::move(fn)](onnx_compute::ComputeContext &ctx, const NodeProto &node,
                                        std::size_t node_index) {
                  nb::gil_scoped_acquire lock;
                  py_fn(nb::cast(&ctx, nb::rv_policy::reference),
                        nb::cast(&node, nb::rv_policy::reference), node_index);
                });
          },
          nb::arg("domain"), nb::arg("op_type"), nb::arg("fn"),
          "Registers a Python callback for ``(domain, op_type)``. The callback receives "
          "``(ctx, node, node_index)`` and can set custom shape tags through ``ctx``.")
      .def(
          "has_custom_value_tag_function",
          [](const onnx_compute::ComputeContext &c, const std::string &domain,
             const std::string &op_type) {
            return c.GetCustomValueTagFunction(domain, op_type) != nullptr;
          },
          nb::arg("domain"), nb::arg("op_type"),
          "Returns whether a custom callback is registered for ``(domain, op_type)``.")
      .def("remove_custom_value_tag_function",
           &onnx_compute::ComputeContext::RemoveCustomValueTagFunction, nb::arg("domain"),
           nb::arg("op_type"),
           "Removes a custom callback for ``(domain, op_type)`` and returns whether one "
           "was removed.")
      .def("clear_custom_value_tag_functions",
           &onnx_compute::ComputeContext::ClearCustomValueTagFunctions,
           "Removes every registered custom callback.")
      .def(
          "custom_value_tag_keys",
          [](const onnx_compute::ComputeContext &c) -> nb::list {
            nb::list out;
            for (const auto &kv : c.CustomValueTagFunctions())
              out.append(kv.first);
            return out;
          },
          "Returns registered custom callback keys as ``\"<domain>:<op_type>\"``.")
      .def(
          "compute_inplace_reuse_graph",
          [](onnx_compute::ComputeContext &self, const GraphProto &graph,
             const onnx_shapes::ShapesContext &ctx, bool allow_input_overwrite,
             const std::unordered_map<std::string, std::string> &value_tags) {
            self.ComputeInPlaceReuseGraph(graph, ctx, allow_input_overwrite, value_tags);
          },
          nb::arg("graph"), nb::arg("ctx"), nb::arg("allow_input_overwrite") = false,
          nb::arg("value_tags") = std::unordered_map<std::string, std::string>{},
          "Guesses, for every node of ``graph``, which outputs reuse which input buffers in "
          "place, using the shapes already inferred into ``ctx``, and stores the result in this "
          "context (replacing any previous result).\n\n"
          "By default declared graph inputs are never overwritten in place; set "
          "``allow_input_overwrite=True`` to let an input be reused like an intermediate.\n\n"
          "When ``value_tags`` is provided (a ``{name: tag}`` dict such as the one returned by "
          ":func:`compute_value_and_node_tags`), released values that carry the ``\"shape\"`` tag "
          "are also stored in ``release_after_shape_tagged`` and written to "
          "``onnx_light.release_after_shape_tag`` by :meth:`write_to_metadata`. When "
          "``value_tags`` is omitted, this method reuses the last tags stored by "
          ":meth:`compute_value_and_node_tags` on the same context, if any.")
      .def(
          "compute_release_after_shape_tagged",
          [](const onnx_compute::ComputeContext &self) { return self.ReleaseAfterShapeTagged(); },
          "Returns ``release_after_shape_tagged``.")
      .def_prop_rw(
          "events_enabled",
          [](const onnx_compute::ComputeContext &self) { return self.events_enabled(); },
          [](onnx_compute::ComputeContext &self, bool enabled) {
            self.set_events_enabled(enabled);
          },
          "When ``True``, :meth:`compute_inplace_reuse_graph` appends one "
          ":class:`ComputeEvent` per in-place decision, release decision and "
          "shape-tagged release decision. Default is ``False``.")
      .def(
          "events",
          [](const onnx_compute::ComputeContext &self) {
            const auto &events = self.Events();
            nb::list out;
            for (const auto &ev : events) {
              out.append(ev);
            }
            return out;
          },
          "Returns the append-only decision log as a list of "
          ":class:`ComputeEvent` entries.")
      .def(
          "clear_events", [](onnx_compute::ComputeContext &self) { self.ClearEvents(); },
          "Empties the decision log without touching computed reuse results.")
      .def_prop_ro(
          "reuse", [](const onnx_compute::ComputeContext &self) { return self.Reuse(); },
          "The per-node reuse opportunities as a list (one entry per node, same order as "
          "``graph.node``); each entry is a list of :class:`InPlaceReuse`.")
      .def(
          "node_reuse",
          [](const onnx_compute::ComputeContext &self, std::size_t node_index) {
            return self.NodeReuse(node_index);
          },
          nb::arg("node_index"),
          "Returns the list of :class:`InPlaceReuse` opportunities discovered for the node at "
          "``node_index``. Raises ``IndexError`` when ``node_index`` is out of bounds.")
      .def_prop_ro(
          "release_after_shape_tagged",
          [](const onnx_compute::ComputeContext &self) { return self.ReleaseAfterShapeTagged(); },
          "The per-node shape-tagged releasable values. When ``compute_inplace_reuse_graph`` "
          "was called with an explicit non-empty ``value_tags`` argument (or after "
          "``compute_value_and_node_tags`` populated this context), this is a list with one "
          "entry per node (same order as ``graph.node``), where each entry is a list of value "
          "names that carry the ``\"shape\"`` tag. Otherwise this list is itself empty.")
      .def(
          "node_release_after_shape_tagged",
          [](const onnx_compute::ComputeContext &self, std::size_t node_index) {
            return self.NodeReleaseAfterShapeTagged(node_index);
          },
          nb::arg("node_index"),
          "Returns the list of shape-tagged releasable value names for the node at "
          "``node_index``. Raises ``IndexError`` when ``node_index`` is out of bounds.")
      .def_prop_ro(
          "memory", [](const onnx_compute::ComputeContext &self) { return self.Memory(); },
          "The per-node memory snapshots as a list (one entry per node, same order as "
          "``graph.node``). Each entry is a ``dict[str, int | str | dict[str, int | str]]`` "
          "describing the live input/initializer/intermediate buffers plus the extra output "
          "allocation needed by that node.")
      .def(
          "node_memory",
          [](const onnx_compute::ComputeContext &self, std::size_t node_index) {
            return self.NodeMemory(node_index);
          },
          nb::arg("node_index"),
          "Returns the memory snapshot ``dict`` computed for the node at ``node_index``. "
          "Raises ``IndexError`` when ``node_index`` is out of bounds.")
      .def(
          "write_to_metadata",
          [](const onnx_compute::ComputeContext &self, GraphProto &graph) {
            self.WriteToMetadata(graph);
          },
          nb::arg("graph"),
          "Records the computed opportunities into each node's ``metadata_props`` under the keys "
          "``onnx_light.inplace_reuse``, ``onnx_light.release_after``, "
          "``onnx_light.not_used_after``, and (when value tags were provided to "
          "``compute_inplace_reuse_graph``) "
          "``onnx_light.release_after_shape_tag``. The ``GraphProto`` is "
          "mutated in place and must be the same graph passed to "
          "``compute_inplace_reuse_graph``.")
      .def(
          "compute_shapes",
          [](onnx_compute::ComputeContext &self, const GraphProto &graph)
              -> const onnx_shapes::ShapesContext & { return self.ComputeShapes(graph); },
          nb::arg("graph"), nb::rv_policy::reference_internal,
          "Runs shape inference on ``graph`` and stores the result in the "
          ":class:`ShapesContext` owned by this context (also returned).")
      .def(
          "compute_shapes",
          [](onnx_compute::ComputeContext &self, const ModelProto &model,
             bool prefill_with_value_info_output) -> const onnx_shapes::ShapesContext & {
            return self.ComputeShapes(model, prefill_with_value_info_output);
          },
          nb::arg("model"), nb::arg("prefill_with_value_info_output") = false,
          nb::rv_policy::reference_internal,
          "Runs shape inference on ``model.graph`` (recording opset versions and local functions "
          "from ``model``) and stores the result in the :class:`ShapesContext` owned by this "
          "context (also returned).")
      .def(
          "compute_shapes",
          [](onnx_compute::ComputeContext &self, const FunctionProto &function,
             const onnx_compute::ComputeContext::InputShapes &input_shapes)
              -> const onnx_shapes::ShapesContext & {
            return self.ComputeShapes(function, input_shapes);
          },
          nb::arg("function"),
          nb::arg("input_shapes") = onnx_compute::ComputeContext::InputShapes{},
          nb::rv_policy::reference_internal,
          "Runs shape inference on the body of ``function`` and stores the result in the "
          ":class:`ShapesContext` owned by this context (also returned). A ``FunctionProto`` only "
          "names its inputs, so their shapes/types must be supplied through ``input_shapes`` (a "
          "``dict`` mapping value name to :class:`SymTensor`). Raises ``ValueError`` when an input "
          "consumed by a node is missing from ``input_shapes`` and not produced by an earlier "
          "node.")
      .def(
          "compute_shapes",
          [](onnx_compute::ComputeContext &self, nb::handle nodes,
             const onnx_compute::ComputeContext::InputShapes &input_shapes)
              -> const onnx_shapes::ShapesContext & {
            if (nb::isinstance<utils::RepeatedProtoField<NodeProto>>(nodes)) {
              return self.ComputeShapes(nb::cast<utils::RepeatedProtoField<NodeProto> &>(nodes),
                                        input_shapes);
            }
            utils::RepeatedProtoField<NodeProto> copied;
            for (nb::handle h : nb::borrow<nb::iterable>(nodes)) {
              copied.push_back(nb::cast<const NodeProto &>(h));
            }
            return self.ComputeShapes(copied, input_shapes);
          },
          nb::arg("nodes"), nb::arg("input_shapes") = onnx_compute::ComputeContext::InputShapes{},
          nb::rv_policy::reference_internal,
          "Runs shape inference on the node list ``nodes`` and stores the result in the "
          ":class:`ShapesContext` owned by this context (also returned). A bare node list has no "
          "declared inputs, so the shapes/types of every value not produced by the list must be "
          "supplied through ``input_shapes`` (a ``dict`` mapping value name to "
          ":class:`SymTensor`). "
          "Raises ``ValueError`` when an input consumed by a node is missing from ``input_shapes`` "
          "and not produced by an earlier node.")
      .def_prop_ro(
          "shapes",
          [](onnx_compute::ComputeContext &self) -> const onnx_shapes::ShapesContext & {
            return self.Shapes();
          },
          nb::rv_policy::reference_internal,
          "The :class:`ShapesContext` owned by this context, populated by "
          ":meth:`compute_shapes` / :meth:`compute`.")
      .def(
          "compute_peak_memory",
          [](onnx_compute::ComputeContext &self, const GraphProto &graph, Device device) {
            return self.ComputePeakMemory(graph, device);
          },
          nb::arg("graph"), nb::arg("device") = Device::kUndefined,
          "Computes the per-node peak scratch memory of ``graph`` using the shapes already "
          "inferred into this context, storing (and returning) one ``int`` per node.")
      .def_prop_ro(
          "peak_memory", [](const onnx_compute::ComputeContext &self) { return self.PeakMemory(); },
          "The per-node peak-memory estimates computed by :meth:`compute_peak_memory`, as a list "
          "with one ``int`` per node. Empty before it has been called.")
      .def(
          "node_peak_memory",
          [](const onnx_compute::ComputeContext &self, std::size_t node_index) {
            return self.NodePeakMemory(node_index);
          },
          nb::arg("node_index"),
          "Returns the peak-memory estimate for the node at ``node_index``. Raises ``IndexError`` "
          "when ``node_index`` is out of bounds.")
      .def(
          "compute",
          [](onnx_compute::ComputeContext &self, const GraphProto &graph, Device device,
             bool allow_input_overwrite) { self.Compute(graph, device, allow_input_overwrite); },
          nb::arg("graph"), nb::arg("device") = Device::kUndefined,
          nb::arg("allow_input_overwrite") = false,
          "Runs every analysis on ``graph`` in order (shape inference, value/node tagging, "
          "in-place reuse with release-after and shape-tag classification, and per-node peak "
          "memory) and stores all results in this context.")
      .def(
          "compute",
          [](onnx_compute::ComputeContext &self, const ModelProto &model, Device device,
             bool allow_input_overwrite, bool prefill_with_value_info_output) {
            self.Compute(model, device, allow_input_overwrite, prefill_with_value_info_output);
          },
          nb::arg("model"), nb::arg("device") = Device::kUndefined,
          nb::arg("allow_input_overwrite") = false,
          nb::arg("prefill_with_value_info_output") = false,
          "Same as ``compute(graph, ...)`` but seeds shape inference from ``model`` (opset "
          "versions and local functions) before analysing ``model.graph``.")
      .def(
          "write_to_graph",
          [](const onnx_compute::ComputeContext &self, GraphProto &graph) {
            self.WriteToGraph(graph);
          },
          nb::arg("graph"),
          "Pushes every computed result into ``graph``: the inferred shapes into "
          "``graph.value_info`` / outputs, the in-place / release / shape-tag information into "
          "node ``metadata_props`` (see :meth:`write_to_metadata`) and the per-node peak-memory "
          "estimates under ``onnx_light.peak_memory``.")
      .def(
          "write_to_model",
          [](const onnx_compute::ComputeContext &self, ModelProto &model) {
            self.WriteToModel(model);
          },
          nb::arg("model"), "Same as :meth:`write_to_graph` applied to ``model.graph``.")
      .def("clear", &onnx_compute::ComputeContext::Clear, "Empties the stored result.")
      .def("__len__", [](const onnx_compute::ComputeContext &self) { return self.Size(); });

  shape_mod.def(
      "compute_inplace_reuse",
      [](const onnx_shapes::ShapesContext &ctx, const GraphProto &graph,
         bool allow_input_overwrite) {
        return onnx_compute::ComputeInPlaceReuse(graph, ctx, allow_input_overwrite);
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
      [](const onnx_shapes::ShapesContext &ctx, GraphProto &graph,
         const std::unordered_map<std::string, std::string> &value_tags) {
        onnx_compute::WriteInPlaceReuseToMetadata(graph, ctx, value_tags);
      },
      nb::arg("ctx"), nb::arg("graph"),
      nb::arg("value_tags") = std::unordered_map<std::string, std::string>{},
      "Computes the in-place reuse opportunities for ``graph`` (see "
      ":func:`compute_inplace_reuse`) and records them into each node's ``metadata_props`` "
      "under the keys ``onnx_light.inplace_reuse``, ``onnx_light.release_after`` and "
      "``onnx_light.not_used_after``. The ``GraphProto`` is mutated in place.\n\n"
      "For every node with at least one opportunity, a single metadata entry is added (or "
      "updated in place if the key already exists) whose value lists the opportunities as "
      "``output_index:input_index:kind`` triplets separated by ``;`` (``kind`` being ``equal`` "
      "or ``greater``). For every node with releasable last-use inputs, one metadata entry is "
      "added under ``onnx_light.release_after`` as a ``;``-separated name list. For every node "
      "where declared graph inputs or initializers reach their last use, one metadata entry is "
      "added under ``onnx_light.not_used_after``.\n\n"
      "When ``value_tags`` is provided (a ``{name: tag}`` dict such as the one returned by "
      ":func:`compute_value_and_node_tags`), released values that carry the ``\"shape\"`` tag are "
      "also written under ``onnx_light.release_after_shape_tag``.");

  shape_mod.def(
      "write_peak_memory_to_metadata",
      [](const onnx_shapes::ShapesContext &ctx, GraphProto &graph, Device device) {
        onnx_compute::WritePeakMemoryToMetadata(graph, ctx, device);
      },
      nb::arg("ctx"), nb::arg("graph"), nb::arg("device") = Device::kUndefined,
      "Computes the estimated peak scratch memory for every node of ``graph`` using the "
      "shapes already inferred into ``ctx``, and records the result in each node's "
      "``metadata_props`` under the key ``onnx_light.peak_memory``. Nodes whose estimated "
      "peak memory is zero (either the operator has no registered peak-memory function or "
      "all relevant input shapes are symbolic) are left untouched.\n\n"
      "The peak memory accounts only for the extra scratch/working memory an operator's "
      "computation allocates, not the memory already accounted for by its inputs and outputs.\n\n"
      "``device`` defaults to :attr:`Device.kUndefined` (no device information).");

  shape_mod.def(
      "compute_value_and_node_tags",
      [](const GraphProto &graph, int verbose) {
        (void)verbose;
        onnx_compute::ComputeContext ctx;
        const auto inferred = ctx.ComputeValueAndNodeTags(graph);
        return nb::make_tuple(inferred.first, inferred.second);
      },
      nb::arg("graph"), nb::arg("verbose") = 0,
      "Computes semantic ``shape``/``axes``/``weight``/``ambiguous`` tags for values and nodes in "
      "``graph``. "
      "``verbose`` is currently accepted for API compatibility and has no effect.");
  shape_mod.def(
      "compute_value_and_node_tags",
      [](const FunctionProto &function, int verbose) {
        (void)verbose;
        onnx_compute::ComputeContext ctx;
        const auto inferred = ctx.ComputeValueAndNodeTags(function);
        return nb::make_tuple(inferred.first, inferred.second);
      },
      nb::arg("function"), nb::arg("verbose") = 0,
      "Computes semantic ``shape``/``axes``/``weight``/``ambiguous`` tags for values and nodes in "
      "``function``. ``verbose`` is currently accepted for API compatibility and has no effect.");
  shape_mod.def(
      "compute_value_and_node_tags",
      [with_node_list](nb::handle nodes, int verbose) {
        (void)verbose;
        return with_node_list(nodes, [](const auto &typed_nodes) {
          onnx_compute::ComputeContext ctx;
          const auto inferred = ctx.ComputeValueAndNodeTags(typed_nodes);
          return nb::make_tuple(inferred.first, inferred.second);
        });
      },
      nb::arg("nodes"), nb::arg("verbose") = 0,
      "Computes semantic ``shape``/``axes``/``weight``/``ambiguous`` tags for a node list. "
      "``verbose`` is "
      "currently accepted for API compatibility and has no effect.");

  shape_mod.def(
      "write_value_and_node_tags_to_metadata",
      [](GraphProto &graph) { onnx_compute::WriteValueAndNodeTagsToMetadata(graph); },
      nb::arg("graph"), "Writes inferred value/node tags into graph metadata.");
  shape_mod.def(
      "write_value_and_node_tags_to_metadata",
      [](FunctionProto &function) { onnx_compute::WriteValueAndNodeTagsToMetadata(function); },
      nb::arg("function"), "Writes inferred value/node tags into function metadata.");
  shape_mod.def(
      "write_value_and_node_tags_to_metadata",
      [](ModelProto &model) { onnx_compute::WriteValueAndNodeTagsToMetadata(model); },
      nb::arg("model"), "Writes inferred value/node tags into ``model.graph`` metadata.");
  shape_mod.def(
      "write_value_and_node_tags_to_metadata",
      [with_node_list](nb::handle nodes) {
        const auto inferred = with_node_list(nodes, [](const auto &typed_nodes) {
          return onnx_compute::InferValueAndNodeTags(typed_nodes);
        });
        const auto &node_tags = inferred.second;
        std::size_t i = 0;
        for (nb::handle h : nb::borrow<nb::iterable>(nodes)) {
          if (i < node_tags.size() && !node_tags[i].empty()) {
            nb::cast<NodeProto &>(h).add_metadata(onnx_compute::kNodeTagMetadataKey, node_tags[i]);
          }
          ++i;
        }
      },
      nb::arg("nodes"), "Writes inferred node tags for a node list.");

  shape_mod.def(
      "write_constant_info_to_metadata",
      [](GraphProto &graph) { onnx_compute::WriteConstantInfoToMetadata(graph); },
      nb::arg("graph"),
      "Writes constant-value / constant-node information into graph metadata.");
  shape_mod.def(
      "write_constant_info_to_metadata",
      [](FunctionProto &function) { onnx_compute::WriteConstantInfoToMetadata(function); },
      nb::arg("function"),
      "Writes constant-value / constant-node information into function metadata.");
  shape_mod.def(
      "write_constant_info_to_metadata",
      [](ModelProto &model) { onnx_compute::WriteConstantInfoToMetadata(model); }, nb::arg("model"),
      "Writes constant-value / constant-node information into ``model.graph`` metadata.");
}

namespace {

// Builds a standalone AttributeProto from a Python value. Supports scalars
// (int, float, str, bytes), homogeneous lists of those, and the proto types
// AttributeProto (passed through), TensorProto and GraphProto.
AttributeProto PyValueToAttribute(const std::string &name, nb::handle value) {
  if (nb::isinstance<AttributeProto>(value)) {
    AttributeProto attribute = nb::cast<const AttributeProto &>(value);
    attribute.set_name(name);
    return attribute;
  }
  if (nb::isinstance<TensorProto>(value)) {
    AttributeProto attribute;
    attribute.set_name(name);
    attribute.set_type(AttributeProto::AttributeType::TENSOR);
    *attribute.mutable_t() = nb::cast<const TensorProto &>(value);
    return attribute;
  }
  if (nb::isinstance<GraphProto>(value)) {
    AttributeProto attribute;
    attribute.set_name(name);
    attribute.set_type(AttributeProto::AttributeType::GRAPH);
    *attribute.mutable_g() = nb::cast<const GraphProto &>(value);
    return attribute;
  }
  NodeProto scratch;
  const char *key = name.c_str();
  if (nb::isinstance<nb::bool_>(value)) {
    ::onnx_light::AddAttribute(scratch, key, static_cast<int64_t>(nb::cast<bool>(value)));
  } else if (nb::isinstance<nb::int_>(value)) {
    ::onnx_light::AddAttribute(scratch, key, nb::cast<int64_t>(value));
  } else if (nb::isinstance<nb::float_>(value)) {
    ::onnx_light::AddAttribute(scratch, key, nb::cast<float>(value));
  } else if (nb::isinstance<nb::str>(value) || nb::isinstance<nb::bytes>(value)) {
    ::onnx_light::AddAttribute(scratch, key, nb::cast<std::string>(value));
  } else if (nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value)) {
    nb::list items = nb::list(value);
    if (items.size() == 0) {
      ::onnx_light::AddAttribute(scratch, key, std::vector<int64_t>{});
    } else if (nb::isinstance<nb::float_>(items[0]) && !nb::isinstance<nb::bool_>(items[0])) {
      ::onnx_light::AddAttribute(scratch, key, nb::cast<std::vector<float>>(items));
    } else if (nb::isinstance<nb::str>(items[0]) || nb::isinstance<nb::bytes>(items[0])) {
      ::onnx_light::AddAttribute(scratch, key, nb::cast<std::vector<std::string>>(items));
    } else {
      ::onnx_light::AddAttribute(scratch, key, nb::cast<std::vector<int64_t>>(items));
    }
  } else {
    throw core::builder::BuilderError("GraphBuilder: unsupported attribute type for '" + name +
                                      "'.");
  }
  return scratch.attribute(0);
}

// Converts the ``attributes`` argument of ``make_node`` into a vector of
// AttributeProto. ``attributes`` may be ``None``, a mapping ``name -> value``
// or a list of ready-made AttributeProto objects.
std::vector<AttributeProto> PyAttributesToVector(nb::handle attributes) {
  std::vector<AttributeProto> result;
  if (attributes.is_none()) {
    return result;
  }
  if (nb::isinstance<nb::dict>(attributes)) {
    for (auto item : nb::cast<nb::dict>(attributes)) {
      result.push_back(PyValueToAttribute(nb::cast<std::string>(item.first), item.second));
    }
    return result;
  }
  for (nb::handle item : nb::borrow<nb::iterable>(attributes)) {
    result.push_back(nb::cast<AttributeProto>(item));
  }
  return result;
}

} // namespace

void AddOnnxPyBuilder(nb::module_ &m) {
  using core::builder::BuilderError;
  using core::builder::GraphBuilder;
  using ::onnx_light::core::symbolic::Device;
  using ::onnx_light::core::symbolic::SymTensor;

  auto builder_mod = m.def_submodule("builder");
  builder_mod.doc() =
      "Incremental ONNX graph builder: accumulates inputs, initializers, nodes and "
      "outputs while keeping shapes, in-place reuse, value tags and peak memory up to "
      "date, then finalises into a model, graph or function.";

  nb::register_exception_translator([](const std::exception_ptr &p, void *) {
    try {
      std::rethrow_exception(p);
    } catch (const BuilderError &e) {
      PyErr_SetString(PyExc_ValueError, e.what());
    }
  });

  // The built-in ONNX operator schemas live in the ``onnx_op`` library, which
  // this extension deliberately does not link against. When the caller wants
  // schema-driven opset resolution and node validation, it injects a schema
  // provider (see ``onnx_core/graph_builder.py``, which wires the schemas
  // exposed by the ``_onnxpyprotoop`` extension).
  nb::class_<GraphBuilder>(builder_mod, "GraphBuilder",
                           "Incrementally builds an ONNX graph, model or function.")
      .def(
          "__init__",
          [](GraphBuilder *self, const ModelProto &model, nb::object schema_lookup) {
            if (schema_lookup.is_none()) {
              new (self) GraphBuilder(model, GraphBuilder::SchemaLookupFn{});
              return;
            }
            auto fn = nb::cast<GraphBuilder::SchemaLookupFn>(schema_lookup);
            new (self) GraphBuilder(model, std::move(fn));
          },
          nb::arg("model"), nb::arg("schema_lookup") = nb::none(),
          "Constructs a builder by importing ``model`` node-by-node. GRAPH/GRAPHS attributes "
          "are represented in-builder as ``*_ref`` attributes that point to nested subgraph "
          "builders, and are materialized back on export.")
      .def(
          "__init__",
          [](GraphBuilder *self, const std::string &name, nb::object schema_lookup) {
            if (schema_lookup.is_none()) {
              new (self) GraphBuilder(name, GraphBuilder::SchemaLookupFn{});
              return;
            }
            auto fn = nb::cast<GraphBuilder::SchemaLookupFn>(schema_lookup);
            new (self) GraphBuilder(name, std::move(fn));
          },
          nb::arg("name") = "graph", nb::arg("schema_lookup") = nb::none(),
          "Constructs an empty builder. ``schema_lookup`` is an optional callable "
          "``op_type -> list[LightOpSchema]`` used to resolve opsets and validate nodes; "
          "when omitted no schema-based validation is performed.")
      .def("set_opset_version", &GraphBuilder::SetOpsetVersion, nb::arg("domain"),
           nb::arg("version"),
           "Records the opset version to use for ``domain`` (empty string for the default "
           "ONNX domain).")
      .def("opset_version", &GraphBuilder::OpsetVersion, nb::arg("domain") = "",
           "Returns the opset version recorded for ``domain``.")
      .def("opset_versions", &GraphBuilder::OpsetVersions,
           "Returns the recorded ``domain -> opset version`` mapping.")
      .def("has_name", &GraphBuilder::HasName, nb::arg("name"),
           "Returns True when ``name`` has already been handed out.")
      .def("unique_name", &GraphBuilder::UniqueName, nb::arg("prefix") = "n",
           "Returns and records a fresh, unused name starting with ``prefix``.")
      .def("make_initializer", &GraphBuilder::MakeInitializer, nb::arg("tensor"),
           "Appends ``tensor`` (which may carry external data) as an initializer and "
           "returns its name.")
      .def(
          "make_external_initializer",
          [](GraphBuilder &self, const std::string &name, int dtype,
             const std::vector<int64_t> &dims, const std::string &location, int64_t offset,
             int64_t length) {
            using ::onnx_light::core::symbolic::DataTypeToTensorType;
            return self.MakeExternalInitializer(
                name, DataTypeToTensorType(static_cast<TensorProto::DataType>(dtype)), dims,
                location, offset, length);
          },
          nb::arg("name"), nb::arg("dtype"), nb::arg("dims"), nb::arg("location"),
          nb::arg("offset") = 0, nb::arg("length") = 0,
          "Appends an initializer whose data lives in an external file and returns its name.")
      .def(
          "make_input",
          [](GraphBuilder &self, const ValueInfoProto &value_info) {
            return self.MakeInput(value_info);
          },
          nb::arg("value_info"), "Declares a graph input from a ready-made ValueInfoProto.")
      .def(
          "make_input",
          [](GraphBuilder &self, const std::string &name, const SymTensor &type) {
            return self.MakeInput(name, type);
          },
          nb::arg("name"), nb::arg("type"), "Declares a graph input described by ``type``.")
      .def(
          "make_input",
          [](GraphBuilder &self, const std::string &name, int dtype,
             const std::vector<nb::object> &shape) {
            using ::onnx_light::core::symbolic::DataTypeToTensorType;
            using ::onnx_light::core::symbolic::SymDim;
            using ::onnx_light::core::symbolic::SymShape;
            SymShape sym;
            for (const nb::object &d : shape) {
              if (nb::isinstance<nb::int_>(d)) {
                sym.PushBack(SymDim(nb::cast<int64_t>(d)));
              } else {
                sym.PushBack(SymDim(nb::cast<std::string>(d)));
              }
            }
            return self.MakeInput(
                name, DataTypeToTensorType(static_cast<TensorProto::DataType>(dtype)), sym);
          },
          nb::arg("name"), nb::arg("dtype"), nb::arg("shape"),
          "Declares a graph input with element type ``dtype`` (a ``TensorProto.DataType`` "
          "integer) and ``shape`` (an iterable of ``int`` or ``str``).")
      .def(
          "make_output",
          [](GraphBuilder &self, const ValueInfoProto &value_info) { self.MakeOutput(value_info); },
          nb::arg("value_info"), "Declares a graph output from a ready-made ValueInfoProto.")
      .def(
          "make_output",
          [](GraphBuilder &self, const std::string &name, const SymTensor &type) {
            self.MakeOutput(name, type);
          },
          nb::arg("name"), nb::arg("type"),
          "Declares ``name`` as a graph output described by ``type``.")
      .def(
          "make_output", [](GraphBuilder &self, const std::string &name) { self.MakeOutput(name); },
          nb::arg("name"),
          "Declares ``name`` as a graph output; its type is filled in at finalisation.")
      .def(
          "make_node",
          [](GraphBuilder &self, const std::string &op_type, const std::vector<std::string> &inputs,
             nb::object outputs, const std::string &domain, const std::string &name,
             nb::object attributes) {
            std::vector<std::string> resolved_outputs;
            if (!outputs.is_none()) {
              if (nb::isinstance<nb::str>(outputs)) {
                resolved_outputs.push_back(nb::cast<std::string>(outputs));
              } else {
                resolved_outputs = nb::cast<std::vector<std::string>>(outputs);
              }
            }
            return self.MakeNode(op_type, inputs, resolved_outputs, domain, name,
                                 PyAttributesToVector(attributes));
          },
          nb::arg("op_type"), nb::arg("inputs"), nb::arg("outputs") = nb::none(),
          nb::arg("domain") = "", nb::arg("name") = "", nb::arg("attributes") = nb::none(),
          "Appends a node and returns its (possibly generated) output names. ``attributes`` "
          "may be a mapping ``name -> value`` (scalars, lists, TensorProto, GraphProto) or a "
          "list of AttributeProto objects.")
      .def("has_shape", &GraphBuilder::HasShape, nb::arg("name"),
           "Returns True when the shape of ``name`` has been inferred.")
      .def("get_shape", &GraphBuilder::GetShape, nb::arg("name"), nb::rv_policy::reference_internal,
           "Returns the inferred descriptor of ``name``.")
      .def("remove_unused_nodes", &GraphBuilder::RemoveUnusedNodes,
           "Recursively removes dead-end (unused) nodes, descending into nested subgraphs and "
           "local functions, and returns the total number of nodes removed.")
      .def("remove_duplicate_initializers", &GraphBuilder::RemoveDuplicateInitializers,
           "Recursively removes duplicated initializers (initializers with byte-for-byte "
           "identical content), rewriting every reference to a dropped duplicate to the surviving "
           "initializer. The deduplication spans the enclosing graph and its subgraphs (a "
           "subgraph body sees the enclosing scope), while local functions are deduplicated on "
           "their own, and returns the total number of initializers removed.")
      .def("remove_identity_nodes", &GraphBuilder::RemoveIdentityNodes,
           "Recursively removes default-domain Identity nodes, rewriting every reference to a "
           "dropped identity's output to its input and collapsing chains of identities in a "
           "single pass. An Identity whose output is a declared graph output is kept. The removal "
           "descends into nested subgraphs and local functions, and returns the total number of "
           "Identity nodes removed.")
      .def("remove_duplicate_nodes", &GraphBuilder::RemoveDuplicateNodes,
           "Recursively removes duplicated nodes (common subexpressions): nodes sharing the same "
           "operator type, domain, inputs and attributes compute the same value, so every later "
           "duplicate is dropped and each reference to its output is rewritten to the surviving "
           "node's matching output. Inputs are resolved against earlier-dropped duplicates in a "
           "single topological pass, so a whole duplicated branch collapses at once. A node whose "
           "output is a declared graph output is kept, and nodes referencing control-flow "
           "subgraphs are never merged. The removal descends into nested subgraphs and local "
           "functions, and returns the total number of nodes removed.")
      .def("inline_local_functions", &GraphBuilder::InlineLocalFunctions,
           nb::arg("include") = std::vector<std::pair<std::string, std::string>>{},
           nb::arg("exclude") = std::vector<std::pair<std::string, std::string>>{},
           "Inlines every call to a local function into the calling graph: the call node is "
           "replaced by a renamed copy of the function body (formal inputs/outputs rewired to the "
           "call inputs/outputs, every other body value given a fresh name, and ``ref_attr_name`` "
           "attributes resolved against the call attributes). The expansion runs to a fixed point "
           "and descends into nested subgraphs; local function definitions left without any caller "
           "are dropped. ``include`` and ``exclude`` are lists of ``(domain, name)`` tuples: a "
           "tuple matches a function when both components match, where an empty domain matches "
           "every domain (all functions sharing the name) and an empty name matches every name "
           "(all functions in the domain). When ``include`` is non-empty only the matched "
           "functions are inlined; when ``exclude`` is non-empty every function except the matched "
           "ones is inlined; passing both raises an error. Returns the total number of call nodes "
           "inlined.")
      .def(
          "build_graph", [](const GraphBuilder &self) { return self.BuildGraph(); },
          "Assembles the accumulated inputs, initializers, nodes and outputs into a "
          "GraphProto without running the finalisation analyses.")
      .def(
          "make_local_function",
          [](GraphBuilder &self, const std::string &name, const std::string &domain)
              -> GraphBuilder & { return self.MakeLocalFunction(name, domain); },
          nb::arg("name"), nb::arg("domain") = "", nb::rv_policy::reference_internal,
          "Creates and returns a nested builder for a local function; local functions are "
          "emitted into the produced ModelProto.")
      .def("has_local_function", &GraphBuilder::HasLocalFunction, nb::arg("name"),
           "Returns True when a local function named ``name`` exists.")
      .def(
          "make_subgraph",
          [](GraphBuilder &self, const std::string &name) -> GraphBuilder & {
            return self.MakeSubgraph(name);
          },
          nb::arg("name"), nb::rv_policy::reference_internal,
          "Creates and returns a nested builder for a subgraph (control-flow body).")
      .def("to_string", &GraphBuilder::ToString,
           "Returns a comprehensive, human-readable description of the builder content.")
      .def("__str__", &GraphBuilder::ToString)
      .def_prop_rw(
          "device", [](const GraphBuilder &self) { return self.device(); },
          [](GraphBuilder &self, Device device) { self.set_device(device); },
          "Logical device used for the peak-memory analysis at finalisation.")
      .def("to_graph", &GraphBuilder::ToGraph, "Returns the finalised GraphProto.")
      .def("to_model", &GraphBuilder::ToModel, nb::arg("ir_version") = 0,
           "Returns the finalised graph wrapped in a ModelProto.")
      .def("to_function", &GraphBuilder::ToFunction, nb::arg("domain") = "",
           "Returns the finalised nodes wrapped in a FunctionProto.")
      .def(
          "to_onnx",
          [](GraphBuilder &self, const std::string &kind, int64_t ir_version,
             const std::string &domain) -> nb::object {
            if (kind == "model") {
              return nb::cast(self.ToModel(ir_version));
            }
            if (kind == "graph") {
              return nb::cast(self.ToGraph());
            }
            if (kind == "function") {
              return nb::cast(self.ToFunction(domain));
            }
            throw BuilderError("GraphBuilder: unknown kind '" + kind +
                               "'; expected 'model', 'graph' or 'function'.");
          },
          nb::arg("kind") = "model", nb::arg("ir_version") = 0, nb::arg("domain") = "",
          "Finalises the builder into a ``model`` (default), ``graph`` or ``function``.");
}
