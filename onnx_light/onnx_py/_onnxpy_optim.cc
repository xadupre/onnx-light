#include "onnx_optim/expressions.h"
#include "onnx_optim/shapes/shape_inference.h"
#include "onnx_optim/shapes/shapes_context.h"
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <vector>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;

NB_MODULE(_onnxpyoptim, m) {
  m.doc() = "onnx optim bindings from python: symbolic dimension expressions and "
            "shape inference helpers (operating on the same proto format).";
  AddOnnxPyExpressions(m);
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
  // Top-level shape-inference helpers operating on a full ModelProto.
  // -----------------------------------------------------------------------
  {
    auto shape_mod = m.def_submodule("shape_inference");
    shape_mod.doc() = "Top-level shape-inference helpers running on ModelProto and GraphProto.";

    shape_mod.def(
        "infer_shapes_model",
        [](ModelProto &model) { onnx_optim::shapes::InferShapesModel(model); }, nb::arg("model"),
        "Runs shape inference on ``model`` and writes the inferred element types and shapes "
        "back into ``model.graph.output`` and ``model.graph.value_info``. The ModelProto is "
        "mutated in place.");
  }
}
