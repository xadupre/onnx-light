// SPDX-License-Identifier: Apache-2.0
//
// C++ implementation of symbolic dimension expression utilities.
// Ported from yobx/xexpressions (https://github.com/xadupre/yet-another-onnx-builder).

/**
 * @file expressions.h
 * @brief Symbolic dimension-expression utilities for ONNX shape inference.
 *
 * Provides a lightweight AST-based library for parsing, simplifying,
 * evaluating, and renaming symbolic shape expressions such as those produced
 * during ONNX shape inference (e.g. ``"2*batch//2"`` → ``"batch"``).
 *
 * ## Overview
 *
 * Shape expressions are strings that contain integer constants, symbolic
 * variable names, and the arithmetic operators `+`, `-`, `*`, `//`, `%`,
 * `^` (encodes `max`), and `&` (encodes `min`).
 *
 * The typical workflow is:
 *
 * @code{.cpp}
 * using namespace onnx_light::expressions;
 *
 * // Simplify a string expression.
 * SimplifyResult r = simplify_expression("2*batch//batch");
 * // r holds int64_t(2) because the expression reduces to a constant.
 *
 * // Evaluate with concrete variable assignments.
 * int64_t v = evaluate_expression("x + y", {{"x", 3}, {"y", 5}});
 * // v == 8
 *
 * // Rename variables.
 * std::string s = rename_expression("s0 + seq_len", {{"s0", "batch"}});
 * // s == "batch+seq_len"
 *
 * // Arithmetic on symbolic dimensions.
 * DimType d = dim_add(DimType{"batch"}, DimType{int64_t{1}});
 * // d holds std::string("1+batch")
 * @endcode
 */

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace onnx_light {
namespace expressions {

// ─────────────────────────── operator kinds ───────────────────────────

/**
 * @brief Binary operator kind used in the expression AST.
 *
 * The `^` and `&` operators are borrowed from Python's bitwise-xor / bitwise-and
 * syntax and are re-interpreted as `max` and `min` in this expression system,
 * matching the convention from `yobx/xexpressions`.
 */
enum class BinOpKind {
  Add,      ///< Addition: `a + b`.
  Sub,      ///< Subtraction: `a - b`.
  Mult,     ///< Multiplication: `a * b`.
  FloorDiv, ///< Floor (integer) division: `a // b`.
  Mod,      ///< Modulo: `a % b`.
  BitXor,   ///< Encodes `max(a, b)` using the `^` syntax.
  BitAnd,   ///< Encodes `min(a, b)` using the `&` syntax.
};

/**
 * @brief Unary operator kind used in the expression AST.
 */
enum class UnaryOpKind {
  USub, ///< Unary minus: `-a`.
  UAdd, ///< Unary plus: `+a` (identity).
};

// ──────────────────────────── AST nodes ───────────────────────────────

struct Node;
/**
 * @brief Owning pointer to an AST node.
 *
 * All AST construction and transformation functions return `NodePtr` values;
 * the tree owns its children and is freed when the root `NodePtr` goes out of
 * scope.
 */
using NodePtr = std::unique_ptr<Node>;

/**
 * @brief Abstract base class for all expression AST nodes.
 *
 * Every concrete node type derives from `Node` and overrides `clone()` to
 * produce a deep copy.  Nodes are always heap-allocated and owned by
 * `NodePtr`.
 */
struct Node {
  /// @brief Destroys the node and recursively frees all child nodes.
  virtual ~Node() = default;
  /**
   * @brief Returns a deep copy of this node and its entire sub-tree.
   * @returns An owning pointer to the cloned node.
   */
  virtual NodePtr clone() const = 0;
};

/**
 * @brief Leaf node representing a signed 64-bit integer constant.
 *
 * @code{.cpp}
 * auto c = std::make_unique<Constant>(42);
 * // unparse(*c) == "42"
 * @endcode
 */
struct Constant : Node {
  int64_t value; ///< The integer value of this constant.
  /**
   * @brief Constructs a Constant node.
   * @param v The integer value.
   */
  explicit Constant(int64_t v) : value(v) {}
  NodePtr clone() const override { return std::make_unique<Constant>(value); }
};

/**
 * @brief Leaf node representing a symbolic variable reference.
 *
 * @code{.cpp}
 * auto n = std::make_unique<Name>("batch");
 * // unparse(*n) == "batch"
 * @endcode
 */
struct Name : Node {
  std::string id; ///< The variable name (e.g. `"batch"`, `"seq_length"`).
  /**
   * @brief Constructs a Name node.
   * @param s The variable name string.
   */
  explicit Name(std::string s) : id(std::move(s)) {}
  NodePtr clone() const override { return std::make_unique<Name>(id); }
};

/**
 * @brief Interior node representing a binary arithmetic operation.
 *
 * @code{.cpp}
 * auto b = std::make_unique<BinOp>(
 *     std::make_unique<Name>("a"),
 *     BinOpKind::Add,
 *     std::make_unique<Constant>(1));
 * // unparse(*b) == "a+1"
 * @endcode
 */
struct BinOp : Node {
  NodePtr left;  ///< Left-hand operand.
  BinOpKind op;  ///< The binary operator.
  NodePtr right; ///< Right-hand operand.
  /**
   * @brief Constructs a BinOp node.
   * @param l Left operand (ownership transferred).
   * @param o The operator kind.
   * @param r Right operand (ownership transferred).
   */
  BinOp(NodePtr l, BinOpKind o, NodePtr r) : left(std::move(l)), op(o), right(std::move(r)) {}
  NodePtr clone() const override {
    return std::make_unique<BinOp>(left->clone(), op, right->clone());
  }
};

/**
 * @brief Interior node representing a unary arithmetic operation.
 *
 * @code{.cpp}
 * auto u = std::make_unique<UnaryOp>(UnaryOpKind::USub,
 *                                    std::make_unique<Name>("x"));
 * // unparse(*u) == "-x"
 * @endcode
 */
struct UnaryOp : Node {
  UnaryOpKind op;  ///< The unary operator.
  NodePtr operand; ///< The operand.
  /**
   * @brief Constructs a UnaryOp node.
   * @param o The operator kind.
   * @param n The operand (ownership transferred).
   */
  UnaryOp(UnaryOpKind o, NodePtr n) : op(o), operand(std::move(n)) {}
  NodePtr clone() const override { return std::make_unique<UnaryOp>(op, operand->clone()); }
};

/**
 * @brief Interior node representing a function call (e.g. `CeilToInt`, `Max`).
 *
 * The only function calls understood by `evaluate_expression` are
 * `CeilToInt(n, div)`, which computes ceiling division.  `Max(a, b)` is
 * syntactic sugar that `MaxToXorTransformer` rewrites to `a ^ b` before
 * evaluation.
 *
 * @code{.cpp}
 * // After MaxToXorTransformer, Max(a, b) becomes BinOp(a, BitXor, b).
 * // CeilToInt(n, 2) is evaluated as (n % 2 == 0) ? n/2 : n/2+1.
 * @endcode
 */
struct Call : Node {
  std::string func;          ///< The function name (e.g. `"CeilToInt"`, `"Max"`).
  std::vector<NodePtr> args; ///< Positional arguments (ownership held).
  /**
   * @brief Constructs a Call node.
   * @param f The function name.
   * @param a The argument list (ownership transferred).
   */
  Call(std::string f, std::vector<NodePtr> a) : func(std::move(f)), args(std::move(a)) {}
  NodePtr clone() const override {
    std::vector<NodePtr> cloned;
    cloned.reserve(args.size());
    for (const auto &a : args)
      cloned.push_back(a->clone());
    return std::make_unique<Call>(func, std::move(cloned));
  }
};

// ──────────────────────── parse / unparse ─────────────────────────────

/**
 * @brief Parses @p expr into an AST.
 *
 * The grammar follows Python operator precedence for the supported subset of
 * operators: `+`, `-`, `*`, `//`, `%`, `^` (max), `&` (min), unary `+`/`-`,
 * parentheses, and function calls with comma-separated arguments.
 *
 * Operator precedence (low to high):
 * - `^` (BitXor / max)
 * - `&` (BitAnd / min)
 * - `+`, `-`
 * - `*`, `//`, `%`
 * - unary `-`, `+`
 * - atoms (constants, names, parenthesised sub-expressions, calls)
 *
 * @param expr The expression string to parse.
 * @returns An owning pointer to the root of the parsed AST.
 * @throws std::runtime_error if the input contains a lexical or syntax error.
 *
 * @code{.cpp}
 * auto tree = parse("2*batch//batch");
 * // tree is a BinOp(BinOp(Constant(2)*Name("batch")), FloorDiv, Name("batch"))
 * @endcode
 */
NodePtr parse(const std::string &expr);

/**
 * @brief Converts @p node back to a canonical string expression.
 *
 * Follows Python's `ast.unparse` parenthesisation rules: operator precedence
 * determines where parentheses are inserted so that the output round-trips
 * through `parse()` to an equivalent AST.
 *
 * @param node The root AST node to convert.
 * @returns A string representation of the expression without extra spaces.
 *
 * @code{.cpp}
 * auto tree = parse("(a + b) * c");
 * std::string s = unparse(*tree);
 * // s == "(a+b)*c"
 * @endcode
 */
std::string unparse(const Node &node);

// ─────────────────────── simplification result ────────────────────────

/**
 * @brief Return type of simplify_expression.
 *
 * Holds either an `int64_t` when the expression reduces to a pure numeric
 * constant, or a `std::string` when symbolic variables remain after
 * simplification.
 *
 * Use `std::holds_alternative<int64_t>(r)` to check which case applies, or
 * call `simplify_result_to_string()` for a uniform string representation.
 *
 * @code{.cpp}
 * SimplifyResult r = simplify_expression("2*batch//batch");
 * assert(std::holds_alternative<int64_t>(r));
 * assert(std::get<int64_t>(r) == 2);
 *
 * SimplifyResult s = simplify_expression("a + b");
 * assert(std::holds_alternative<std::string>(s));
 * assert(std::get<std::string>(s) == "a+b");
 * @endcode
 */
using SimplifyResult = std::variant<int64_t, std::string>;

/**
 * @brief Returns a string representation of a SimplifyResult.
 *
 * Converts an `int64_t` result to its decimal string; returns the `std::string`
 * variant unchanged.
 *
 * @param r The result to convert.
 * @returns Decimal string for integer results; the simplified expression string otherwise.
 */
std::string simplify_result_to_string(const SimplifyResult &r);

// ─────────────────── high-level expression functions ──────────────────

/**
 * @brief Simplifies a symbolic or numeric expression string.
 *
 * Applies a pipeline of AST transformations twice:
 *  1. `CeilToIntTransformer` — expands `CeilToInt(x, n)` to `(x + n - 1) // n`.
 *  2. `SimpleSimplifyTransformer` — folds `x^x → x`, `x + 0 → x`, `x * 1 → x`.
 *  3. `MulDivCancellerTransformer` — cancels common symbolic factors, e.g. `2*x//x → 2`.
 *  4. `ExactMulDivConstantFolderTransformer` — folds `1024*a//2` → `512*a`.
 *  5. `MaxToXorTransformer` — rewrites `Max(a,b)` and `max(a,b)` to `a^b`.
 *  6. `ReorderCommutativeOpsTransformer` — sorts `+`/`*` operands alphabetically.
 *  7. `MaxIntTransformer` — folds `int_const ^ int_const` to `max(a, b)`.
 *
 * A final linear-combination visitor then collects the result as a
 * normalised sum of symbolic terms plus an integer constant.
 *
 * @param expr The expression string to simplify.
 * @returns An `int64_t` when the result is fully numeric, or a simplified
 *          `std::string` otherwise.  Returns `expr` unchanged when it
 *          contains syntax that the parser does not recognise (e.g. `"::"` in
 *          ONNX node names).
 *
 * @code{.cpp}
 * // Fully numeric result:
 * auto r1 = simplify_expression("2*batch//batch");
 * assert(std::get<int64_t>(r1) == 2);
 *
 * // Symbolic result:
 * auto r2 = simplify_expression("a + b - a");
 * assert(std::get<std::string>(r2) == "b");
 *
 * // CeilToInt expansion:
 * auto r3 = simplify_expression("CeilToInt(b+c, 2)");
 * assert(std::get<std::string>(r3) == "(1+b+c)//2");
 * @endcode
 */
SimplifyResult simplify_expression(const std::string &expr);

/**
 * @brief Returns the integer as-is (convenience overload for uniform call sites).
 *
 * @param value An integer that is already fully simplified.
 * @returns A `SimplifyResult` holding @p value.
 */
SimplifyResult simplify_expression(int64_t value);

/**
 * @brief Returns the non-zero coefficient map of the difference @p expr1 - (@p expr2).
 *
 * Builds the combined expression `expr1 - (expr2)`, runs the linear-combination
 * visitor, and returns only those variable coefficients that are non-zero.
 * An empty map indicates that the two expressions are equal under linear
 * arithmetic.
 *
 * @param expr1 The first expression string.
 * @param expr2 The second expression string.
 * @returns A map from variable name (or sub-expression key) to its integer
 *          coefficient in `expr1 - expr2`.  Zero-coefficient terms are omitted.
 *
 * @code{.cpp}
 * auto diff = simplify_two_expressions("s52+seq_length", "s52+s70");
 * // diff == {{"s70", -1}, {"seq_length", 1}}
 *
 * auto same = simplify_two_expressions("e*2", "e+e");
 * // same is empty — the two expressions are equal
 * @endcode
 */
std::map<std::string, int64_t> simplify_two_expressions(const std::string &expr1,
                                                        const std::string &expr2);

/**
 * @brief Evaluates @p expr with the variable assignments in @p context.
 *
 * Supported constructs:
 *  - Signed 64-bit integer constants.
 *  - Variable references resolved via @p context.
 *  - Binary operators `+`, `-`, `*`, `//` (floor division), `%` (modulo),
 *    `^` (max), `&` (min).
 *  - Unary `-`.
 *  - `CeilToInt(n, div)` — ceiling division: `(n % div == 0) ? n/div : n/div + 1`.
 *
 * @param expr    The expression string to evaluate.
 * @param context A map from variable name to its integer value.
 * @returns The integer result of evaluating the expression.
 * @throws std::runtime_error if the expression has a syntax error, references
 *         an unknown variable, or contains an unsupported node type.
 *
 * @code{.cpp}
 * int64_t v = evaluate_expression("x - y", {{"x", 5}, {"y", 6}});
 * // v == -1
 *
 * int64_t c = evaluate_expression("CeilToInt(7, 2)", {});
 * // c == 4
 * @endcode
 */
int64_t evaluate_expression(const std::string &expr,
                            const std::unordered_map<std::string, int64_t> &context);

/**
 * @brief Returns the set of variable names referenced in @p expr.
 *
 * Parses @p expr and walks the AST to collect every `Name` node.  If the
 * expression has a syntax error the function returns `{expr}` (a set
 * containing the original string), matching the Python reference behaviour.
 *
 * @param expr The expression string to scan.
 * @returns An unordered set of variable name strings.  Contains only @p expr
 *          itself when parsing fails.
 *
 * @code{.cpp}
 * auto tokens = parse_expression_tokens("a + b * c");
 * // tokens == {"a", "b", "c"}
 *
 * auto bad = parse_expression_tokens("a +");
 * // bad == {"a +"} (syntax error → original string returned)
 * @endcode
 */
std::unordered_set<std::string> parse_expression_tokens(const std::string &expr);

/**
 * @brief Renames variables in @p expr according to @p mapping.
 *
 * Also converts `Max(a, b)` calls to the `a^b` xor form before renaming.
 * The result has all spaces removed (matching the Python reference output).
 *
 * @param expr    The expression string to rename.
 * @param mapping A map from old variable name to new variable name.
 * @returns The renamed expression string (no spaces).
 * @throws std::runtime_error if @p expr cannot be parsed.
 *
 * @code{.cpp}
 * std::string r = rename_expression("s52 + seq_length", {{"s52", "B"}});
 * // r == "B+seq_length"
 *
 * std::string m = rename_expression("Max(s10, s3)", {{"s10", "E"}, {"s3", "D"}});
 * // m == "E^D"  (Max is rewritten to ^ before renaming)
 * @endcode
 */
std::string rename_expression(const std::string &expr,
                              const std::unordered_map<std::string, std::string> &mapping);

/**
 * @brief Renames variables in @p expression using @p replacements, then simplifies.
 *
 * Applies the following pipeline in order:
 *  1. Parse @p expression.
 *  2. Rewrite `Max(a, b)` → `a ^ b`.
 *  3. Apply the rename mapping.
 *  4. Apply `SimpleSimplifyTransformer`.
 *  5. Unparse and strip spaces.
 *
 * Returns @p expression unchanged if it has a syntax error.
 *
 * @param expression  The expression string to transform.
 * @param replacements A map from old variable name to new variable name.
 * @returns The renamed and simplified expression string (no spaces), or
 *          @p expression unchanged on parse failure.
 *
 * @code{.cpp}
 * std::string r = rename_dynamic_expression("s9+seq_length",
 *     {{"s9", "cache_length"}, {"seq_length", "seq_length"}});
 * // r == "cache_length+seq_length"
 * @endcode
 */
std::string
rename_dynamic_expression(const std::string &expression,
                          const std::unordered_map<std::string, std::string> &replacements);

/**
 * @brief Renames dynamic shape dimensions from internal names to user-visible ones.
 *
 * Frameworks such as `torch.export.export` produce many internal dimension
 * names (e.g. `s0`, `s1`, …) for dynamic shapes.  This function replaces
 * them with the canonical names supplied by the user via @p original.
 *
 * The algorithm iterates over @p constraints; for each entry it finds the
 * intersection of the equivalent-name set with @p original, picks the
 * lexicographically smallest match as the canonical name, and propagates it
 * to all aliases — unless the name starts with @p ban_prefix.
 *
 * @param constraints A map from each dimension name to the set of all dimension
 *                    names that are known to be equal to it (i.e. the equivalence
 *                    class).
 * @param original    The set of user-visible (preferred) dimension names.
 * @param ban_prefix  Names starting with this prefix are never selected as the
 *                    canonical replacement (default: `"DYN"`).
 * @returns A map `{internal_name → canonical_name}` covering all names in
 *          @p original (mapped to themselves) plus every name in @p constraints
 *          that was successfully resolved.
 *
 * @code{.cpp}
 * std::map<std::string, std::unordered_set<std::string>> constraints = {
 *     {"s0", {"batch", "s12"}},
 *     {"s12", {"batch", "s0"}},
 * };
 * std::unordered_set<std::string> original = {"batch"};
 * auto renamed = rename_dynamic_dimensions(constraints, original);
 * // renamed["s0"] == "batch"
 * // renamed["s12"] == "batch"
 * @endcode
 */
std::map<std::string, std::string>
rename_dynamic_dimensions(const std::map<std::string, std::unordered_set<std::string>> &constraints,
                          const std::unordered_set<std::string> &original,
                          const std::string &ban_prefix = "DYN");

// ─────────────────── dimension operation types ────────────────────────

/**
 * @brief A dimension value: either a concrete integer or a symbolic string.
 *
 * `int64_t` is used when the dimension is statically known; `std::string`
 * when it is symbolic (e.g. `"batch"` or `"seq_length+1"`).
 *
 * @code{.cpp}
 * DimType d1 = int64_t{64};       // concrete dimension
 * DimType d2 = std::string{"N"};  // symbolic dimension
 * @endcode
 */
using DimType = std::variant<int64_t, std::string>;

/**
 * @brief Returns a string representation of @p d.
 *
 * Converts an `int64_t` to its decimal string; returns the `std::string`
 * variant unchanged.
 *
 * @param d The dimension value to convert.
 * @returns Decimal string for integer dimensions; the symbol string otherwise.
 */
std::string dim_to_string(const DimType &d);

// ─────────────────────── dimension operations ─────────────────────────

/**
 * @brief Multiplies two dimensions.
 *
 * Returns `a * b` as an `int64_t` when both operands are integers.
 * Otherwise builds the expression `"(a)*(b)"` and simplifies it symbolically.
 *
 * @param a The first dimension (integer or symbolic string).
 * @param b The second dimension (integer or symbolic string).
 * @returns The product as an integer when both are concrete, or as a
 *          simplified string otherwise.
 *
 * @code{.cpp}
 * dim_mul(DimType{int64_t{3}}, DimType{int64_t{4}}) == DimType{int64_t{12}};
 * // dim_mul("n", 2) returns a string containing "n" and "2"
 * @endcode
 */
DimType dim_mul(const DimType &a, const DimType &b);

/**
 * @brief Multiplies a sequence of dimensions.
 *
 * Computes the product of all elements in @p args.  If every element is an
 * `int64_t` the result is an exact integer product; otherwise the expression
 * `"(a0)*(a1)*..."` is built and simplified symbolically.
 *
 * @param args Non-empty vector of dimensions.  Returns `int64_t{1}` for an
 *             empty vector.
 * @returns The product as an integer when all operands are concrete, or as a
 *          simplified string otherwise.
 *
 * @code{.cpp}
 * dim_multi_mul({DimType{int64_t{2}}, DimType{int64_t{3}}, DimType{int64_t{4}}})
 *     == DimType{int64_t{24}};
 * @endcode
 */
DimType dim_multi_mul(const std::vector<DimType> &args);

/**
 * @brief Adds two dimensions.
 *
 * Returns `a + b` as an `int64_t` when both are integers; otherwise builds
 * `"(a)+(b)"` and simplifies.
 *
 * @param a The first dimension.
 * @param b The second dimension.
 * @returns The sum.
 *
 * @code{.cpp}
 * dim_add(DimType{int64_t{3}}, DimType{int64_t{4}}) == DimType{int64_t{7}};
 * @endcode
 */
DimType dim_add(const DimType &a, const DimType &b);

/**
 * @brief Subtracts @p b from @p a.
 *
 * Returns `a - b` as an `int64_t` when both are integers; otherwise builds
 * `"(a)-(b)"` and simplifies.
 *
 * @param a The minuend.
 * @param b The subtrahend.
 * @returns The difference.
 *
 * @code{.cpp}
 * dim_sub(DimType{int64_t{10}}, DimType{int64_t{3}}) == DimType{int64_t{7}};
 * @endcode
 */
DimType dim_sub(const DimType &a, const DimType &b);

/**
 * @brief Floor-divides @p a by @p b.
 *
 * Assumes both values are non-negative (as is typical for ONNX shape
 * dimensions).  Returns `a // b` as an `int64_t` when both are integers;
 * otherwise builds `"(a)//(b)"` and simplifies.
 *
 * @param a The dividend.
 * @param b The divisor.
 * @returns The floor-division result.
 *
 * @code{.cpp}
 * dim_div(DimType{int64_t{7}}, DimType{int64_t{2}}) == DimType{int64_t{3}};
 * dim_div(DimType{std::string{"2*n"}}, DimType{int64_t{2}}) == DimType{std::string{"n"}};
 * @endcode
 */
DimType dim_div(const DimType &a, const DimType &b);

/**
 * @brief Computes @p a modulo @p b.
 *
 * Returns `a % b` as an `int64_t` when both are integers; otherwise builds
 * `"(a)%(b)"` and simplifies.
 *
 * @param a The dividend.
 * @param b The divisor.
 * @returns The remainder.
 *
 * @code{.cpp}
 * dim_mod(DimType{int64_t{10}}, DimType{int64_t{3}}) == DimType{int64_t{1}};
 * @endcode
 */
DimType dim_mod(const DimType &a, const DimType &b);

/**
 * @brief Returns the maximum of @p a and @p b.
 *
 * Returns `max(a, b)` as an `int64_t` when both are integers; otherwise
 * builds `"(a)^(b)"` (the xor encoding of max) and simplifies.
 *
 * @param a The first dimension.
 * @param b The second dimension.
 * @returns The maximum of the two dimensions.
 *
 * @code{.cpp}
 * dim_max(DimType{int64_t{7}}, DimType{int64_t{3}}) == DimType{int64_t{7}};
 * // dim_max("n", "n") simplifies to DimType{std::string{"n"}} (x^x → x)
 * @endcode
 */
DimType dim_max(const DimType &a, const DimType &b);

/**
 * @brief Returns the minimum of @p a and @p b.
 *
 * Returns `min(a, b)` as an `int64_t` when both are integers; otherwise
 * builds `"(a)&(b)"` (the ampersand encoding of min) and simplifies.
 *
 * @param a The first dimension.
 * @param b The second dimension.
 * @returns The minimum of the two dimensions.
 *
 * @code{.cpp}
 * dim_min(DimType{int64_t{2}}, DimType{int64_t{9}}) == DimType{int64_t{2}};
 * @endcode
 */
DimType dim_min(const DimType &a, const DimType &b);

} // namespace expressions
} // namespace onnx_light
