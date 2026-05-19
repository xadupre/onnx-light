// SPDX-License-Identifier: Apache-2.0
//
// C++ implementation of symbolic dimension expression utilities.
// Ported from yobx/xexpressions (https://github.com/xadupre/yet-another-onnx-builder).

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

/** Binary operator kind used in the expression AST. */
enum class BinOpKind {
  Add,      ///< + (addition)
  Sub,      ///< - (subtraction)
  Mult,     ///< * (multiplication)
  FloorDiv, ///< // (floor division)
  Mod,      ///< % (modulo)
  BitXor,   ///< ^ (encodes max in this system)
  BitAnd,   ///< & (encodes min in this system)
};

/** Unary operator kind used in the expression AST. */
enum class UnaryOpKind {
  USub, ///< unary minus  (-)
  UAdd, ///< unary plus   (+)
};

// ──────────────────────────── AST nodes ───────────────────────────────

struct Node;
/** Owning pointer to an AST node. */
using NodePtr = std::unique_ptr<Node>;

/** Base class for all expression AST nodes. */
struct Node {
  virtual ~Node() = default;
  /** Returns a deep copy of this node. */
  virtual NodePtr clone() const = 0;
};

/** Integer constant node. */
struct Constant : Node {
  int64_t value;
  explicit Constant(int64_t v) : value(v) {}
  NodePtr clone() const override { return std::make_unique<Constant>(value); }
};

/** Variable reference node. */
struct Name : Node {
  std::string id;
  explicit Name(std::string s) : id(std::move(s)) {}
  NodePtr clone() const override { return std::make_unique<Name>(id); }
};

/** Binary operation node. */
struct BinOp : Node {
  NodePtr left;
  BinOpKind op;
  NodePtr right;
  BinOp(NodePtr l, BinOpKind o, NodePtr r) : left(std::move(l)), op(o), right(std::move(r)) {}
  NodePtr clone() const override {
    return std::make_unique<BinOp>(left->clone(), op, right->clone());
  }
};

/** Unary operation node. */
struct UnaryOp : Node {
  UnaryOpKind op;
  NodePtr operand;
  UnaryOp(UnaryOpKind o, NodePtr n) : op(o), operand(std::move(n)) {}
  NodePtr clone() const override { return std::make_unique<UnaryOp>(op, operand->clone()); }
};

/** Function call node (e.g. CeilToInt, Max). */
struct Call : Node {
  std::string func;
  std::vector<NodePtr> args;
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
 * Parses @p expr into an AST.
 *
 * The grammar follows Python operator precedence for the supported subset of
 * operators: +, -, *, //, %, ^ (max), & (min), unary +/-, parentheses, and
 * function calls with comma-separated arguments.
 *
 * Throws std::runtime_error on lexical or syntax errors.
 */
NodePtr parse(const std::string &expr);

/**
 * Converts @p node back to a canonical string expression.
 *
 * Follows Python's ast.unparse parenthesisation rules: operator precedence
 * determines where parentheses are inserted so that the output round-trips
 * through parse() to the same AST.
 */
std::string unparse(const Node &node);

// ─────────────────────── simplification result ────────────────────────

/**
 * Return type of simplify_expression.
 *
 * Holds either an int64_t (when the expression reduces to a numeric constant)
 * or a std::string (when symbolic variables remain).
 */
using SimplifyResult = std::variant<int64_t, std::string>;

/** Returns a string representation of a SimplifyResult. */
std::string simplify_result_to_string(const SimplifyResult &r);

// ─────────────────── high-level expression functions ──────────────────

/**
 * Simplifies a symbolic or numeric expression string.
 *
 * Applies a pipeline of AST transformations (CeilToInt expansion, constant
 * folding, mul/div cancellation, max-to-xor conversion, commutative
 * reordering, linear-combination normalisation) and returns either an int64_t
 * when the result is fully numeric, or a simplified string otherwise.
 */
SimplifyResult simplify_expression(const std::string &expr);

/** Returns the integer as-is (convenience overload). */
SimplifyResult simplify_expression(int64_t value);

/**
 * Returns the non-zero coefficient map of the difference @p expr1 - (@p expr2).
 *
 * Keys are variable names (or sub-expression strings); values are their
 * integer coefficients in the difference.  An empty map means the two
 * expressions are equal under linear arithmetic.
 */
std::map<std::string, int64_t> simplify_two_expressions(const std::string &expr1,
                                                        const std::string &expr2);

/**
 * Evaluates @p expr with the variable assignments in @p context.
 *
 * Supports integers, the binary operators +, -, *, //, %, ^ (max), &
 * (min), unary -, and the CeilToInt(n, div) function.
 *
 * Throws std::runtime_error for unknown variables, non-integer constants, or
 * unsupported AST nodes.  Throws std::runtime_error for syntax errors.
 */
int64_t evaluate_expression(const std::string &expr,
                            const std::unordered_map<std::string, int64_t> &context);

/**
 * Returns the set of variable names referenced in @p expr.
 *
 * If @p expr has a syntax error the function returns {@p expr} (a set
 * containing the original string), matching the Python reference behaviour.
 */
std::unordered_set<std::string> parse_expression_tokens(const std::string &expr);

/**
 * Renames variables in @p expr according to @p mapping.
 *
 * Also converts Max(a,b) calls to the a^b xor form before renaming.
 * Throws std::runtime_error if @p expr cannot be parsed.
 */
std::string rename_expression(const std::string &expr,
                              const std::unordered_map<std::string, std::string> &mapping);

/**
 * Renames variables in @p expression using @p replacements, then simplifies.
 *
 * Applies Max-to-xor conversion, the rename mapping, and a lightweight
 * simplification pass, stripping all spaces from the result.
 * Returns @p expression unchanged if it has a syntax error.
 */
std::string
rename_dynamic_expression(const std::string &expression,
                          const std::unordered_map<std::string, std::string> &replacements);

/**
 * Renames dynamic shape dimensions from internal names to user-visible ones.
 *
 * @p constraints maps each dimension name to the set of names that are equal
 * to it.  @p original is the set of preferred (user-visible) names.
 * @p ban_prefix prevents any name starting with that prefix from being used
 * as the canonical replacement.
 *
 * Returns a mapping {internal_name -> canonical_name}.
 */
std::map<std::string, std::string>
rename_dynamic_dimensions(const std::map<std::string, std::unordered_set<std::string>> &constraints,
                          const std::unordered_set<std::string> &original,
                          const std::string &ban_prefix = "DYN");

// ─────────────────── dimension operation types ────────────────────────

/** A dimension value: either a concrete integer or a symbolic string. */
using DimType = std::variant<int64_t, std::string>;

/** Returns a string representation of @p d (integer as decimal, string as-is). */
std::string dim_to_string(const DimType &d);

// ─────────────────────── dimension operations ─────────────────────────

/** Multiplies two dimensions; simplifies symbolically when at least one is a string. */
DimType dim_mul(const DimType &a, const DimType &b);

/**
 * Multiplies a sequence of dimensions.
 *
 * If all values are integers the product is computed exactly; otherwise the
 * expression is built and simplified symbolically.
 */
DimType dim_multi_mul(const std::vector<DimType> &args);

/** Adds two dimensions. */
DimType dim_add(const DimType &a, const DimType &b);

/** Subtracts @p b from @p a. */
DimType dim_sub(const DimType &a, const DimType &b);

/** Floor-divides @p a by @p b (assumes positive values). */
DimType dim_div(const DimType &a, const DimType &b);

/** Computes @p a modulo @p b. */
DimType dim_mod(const DimType &a, const DimType &b);

/** Returns the maximum of @p a and @p b. */
DimType dim_max(const DimType &a, const DimType &b);

/** Returns the minimum of @p a and @p b. */
DimType dim_min(const DimType &a, const DimType &b);

} // namespace expressions
} // namespace onnx_light
