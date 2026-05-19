// SPDX-License-Identifier: Apache-2.0
//
// C++ implementation of symbolic dimension expression utilities.
// Ported from yobx/xexpressions (https://github.com/xadupre/yet-another-onnx-builder).

#include "expressions.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace onnx_light {
namespace expressions {

// ═══════════════════════════════════════════════════════════════════════════
// Tokenizer
// ═══════════════════════════════════════════════════════════════════════════

enum class TokenKind {
  Integer,
  Name,
  Plus,
  Minus,
  Star,
  DoubleSlash,
  Percent,
  Caret,
  Ampersand,
  LParen,
  RParen,
  Comma,
  Eof
};

struct Token {
  TokenKind kind{TokenKind::Eof};
  std::string text;
  int64_t value{0};
};

class Tokenizer {
public:
  explicit Tokenizer(const std::string &input) : input_(input), pos_(0) {}

  Token next() {
    skip_ws();
    if (pos_ >= input_.size())
      return {TokenKind::Eof, ""};

    char c = input_[pos_];

    // Integer literal
    if (std::isdigit(static_cast<unsigned char>(c))) {
      size_t start = pos_;
      while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_])))
        ++pos_;
      Token t;
      t.kind = TokenKind::Integer;
      t.text = input_.substr(start, pos_ - start);
      t.value = std::stoll(t.text);
      return t;
    }

    // Identifier
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      size_t start = pos_;
      while (pos_ < input_.size() &&
             (std::isalnum(static_cast<unsigned char>(input_[pos_])) || input_[pos_] == '_'))
        ++pos_;
      return {TokenKind::Name, input_.substr(start, pos_ - start)};
    }

    // Double slash //
    if (c == '/' && pos_ + 1 < input_.size() && input_[pos_ + 1] == '/') {
      pos_ += 2;
      return {TokenKind::DoubleSlash, "//"};
    }

    ++pos_;
    switch (c) {
    case '+':
      return {TokenKind::Plus, "+"};
    case '-':
      return {TokenKind::Minus, "-"};
    case '*':
      return {TokenKind::Star, "*"};
    case '%':
      return {TokenKind::Percent, "%"};
    case '^':
      return {TokenKind::Caret, "^"};
    case '&':
      return {TokenKind::Ampersand, "&"};
    case '(':
      return {TokenKind::LParen, "("};
    case ')':
      return {TokenKind::RParen, ")"};
    case ',':
      return {TokenKind::Comma, ","};
    default:
      throw std::runtime_error(std::string("Unexpected character '") + c + "' in expression");
    }
  }

  Token peek() {
    size_t saved = pos_;
    Token t = next();
    pos_ = saved;
    return t;
  }

private:
  void skip_ws() {
    while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_])))
      ++pos_;
  }

  const std::string &input_;
  size_t pos_;
};

// ═══════════════════════════════════════════════════════════════════════════
// Recursive-descent parser
//
// Operator precedence (low → high), matching Python's:
//   ^ (BitXor / max)  lowest
//   & (BitAnd / min)
//   + -
//   * // %
//   unary - +
//   atoms            highest
// ═══════════════════════════════════════════════════════════════════════════

class Parser {
public:
  explicit Parser(const std::string &input) : tok_(input) { advance(); }

  NodePtr parse_expr() { return parse_xor(); }

  void expect_eof() const {
    if (cur_.kind != TokenKind::Eof)
      throw std::runtime_error("Unexpected token '" + cur_.text + "' after expression");
  }

private:
  void advance() { cur_ = tok_.next(); }

  NodePtr parse_xor() {
    NodePtr lhs = parse_and();
    while (cur_.kind == TokenKind::Caret) {
      advance();
      NodePtr rhs = parse_and();
      lhs = std::make_unique<BinOp>(std::move(lhs), BinOpKind::BitXor, std::move(rhs));
    }
    return lhs;
  }

  NodePtr parse_and() {
    NodePtr lhs = parse_add();
    while (cur_.kind == TokenKind::Ampersand) {
      advance();
      NodePtr rhs = parse_add();
      lhs = std::make_unique<BinOp>(std::move(lhs), BinOpKind::BitAnd, std::move(rhs));
    }
    return lhs;
  }

  NodePtr parse_add() {
    NodePtr lhs = parse_mul();
    while (cur_.kind == TokenKind::Plus || cur_.kind == TokenKind::Minus) {
      BinOpKind op = (cur_.kind == TokenKind::Plus) ? BinOpKind::Add : BinOpKind::Sub;
      advance();
      NodePtr rhs = parse_mul();
      lhs = std::make_unique<BinOp>(std::move(lhs), op, std::move(rhs));
    }
    return lhs;
  }

  NodePtr parse_mul() {
    NodePtr lhs = parse_unary();
    while (cur_.kind == TokenKind::Star || cur_.kind == TokenKind::DoubleSlash ||
           cur_.kind == TokenKind::Percent) {
      BinOpKind op;
      switch (cur_.kind) {
      case TokenKind::Star:
        op = BinOpKind::Mult;
        break;
      case TokenKind::DoubleSlash:
        op = BinOpKind::FloorDiv;
        break;
      default:
        op = BinOpKind::Mod;
        break;
      }
      advance();
      NodePtr rhs = parse_unary();
      lhs = std::make_unique<BinOp>(std::move(lhs), op, std::move(rhs));
    }
    return lhs;
  }

  NodePtr parse_unary() {
    if (cur_.kind == TokenKind::Minus) {
      advance();
      return std::make_unique<UnaryOp>(UnaryOpKind::USub, parse_unary());
    }
    if (cur_.kind == TokenKind::Plus) {
      advance();
      return std::make_unique<UnaryOp>(UnaryOpKind::UAdd, parse_unary());
    }
    return parse_atom();
  }

  NodePtr parse_atom() {
    if (cur_.kind == TokenKind::Integer) {
      int64_t v = cur_.value;
      advance();
      return std::make_unique<Constant>(v);
    }
    if (cur_.kind == TokenKind::Name) {
      std::string nm = cur_.text;
      advance();
      if (cur_.kind == TokenKind::LParen) {
        advance(); // '('
        std::vector<NodePtr> args;
        if (cur_.kind != TokenKind::RParen) {
          args.push_back(parse_expr());
          while (cur_.kind == TokenKind::Comma) {
            advance();
            args.push_back(parse_expr());
          }
        }
        if (cur_.kind != TokenKind::RParen)
          throw std::runtime_error("Expected ')' after function arguments");
        advance(); // ')'
        return std::make_unique<Call>(nm, std::move(args));
      }
      return std::make_unique<Name>(nm);
    }
    if (cur_.kind == TokenKind::LParen) {
      advance(); // '('
      NodePtr inner = parse_expr();
      if (cur_.kind != TokenKind::RParen)
        throw std::runtime_error("Expected ')'");
      advance(); // ')'
      return inner;
    }
    throw std::runtime_error("Unexpected token '" + cur_.text + "' in expression");
  }

  Tokenizer tok_;
  Token cur_;
};

NodePtr parse(const std::string &expr) {
  Parser p(expr);
  NodePtr n = p.parse_expr();
  p.expect_eof();
  return n;
}

// ═══════════════════════════════════════════════════════════════════════════
// Unparser
// Produces a string that round-trips through parse(), following Python's
// ast.unparse parenthesisation rules.
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// Python precedence values (higher == tighter binding)
int binop_prec(BinOpKind op) {
  switch (op) {
  case BinOpKind::BitXor:
    return 1;
  case BinOpKind::BitAnd:
    return 2;
  case BinOpKind::Add:
  case BinOpKind::Sub:
    return 3;
  case BinOpKind::Mult:
  case BinOpKind::FloorDiv:
  case BinOpKind::Mod:
    return 4;
  }
  return 0;
}

bool binop_commutative(BinOpKind op) {
  return op == BinOpKind::Add || op == BinOpKind::Mult || op == BinOpKind::BitXor ||
         op == BinOpKind::BitAnd;
}

const char *binop_sym(BinOpKind op) {
  switch (op) {
  case BinOpKind::Add:
    return "+";
  case BinOpKind::Sub:
    return "-";
  case BinOpKind::Mult:
    return "*";
  case BinOpKind::FloorDiv:
    return "//";
  case BinOpKind::Mod:
    return "%";
  case BinOpKind::BitXor:
    return "^";
  case BinOpKind::BitAnd:
    return "&";
  }
  return "?";
}

// Effective precedence when used as a child operand.
// Name/Constant/Call never need parentheses (very high precedence).
int node_prec(const Node &node) {
  if (const auto *b = dynamic_cast<const BinOp *>(&node))
    return binop_prec(b->op);
  if (dynamic_cast<const UnaryOp *>(&node))
    return 5; // unary > all binary ops
  return 100; // Constant, Name, Call: atomic
}

std::string unparse_node(const Node &node);

std::string unparse_node(const Node &node) {
  if (const auto *c = dynamic_cast<const Constant *>(&node))
    return std::to_string(c->value);

  if (const auto *n = dynamic_cast<const Name *>(&node))
    return n->id;

  if (const auto *b = dynamic_cast<const BinOp *>(&node)) {
    int prec = binop_prec(b->op);
    bool comm = binop_commutative(b->op);

    // Left operand: parenthesise when its precedence is strictly lower.
    int lp = node_prec(*b->left);
    std::string ls;
    if (lp < prec)
      ls = "(" + unparse_node(*b->left) + ")";
    else
      ls = unparse_node(*b->left);

    // Right operand: parenthesise when strictly lower, or equal for
    // non-commutative operators (right-associativity would change meaning).
    int rp = node_prec(*b->right);
    std::string rs;
    bool rpar = (rp < prec) || (!comm && rp == prec);
    if (rpar)
      rs = "(" + unparse_node(*b->right) + ")";
    else
      rs = unparse_node(*b->right);

    return ls + binop_sym(b->op) + rs;
  }

  if (const auto *u = dynamic_cast<const UnaryOp *>(&node)) {
    const char *sym = (u->op == UnaryOpKind::USub) ? "-" : "+";
    // Wrap operand in parens when it is itself a BinOp (lower prec).
    int op = node_prec(*u->operand);
    std::string operand_s;
    if (op < 5)
      operand_s = "(" + unparse_node(*u->operand) + ")";
    else
      operand_s = unparse_node(*u->operand);
    return std::string(sym) + operand_s;
  }

  if (const auto *call = dynamic_cast<const Call *>(&node)) {
    std::string s = call->func + "(";
    for (size_t i = 0; i < call->args.size(); ++i) {
      if (i > 0)
        s += ",";
      s += unparse_node(*call->args[i]);
    }
    s += ")";
    return s;
  }

  throw std::runtime_error("unparse: unknown node type");
}

} // namespace

std::string unparse(const Node &node) { return unparse_node(node); }

// ═══════════════════════════════════════════════════════════════════════════
// Transformer base
// ═══════════════════════════════════════════════════════════════════════════

class Transformer {
public:
  virtual ~Transformer() = default;

  virtual NodePtr visit(NodePtr node) {
    if (!node)
      return node;
    if (auto *c = dynamic_cast<Constant *>(node.get()))
      return visit_Constant(std::unique_ptr<Constant>(static_cast<Constant *>(node.release())));
    if (auto *n = dynamic_cast<Name *>(node.get()))
      return visit_Name(std::unique_ptr<Name>(static_cast<Name *>(node.release())));
    if (auto *b = dynamic_cast<BinOp *>(node.get()))
      return visit_BinOp(std::unique_ptr<BinOp>(static_cast<BinOp *>(node.release())));
    if (auto *u = dynamic_cast<UnaryOp *>(node.get()))
      return visit_UnaryOp(std::unique_ptr<UnaryOp>(static_cast<UnaryOp *>(node.release())));
    if (auto *call = dynamic_cast<Call *>(node.get()))
      return visit_Call(std::unique_ptr<Call>(static_cast<Call *>(node.release())));
    return node;
  }

  virtual NodePtr visit_Constant(std::unique_ptr<Constant> n) { return n; }
  virtual NodePtr visit_Name(std::unique_ptr<Name> n) { return n; }
  virtual NodePtr visit_BinOp(std::unique_ptr<BinOp> n) { return generic_visit(std::move(n)); }
  virtual NodePtr visit_UnaryOp(std::unique_ptr<UnaryOp> n) { return generic_visit(std::move(n)); }
  virtual NodePtr visit_Call(std::unique_ptr<Call> n) { return generic_visit(std::move(n)); }

  NodePtr generic_visit(NodePtr n) {
    if (auto *b = dynamic_cast<BinOp *>(n.get())) {
      b->left = visit(std::move(b->left));
      b->right = visit(std::move(b->right));
    } else if (auto *u = dynamic_cast<UnaryOp *>(n.get())) {
      u->operand = visit(std::move(u->operand));
    } else if (auto *call = dynamic_cast<Call *>(n.get())) {
      for (auto &a : call->args)
        a = visit(std::move(a));
    }
    return n;
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// CeilToIntTransformer: CeilToInt(x, n) → (x + n − 1) // n
// ═══════════════════════════════════════════════════════════════════════════

class CeilToIntTransformer : public Transformer {
public:
  NodePtr visit_Call(std::unique_ptr<Call> n) override {
    for (auto &a : n->args)
      a = visit(std::move(a));
    if (n->func == "CeilToInt" && n->args.size() == 2) {
      NodePtr x = std::move(n->args[0]);
      NodePtr ndiv = std::move(n->args[1]);
      NodePtr n_minus_1;
      if (const auto *nc = dynamic_cast<const Constant *>(ndiv.get()))
        n_minus_1 = std::make_unique<Constant>(nc->value - 1);
      else
        n_minus_1 =
            std::make_unique<BinOp>(ndiv->clone(), BinOpKind::Sub, std::make_unique<Constant>(1));
      NodePtr numerator =
          std::make_unique<BinOp>(std::move(x), BinOpKind::Add, std::move(n_minus_1));
      return std::make_unique<BinOp>(std::move(numerator), BinOpKind::FloorDiv, std::move(ndiv));
    }
    return n;
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// SimpleSimplifyTransformer
//   x^x → x   |   x+0 → x, 0+x → x   |   x*1 → x, 1*x → x
// ═══════════════════════════════════════════════════════════════════════════

class SimpleSimplifyTransformer : public Transformer {
public:
  NodePtr visit_BinOp(std::unique_ptr<BinOp> n) override {
    n->left = visit(std::move(n->left));
    n->right = visit(std::move(n->right));

    if (n->op == BinOpKind::BitXor) {
      const auto *nl = dynamic_cast<const Name *>(n->left.get());
      const auto *nr = dynamic_cast<const Name *>(n->right.get());
      if (nl && nr && nl->id == nr->id)
        return std::move(n->left);
    }
    if (n->op == BinOpKind::Add) {
      if (const auto *c = dynamic_cast<const Constant *>(n->left.get()))
        if (c->value == 0)
          return std::move(n->right);
      if (const auto *c = dynamic_cast<const Constant *>(n->right.get()))
        if (c->value == 0)
          return std::move(n->left);
    }
    if (n->op == BinOpKind::Mult) {
      if (const auto *c = dynamic_cast<const Constant *>(n->left.get()))
        if (c->value == 1)
          return std::move(n->right);
      if (const auto *c = dynamic_cast<const Constant *>(n->right.get()))
        if (c->value == 1)
          return std::move(n->left);
    }
    return n;
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// MulDivCancellerTransformer
// Cancels common factors in multiplicative chains, e.g. 2*x//x → 2
// ═══════════════════════════════════════════════════════════════════════════

namespace {

void flatten_mul_div(const Node &node, std::vector<NodePtr> &num, std::vector<NodePtr> &den) {
  if (const auto *b = dynamic_cast<const BinOp *>(&node)) {
    if (b->op == BinOpKind::Mult) {
      flatten_mul_div(*b->left, num, den);
      flatten_mul_div(*b->right, num, den);
      return;
    }
    if (b->op == BinOpKind::FloorDiv) {
      std::vector<NodePtr> ln, ld, rn, rd;
      flatten_mul_div(*b->left, ln, ld);
      flatten_mul_div(*b->right, rn, rd);
      for (auto &x : ln)
        num.push_back(std::move(x));
      for (auto &x : ld)
        den.push_back(std::move(x));
      for (auto &x : rn)
        den.push_back(std::move(x)); // swapped
      for (auto &x : rd)
        num.push_back(std::move(x)); // swapped
      return;
    }
  }
  num.push_back(node.clone());
}

NodePtr build_product(std::vector<NodePtr> &factors) {
  if (factors.empty())
    return std::make_unique<Constant>(1);
  NodePtr res = std::move(factors[0]);
  for (size_t i = 1; i < factors.size(); ++i)
    res = std::make_unique<BinOp>(std::move(res), BinOpKind::Mult, std::move(factors[i]));
  return res;
}

} // namespace

class MulDivCancellerTransformer : public Transformer {
public:
  NodePtr visit_BinOp(std::unique_ptr<BinOp> n) override {
    n = std::unique_ptr<BinOp>(static_cast<BinOp *>(generic_visit(std::move(n)).release()));

    if (n->op != BinOpKind::Mult && n->op != BinOpKind::FloorDiv)
      return n;

    std::vector<NodePtr> num, den;
    flatten_mul_div(*n, num, den);

    std::vector<std::string> nk, dk;
    for (const auto &x : num)
      nk.push_back(unparse(*x));
    for (const auto &x : den)
      dk.push_back(unparse(*x));

    std::map<std::string, int> nc, dc;
    for (const auto &k : nk)
      nc[k]++;
    for (const auto &k : dk)
      dc[k]++;

    bool any_common = false;
    for (const auto &[k, _] : nc)
      if (dc.count(k)) {
        any_common = true;
        break;
      }
    if (!any_common)
      return n;

    for (const auto &[k, cnt] : dc) {
      if (nc.count(k)) {
        int cancel = std::min(nc[k], dc[k]);
        nc[k] -= cancel;
        dc[k] -= cancel;
      }
    }

    std::vector<NodePtr> rn, rd;
    std::map<std::string, int> needed_n = nc, needed_d = dc;
    for (size_t i = 0; i < num.size(); ++i) {
      const std::string &k = nk[i];
      if (needed_n.count(k) && needed_n[k] > 0) {
        rn.push_back(std::move(num[i]));
        needed_n[k]--;
      }
    }
    for (size_t i = 0; i < den.size(); ++i) {
      const std::string &k = dk[i];
      if (needed_d.count(k) && needed_d[k] > 0) {
        rd.push_back(std::move(den[i]));
        needed_d[k]--;
      }
    }

    NodePtr numer = build_product(rn);
    if (rd.empty())
      return numer;
    NodePtr denom = build_product(rd);
    return std::make_unique<BinOp>(std::move(numer), BinOpKind::FloorDiv, std::move(denom));
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// ExactMulDivConstantFolderTransformer
// Folds integer constants in mul/div chains when the division is exact.
// e.g. 1024*a//2 → 512*a
// ═══════════════════════════════════════════════════════════════════════════

class ExactMulDivConstantFolderTransformer : public Transformer {
public:
  NodePtr visit_BinOp(std::unique_ptr<BinOp> n) override {
    n = std::unique_ptr<BinOp>(static_cast<BinOp *>(generic_visit(std::move(n)).release()));

    if (n->op != BinOpKind::Mult && n->op != BinOpKind::FloorDiv)
      return n;

    std::vector<NodePtr> num, den;
    flatten_mul_div(*n, num, den);

    int64_t num_c = 1, den_c = 1;
    std::vector<NodePtr> num_other, den_other;

    for (auto &x : num) {
      if (const auto *c = dynamic_cast<const Constant *>(x.get()))
        num_c *= c->value;
      else
        num_other.push_back(std::move(x));
    }
    for (auto &x : den) {
      if (const auto *c = dynamic_cast<const Constant *>(x.get()))
        den_c *= c->value;
      else
        den_other.push_back(std::move(x));
    }

    if (den_c == 0)
      return n;
    if (!den_other.empty())
      return n;
    if (num_c % den_c != 0)
      return n;

    int64_t folded = num_c / den_c;
    std::vector<NodePtr> factors;
    if (folded != 1 || num_other.empty())
      factors.push_back(std::make_unique<Constant>(folded));
    for (auto &x : num_other)
      factors.push_back(std::move(x));

    return build_product(factors);
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// MaxToXorTransformer: max(a,b) / Max(a,b) → a^b
// ═══════════════════════════════════════════════════════════════════════════

class MaxToXorTransformer : public Transformer {
public:
  NodePtr visit_Call(std::unique_ptr<Call> n) override {
    for (auto &a : n->args)
      a = visit(std::move(a));
    if ((n->func == "max" || n->func == "Max") && n->args.size() == 2) {
      return std::make_unique<BinOp>(std::move(n->args[0]), BinOpKind::BitXor,
                                     std::move(n->args[1]));
    }
    return n;
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// ReorderCommutativeOpsTransformer
// Sorts operands of + and * chains alphabetically by their unparse string.
// ═══════════════════════════════════════════════════════════════════════════

namespace {

void flatten_chain(const Node &node, BinOpKind op, std::vector<NodePtr> &out) {
  if (const auto *b = dynamic_cast<const BinOp *>(&node)) {
    if (b->op == op) {
      flatten_chain(*b->left, op, out);
      flatten_chain(*b->right, op, out);
      return;
    }
  }
  out.push_back(node.clone());
}

NodePtr rebuild_chain(std::vector<NodePtr> &ops, BinOpKind op) {
  NodePtr res = std::move(ops[0]);
  for (size_t i = 1; i < ops.size(); ++i)
    res = std::make_unique<BinOp>(std::move(res), op, std::move(ops[i]));
  return res;
}

} // namespace

class ReorderCommutativeOpsTransformer : public Transformer {
public:
  NodePtr visit_BinOp(std::unique_ptr<BinOp> n) override {
    n = std::unique_ptr<BinOp>(static_cast<BinOp *>(generic_visit(std::move(n)).release()));

    if (n->op != BinOpKind::Add && n->op != BinOpKind::Mult)
      return n;

    std::vector<NodePtr> operands;
    flatten_chain(*n, n->op, operands);

    std::sort(operands.begin(), operands.end(),
              [](const NodePtr &a, const NodePtr &b) { return unparse(*a) < unparse(*b); });

    return rebuild_chain(operands, n->op);
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// MaxIntTransformer: integer_constant ^ integer_constant → max result
// ═══════════════════════════════════════════════════════════════════════════

class MaxIntTransformer : public Transformer {
public:
  NodePtr visit_BinOp(std::unique_ptr<BinOp> n) override {
    n->left = visit(std::move(n->left));
    n->right = visit(std::move(n->right));

    if (n->op == BinOpKind::BitXor) {
      const auto *cl = dynamic_cast<const Constant *>(n->left.get());
      const auto *cr = dynamic_cast<const Constant *>(n->right.get());
      if (cl && cr)
        return std::make_unique<Constant>(std::max(cl->value, cr->value));
    }
    return n;
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// ExpressionSimplifierAddVisitor
// Collects a linear combination { variable_string → coefficient } from a
// simplified expression tree; constant terms accumulate separately.
// Mirrors Python's ExpressionSimplifierAddVisitor.make_simplified().
// ═══════════════════════════════════════════════════════════════════════════

struct AddVisitorResult {
  // Ordered list of (variable_key, coefficient) pairs preserving visit order.
  std::vector<std::pair<std::string, int64_t>> coeffs;
  int64_t const_term{0};

  // Returns true when the coefficients map is logically empty (no keys were
  // ever inserted, so the expression is a pure constant).
  bool coeffs_empty() const { return coeffs.empty(); }

  void add_coeff(const std::string &key, int64_t delta) {
    for (auto &[k, v] : coeffs) {
      if (k == key) {
        v += delta;
        return;
      }
    }
    coeffs.emplace_back(key, delta);
  }
};

// Returns true if `var` contains `//` or `%` outside balanced parentheses,
// meaning the expression needs to be wrapped in parens when used as a
// multiplicand (since `*` and `//` share the same precedence level).
static bool needs_mul_parens(const std::string &var) {
  int depth = 0;
  for (size_t i = 0; i < var.size(); ++i) {
    char c = var[i];
    if (c == '(')
      ++depth;
    else if (c == ')')
      --depth;
    else if (depth == 0) {
      if (c == '%')
        return true;
      if (i + 1 < var.size() && var[i] == '/' && var[i + 1] == '/')
        return true;
    }
  }
  return false;
}

static void run_add_visitor(const Node &node, AddVisitorResult &res);

static void run_add_visitor(const Node &node, AddVisitorResult &res) {
  if (const auto *c = dynamic_cast<const Constant *>(&node)) {
    res.const_term += c->value;
    return;
  }
  if (const auto *n = dynamic_cast<const Name *>(&node)) {
    res.add_coeff(n->id, 1);
    return;
  }
  if (const auto *b = dynamic_cast<const BinOp *>(&node)) {
    if (b->op == BinOpKind::Add) {
      run_add_visitor(*b->left, res);
      run_add_visitor(*b->right, res);
      return;
    }
    if (b->op == BinOpKind::Sub) {
      run_add_visitor(*b->left, res);
      AddVisitorResult neg;
      run_add_visitor(*b->right, neg);
      for (const auto &[k, v] : neg.coeffs)
        res.add_coeff(k, -v);
      res.const_term -= neg.const_term;
      return;
    }
    if (b->op == BinOpKind::Mult) {
      const auto *cl = dynamic_cast<const Constant *>(b->left.get());
      const auto *cr = dynamic_cast<const Constant *>(b->right.get());
      if (cl || cr) {
        const Node *other = cl ? b->right.get() : b->left.get();
        int64_t value = cl ? cl->value : cr->value;
        AddVisitorResult simp;
        run_add_visitor(*other, simp);
        for (const auto &[k, v] : simp.coeffs)
          res.add_coeff(k, value * v);
        res.const_term += simp.const_term * value;
        return;
      }
      // Both sides symbolic: fall through to generic_visit
    }
    // For UnaryOp and other BinOps: generic_visit
  }
  if (const auto *u = dynamic_cast<const UnaryOp *>(&node)) {
    if (u->op == UnaryOpKind::USub) {
      AddVisitorResult neg;
      run_add_visitor(*u->operand, neg);
      for (const auto &[k, v] : neg.coeffs)
        res.add_coeff(k, -v);
      res.const_term -= neg.const_term;
      return;
    }
    if (u->op == UnaryOpKind::UAdd) {
      run_add_visitor(*u->operand, res);
      return;
    }
  }
  // generic_visit: treat the whole node as a single symbolic variable
  std::string s = unparse(node);
  res.add_coeff(s, 1);
}

// Converts an AddVisitorResult into the final simplified string (or int).
// Returns int64_t when no variables remain (coefficients map was never
// populated — i.e. the expression was a pure constant).
static SimplifyResult make_simplified(const AddVisitorResult &res) {
  if (res.coeffs_empty())
    return res.const_term;

  std::string result;
  for (const auto &[var, coeff] : res.coeffs) {
    if (coeff == 0)
      continue;
    if (coeff == 1) {
      result += "+";
      result += var;
    } else if (coeff == -1) {
      result += "-";
      result += var;
    } else {
      result += (coeff > 0) ? "+" : "";
      result += std::to_string(coeff);
      result += "*";
      if (needs_mul_parens(var)) {
        result += "(";
        result += var;
        result += ")";
      } else {
        result += var;
      }
    }
  }
  if (res.const_term != 0) {
    result += (res.const_term > 0) ? "+" : "";
    result += std::to_string(res.const_term);
  }

  // Strip leading '+' and handle empty result.
  std::string out;
  if (result.empty())
    out = "0";
  else if (result[0] == '+')
    out = result.substr(1);
  else
    out = result;

  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// RenameTransformer
// ═══════════════════════════════════════════════════════════════════════════

class RenameTransformer : public Transformer {
public:
  explicit RenameTransformer(const std::unordered_map<std::string, std::string> &mapping)
      : mapping_(mapping) {}

  NodePtr visit_Name(std::unique_ptr<Name> n) override {
    auto it = mapping_.find(n->id);
    if (it != mapping_.end())
      n->id = it->second;
    return n;
  }

private:
  const std::unordered_map<std::string, std::string> &mapping_;
};

// ═══════════════════════════════════════════════════════════════════════════
// TokenCollectorVisitor
// ═══════════════════════════════════════════════════════════════════════════

static void collect_tokens(const Node &node, std::unordered_set<std::string> &out) {
  if (const auto *n = dynamic_cast<const Name *>(&node)) {
    out.insert(n->id);
    return;
  }
  if (const auto *b = dynamic_cast<const BinOp *>(&node)) {
    collect_tokens(*b->left, out);
    collect_tokens(*b->right, out);
    return;
  }
  if (const auto *u = dynamic_cast<const UnaryOp *>(&node)) {
    collect_tokens(*u->operand, out);
    return;
  }
  if (const auto *call = dynamic_cast<const Call *>(&node)) {
    for (const auto &a : call->args)
      collect_tokens(*a, out);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Simplification pipeline (mirrors the Python simplify_expression pipeline)
// ═══════════════════════════════════════════════════════════════════════════

static NodePtr apply_pipeline(NodePtr node) {
  CeilToIntTransformer ceil_tr;
  SimpleSimplifyTransformer simple_tr;
  MulDivCancellerTransformer muldiv_tr;
  ExactMulDivConstantFolderTransformer fold_tr;
  MaxToXorTransformer max_tr;
  ReorderCommutativeOpsTransformer reorder_tr;
  MaxIntTransformer maxint_tr;

  // Apply pipeline twice (matches the Python implementation)
  for (int pass = 0; pass < 2; ++pass) {
    node = ceil_tr.visit(std::move(node));
    node = simple_tr.visit(std::move(node));
    node = muldiv_tr.visit(std::move(node));
    node = fold_tr.visit(std::move(node));
    node = muldiv_tr.visit(std::move(node));
    node = max_tr.visit(std::move(node));
    // SimplifyParensTransformer is a no-op; omitted
    node = reorder_tr.visit(std::move(node));
    // StringToIntTransformer converts string constants to ints (not applicable here)
    node = maxint_tr.visit(std::move(node));
  }
  return node;
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

std::string simplify_result_to_string(const SimplifyResult &r) {
  if (std::holds_alternative<int64_t>(r))
    return std::to_string(std::get<int64_t>(r));
  return std::get<std::string>(r);
}

SimplifyResult simplify_expression(int64_t value) { return value; }

SimplifyResult simplify_expression(const std::string &expr) {
  NodePtr tree;
  try {
    tree = parse(expr);
  } catch (const std::runtime_error &) {
    // Unparseable expression: return unchanged (matches Python behaviour for
    // expressions with invalid syntax like ONNX '::' node names)
    return expr;
  }

  tree = apply_pipeline(std::move(tree));

  AddVisitorResult res;
  run_add_visitor(*tree, res);
  return make_simplified(res);
}

std::map<std::string, int64_t> simplify_two_expressions(const std::string &expr1,
                                                        const std::string &expr2) {
  // Build expr1 - (expr2) and run the add-visitor directly (no pipeline
  // transformations — matches the Python implementation).
  std::string combined = expr1 + "-(" + expr2 + ")";
  NodePtr tree = parse(combined);

  AddVisitorResult res;
  run_add_visitor(*tree, res);

  std::map<std::string, int64_t> out;
  for (const auto &[k, v] : res.coeffs)
    if (v != 0)
      out[k] = v;
  return out;
}

// ─────────────────────────────────────────────────────────────────────────

static int64_t eval_node(const Node &node, const std::unordered_map<std::string, int64_t> &ctx,
                         const std::string &expr);

static int64_t eval_node(const Node &node, const std::unordered_map<std::string, int64_t> &ctx,
                         const std::string &expr) {
  if (const auto *c = dynamic_cast<const Constant *>(&node))
    return c->value;

  if (const auto *n = dynamic_cast<const Name *>(&node)) {
    auto it = ctx.find(n->id);
    if (it == ctx.end())
      throw std::runtime_error("Unknown variable '" + n->id + "' in expression '" + expr + "'");
    return it->second;
  }

  if (const auto *b = dynamic_cast<const BinOp *>(&node)) {
    int64_t l = eval_node(*b->left, ctx, expr);
    int64_t r = eval_node(*b->right, ctx, expr);
    switch (b->op) {
    case BinOpKind::Add:
      return l + r;
    case BinOpKind::Sub:
      return l - r;
    case BinOpKind::Mult:
      return l * r;
    case BinOpKind::FloorDiv:
      if (r == 0)
        throw std::runtime_error("Division by zero in expression '" + expr + "'");
      return l / r;
    case BinOpKind::Mod:
      if (r == 0)
        throw std::runtime_error("Modulo by zero in expression '" + expr + "'");
      return l % r;
    case BinOpKind::BitXor:
      return std::max(l, r);
    case BinOpKind::BitAnd:
      return std::min(l, r);
    }
  }

  if (const auto *u = dynamic_cast<const UnaryOp *>(&node)) {
    int64_t v = eval_node(*u->operand, ctx, expr);
    return (u->op == UnaryOpKind::USub) ? -v : v;
  }

  if (const auto *call = dynamic_cast<const Call *>(&node)) {
    if (call->func == "CeilToInt" && call->args.size() == 2) {
      int64_t n = eval_node(*call->args[0], ctx, expr);
      int64_t d = eval_node(*call->args[1], ctx, expr);
      if (d == 0)
        throw std::runtime_error("Division by zero in CeilToInt in expression '" + expr + "'");
      return (n % d == 0) ? n / d : n / d + 1;
    }
    throw std::runtime_error("Unsupported function '" + call->func + "' in expression '" + expr +
                             "'");
  }

  throw std::runtime_error("Unsupported AST node type in expression '" + expr + "'");
}

int64_t evaluate_expression(const std::string &expr,
                            const std::unordered_map<std::string, int64_t> &context) {
  NodePtr tree = parse(expr); // propagates SyntaxError as runtime_error
  return eval_node(*tree, context, expr);
}

// ─────────────────────────────────────────────────────────────────────────

std::unordered_set<std::string> parse_expression_tokens(const std::string &expr) {
  NodePtr tree;
  try {
    tree = parse(expr);
  } catch (const std::runtime_error &) {
    return {expr};
  }
  std::unordered_set<std::string> out;
  collect_tokens(*tree, out);
  return out;
}

// ─────────────────────────────────────────────────────────────────────────

std::string rename_expression(const std::string &expr,
                              const std::unordered_map<std::string, std::string> &mapping) {
  NodePtr tree = parse(expr);

  MaxToXorTransformer max_tr;
  RenameTransformer rename_tr(mapping);

  tree = rename_tr.visit(max_tr.visit(std::move(tree)));

  // Remove spaces from output (matches Python behaviour)
  std::string out = unparse(*tree);
  out.erase(std::remove(out.begin(), out.end(), ' '), out.end());
  return out;
}

// ─────────────────────────────────────────────────────────────────────────

std::string
rename_dynamic_expression(const std::string &expression,
                          const std::unordered_map<std::string, std::string> &replacements) {
  NodePtr tree;
  try {
    tree = parse(expression);
  } catch (const std::runtime_error &) {
    return expression;
  }

  MaxToXorTransformer max_tr;
  RenameTransformer rename_tr(replacements);
  SimpleSimplifyTransformer simple_tr;

  tree = simple_tr.visit(rename_tr.visit(max_tr.visit(std::move(tree))));

  std::string out = unparse(*tree);
  out.erase(std::remove(out.begin(), out.end(), ' '), out.end());
  return out;
}

// ─────────────────────────────────────────────────────────────────────────

std::map<std::string, std::string>
rename_dynamic_dimensions(const std::map<std::string, std::unordered_set<std::string>> &constraints,
                          const std::unordered_set<std::string> &original,
                          const std::string &ban_prefix) {
  std::map<std::string, std::string> replacements;
  for (const auto &s : original)
    replacements[s] = s;

  std::unordered_set<std::string> all_values;
  for (const auto &[k, _] : constraints)
    all_values.insert(k);
  for (const auto &s : original)
    all_values.insert(s);

  std::unordered_set<std::string> not_done;
  for (const auto &[k, _] : constraints)
    not_done.insert(k);

  int max_iter = static_cast<int>(replacements.size());
  while (!not_done.empty() && max_iter > 0) {
    --max_iter;
    for (const auto &[k, v] : constraints) {
      // Find intersection of v and original
      std::vector<std::string> common;
      for (const auto &s : v)
        if (original.count(s))
          common.push_back(s);
      if (common.empty())
        continue;
      std::sort(common.begin(), common.end());
      const std::string &by = common[0];
      if (!ban_prefix.empty() && by.size() >= ban_prefix.size() &&
          by.substr(0, ban_prefix.size()) == ban_prefix)
        continue;
      replacements[k] = by;
      for (const auto &vv : v)
        if (!replacements.count(vv))
          replacements[vv] = by;
    }
    // Recompute not_done
    not_done.clear();
    for (const auto &s : all_values)
      if (!replacements.count(s))
        not_done.insert(s);
  }

  return replacements;
}

// ═══════════════════════════════════════════════════════════════════════════
// Dimension operations
// ═══════════════════════════════════════════════════════════════════════════

std::string dim_to_string(const DimType &d) {
  if (std::holds_alternative<int64_t>(d))
    return std::to_string(std::get<int64_t>(d));
  return std::get<std::string>(d);
}

static DimType simplify_dim(const std::string &expr) {
  auto r = simplify_expression(expr);
  if (std::holds_alternative<int64_t>(r))
    return std::get<int64_t>(r);
  return std::get<std::string>(r);
}

DimType dim_add(const DimType &a, const DimType &b) {
  if (std::holds_alternative<int64_t>(a) && std::holds_alternative<int64_t>(b))
    return std::get<int64_t>(a) + std::get<int64_t>(b);
  return simplify_dim("(" + dim_to_string(a) + ")+(" + dim_to_string(b) + ")");
}

DimType dim_sub(const DimType &a, const DimType &b) {
  if (std::holds_alternative<int64_t>(a) && std::holds_alternative<int64_t>(b))
    return std::get<int64_t>(a) - std::get<int64_t>(b);
  return simplify_dim("(" + dim_to_string(a) + ")-(" + dim_to_string(b) + ")");
}

DimType dim_mul(const DimType &a, const DimType &b) {
  if (std::holds_alternative<int64_t>(a) && std::holds_alternative<int64_t>(b))
    return std::get<int64_t>(a) * std::get<int64_t>(b);
  return simplify_dim("(" + dim_to_string(a) + ")*(" + dim_to_string(b) + ")");
}

DimType dim_multi_mul(const std::vector<DimType> &args) {
  bool all_int = true;
  for (const auto &a : args)
    if (!std::holds_alternative<int64_t>(a)) {
      all_int = false;
      break;
    }
  if (all_int) {
    int64_t prod = 1;
    for (const auto &a : args)
      prod *= std::get<int64_t>(a);
    return prod;
  }
  if (args.empty())
    return int64_t{1};
  std::string expr;
  for (size_t i = 0; i < args.size(); ++i) {
    if (i > 0)
      expr += "*";
    expr += "(" + dim_to_string(args[i]) + ")";
  }
  return simplify_dim(expr);
}

DimType dim_div(const DimType &a, const DimType &b) {
  if (std::holds_alternative<int64_t>(a) && std::holds_alternative<int64_t>(b))
    return std::get<int64_t>(a) / std::get<int64_t>(b);
  return simplify_dim("(" + dim_to_string(a) + ")//(" + dim_to_string(b) + ")");
}

DimType dim_mod(const DimType &a, const DimType &b) {
  if (std::holds_alternative<int64_t>(a) && std::holds_alternative<int64_t>(b))
    return std::get<int64_t>(a) % std::get<int64_t>(b);
  return simplify_dim("(" + dim_to_string(a) + ")%(" + dim_to_string(b) + ")");
}

DimType dim_max(const DimType &a, const DimType &b) {
  if (std::holds_alternative<int64_t>(a) && std::holds_alternative<int64_t>(b))
    return std::max(std::get<int64_t>(a), std::get<int64_t>(b));
  return simplify_dim("(" + dim_to_string(a) + ")^(" + dim_to_string(b) + ")");
}

DimType dim_min(const DimType &a, const DimType &b) {
  if (std::holds_alternative<int64_t>(a) && std::holds_alternative<int64_t>(b))
    return std::min(std::get<int64_t>(a), std::get<int64_t>(b));
  return simplify_dim("(" + dim_to_string(a) + ")&(" + dim_to_string(b) + ")");
}

} // namespace expressions
} // namespace onnx_light
