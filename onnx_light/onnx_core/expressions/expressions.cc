// SPDX-License-Identifier: Apache-2.0
//
// C++ implementation of symbolic dimension expression utilities.
// Ported from yobx/xexpressions (https://github.com/xadupre/yet-another-onnx-builder).

#include "expressions.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <charconv>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace onnx_light::core::expressions {

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
  SlashColon,
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

namespace {

bool TryParseInt64Literal(const std::string &text, int64_t &value) {
  const char *begin = text.data();
  const char *end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, value);
  return parsed.ec == std::errc() && parsed.ptr == end;
}

bool TryParseExpression(const std::string &expr, NodePtr &tree) {
  try {
    tree = parse(expr);
    return true;
  } catch (const std::runtime_error &) {
    tree.reset();
    return false;
  }
}

} // namespace

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
      if (!TryParseInt64Literal(t.text, t.value)) {
        throw std::runtime_error("Integer literal out of range in expression");
      }
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

    // Double slash // or exact-division slash-colon /:
    if (c == '/') {
      if (pos_ + 1 < input_.size() && input_[pos_ + 1] == '/') {
        pos_ += 2;
        return {TokenKind::DoubleSlash, "//"};
      }
      if (pos_ + 1 < input_.size() && input_[pos_ + 1] == ':') {
        pos_ += 2;
        return {TokenKind::SlashColon, "/:"};
      }
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
           cur_.kind == TokenKind::SlashColon || cur_.kind == TokenKind::Percent) {
      BinOpKind op;
      switch (cur_.kind) {
      case TokenKind::Star:
        op = BinOpKind::Mult;
        break;
      case TokenKind::DoubleSlash:
        op = BinOpKind::FloorDiv;
        break;
      case TokenKind::SlashColon:
        op = BinOpKind::ExactDiv;
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
constexpr int binop_prec(BinOpKind op) {
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
  case BinOpKind::ExactDiv:
  case BinOpKind::Mod:
    return 4;
  }
  return 0;
}

bool binop_commutative(BinOpKind op) {
  return op == BinOpKind::Add || op == BinOpKind::Mult || op == BinOpKind::BitXor ||
         op == BinOpKind::BitAnd;
}

constexpr const char *binop_sym(BinOpKind op) {
  switch (op) {
  case BinOpKind::Add:
    return "+";
  case BinOpKind::Sub:
    return "-";
  case BinOpKind::Mult:
    return "*";
  case BinOpKind::FloorDiv:
    return "//";
  case BinOpKind::ExactDiv:
    return "/:";
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
    if (dynamic_cast<Constant *>(node.get()))
      return visit_Constant(std::unique_ptr<Constant>(static_cast<Constant *>(node.release())));
    if (dynamic_cast<Name *>(node.get()))
      return visit_Name(std::unique_ptr<Name>(static_cast<Name *>(node.release())));
    if (dynamic_cast<BinOp *>(node.get()))
      return visit_BinOp(std::unique_ptr<BinOp>(static_cast<BinOp *>(node.release())));
    if (dynamic_cast<UnaryOp *>(node.get()))
      return visit_UnaryOp(std::unique_ptr<UnaryOp>(static_cast<UnaryOp *>(node.release())));
    if (dynamic_cast<Call *>(node.get()))
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

// Returns whether a floor division appears on the multiplicative spine of
// `node`, i.e. as a factor of a `*` chain. Floor division is not exact, so
// `a*(x//b)` must not be flattened into `(a*x)//b`: the equality holds only
// when `x` is a multiple of `b` (e.g. `2*(3//2) == 2`, not `3`).
// Exact division (`/:`) is excluded because it commutes with multiplication
// by definition: `c*(x/:b) == (c*x)/:b` always holds.
bool has_floordiv_factor(const Node &node) {
  if (const auto *b = dynamic_cast<const BinOp *>(&node)) {
    if (b->op == BinOpKind::FloorDiv)
      return true;
    if (b->op == BinOpKind::Mult)
      return has_floordiv_factor(*b->left) || has_floordiv_factor(*b->right);
  }
  return false;
}

void flatten_mul_div(const Node &node, std::vector<NodePtr> &num, std::vector<NodePtr> &den) {
  if (const auto *b = dynamic_cast<const BinOp *>(&node)) {
    if (b->op == BinOpKind::Mult) {
      // A floor division used as a multiplicative factor cannot be flattened
      // into the global numerator/denominator: doing so would move the factor
      // across the (non-exact) division boundary. Keep the product atomic.
      if (has_floordiv_factor(*b->left) || has_floordiv_factor(*b->right)) {
        num.push_back(node.clone());
        return;
      }
      flatten_mul_div(*b->left, num, den);
      flatten_mul_div(*b->right, num, den);
      return;
    }
    if (b->op == BinOpKind::FloorDiv || b->op == BinOpKind::ExactDiv) {
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

    if (n->op != BinOpKind::Mult && n->op != BinOpKind::FloorDiv && n->op != BinOpKind::ExactDiv)
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

    NodePtr numerator = build_product(rn);
    if (rd.empty())
      return numerator;
    NodePtr denom = build_product(rd);
    // Preserve ExactDiv annotation if the root operation was an exact division.
    BinOpKind div_op = (n->op == BinOpKind::ExactDiv) ? BinOpKind::ExactDiv : BinOpKind::FloorDiv;
    return std::make_unique<BinOp>(std::move(numerator), div_op, std::move(denom));
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

    if (n->op != BinOpKind::Mult && n->op != BinOpKind::FloorDiv && n->op != BinOpKind::ExactDiv)
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

    if (num_c % den_c == 0) {
      // Denominator divides the numerator constant exactly: fold completely.
      int64_t folded = num_c / den_c;
      std::vector<NodePtr> factors;
      if (folded != 1 || num_other.empty())
        factors.push_back(std::make_unique<Constant>(folded));
      for (auto &x : num_other)
        factors.push_back(std::move(x));
      return build_product(factors);
    }

    // Partial cancellation via GCD: collapse multiple floor-division constants
    // into one, e.g. d//5//2 → d//10, or 6*a//4 → 3*a//2.
    // The identity floor(floor(x/a)/b) == floor(x/(a*b)) makes this valid.
    int64_t g = std::gcd(std::abs(num_c), std::abs(den_c));
    int64_t new_num_c = num_c / g;
    int64_t new_den_c = den_c / g;

    // Build the (reduced) numerator.
    std::vector<NodePtr> factors;
    if (new_num_c != 1 || num_other.empty())
      factors.push_back(std::make_unique<Constant>(new_num_c));
    for (auto &x : num_other)
      factors.push_back(std::move(x));
    NodePtr numerator = build_product(factors);

    if (new_den_c == 1)
      return numerator;

    // n->op is FloorDiv or ExactDiv here: Mult would have den_c==1 and gone
    // through the exact-fold branch above (num_c % 1 == 0 is always true).
    BinOpKind div_op = (n->op == BinOpKind::ExactDiv) ? BinOpKind::ExactDiv : BinOpKind::FloorDiv;
    return std::make_unique<BinOp>(std::move(numerator), div_op,
                                   std::make_unique<Constant>(new_den_c));
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// DistributeFloorDivOverAddTransformer
// Distributes a floor-division by an integer constant over additive terms
// when every term is exactly divisible by the divisor, e.g.
//   (2*b + 2*c) // 2 → b + c
//   (4*a - 2*b) // 2 → 2*a - b
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// Attempts to exactly divide the expression rooted at `x` by the positive
// integer `d`. Returns the resulting expression on success, or nullptr if
// the division cannot be performed exactly without introducing a residue.
NodePtr try_exact_divide(const Node &x, int64_t d) {
  if (d == 1)
    return x.clone();
  if (const auto *c = dynamic_cast<const Constant *>(&x)) {
    if (c->value % d != 0)
      return nullptr;
    return std::make_unique<Constant>(c->value / d);
  }
  if (const auto *b = dynamic_cast<const BinOp *>(&x)) {
    if (b->op == BinOpKind::Add || b->op == BinOpKind::Sub) {
      auto l = try_exact_divide(*b->left, d);
      if (!l)
        return nullptr;
      auto r = try_exact_divide(*b->right, d);
      if (!r)
        return nullptr;
      return std::make_unique<BinOp>(std::move(l), b->op, std::move(r));
    }
    if (b->op == BinOpKind::Mult) {
      if (const auto *cl = dynamic_cast<const Constant *>(b->left.get())) {
        if (cl->value % d == 0) {
          int64_t nc = cl->value / d;
          if (nc == 1)
            return b->right->clone();
          return std::make_unique<BinOp>(std::make_unique<Constant>(nc), BinOpKind::Mult,
                                         b->right->clone());
        }
      }
      if (const auto *cr = dynamic_cast<const Constant *>(b->right.get())) {
        if (cr->value % d == 0) {
          int64_t nc = cr->value / d;
          if (nc == 1)
            return b->left->clone();
          return std::make_unique<BinOp>(b->left->clone(), BinOpKind::Mult,
                                         std::make_unique<Constant>(nc));
        }
      }
      // Recurse into sub-multiplications (e.g. 2*(b+c) under a chain).
      if (auto l = try_exact_divide(*b->left, d))
        return std::make_unique<BinOp>(std::move(l), BinOpKind::Mult, b->right->clone());
      if (auto r = try_exact_divide(*b->right, d))
        return std::make_unique<BinOp>(b->left->clone(), BinOpKind::Mult, std::move(r));
    }
  }
  if (const auto *u = dynamic_cast<const UnaryOp *>(&x)) {
    if (u->op == UnaryOpKind::USub) {
      auto o = try_exact_divide(*u->operand, d);
      if (!o)
        return nullptr;
      return std::make_unique<UnaryOp>(UnaryOpKind::USub, std::move(o));
    }
    if (u->op == UnaryOpKind::UAdd) {
      auto o = try_exact_divide(*u->operand, d);
      if (!o)
        return nullptr;
      return std::make_unique<UnaryOp>(UnaryOpKind::UAdd, std::move(o));
    }
  }
  return nullptr;
}

// Splits an expression `x` into a divisible quotient `q` and an integer
// residual `r` such that `x == q * d + r`, where every multiplicative term
// inside `q` is exactly divisible by `d`. Constant leaves that are not
// divisible by `d` are folded into `r`. Returns false if the expression
// cannot be split this way (e.g. a non-divisible symbolic factor).
bool try_split_for_division(const Node &x, int64_t d, NodePtr &quotient, int64_t &residual) {
  if (auto q = try_exact_divide(x, d)) {
    quotient = std::move(q);
    residual = 0;
    return true;
  }
  if (const auto *c = dynamic_cast<const Constant *>(&x)) {
    quotient = std::make_unique<Constant>(0);
    residual = c->value;
    return true;
  }
  if (const auto *b = dynamic_cast<const BinOp *>(&x)) {
    if (b->op == BinOpKind::Add || b->op == BinOpKind::Sub) {
      NodePtr lq, rq;
      int64_t lr = 0, rr = 0;
      if (!try_split_for_division(*b->left, d, lq, lr))
        return false;
      if (!try_split_for_division(*b->right, d, rq, rr))
        return false;
      quotient = std::make_unique<BinOp>(std::move(lq), b->op, std::move(rq));
      residual = (b->op == BinOpKind::Add) ? (lr + rr) : (lr - rr);
      return true;
    }
  }
  if (const auto *u = dynamic_cast<const UnaryOp *>(&x)) {
    if (u->op == UnaryOpKind::USub) {
      NodePtr q;
      int64_t r = 0;
      if (!try_split_for_division(*u->operand, d, q, r))
        return false;
      quotient = std::make_unique<UnaryOp>(UnaryOpKind::USub, std::move(q));
      residual = -r;
      return true;
    }
    if (u->op == UnaryOpKind::UAdd)
      return try_split_for_division(*u->operand, d, quotient, residual);
  }
  return false;
}

// Python-style floor division for int64 (rounds toward negative infinity).
int64_t floor_div_i64(int64_t a, int64_t b) {
  int64_t q = a / b;
  int64_t r = a % b;
  if ((r != 0) && ((r < 0) != (b < 0)))
    --q;
  return q;
}

bool checked_add_i64(int64_t a, int64_t b, int64_t &out) {
  if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b))
    return false;
  out = a + b;
  return true;
}

bool checked_mul_i64(int64_t a, int64_t b, int64_t &out) {
  if (a == 0 || b == 0) {
    out = 0;
    return true;
  }
  if (a > 0) {
    if (b > 0) {
      if (a > INT64_MAX / b)
        return false;
    } else if (b < INT64_MIN / a) {
      return false;
    }
  } else if (b > 0) {
    if (a < INT64_MIN / b)
      return false;
  } else if (a != 0 && b < INT64_MAX / a) {
    return false;
  }
  out = a * b;
  return true;
}

/// Returns whether the node represents the constant value zero.
inline bool is_constant_zero(const Node &n) {
  const auto *c = dynamic_cast<const Constant *>(&n);
  return c && c->value == 0;
}

/// Returns true when `node` is `x+d` or `x-d` with the given denominator `d`.
/// Returns false for `x-INT64_MIN` to avoid signed-overflow when negating the
/// right-hand constant offset.
bool has_single_step_floordiv_offset(const Node &node, int64_t d) {
  const auto *left_bin = dynamic_cast<const BinOp *>(&node);
  if (!left_bin || (left_bin->op != BinOpKind::Add && left_bin->op != BinOpKind::Sub))
    return false;

  int64_t constant_offset = 0;
  bool has_constant_offset = false;
  if (const auto *cr = dynamic_cast<const Constant *>(left_bin->right.get())) {
    if (left_bin->op == BinOpKind::Sub && cr->value == std::numeric_limits<int64_t>::min()) {
      // Avoid overflow when computing -INT64_MIN.
      return false;
    }
    constant_offset = (left_bin->op == BinOpKind::Add) ? cr->value : -cr->value;
    has_constant_offset = true;
  } else if (left_bin->op == BinOpKind::Add) {
    if (const auto *cl = dynamic_cast<const Constant *>(left_bin->left.get())) {
      constant_offset = cl->value;
      has_constant_offset = true;
    }
  } else if (left_bin->op == BinOpKind::Sub) {
    // `c-x` is not the `x±c` pattern this guard handles.
    return false;
  }
  return has_constant_offset && (constant_offset == d || constant_offset == -d);
}

/// Decomposes the input `node` into `symbolic + offset`.
/// @param node The input expression to decompose.
/// @param symbolic The output non-constant expression term.
/// @param offset The output folded integer constant term.
void split_symbolic_and_offset(const Node &node, NodePtr &symbolic, int64_t &offset) {
  if (const auto *c = dynamic_cast<const Constant *>(&node)) {
    symbolic = std::make_unique<Constant>(0);
    offset = c->value;
    return;
  }
  if (const auto *b = dynamic_cast<const BinOp *>(&node)) {
    if (b->op == BinOpKind::Add || b->op == BinOpKind::Sub) {
      NodePtr ls, rs;
      int64_t lo = 0, ro = 0;
      split_symbolic_and_offset(*b->left, ls, lo);
      split_symbolic_and_offset(*b->right, rs, ro);
      offset = (b->op == BinOpKind::Add) ? (lo + ro) : (lo - ro);
      if (is_constant_zero(*ls)) {
        symbolic = (b->op == BinOpKind::Add)
                       ? std::move(rs)
                       : std::make_unique<UnaryOp>(UnaryOpKind::USub, std::move(rs));
      } else if (is_constant_zero(*rs)) {
        symbolic = std::move(ls);
      } else {
        symbolic = std::make_unique<BinOp>(std::move(ls), b->op, std::move(rs));
      }
      return;
    }
  }
  if (const auto *u = dynamic_cast<const UnaryOp *>(&node)) {
    NodePtr os;
    int64_t oo = 0;
    split_symbolic_and_offset(*u->operand, os, oo);
    if (u->op == UnaryOpKind::USub) {
      offset = -oo;
      symbolic = is_constant_zero(*os)
                     ? std::move(os)
                     : std::make_unique<UnaryOp>(UnaryOpKind::USub, std::move(os));
      return;
    }
    if (u->op == UnaryOpKind::UAdd) {
      symbolic = std::move(os);
      offset = oo;
      return;
    }
  }
  symbolic = node.clone();
  offset = 0;
}

} // namespace

class DistributeFloorDivOverAddTransformer : public Transformer {
public:
  NodePtr visit_BinOp(std::unique_ptr<BinOp> n) override {
    n = std::unique_ptr<BinOp>(static_cast<BinOp *>(generic_visit(std::move(n)).release()));
    if (n->op != BinOpKind::FloorDiv && n->op != BinOpKind::ExactDiv)
      return n;
    const auto *dc = dynamic_cast<const Constant *>(n->right.get());
    if (!dc || dc->value == 0)
      return n;
    int64_t d = dc->value;
    if (d < 0)
      return n;
    // Keep (x+d)//d and (x-d)//d unchanged so non-contiguous ring cases (offsets
    // that do not cover every residue class, e.g. x//d + (x+d)//d) are preserved
    // for the dedicated ring pass.
    if (n->op == BinOpKind::FloorDiv && has_single_step_floordiv_offset(*n->left, d))
      return n;
    if (n->op == BinOpKind::FloorDiv || n->op == BinOpKind::ExactDiv) {
      NodePtr symbolic;
      int64_t offset = 0;
      split_symbolic_and_offset(*n->left, symbolic, offset);
      if (offset != 0 && offset % d == 0) {
        int64_t offset_quotient = offset / d;
        if (is_constant_zero(*symbolic))
          return std::make_unique<Constant>(offset_quotient);
        auto base =
            std::make_unique<BinOp>(std::move(symbolic), n->op, std::make_unique<Constant>(d));
        if (offset_quotient == 0)
          return base;
        return std::make_unique<BinOp>(std::move(base), BinOpKind::Add,
                                       std::make_unique<Constant>(offset_quotient));
      }
    }
    if (auto r = try_exact_divide(*n->left, d))
      return r;
    // Fallback: split numerator into (quotient * d) + constant_residual.
    // Then floor((q*d + r) / d) == q + floor(r/d), valid because every
    // non-constant term in the numerator is an exact multiple of d.
    NodePtr q;
    int64_t r = 0;
    if (try_split_for_division(*n->left, d, q, r)) {
      int64_t add = floor_div_i64(r, d);
      if (add == 0)
        return q;
      return std::make_unique<BinOp>(std::move(q), BinOpKind::Add, std::make_unique<Constant>(add));
    }
    return n;
  }
};

class FoldAddIntoFloorDivTransformer : public Transformer {
public:
  NodePtr visit_BinOp(std::unique_ptr<BinOp> n) override {
    n = std::unique_ptr<BinOp>(static_cast<BinOp *>(generic_visit(std::move(n)).release()));
    if (n->op != BinOpKind::Add && n->op != BinOpKind::Sub)
      return n;

    const BinOp *fd = nullptr;
    int64_t outer_constant = 0;

    if (n->op == BinOpKind::Add) {
      if (const auto *c = dynamic_cast<const Constant *>(n->left.get())) {
        fd = dynamic_cast<const BinOp *>(n->right.get());
        outer_constant = c->value;
      } else if (const auto *c = dynamic_cast<const Constant *>(n->right.get())) {
        fd = dynamic_cast<const BinOp *>(n->left.get());
        outer_constant = c->value;
      }
    } else if (const auto *c = dynamic_cast<const Constant *>(n->right.get())) {
      fd = dynamic_cast<const BinOp *>(n->left.get());
      if (c->value == INT64_MIN)
        return n;
      outer_constant = -c->value;
    }

    if (!fd || fd->op != BinOpKind::FloorDiv)
      return n;

    const auto *denom = dynamic_cast<const Constant *>(fd->right.get());
    if (!denom || denom->value == 0)
      return n;

    NodePtr symbolic;
    int64_t inner_offset = 0;
    split_symbolic_and_offset(*fd->left, symbolic, inner_offset);
    int64_t outer_shift = 0;
    if (!checked_mul_i64(outer_constant, denom->value, outer_shift))
      return n;
    int64_t shifted_offset = 0;
    if (!checked_add_i64(inner_offset, outer_shift, shifted_offset))
      return n;
    // Keep divisible offsets in the q + k form produced by
    // DistributeFloorDivOverAddTransformer. Folding them back into the
    // numerator here would only oscillate between equivalent representations
    // across simplification passes.
    if (shifted_offset % denom->value == 0)
      return n;

    NodePtr numerator;
    if (is_constant_zero(*symbolic)) {
      numerator = std::make_unique<Constant>(shifted_offset);
    } else if (shifted_offset > 0) {
      numerator = std::make_unique<BinOp>(std::move(symbolic), BinOpKind::Add,
                                          std::make_unique<Constant>(shifted_offset));
    } else if (shifted_offset < 0) {
      numerator = std::make_unique<BinOp>(std::move(symbolic), BinOpKind::Sub,
                                          std::make_unique<Constant>(-shifted_offset));
    } else {
      numerator = std::move(symbolic);
    }

    return std::make_unique<BinOp>(std::move(numerator), BinOpKind::FloorDiv,
                                   std::make_unique<Constant>(denom->value));
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// NestedFloorDivTransformer
// Simplifies (x // a) // b → x // (a * b) when both a and b are positive
// integer constants.  The identity holds for all integer x and positive
// integer a, b: floor(floor(x/a)/b) == floor(x/(a*b)).
// ═══════════════════════════════════════════════════════════════════════════

class NestedFloorDivTransformer : public Transformer {
public:
  NodePtr visit_BinOp(std::unique_ptr<BinOp> n) override {
    n = std::unique_ptr<BinOp>(static_cast<BinOp *>(generic_visit(std::move(n)).release()));

    if (n->op != BinOpKind::FloorDiv)
      return n;
    const auto *outer_d = dynamic_cast<const Constant *>(n->right.get());
    if (!outer_d || outer_d->value <= 0)
      return n;

    const auto *inner = dynamic_cast<const BinOp *>(n->left.get());
    if (!inner || inner->op != BinOpKind::FloorDiv)
      return n;
    const auto *inner_d = dynamic_cast<const Constant *>(inner->right.get());
    if (!inner_d || inner_d->value <= 0)
      return n;

    // Guard against overflow before multiplying the two divisors.
    int64_t a = inner_d->value;
    int64_t b = outer_d->value;
    if (b > INT64_MAX / a)
      return n;

    return std::make_unique<BinOp>(inner->left->clone(), BinOpKind::FloorDiv,
                                   std::make_unique<Constant>(a * b));
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
// FloorDivAddRingTransformer
// Recognises sums of floor-divisions sharing the same denominator ``n`` whose
// numerators only differ by a contiguous range of integer constants of length
// ``n``.  The identity
//
//   sum_{i=0..n-1} floor((y + i) / n) = y          (for integer y)
//
// lets us collapse such a group to ``s + k`` where ``s`` is the common
// symbolic part and ``k`` is the smallest of the integer offsets.
//
// Example: ``(b + c) // 2 + (1 + b + c) // 2  →  b + c``.
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// Splits an expression into a symbolic part and an additive integer offset
// such that ``node == symbolic + offset``.  Walks Add/Sub chains and folds
// every numeric leaf into ``offset``; the remaining symbolic terms are
// reassembled into ``symbolic`` preserving their original signs.  When the
// expression is a pure constant, ``symbolic`` is set to ``Constant(0)``.
void split_const_offset(const Node &node, NodePtr &symbolic, int64_t &offset) {
  std::vector<std::pair<NodePtr, int>> sym_terms; // (cloned node, sign)
  offset = 0;
  std::function<void(const Node &, int)> walk = [&](const Node &n, int sign) {
    if (const auto *c = dynamic_cast<const Constant *>(&n)) {
      offset += sign * c->value;
      return;
    }
    if (const auto *b = dynamic_cast<const BinOp *>(&n)) {
      if (b->op == BinOpKind::Add) {
        walk(*b->left, sign);
        walk(*b->right, sign);
        return;
      }
      if (b->op == BinOpKind::Sub) {
        walk(*b->left, sign);
        walk(*b->right, -sign);
        return;
      }
    }
    if (const auto *u = dynamic_cast<const UnaryOp *>(&n)) {
      if (u->op == UnaryOpKind::UAdd) {
        walk(*u->operand, sign);
        return;
      }
      if (u->op == UnaryOpKind::USub) {
        walk(*u->operand, -sign);
        return;
      }
    }
    sym_terms.emplace_back(n.clone(), sign);
  };
  walk(node, 1);

  if (sym_terms.empty()) {
    symbolic = std::make_unique<Constant>(0);
    return;
  }
  // Reassemble.  Sort by unparse so that semantically-equal symbolic parts
  // produce identical keys regardless of source ordering.
  std::sort(sym_terms.begin(), sym_terms.end(),
            [](const std::pair<NodePtr, int> &a, const std::pair<NodePtr, int> &b) {
              return unparse(*a.first) < unparse(*b.first);
            });
  NodePtr res;
  if (sym_terms[0].second >= 0)
    res = std::move(sym_terms[0].first);
  else
    res = std::make_unique<UnaryOp>(UnaryOpKind::USub, std::move(sym_terms[0].first));
  for (size_t i = 1; i < sym_terms.size(); ++i) {
    BinOpKind op = (sym_terms[i].second >= 0) ? BinOpKind::Add : BinOpKind::Sub;
    res = std::make_unique<BinOp>(std::move(res), op, std::move(sym_terms[i].first));
  }
  symbolic = std::move(res);
}

} // namespace

class FloorDivAddRingTransformer : public Transformer {
public:
  NodePtr visit_BinOp(std::unique_ptr<BinOp> n) override {
    n = std::unique_ptr<BinOp>(static_cast<BinOp *>(generic_visit(std::move(n)).release()));
    if (n->op != BinOpKind::Add)
      return n;

    std::vector<NodePtr> terms;
    flatten_chain(*n, BinOpKind::Add, terms);

    while (try_combine_ring(terms)) {
      // keep combining
    }

    if (terms.size() == 1)
      return std::move(terms[0]);
    return rebuild_chain(terms, BinOpKind::Add);
  }

private:
  // Identifies a floor-div ring group within ``terms`` and replaces it with
  // its collapsed form.  Returns true when a replacement was performed.
  static bool try_combine_ring(std::vector<NodePtr> &terms) {
    struct FDInfo {
      size_t idx;          // position in ``terms``
      int64_t denom;       // floor-div denominator
      NodePtr symbolic;    // symbolic part of the numerator
      int64_t offset;      // integer offset of the numerator
      std::string sym_key; // unparse(symbolic) for grouping
    };
    std::vector<FDInfo> infos;
    for (size_t i = 0; i < terms.size(); ++i) {
      const auto *b = dynamic_cast<const BinOp *>(terms[i].get());
      if (!b || (b->op != BinOpKind::FloorDiv && b->op != BinOpKind::ExactDiv))
        continue;
      const auto *c = dynamic_cast<const Constant *>(b->right.get());
      if (!c || c->value <= 1)
        continue;
      NodePtr sym;
      int64_t off = 0;
      split_const_offset(*b->left, sym, off);
      std::string key = unparse(*sym);
      infos.push_back({i, c->value, std::move(sym), off, std::move(key)});
    }
    if (infos.empty())
      return false;

    std::map<std::pair<int64_t, std::string>, std::vector<size_t>> groups;
    for (size_t j = 0; j < infos.size(); ++j)
      groups[{infos[j].denom, infos[j].sym_key}].push_back(j);

    for (auto &kv : groups) {
      int64_t denom = kv.first.first;
      auto &list = kv.second;
      if (static_cast<int64_t>(list.size()) < denom)
        continue;
      std::sort(list.begin(), list.end(),
                [&](size_t a, size_t b) { return infos[a].offset < infos[b].offset; });
      for (size_t start = 0; start + static_cast<size_t>(denom) <= list.size(); ++start) {
        int64_t base = infos[list[start]].offset;
        bool ok = true;
        for (int64_t i = 1; i < denom; ++i) {
          if (infos[list[start + static_cast<size_t>(i)]].offset != base + i) {
            ok = false;
            break;
          }
        }
        if (!ok)
          continue;
        // Collect original indices and erase them (largest first).
        std::vector<size_t> orig;
        orig.reserve(static_cast<size_t>(denom));
        for (int64_t i = 0; i < denom; ++i)
          orig.push_back(infos[list[start + static_cast<size_t>(i)]].idx);
        std::sort(orig.begin(), orig.end(), std::greater<size_t>());
        NodePtr sym_clone = infos[list[start]].symbolic->clone();
        NodePtr replacement;
        if (base == 0) {
          replacement = std::move(sym_clone);
        } else {
          replacement = std::make_unique<BinOp>(std::move(sym_clone), BinOpKind::Add,
                                                std::make_unique<Constant>(base));
        }
        for (size_t oi : orig)
          terms.erase(terms.begin() + static_cast<std::ptrdiff_t>(oi));
        terms.push_back(std::move(replacement));
        return true;
      }
    }
    return false;
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

// Returns true if `var` contains `//`, `/:`, or `%` outside balanced parentheses,
// meaning the expression needs to be wrapped in parens when used as a
// multiplicand (since `*`, `//`, and `/:` share the same precedence level).
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
      if (i + 1 < var.size() && var[i] == '/' && (var[i + 1] == '/' || var[i + 1] == ':'))
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
      // Neither direct child is a constant.  Flatten the whole multiplication
      // chain to extract any embedded integer coefficient.  For example,
      // ``4096*batch_size*sequence_length`` parses as
      //   Mult(Mult(4096, batch_size), sequence_length)
      // so the direct-child check above misses the constant 4096.  By
      // collecting and separating all factors we can still combine
      // ``4096*a*b + 8*a*b`` into ``4104*a*b``.
      {
        std::vector<NodePtr> num, den;
        flatten_mul_div(*b, num, den);
        if (den.empty()) {
          // Single pass: accumulate the integer coefficient and collect each
          // symbolic factor exactly once, pairing it with its unparsed key
          // immediately so no second cast or second unparse is needed.
          int64_t num_c = 1;
          std::vector<std::pair<std::string, NodePtr>> sym_keyed;
          sym_keyed.reserve(num.size());
          for (auto &x : num) {
            if (const auto *c = dynamic_cast<const Constant *>(x.get()))
              num_c *= c->value;
            else
              sym_keyed.emplace_back(unparse(*x), std::move(x));
          }
          if (sym_keyed.size() < num.size()) {
            // At least one constant factor was extracted.
            if (sym_keyed.empty()) {
              res.const_term += num_c;
              return;
            }
            // Sort symbolic factors by their pre-computed key for a canonical form.
            std::sort(sym_keyed.begin(), sym_keyed.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
            NodePtr symbolic = std::move(sym_keyed[0].second);
            for (size_t i = 1; i < sym_keyed.size(); ++i)
              symbolic = std::make_unique<BinOp>(std::move(symbolic), BinOpKind::Mult,
                                                 std::move(sym_keyed[i].second));
            AddVisitorResult simp;
            run_add_visitor(*symbolic, simp);
            for (const auto &[k, v] : simp.coeffs)
              res.add_coeff(k, num_c * v);
            res.const_term += simp.const_term * num_c;
            return;
          }
        }
      }
      // All factors are symbolic (no extractable constant): fall through to generic_visit
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

namespace {

bool TryDifferenceLinearCombination(const std::string &expr1, const std::string &expr2,
                                    AddVisitorResult &out) {
  NodePtr tree;
  if (!TryParseExpression(expr1 + "-(" + expr2 + ")", tree)) {
    return false;
  }
  try {
    out = AddVisitorResult{};
    run_add_visitor(*tree, out);
    return true;
  } catch (const std::runtime_error &) {
    out = AddVisitorResult{};
    return false;
  }
}

} // namespace

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
  // ``match_subexpressions`` enables replacing whole compound subexpressions
  // (e.g. ``past_seq+seq``) whose textual form is a key of ``mapping`` with the
  // mapped name. When disabled (the default) only leaf identifiers are renamed,
  // preserving the original behavior of :func:`rename_expression`.
  explicit RenameTransformer(const std::unordered_map<std::string, std::string> &mapping,
                             bool match_subexpressions = false)
      : mapping_(mapping), match_subexpressions_(match_subexpressions) {}

  NodePtr visit(NodePtr node) override {
    if (match_subexpressions_ && node != nullptr && dynamic_cast<Name *>(node.get()) == nullptr &&
        dynamic_cast<Constant *>(node.get()) == nullptr) {
      // A compound (non-leaf) subexpression whose textual form matches a
      // mapping key collapses to the mapped name without descending further.
      auto it = mapping_.find(unparse(*node));
      if (it != mapping_.end()) {
        return std::make_unique<Name>(it->second);
      }
    }
    return Transformer::visit(std::move(node));
  }

  NodePtr visit_Name(std::unique_ptr<Name> n) override {
    auto it = mapping_.find(n->id);
    if (it != mapping_.end())
      n->id = it->second;
    return n;
  }

private:
  const std::unordered_map<std::string, std::string> &mapping_;
  bool match_subexpressions_;
};

// Collapses ``broadcast(x, x)`` to ``x``: once an equality constraint has been
// applied the two operands of a synthesized broadcast expression may become
// textually identical, in which case the broadcast is a no-op.
class BroadcastSimplifyTransformer : public Transformer {
public:
  NodePtr visit_Call(std::unique_ptr<Call> n) override {
    n = std::unique_ptr<Call>(static_cast<Call *>(generic_visit(std::move(n)).release()));
    if (n->func == "broadcast" && n->args.size() == 2 &&
        unparse(*n->args[0]) == unparse(*n->args[1])) {
      return std::move(n->args[0]);
    }
    return n;
  }
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
  DistributeFloorDivOverAddTransformer distrib_tr;
  FoldAddIntoFloorDivTransformer fold_add_floordiv_tr;
  NestedFloorDivTransformer nested_fd_tr;
  MaxToXorTransformer max_tr;
  ReorderCommutativeOpsTransformer reorder_tr;
  MaxIntTransformer maxint_tr;
  FloorDivAddRingTransformer fd_ring_tr;

  // Apply pipeline twice (matches the Python implementation)
  for (int pass = 0; pass < 2; ++pass) {
    node = ceil_tr.visit(std::move(node));
    node = simple_tr.visit(std::move(node));
    node = muldiv_tr.visit(std::move(node));
    node = fold_tr.visit(std::move(node));
    node = muldiv_tr.visit(std::move(node));
    node = nested_fd_tr.visit(std::move(node));
    node = fd_ring_tr.visit(std::move(node));
    node = distrib_tr.visit(std::move(node));
    node = fold_add_floordiv_tr.visit(std::move(node));
    node = max_tr.visit(std::move(node));
    // SimplifyParensTransformer is a no-op; omitted
    node = reorder_tr.visit(std::move(node));
    // StringToIntTransformer converts string constants to ints (not applicable here)
    node = maxint_tr.visit(std::move(node));
    node = fd_ring_tr.visit(std::move(node));
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
  if (!TryParseExpression(expr, tree)) {
    // Unparsable expression: return unchanged (matches Python behaviour for
    // expressions with invalid syntax like ONNX '::' node names)
    return expr;
  }

  tree = apply_pipeline(std::move(tree));

  AddVisitorResult res;
  run_add_visitor(*tree, res);
  return make_simplified(res);
}

// Builds `expr1 - (expr2)` and runs the add-visitor directly (no pipeline
// transformations — matches the Python implementation), returning the full
// linear combination (variable coefficients and constant term).
static AddVisitorResult difference_linear_combination(const std::string &expr1,
                                                      const std::string &expr2) {
  std::string combined = expr1 + "-(" + expr2 + ")";
  NodePtr tree = parse(combined);

  AddVisitorResult res;
  run_add_visitor(*tree, res);
  return res;
}

std::map<std::string, int64_t> simplify_two_expressions(const std::string &expr1,
                                                        const std::string &expr2) {
  AddVisitorResult res = difference_linear_combination(expr1, expr2);

  std::map<std::string, int64_t> out;
  for (const auto &[k, v] : res.coeffs)
    if (v != 0)
      out[k] = v;
  return out;
}

bool try_simplify_two_expressions(const std::string &expr1, const std::string &expr2,
                                  std::map<std::string, int64_t> &out) {
  AddVisitorResult res;
  if (!TryDifferenceLinearCombination(expr1, expr2, res)) {
    out.clear();
    return false;
  }

  out.clear();
  for (const auto &[k, v] : res.coeffs)
    if (v != 0)
      out[k] = v;
  return true;
}

ExpressionComparison compare_expressions(const std::string &expr1, const std::string &expr2) {
  // Collect the linear combination of expr1 - (expr2). simplify_two_expressions
  // discards the constant term, which is exactly what decides Greater/Smaller/
  // Equal here, so reuse the shared helper that keeps it.
  AddVisitorResult res = difference_linear_combination(expr1, expr2);

  // Inspect the non-zero token coefficients of expr1 - expr2. Every token is
  // assumed to be positive or null, so the minimum of the difference over all
  // non-negative token assignments is reached when all tokens are zero, which
  // equals the constant term.
  bool any_positive_coeff = false;
  bool any_negative_coeff = false;
  for (const auto &[k, v] : res.coeffs) {
    (void)k;
    if (v > 0)
      any_positive_coeff = true;
    else if (v < 0)
      any_negative_coeff = true;
  }

  CompareResult result;
  if (!any_positive_coeff && !any_negative_coeff) {
    // Pure constant difference.
    if (res.const_term > 0)
      result = CompareResult::Greater;
    else if (res.const_term < 0)
      result = CompareResult::Smaller;
    else
      result = CompareResult::Equal;
  } else if (!any_negative_coeff && res.const_term > 0) {
    // All token coefficients are non-negative and the constant is strictly
    // positive: the difference is strictly positive for any non-negative tokens.
    result = CompareResult::Greater;
  } else if (!any_positive_coeff && res.const_term < 0) {
    // All token coefficients are non-positive and the constant is strictly
    // negative: the difference is strictly negative for any non-negative tokens.
    result = CompareResult::Smaller;
  } else {
    result = CompareResult::Unknown;
  }

  // Always report the simplified difference expr2 - expr1.
  SimplifyResult difference = simplify_expression("(" + expr2 + ")-(" + expr1 + ")");
  return ExpressionComparison{result, difference};
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
      return floor_div_i64(l, r);
    case BinOpKind::ExactDiv:
      if (r == 0)
        throw std::runtime_error("Division by zero in expression '" + expr + "'");
      if (l % r != 0)
        throw std::runtime_error("Exact division has remainder in expression '" + expr + "'");
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
  if (!TryParseExpression(expr, tree)) {
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
  if (!TryParseExpression(expression, tree)) {
    return expression;
  }

  MaxToXorTransformer max_tr;
  RenameTransformer rename_tr(replacements, /*match_subexpressions=*/true);
  BroadcastSimplifyTransformer broadcast_tr;
  SimpleSimplifyTransformer simple_tr;

  tree = simple_tr.visit(broadcast_tr.visit(rename_tr.visit(max_tr.visit(std::move(tree)))));

  std::string out = unparse(*tree);
  out.erase(std::remove(out.begin(), out.end(), ' '), out.end());
  return out;
}

// ─────────────────────────────────────────────────────────────────────────

std::unordered_map<std::string, std::string> rename_dynamic_dimensions(
    const std::unordered_map<std::string, std::unordered_set<std::string>> &constraints,
    const std::unordered_set<std::string> &original, const std::string &ban_prefix) {
  std::unordered_map<std::string, std::string> replacements;
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
      // Preferred names keep their identity mapping. Without this guard, two
      // preferred names linked by a constraint would be swapped (each
      // overwriting the other's identity), which destroys both names instead
      // of leaving them alone.
      if (!original.count(k))
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

bool is_zero_dim(const DimType &d) {
  return std::holds_alternative<int64_t>(d) && std::get<int64_t>(d) == 0;
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

DimType dim_exact_div(const DimType &a, const DimType &b) {
  if (std::holds_alternative<int64_t>(a) && std::holds_alternative<int64_t>(b)) {
    int64_t av = std::get<int64_t>(a);
    int64_t bv = std::get<int64_t>(b);
    if (bv == 0)
      throw std::runtime_error("dim_exact_div: division by zero");
    if (av % bv != 0)
      throw std::runtime_error("dim_exact_div: division is not exact (" + std::to_string(av) +
                               " % " + std::to_string(bv) + " != 0)");
    return av / bv;
  }
  return simplify_dim("(" + dim_to_string(a) + ")/:(" + dim_to_string(b) + ")");
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

// ═══════════════════════════════════════════════════════════════════════════
// Dimension range inference
// ═══════════════════════════════════════════════════════════════════════════

// Returns {variable_name, product_of_divisors} when simplified_str matches the
// pattern  var // d1 // d2 // … // dn  where every divisor di is a strictly
// positive integer literal.  Returns nullopt for any other shape (multiple
// variables, non-integer divisors, arithmetic at the top level, etc.).
static std::optional<std::pair<std::string, int64_t>>
extract_floordiv_chain(const std::string &simplified_str) {
  NodePtr node;
  if (!TryParseExpression(simplified_str, node)) {
    return std::nullopt;
  }

  int64_t product = 1;
  const Node *current = node.get();

  // Walk the left-spine of floor-division nodes.
  while (true) {
    const auto *binop = dynamic_cast<const BinOp *>(current);
    if (binop && binop->op == BinOpKind::FloorDiv) {
      const auto *rhs_const = dynamic_cast<const Constant *>(binop->right.get());
      if (!rhs_const || rhs_const->value <= 0)
        return std::nullopt;
      product *= rhs_const->value;
      current = binop->left.get();
    } else {
      break;
    }
  }

  // The leaf must be a single Name node (bare variable).
  const auto *name_node = dynamic_cast<const Name *>(current);
  if (!name_node)
    return std::nullopt;

  return std::make_pair(name_node->id, product);
}

std::unordered_map<std::string, DimRange>
dim_ranges_from_expressions(const std::vector<std::pair<std::string, std::string>> &equalities,
                            const std::vector<std::string> &tokens) {
  std::unordered_map<std::string, DimRange> ranges;

  // Try to interpret chain_side as a floor-div chain of one variable and
  // compute the variable's range using value_side as the bound expression.
  auto process_side = [&](const std::string &chain_side, const std::string &value_side) {
    // Simplify the chain side; skip if it reduces to an integer.
    auto chain_simplified = simplify_expression(chain_side);
    if (std::holds_alternative<int64_t>(chain_simplified))
      return;
    const std::string &chain_str = std::get<std::string>(chain_simplified);

    // Extract the floor-div chain pattern.
    auto chain = extract_floordiv_chain(chain_str);
    if (!chain)
      return;
    const auto &[var_name, product] = *chain;

    // Simplify the value (bound) side.
    DimType value_dim;
    auto val_simplified = simplify_expression(value_side);
    if (std::holds_alternative<int64_t>(val_simplified)) {
      value_dim = std::get<int64_t>(val_simplified);
    } else {
      value_dim = std::get<std::string>(val_simplified);
    }

    if (product == 1) {
      // Direct equality: var == value  →  var ∈ [value, value].
      ranges[var_name] = {value_dim, value_dim};
    } else {
      // var // product == value  →  var ∈ [value·product, value·product + product − 1].
      DimType lower = dim_mul(value_dim, DimType{product});
      DimType upper = dim_add(lower, DimType{product - 1});
      ranges[var_name] = {lower, upper};
    }
  };

  for (const auto &[lhs, rhs] : equalities) {
    process_side(lhs, rhs);
    process_side(rhs, lhs);
  }

  if (tokens.empty())
    return ranges;

  // Filter to the requested subset of tokens.
  std::unordered_map<std::string, DimRange> filtered;
  for (const auto &tok : tokens) {
    auto it = ranges.find(tok);
    if (it != ranges.end())
      filtered[tok] = it->second;
  }
  return filtered;
}

} // namespace onnx_light::core::expressions
