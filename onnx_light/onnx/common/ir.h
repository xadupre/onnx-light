// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

/**
 * @file ir.h
 * @brief Lightweight in-memory intermediate representation (IR) for ONNX graphs.
 *
 * Provides the core graph data structures used by the ONNX version converter
 * and related tools:
 *
 * - @ref ONNX_NAMESPACE::Graph  – owns all nodes and values of one function.
 * - @ref ONNX_NAMESPACE::Node   – a single computation step with typed attributes.
 * - @ref ONNX_NAMESPACE::Value  – a typed edge (SSA value) connecting nodes.
 * - @ref ONNX_NAMESPACE::OpSetID – a lightweight (domain, version) pair without
 *   protobuf overhead.
 *
 * All raw pointers returned by this API are non-owning; the @c Graph is the
 * sole owner of every @c Node and @c Value it creates.  Destroying a @c Graph
 * invalidates all pointers into it.
 */

#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/common/array_ref.h"
#include "onnx/common/assertions.h"
#include "onnx/common/common.h"
#include "onnx/common/graph_node_list.h"
#include "onnx/common/tensor.h"
#include "onnx/string_utils.h"
#include "onnx/version_converter/internal_symbol.h"

#ifndef ONNX_DISALLOW_COPY_AND_ASSIGN
#define ONNX_DISALLOW_COPY_AND_ASSIGN(TypeName)                                                    \
  TypeName(const TypeName &) = delete;                                                             \
  TypeName &operator=(const TypeName &) = delete
#endif // ONNX_DISALLOW_COPY_AND_ASSIGN

namespace ONNX_NAMESPACE {

// internal/private API
static inline std::string toVarName(size_t i) {
  std::ostringstream oss;
  oss << "_v_" << i;
  return oss.str();
}

/**
 * @brief Lightweight computation graph that owns all its nodes and values.
 *
 * Uses a simple ownership model: the graph owns all nodes inside it and all
 * references within the graph are raw pointers.  Destroying the Graph
 * invalidates every pointer to any node or value in that graph.
 */
struct Graph;

/**
 * @brief Base class of every IR node.
 *
 * Represents a single computation step and its data dependencies expressed as
 * a list of input @ref Value pointers.  Also carries typed attributes (see
 * @ref Attributes).
 */
struct Node;

/**
 * @brief A typed SSA value connecting nodes in the IR graph.
 *
 * Represents an input or output of a @ref Node that carries either a tensor
 * type or an opaque handle, as indicated by its element type and optional
 * shape information.
 */
struct Value;

/**
 * @brief RAII guard that invokes a callable destructor when it goes out of scope.
 *
 * Non-copyable and non-movable.  Typically used to restore transient graph
 * state (e.g. stage number) set for the duration of a code region.
 */
class ResourceGuard final {
  std::function<void()> destructor_;

public:
  ONNX_DISALLOW_COPY_AND_ASSIGN(ResourceGuard);
  ResourceGuard(ResourceGuard &&) = delete;
  ResourceGuard &operator=(ResourceGuard &&) = delete;

  explicit ResourceGuard(std::function<void()> destructor) : destructor_(std::move(destructor)) {}

  /** @brief Invokes the destructor callable. */
  ~ResourceGuard() { destructor_(); }
};

/**
 * @brief Represents a single dimension of a tensor shape.
 *
 * A dimension is one of:
 * - *unknown* – neither an integer nor a symbolic parameter (default).
 * - *symbolic* – a named parameter string (e.g. @c "batch_size").
 * - *static*   – a concrete non-negative @c int64_t value.
 */
struct Dimension final {
  /** @brief Constructs an unknown dimension. */
  Dimension() : is_unknown(true), is_int(false), dim(-1) {}
  /**
   * @brief Constructs a symbolic dimension.
   * @param param Symbolic parameter name.
   */
  explicit Dimension(std::string param)
      : is_unknown(false), is_int(false), dim(-1), param(std::move(param)) {}
  /**
   * @brief Constructs a static integer dimension.
   * @param dim Non-negative size of this dimension.
   */
  explicit Dimension(int64_t dim) : is_unknown(false), is_int(true), dim(dim) {}

  /** @brief True when the dimension is completely unknown. */
  bool is_unknown;
  /** @brief True when @ref dim holds a concrete integer value. */
  bool is_int;
  /** @brief The concrete size; meaningful only when @ref is_int is true. */
  int64_t dim;
  /** @brief The symbolic parameter name; meaningful only when @ref is_int and @ref is_unknown are
   * both false. */
  std::string param;
};

/**
 * @brief Discriminator tag for the concrete type of an attribute value.
 *
 * Each enumerator corresponds to a scalar or vector ONNX attribute type:
 * float (@c f), float list (@c fs), integer (@c i), integer list (@c is),
 * string (@c s), string list (@c ss), tensor (@c t), tensor list (@c ts),
 * subgraph (@c g), subgraph list (@c gs), type-proto (@c tp), and
 * type-proto list (@c tps).
 */
enum class AttributeKind : uint8_t {
  // float, float list, int, int list, string, string list,
  // tensor, tensor list, subgraph, subgraph list. type proto, type proto list
  f,
  fs,
  i,
  is,
  s,
  ss,
  t,
  ts,
  g,
  gs,
  tp,
  tps
};

/**
 * @brief Converts an @ref AttributeKind enumerator to its string name.
 * @param kind The attribute kind.
 * @return A null-terminated string such as @c "f", @c "fs", @c "i", etc.
 */
static inline const char *toString(AttributeKind kind) {
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  static constexpr const char *names[] = {"f", "fs", "i", "is", "s",  "ss",
                                          "t", "ts", "g", "gs", "tp", "tps"};
  ONNX_ASSERT(size_t(kind) < std::size(names))
  return names[static_cast<int>(kind)];
}

/**
 * @brief Abstract base class for a named attribute value stored on a @ref Node.
 *
 * Concrete subtypes are @ref ScalarAttributeValue and @ref VectorAttributeValue,
 * instantiated for each supported ONNX attribute kind.
 */
struct AttributeValue {
  /**
   * @brief Constructs an attribute with the given name.
   * @param name Symbol identifying the attribute.
   */
  explicit AttributeValue(Symbol name) : name(name) {}
  /** @brief Owning pointer alias for convenience. */
  using Ptr = std::unique_ptr<AttributeValue>;
  /** @brief The attribute's symbolic name. */
  Symbol name;
  /** @brief Returns the discriminator kind of this attribute. */
  virtual AttributeKind kind() const = 0;
  /** @brief Returns a deep copy of this attribute. */
  virtual Ptr clone() const = 0;
  virtual ~AttributeValue() = default;
};

/**
 * @brief Holds a single scalar attribute value of type @p T and kind @p Kind.
 * @tparam T  The C++ value type (e.g. @c double, @c int64_t, @c std::string).
 * @tparam Kind The @ref AttributeKind tag for this type.
 */
template <typename T, AttributeKind Kind>
struct ScalarAttributeValue final : public AttributeValue {
  using ConstructorType = const T &;
  using ValueType = T;
  ScalarAttributeValue(Symbol name, ConstructorType value_)
      : AttributeValue(name), value_(std::move(value_)) {}
  /** @brief Returns a mutable reference to the stored scalar value. */
  ValueType &value() { return value_; }
  /** @brief Returns a deep copy wrapped in a @c Ptr. */
  Ptr clone() const override { return std::make_unique<ScalarAttributeValue>(name, value_); }
  /** @brief Returns @p Kind. */
  AttributeKind kind() const override { return Kind; }

private:
  ValueType value_;
};

/**
 * @brief Holds a vector attribute value of element type @p T and kind @p Kind.
 * @tparam T  The C++ element type (e.g. @c double, @c int64_t).
 * @tparam Kind The @ref AttributeKind tag for this list type.
 */
template <typename T, AttributeKind Kind>
struct VectorAttributeValue final : public AttributeValue {
  using ConstructorType = const std::vector<T> &&;
  using ValueType = std::vector<T>;
  VectorAttributeValue(Symbol name, ValueType value_)
      : AttributeValue(name), value_(std::move(value_)) {}
  /** @brief Returns a mutable reference to the stored vector. */
  ValueType &value() { return value_; }
  /** @brief Returns @p Kind. */
  AttributeKind kind() const override { return Kind; }
  /** @brief Returns a deep copy wrapped in a @c unique_ptr. */
  std::unique_ptr<AttributeValue> clone() const override {
    return std::make_unique<VectorAttributeValue>(name, ValueType(value_));
  }

private:
  ValueType value_;
};

/** @brief Scalar float attribute (@c double storage, kind @c f). */
using FloatAttr = ScalarAttributeValue<double, AttributeKind::f>;
/** @brief Float-list attribute (@c double element storage, kind @c fs). */
using FloatsAttr = VectorAttributeValue<double, AttributeKind::fs>;
/** @brief Scalar integer attribute (@c int64_t storage, kind @c i). */
using IntAttr = ScalarAttributeValue<int64_t, AttributeKind::i>;
/** @brief Integer-list attribute (@c int64_t element storage, kind @c is). */
using IntsAttr = VectorAttributeValue<int64_t, AttributeKind::is>;
/** @brief Scalar string attribute (@c std::string storage, kind @c s). */
using StringAttr = ScalarAttributeValue<std::string, AttributeKind::s>;
/** @brief String-list attribute (@c std::string element storage, kind @c ss). */
using StringsAttr = VectorAttributeValue<std::string, AttributeKind::ss>;
/** @brief Scalar tensor attribute (kind @c t). */
using TensorAttr = ScalarAttributeValue<Tensor, AttributeKind::t>;
/** @brief Tensor-list attribute (kind @c ts). */
using TensorsAttr = VectorAttributeValue<Tensor, AttributeKind::ts>;
/** @brief Scalar subgraph attribute (kind @c g). */
using GraphAttr = ScalarAttributeValue<std::shared_ptr<Graph>, AttributeKind::g>;
/** @brief Subgraph-list attribute (kind @c gs). */
using GraphsAttr = VectorAttributeValue<std::shared_ptr<Graph>, AttributeKind::gs>;
/** @brief Scalar type-proto attribute (kind @c tp). */
using TypeProtoAttr = ScalarAttributeValue<TypeProto, AttributeKind::tp>;
/** @brief Type-proto-list attribute (kind @c tps). */
using TypeProtosAttr = VectorAttributeValue<TypeProto, AttributeKind::tps>;

/**
 * @brief CRTP mixin that adds typed attribute storage to a node class.
 *
 * Enables method chaining via a @p Derived* return type, e.g.:
 * @code
 * Node *n = g->create(kSelect)->i_(kOffset, 3)->f_(kValue, 3.5f);
 * @endcode
 *
 * Attribute lookup is O(n) in the number of attributes to keep deterministic
 * ordering; nodes are not expected to carry a large number of attributes.
 *
 * @tparam Derived The concrete node type that inherits this mixin.
 */
template <typename Derived> struct Attributes {
  // NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
  Attributes() = default;

  /**
   * @brief Replaces all attributes of this object with deep copies from @p rhs.
   * @param rhs Source object whose attributes are cloned.
   */
  void copyAttributes(const Attributes &rhs) {
    values_.clear();
    values_.reserve(rhs.values_.size());
    for (const auto &i : rhs.values_) {
      values_.push_back(i->clone());
    }
  }
  /**
   * @brief Returns true if an attribute with the given name exists.
   * @param name Attribute symbol to look up.
   */
  bool hasAttribute(Symbol name) const { return find(name, false) != values_.end(); }
  /**
   * @brief Returns the kind of the attribute with the given name.
   * @param name Attribute symbol (must exist).
   */
  AttributeKind kindOf(Symbol name) const { return (*find(name, true))->kind(); }
  /**
   * @brief Removes the attribute with the given name and returns @c this.
   * @param name Attribute symbol (must exist).
   */
  Derived *removeAttribute(Symbol name) {
    values_.erase(find(name, true));
    return This();
  }
  /** @brief Returns true if any attributes are set. */
  bool hasAttributes() const { return !values_.empty(); }
  /**
   * @brief Returns the names of all set attributes in insertion order.
   */
  // The names are returned in order, since name actually is the index.
  std::vector<Symbol> attributeNames() const {
    std::vector<Symbol> names;
    names.reserve(values_.size());
    for (const auto &a : values_)
      names.push_back(a->name);
    return names;
  }

#define CREATE_ACCESSOR(Kind, method)                                                              \
  Derived *method##_(Symbol name, Kind##Attr::ConstructorType v) {                                 \
    return set<Kind##Attr>(name, std::forward<Kind##Attr::ConstructorType>(v));                    \
  }                                                                                                \
  const Kind##Attr::ValueType &method(Symbol name) const { return get<Kind##Attr>(name); }
  CREATE_ACCESSOR(Float, f)
  CREATE_ACCESSOR(Floats, fs)
  CREATE_ACCESSOR(String, s)
  CREATE_ACCESSOR(Strings, ss)
  CREATE_ACCESSOR(Int, i)
  CREATE_ACCESSOR(Ints, is)
  CREATE_ACCESSOR(Tensor, t)
  CREATE_ACCESSOR(Tensors, ts)
  CREATE_ACCESSOR(Graph, g)
  CREATE_ACCESSOR(Graphs, gs)
  CREATE_ACCESSOR(TypeProto, tp)
  CREATE_ACCESSOR(TypeProtos, tps)

#undef CREATE_ACCESSOR

private:
  Derived *This() { return static_cast<Derived *>(this); }
  template <typename T> Derived *set(Symbol name, typename T::ConstructorType v) {
    auto it = find(name, false);
    auto nv = std::make_unique<T>(name, std::forward<typename T::ConstructorType>(v));
    if (it == values_.end()) {
      values_.push_back(std::move(nv));
    } else {
      *it = std::move(nv);
    }
    return This();
  }
  template <typename T> typename T::ValueType &get(Symbol name) const {
    auto it = find(name, true);
    T *child = static_cast<T *>(it->get());
    return child->value();
  }
  using AVPtr = AttributeValue::Ptr;
  // NB: For determinism, we use a vector rather than a hash map.  This does
  // mean that lookups are O(n), so you shouldn't use Attributes to store
  // a big pile of messages.
  std::vector<AVPtr> values_;
  using iterator = std::vector<AVPtr>::iterator;
  iterator find(Symbol name, bool required) {
    auto it = std::find_if(values_.begin(), values_.end(),
                           [&](const AVPtr &v) { return v->name == name; });
    ONNX_ASSERT(!required || it != values_.end())
    return it;
  }
  using const_iterator = std::vector<AVPtr>::const_iterator;
  const_iterator find(Symbol name, bool required) const {
    auto it = std::find_if(values_.begin(), values_.end(),
                           [&](const AVPtr &v) { return v->name == name; });
    ONNX_ASSERTM(!required || it != values_.end(), "%s:%u: %s: required undefined attribute '%s'",
                 __FILE__, __LINE__, __func__, name.toString())
    return it;
  }
};

/**
 * @brief Records a single use of a @ref Value by a @ref Node.
 *
 * A @c Use links a producer @ref Value to the @ref Node that consumes it
 * and the position (offset) in that node's input list where the value appears.
 */
struct Use final {
  /**
   * @brief Constructs a use record.
   * @param user   The node that consumes the value.
   * @param offset Index into @p user's input list.
   */
  Use(Node *user, size_t offset) : user(user), offset(offset) {}
  /** @brief The consuming node. */
  Node *user;
  /** @brief Index of this value in @ref user's input list. */
  size_t offset;
};

/**
 * @brief Returns true if two @ref Use records refer to the same (node, offset) pair.
 */
static inline bool operator==(const Use &a, const Use &b) {
  return a.user == b.user && a.offset == b.offset;
}

/** @brief Ordered list of @ref Node pointers. */
using node_list = std::vector<Node *>;
/** @brief Ordered list of @ref Value pointers. */
using value_list = std::vector<Value *>;
/** @brief List of @ref Use records for a value. */
using use_list = std::vector<Use>;
/** @brief Symbolic kind identifier for a @ref Node. */
using NodeKind = Symbol;

/**
 * @brief A typed SSA value (edge) in the IR graph.
 *
 * Each @c Value is produced by exactly one @ref Node (its @ref node()) and may
 * be consumed by zero or more nodes, tracked via @ref uses().  The value
 * carries optional type information: ONNX element type (@ref elemType()) and
 * shape (@ref sizes()).
 *
 * @c Value objects are owned by their @ref Graph; do not delete them manually.
 */
struct Value final {
  ONNX_DISALLOW_COPY_AND_ASSIGN(Value);
  /**
   * @brief Constructs a value produced by @p node_ at output position @p offset_.
   */
  Value(Node *node_, size_t offset_);
  Value(Value &&) = default;
  Value &operator=(Value &&) = default;
  ~Value() = default;

private:
  friend struct Node;
  friend struct Graph;
  Node *node_;
  size_t offset_;
  size_t unique_ = 0; // unique id
  size_t stage_ = 0;  // 0-forward, 1-backward, 2-double-backward,...
  use_list uses_in_current_graph_;
  bool has_unique_name_{false};
  std::string unique_name_;
  int32_t elem_type_{TensorProto::DataType::UNDEFINED};
  bool has_sizes_{false};
  std::vector<Dimension> sizes_;
  std::unique_ptr<TypeProto> type_;

public:
  /**
   * @brief Sets the ONNX element type and returns @c this for chaining.
   * @param elem_type ONNX DataType enum value.
   */
  Value *setElemType(int32_t elem_type) {
    elem_type_ = elem_type;
    return this;
  }
  /** @brief Returns the ONNX element type. */
  int32_t elemType() const { return elem_type_; }
  /** @brief Returns true if shape information has been set. */
  bool has_sizes() const { return has_sizes_; }
  /**
   * @brief Sets the shape of this value and returns @c this for chaining.
   * @param sizes Per-dimension size descriptors.
   */
  Value *setSizes(std::vector<Dimension> sizes) {
    has_sizes_ = true;
    sizes_ = std::move(sizes);
    return this;
  }
  /**
   * @brief Clears any previously set shape information and returns @c this.
   */
  Value *wipeSizes() {
    has_sizes_ = false;
    sizes_ = std::vector<Dimension>();
    return this;
  }
  /** @brief Returns the shape dimensions (may be empty if shape is unknown). */
  const std::vector<Dimension> &sizes() const { return sizes_; }
  /** @brief Returns the graph-unique numeric identifier of this value. */
  size_t unique() const { return unique_; }
  /** @brief Returns true if an explicit string name has been assigned. */
  bool has_unique_name() const { return has_unique_name_; }
  /**
   * @brief Returns the string name of this value.
   *
   * If no explicit name has been set, returns a generated name of the form
   * @c "_v_N" where @c N is @ref unique().
   */
  std::string uniqueName() const {
    if (has_unique_name())
      return unique_name_;
    return toVarName(unique());
  }
  /**
   * @brief Sets the string name of this value.
   * @param name                 New name for this value.
   * @param update_related_names When true, propagates the rename to
   *                             initializer entries and captured nodes in
   *                             subgraphs.
   */
  Value *setUniqueName(const std::string &name, bool update_related_names = true);
  /**
   * @brief Sets the stage index (forward=0, backward=1, …) and returns @c this.
   * @param s New stage index.
   */
  Value *setStage(size_t s) {
    stage_ = s;
    return this;
  }
  /** @brief Returns the stage index. */
  size_t stage() const { return stage_; }
  /** @brief Returns the node that produces this value. */
  Node *node() { return node_; }
  /** @brief Returns the output offset within the producing node. */
  size_t offset() const { return offset_; }
  /** @brief Returns the node that produces this value (const overload). */
  const Node *node() const { return node_; }
  /** @brief Returns the owning graph. */
  Graph *owningGraph();
  /** @brief Returns the owning graph (const overload). */
  const Graph *owningGraph() const;
  /**
   * @brief Returns all recorded uses of this value in the current graph.
   */
  use_list uses() const;

  /**
   * @brief Replaces all uses of this value with @p newValue.
   *
   * Also propagates element type and size information to @p newValue, and
   * renames captured references in subgraphs if this value is a graph output.
   *
   * Given:   %3 = f(%1, %2)
   *          %4 = g(%3)
   *          %5 = h(%3, %3)
   * Execute: %3.replaceAllUsesWith(%6)
   * Result:  %3 = f(%1, %2)
   *          %4 = g(%6)
   *          %5 = h(%6, %6)
   *
   * @param newValue The replacement value (must belong to the same graph).
   */
  void replaceAllUsesWith(Value *newValue);

  /**
   * @brief Copies element type, shape, and optional name from @p from.
   * @param from Source value whose metadata is copied.
   * @return @c this for chaining.
   */
  Value *copyMetadata(const Value *from) {
    setElemType(from->elemType());
    setSizes(from->sizes());
    if (from->has_unique_name()) {
      setUniqueName(from->uniqueName());
    }
    return this;
  }

  /** @brief Returns a mutable reference to the optional TypeProto for this value. */
  std::unique_ptr<TypeProto> &type() { return type_; }
};

/**
 * @brief A single computation node in the IR graph.
 *
 * A @c Node has a symbolic @ref kind() that identifies the operation, an
 * ordered list of input @ref Value pointers, and a list of output @ref Value
 * objects it produces.  It also inherits typed attribute storage from
 * @ref Attributes<Node>.
 *
 * Nodes are maintained in topological order via an intrusive doubly-linked
 * list (@ref next_in_graph).  The @c Return / @c Param pseudo-nodes serve as
 * sentinels and are not part of the logical computation.
 *
 * @c Node objects are owned by their @ref Graph; never delete them manually —
 * use @ref destroy() instead.
 */
struct Node : public Attributes<Node> {
  ONNX_DISALLOW_COPY_AND_ASSIGN(Node);
  friend struct Graph;
  friend struct Value;

  // Each node (except Return/Param) is associated with exactly one place in
  // the graph's node list.  The list is a circular doubly-linked list where
  // the Return node acts as sentinel (never null).  Index 0 is the forward
  // (next) pointer; index 1 is the backward (prev) pointer.  A single array
  // of two pointers allows the same iterator class to handle both directions.
  // The list represents a topological sort.

  /**
   * @brief Returns a reference to the linked-list pointer in direction @p i.
   * @param i 0 for @c next, 1 for @c prev (see @c kNextDirection / @c kPrevDirection).
   */
  Node *&next_in_graph(size_t i) { return next_in_graph_[i]; }
  /** @brief Returns a const reference to the linked-list pointer in direction @p i. */
  Node *const &next_in_graph(size_t i) const { return next_in_graph_[i]; }
  /** @brief Returns a mutable reference to the forward (next) linked-list pointer. */
  Node *&next() { return next_in_graph_[kNextDirection]; }
  /** @brief Returns a mutable reference to the backward (prev) linked-list pointer. */
  Node *&prev() { return next_in_graph_[kPrevDirection]; }
  /** @brief Returns a const reference to the forward (next) linked-list pointer. */
  Node *const &next() const { return next_in_graph_[kNextDirection]; }
  /** @brief Returns a const reference to the backward (prev) linked-list pointer. */
  Node *const &prev() const { return next_in_graph_[kPrevDirection]; }

private:
  std::array<Node *, 2> next_in_graph_{nullptr, nullptr};
  const NodeKind kind_;
  std::vector<Value *> inputs_;
  std::vector<Value *> outputs_;
  Graph *graph_;
  size_t stage_;
  bool has_name_{false};
  std::string name_;
  bool has_domain_{false};
  std::string domain_;
  bool has_doc_string_{false};
  std::string doc_string_;
  bool has_overload_{false};
  std::string overload_;

protected:
  Node(Graph *graph_, NodeKind kind_); // defined after graph

public:
  /** @brief Returns true if a node name is set. */
  bool has_name() const { return has_name_; }
  /** @brief Returns the node name (meaningful only when @ref has_name() is true). */
  const std::string &name() const { return name_; }
  /**
   * @brief Sets the node name.
   * @param name Node name string.
   */
  void setName(std::string name) {
    has_name_ = true;
    name_ = std::move(name);
  }
  /** @brief Returns true if an operator domain is set. */
  bool has_domain() const { return has_domain_; }
  /** @brief Returns the operator domain (meaningful only when @ref has_domain() is true). */
  const std::string &domain() const { return domain_; }
  /**
   * @brief Sets the operator domain.
   * @param domain Domain string (e.g. @c "" for the default ONNX domain).
   */
  void setDomain(std::string domain) {
    has_domain_ = true;
    domain_ = std::move(domain);
  }
  /** @brief Returns true if an overload string is set. */
  bool has_overload() const { return has_overload_; }
  /** @brief Returns the overload string (meaningful only when @ref has_overload() is true). */
  const std::string &overload() const { return overload_; }
  /**
   * @brief Sets the overload string.
   * @param overload Overload identifier.
   */
  void setOverload(std::string overload) {
    has_overload_ = true;
    overload_ = std::move(overload);
  }
  /** @brief Returns true if a documentation string is set. */
  bool has_doc_string() const { return has_doc_string_; }
  /** @brief Returns the documentation string. */
  const std::string &docString() const { return doc_string_; }
  /**
   * @brief Sets the documentation string.
   * @param doc_string Documentation text.
   */
  void setDocString(std::string doc_string) {
    has_doc_string_ = true;
    doc_string_ = std::move(doc_string);
  }
  /** @brief Returns the symbolic kind of this node (e.g. @c kAdd, @c kConv). */
  NodeKind kind() const { return kind_; }
  /** @brief Returns the owning graph. */
  Graph *owningGraph() { return graph_; }
  /** @brief Returns the owning graph (const overload). */
  const Graph *owningGraph() const { return graph_; }
  /** @brief Returns the current stage index of this node. */
  size_t stage() const { return stage_; }
  /**
   * @brief Sets the stage index and returns @c this.
   * @param s New stage index (0 = forward, 1 = backward, …).
   */
  Node *setStage(size_t s) {
    stage_ = s;
    return this;
  }
  /**
   * @brief Returns an @c ArrayRef over this node's input values.
   *
   * @warning The returned reference is invalidated if the input list is
   *          resized (e.g. via @ref addInput).
   */
  ArrayRef<Value *> inputs() { return inputs_; }
  /** @brief Returns a const @c ArrayRef over this node's input values. */
  ArrayRef<const Value *> inputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {inputs_.data(), inputs_.size()};
  }
  /**
   * @brief Returns an @c ArrayRef over this node's output values.
   *
   * @warning The returned reference is invalidated if the output list is
   *          resized.
   */
  ArrayRef<Value *> outputs() { return outputs_; }
  /** @brief Returns a const @c ArrayRef over this node's output values. */
  ArrayRef<const Value *> outputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {outputs_.data(), outputs_.size()};
  }
  /** @brief Returns true if any output of this node has at least one use. */
  bool hasUses() const {
    for (const auto *o : outputs()) {
      if (!o->uses().empty())
        return true;
    }
    return false;
  }
  /**
   * @brief Replaces all uses of every output of @c this with the corresponding
   *        output of @p n.
   *
   * @p n must have the same number of outputs as @c this.
   * @param n Replacement node.
   */
  void replaceAllUsesWith(Node *n) {
    ONNX_ASSERT(outputs().size() == n->outputs().size())
    size_t nOutputs = outputs().size();
    for (size_t i = 0; i < nOutputs; i++) {
      outputs()[i]->replaceAllUsesWith(n->outputs()[i]);
    }
  }
  /**
   * @brief Returns the sole input of this node.
   *
   * Asserts that the node has exactly one input.
   */
  // lots of things like chunk have a single input or single output, so we have a
  // helper to make accessing it easier
  Value *input() {
    ONNX_ASSERT(inputs_.size() == 1)
    return inputs_.at(0);
  }
  /**
   * @brief Returns the sole output of this node.
   *
   * Asserts that the node has exactly one output.
   */
  Value *output() {
    ONNX_ASSERT(outputs_.size() == 1)
    return outputs_.at(0);
  }
  /** @brief Returns the sole input of this node (const overload). */
  const Value *input() const {
    ONNX_ASSERT(inputs_.size() == 1)
    return inputs_.at(0);
  }
  /** @brief Returns the sole output of this node (const overload). */
  const Value *output() const {
    ONNX_ASSERT(outputs_.size() == 1)
    return outputs_.at(0);
  }
  /**
   * @brief Returns the input at index @p i (bounds-checked).
   * @param i Zero-based input index.
   */
  Value *input(size_t i) { return inputs_.at(i); }
  /** @brief Returns the input at index @p i (const overload). */
  const Value *input(size_t i) const { return inputs_.at(i); }

  // Graphs

  // Note [Topological invariant]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // We always maintain an up-to-date topological ordering of all nodes via
  // the next()/prev() links.  All transformations to graphs must preserve
  // this topological ordering: for example, it is only valid to 'addInput'
  // with an input which is topologically before the current node.
  //
  // Usually, it is obvious whether or not topological order is maintained;
  // for example, if you are adding nodes to the end of the topsort, it's
  // impossible for them to refer to inputs that are not in the topsort.
  // If it is not obvious, please comment accordingly.

  /**
   * @brief Appends @p node as an input to @c this and returns it for chaining.
   *
   * Given:   %3 = f(%1, %2)
   * Execute: %3.addInput(%4)
   * Result:  %3 = f(%1, %2, %4)
   *
   * @param node Input value to append (must belong to the same graph).
   * @return The appended value @p node.
   */
  Value *addInput(Value *node) {
    ONNX_ASSERT(graph_ == node->owningGraph())
    node->uses_in_current_graph_.emplace_back(this, inputs_.size());
    inputs_.push_back(node);
    return node;
  }

  /**
   * @brief Replaces the input at position @p i with @p newValue.
   *
   * Given:   %3 = f(%1, %2)
   * Execute: %3.replaceInput(1, %4)
   * Result:  %3 = f(%1, %4)
   *
   * @param i        Zero-based index of the input to replace.
   * @param newValue Replacement value (must belong to the same graph).
   * @return The old input value at position @p i.
   */
  Value *replaceInput(size_t i, Value *newValue) {
    ONNX_ASSERT(newValue->owningGraph() == graph_)
    Value *old = dropInput(i);
    inputs_[i] = newValue;
    newValue->uses_in_current_graph_.emplace_back(this, i);
    return old;
  }

  /**
   * @brief Replaces all occurrences of @p from in the input list with @p to.
   *
   * Corresponds to LLVM's @c replaceUsesOfWith.
   *
   * Given:   %3 = f(%1, %2, %1)
   * Execute: %3.replaceInputWith(%1, %4)
   * Result:  %3 = f(%4, %2, %4)
   *
   * @param from Value to replace (must belong to the same graph).
   * @param to   Replacement value (must belong to the same graph).
   */
  void replaceInputWith(Value *from, Value *to) {
    ONNX_ASSERT(from->owningGraph() == graph_)
    ONNX_ASSERT(to->owningGraph() == graph_)
    size_t i = 0;
    for (const auto *input : inputs()) {
      if (input == from)
        replaceInput(i, to);
      i++;
    }
  }

  /**
   * @brief Appends a new output @ref Value to this node and returns it.
   */
  Value *addOutput() {
    outputs_.push_back(new Value(this, outputs_.size()));
    return outputs_.back();
  }

  /**
   * @brief Removes the output at index @p i.
   * @param i Zero-based output index (the output must have no uses).
   */
  void eraseOutput(size_t i);

  /**
   * @brief Inserts an unattached node @em before @p n in topological order.
   *
   * Given:   %3 = f(%1, %2)
   *          %4 = g(%3)
   * and unattached: %5 = h(%1)
   * Execute: %5.insertBefore(%4)
   * Result:  %3 = f(%1, %2)
   *          %5 = h(%1)
   *          %4 = g(%3)
   *
   * @param n An existing node already in the graph list.
   * @return @c this for chaining.
   */
  Node *insertBefore(Node *n) {
    ONNX_ASSERT(n->inGraphList())
    insertAfter(n->prev());
    return this;
  }

  /**
   * @brief Inserts an unattached node @em after @p n in topological order.
   *
   * Given: %3 = f(%1, %2)
   *        %4 = g(%3)
   * and unattached: %5 = h(%1)
   * Execute: %5.insertAfter(%4)
   * Result:  %3 = f(%1, %2)
   *          %4 = g(%3)
   *          %5 = h(%1)
   *
   * @param n An existing node already in the graph list.
   * @return @c this for chaining.
   */
  Node *insertAfter(Node *n) {
    ONNX_ASSERT(!inGraphList() && n->inGraphList())
    Node *next = n->next();
    n->next() = this;
    this->prev() = n;
    this->next() = next;
    next->prev() = this;
    return this;
  }

  /**
   * @brief Moves @c this (already in the graph) to appear after @p n.
   *
   * Given: %2 = f(%1)
   *        %3 = g(%1)
   * Execute: %2.moveAfter(%3)
   * Result: %3 = g(%1)
   *         %2 = f(%1)
   *
   * @param n Target node (must be in the graph).
   */
  void moveAfter(Node *n) {
    removeFromList();
    insertAfter(n);
  }

  /**
   * @brief Moves @c this (already in the graph) to appear before @p n.
   *
   * Given: %2 = f(%1)
   *        %3 = g(%1)
   * Execute: %3.moveBefore(%2)
   * Result: %3 = g(%1)
   *         %2 = f(%1)
   *
   * @param n Target node (must be in the graph).
   */
  void moveBefore(Node *n) {
    removeFromList();
    insertBefore(n);
  }

  /**
   * @brief Removes the input at index @p i.
   *
   * @warning This is O(n) in the number of inputs; avoid calling it in a loop.
   *
   * Given: %3 = f(%1, %2)
   * Execute: %3.removeInput(1)
   * Result: %3 = f(%1)
   *
   * @param i Zero-based index of the input to remove.
   */
  void removeInput(size_t i) {
    dropInput(i);
    // everything after this input shifts left,
    // so we need to update their use offsets to match
    for (size_t j = i + 1; j < inputs_.size(); j++) {
      auto it = findUseForInput(j);
      it->offset--;
    }
    inputs_.erase(inputs_.begin() + i);
  }

  /**
   * @brief Removes all inputs from this node.
   *
   * Given: %3 = f(%1, %2)
   * Execute: %3.removeAllInputs()
   * Result: %3 = f()
   */
  void removeAllInputs() {
    for (size_t i = 0; i < inputs().size(); ++i)
      dropInput(i);
    inputs_.clear();
  }

  /**
   * @brief Returns true if @c this appears before @p n in topological order.
   * @param n Node to compare against.
   */
  bool isBefore(const Node *n);

  /**
   * @brief Returns a forward iterator to the node-list position of @c this.
   *
   * Useful for resuming a graph traversal starting at a known node.
   */
  graph_node_list_iterator iterator();
  /** @brief Returns a reverse iterator to the node-list position of @c this. */
  graph_node_list_iterator reverseIterator();
  /** @brief Returns a const forward iterator to the node-list position of @c this. */
  const_graph_node_list_iterator iterator() const;
  /** @brief Returns a const reverse iterator to the node-list position of @c this. */
  const_graph_node_list_iterator reverseIterator() const;

  /**
   * @brief Removes @c this from the node list and deallocates it.
   *
   * @pre No outputs of @c this may have any remaining uses.
   *
   * Given: %2 = f(%1)
   *        %3 = g(%1)
   * Execute: %2.destroy()
   * Result: %3 = g(%1)
   */
  void destroy();

  /**
   * @brief Dynamically casts @c this to @p T; returns @c nullptr if the kind
   *        does not match.
   *
   * Example: @code if (auto *s = n->cast<Select>()) { ... } @endcode
   *
   * @tparam T Subclass with a static @c Kind member to compare against.
   */
  template <typename T> T *cast() {
    if (T::Kind == kind())
      return static_cast<T *>(this);
    return nullptr;
  }
  /** @brief Const overload of @ref cast(). */
  template <typename T> const T *cast() const {
    if (T::Kind == kind())
      return static_cast<const T *>(this);
    return nullptr;
  }
  /**
   * @brief Asserts that the kind matches @p T and returns a typed pointer.
   *
   * Aborts with a descriptive message if the kind does not match.
   * @tparam T Subclass with a static @c Kind member.
   */
  template <typename T> T *expect() {
    ONNX_ASSERTM(T::Kind == kind(), "expected a %s but found a %s", T::Kind.toString(),
                 kind().toString())
    return static_cast<T *>(this);
  }
  /** @brief Const overload of @ref expect(). */
  template <typename T> const T *expect() const {
    ONNX_ASSERTM(T::Kind == kind(), "expected a %s but found a %s", T::Kind.toString(),
                 kind().toString())
    return static_cast<const T *>(this);
  }

  virtual ~Node() = default;

private:
  // Lookup iterator in use list of _input i_ that corresponds to its use of _this_
  use_list::iterator findUseForInput(size_t i) {
    auto &input_uses = inputs_[i]->uses_in_current_graph_;
    // O(N) on the use list, but unless we get nodes with +100 uses
    // vector traversal still is probably faster than linked list
    auto use_it = std::find(input_uses.begin(), input_uses.end(), Use(this, i));
    ONNX_ASSERT(use_it != input_uses.end())
    return use_it;
  }

  // remove the use of input i, this sets input i to nullptr, but
  // is only used internally to Node before setting it to a new value
  // or erasing the entry from the list.
  Value *dropInput(size_t i) {
    ONNX_ASSERT(i < inputs_.size())
    auto *input_node = inputs_[i];
    auto use_it = findUseForInput(i);
    input_node->uses_in_current_graph_.erase(use_it);
    inputs_[i] = nullptr;
    return input_node;
  }

  bool inGraphList() const {
    ONNX_ASSERT(next() != nullptr || prev() == nullptr)
    return next() != nullptr;
  }
  void removeFromList() {
    ONNX_ASSERT(inGraphList())
    Node *next = this->next();
    Node *prev = this->prev();
    prev->next() = next;
    next->prev() = prev;
    this->next() = nullptr;
    this->prev() = nullptr;
  }

protected:
  /**
   * @brief Allocates a new node of the same concrete type in graph @p g.
   *
   * Used by @c createClone to create a fresh instance in a different graph.
   * Subclasses must override if they introduce additional state.
   */
  virtual Node *allocNewInstance(Graph *g) { return new Node(g, kind()); }
  /**
   * @brief Copies all attribute values from @p s into @c this.
   *
   * Subclasses should extend to also copy any additional state they introduce.
   * @c this will have been allocated via @c s->allocNewInstance(g).
   *
   * @note Stage information is @em not cloned; set it explicitly if needed.
   * @param s Source node (same concrete type as @c this).
   */
  virtual void cloneFrom(Node *s) { copyAttributes(*s); }
};

/**
 * @brief Lightweight (domain, version) pair identifying an ONNX operator set.
 *
 * Provides the same information as @c OperatorSetIdProto but without protobuf
 * overhead, resulting in simpler and more readable code.
 */
class OpSetID final {
private:
  std::string domain_;
  int64_t version_;

public:
  /**
   * @brief Constructs an @c OpSetID from a protobuf @c OperatorSetIdProto.
   * @param proto Source protobuf message.
   */
  explicit OpSetID(const OperatorSetIdProto &proto)
      : domain_(std::string(proto.domain().data(), proto.domain().size())),
        version_(proto.version()) {}

  /**
   * @brief Constructs an @c OpSetID for the default ONNX domain at the given version.
   * @param version Operator-set version number.
   */
  explicit OpSetID(const int64_t version) : version_(version) {}

  /**
   * @brief Constructs an @c OpSetID for the specified domain and version.
   * @param domain  Domain string (e.g. @c "" or @c "ai.onnx.ml").
   * @param version Operator-set version number.
   */
  explicit OpSetID(std::string domain, int64_t version)
      : domain_(std::move(domain)), version_(version) {}

  /**
   * @brief Returns a string representation in the form @c "<domain>$<version>".
   */
  std::string toString() const { return domain_ + "$" + ONNX_NAMESPACE::to_string(version_); }

  /**
   * @brief Parses a string in the form @c "<domain>$<version>" into an @c OpSetID.
   * @param target String to parse; must contain exactly one @c '$' separator.
   * @throws std::runtime_error if the format is invalid.
   */
  static OpSetID fromString(const std::string &target) {
    ONNX_TRY {
      auto pos = target.find('$');
      if (pos == std::string::npos) {
        ONNX_THROW("Invalid OpSetID string '", target,
                   "': must be in the form \"<domain>$<version>\"");
      }
      std::string new_domain = target.substr(0, pos);
      const char *version_start = target.data() + pos + 1;
      const char *version_end = target.data() + target.size();
      int64_t new_version = 0;
      auto result = std::from_chars(version_start, version_end, new_version);
      if (result.ec != std::errc{} || result.ptr != version_end) {
        ONNX_THROW("Invalid OpSetID string '", target,
                   "': must be in the form \"<domain>$<version>\"");
      }
      return OpSetID(new_domain, new_version);
    }
    ONNX_CATCH(const std::runtime_error &e) {
      ONNX_HANDLE_EXCEPTION([&]() { ONNX_ASSERTM(false, "Error in fromString: %s", e.what()) });
    }

    // The control will never reach here.
    // In the default build where exceptions are turned on in case of any error
    // the control will enter catch block where an exception will be thrown again.
    // In case of "no exception build" the code aborts at the site of first exception.
    // Adding this to appease the warning "control may reach end of non-void function"
    // as the mac build fails when ONNX_WERROR==ON
    return OpSetID("", 0);
  }

  /** @brief Returns the operator set domain. */
  const std::string &domain() const { return domain_; }

  /** @brief Returns the operator set version number. */
  int64_t version() const { return version_; }

  /**
   * @brief Increments the version by @p step.
   * @param step Amount to add to the current version.
   */
  void incrementVersion(int64_t step) { version_ += step; }

  /**
   * @brief Sets the version to @p newVal.
   * @param newVal New version number.
   */
  void setVersion(int64_t newVal) { version_ = newVal; }
};

/**
 * @brief Computation graph owning all its nodes and values.
 *
 * A @c Graph owns every @ref Node and @ref Value it creates.  All pointers
 * returned by its API are non-owning; destroying the graph invalidates them.
 *
 * The graph maintains two pseudo-nodes:
 * - @em input_ — its outputs represent the graph's formal input @ref Value objects.
 * - @em output_ — its inputs represent the graph's output @ref Value objects; also
 *   serves as the sentinel for the circular topological node list.
 *
 * A separate @em initializer_node_ holds @ref Value objects for IR ≥ 4
 * initializers that are not required to appear in the input list.
 */
struct Graph final {
  ONNX_DISALLOW_COPY_AND_ASSIGN(Graph);
  friend struct Node;
  friend struct Value;

private:
  // only used to keep track of allocated nodes
  // actual representation of Graph is done with
  // inputs, outputs, nodes

  std::unordered_set<const Node *> all_nodes;
  std::unordered_set<const Value *> all_values;
  size_t next_unique_{0};

  size_t new_node_stage_{0};

  // holds outputs in a way that can be reflected
  // as a Use object
  // also used as the beginning/end of the circular node list to avoid
  // having corner cases where the list is empty.
  Node *const output_;
  Node *const input_;
  // Create an independent node list for those initializers do not exist in input
  Node *const initializer_node_;

  std::vector<Tensor> initializers_;
  std::vector<std::string> initializer_names_;

  bool has_name_{false};
  std::string name_;
  bool has_doc_string_{false};
  std::string doc_string_;

  std::vector<OpSetID> opset_versions_;

  bool isNameUnique(const std::string &name) const {
    if (std::find(initializer_names_.cbegin(), initializer_names_.cend(), name) !=
        initializer_names_.cend()) {
      return false;
    }
    const auto f = [&name](const Value *v) { return v->uniqueName() == name; };
    for (const Node *node : all_nodes) {
      for (const auto &attr : node->attributeNames()) {
        if (node->kindOf(attr) == AttributeKind::g) {
          const auto &subgraph = node->g(attr);
          if (!subgraph->isNameUnique(name)) {
            return false;
          }
        } else if (node->kindOf(attr) == AttributeKind::gs) {
          for (const auto &subgraph : node->gs(attr)) {
            if (!subgraph->isNameUnique(name)) {
              return false;
            }
          }
        }
      }
      const auto *const found_in = std::find_if(node->inputs().begin(), node->inputs().end(), f);
      if (found_in != node->inputs().end()) {
        return false;
      }
      const auto *const found_out = std::find_if(node->outputs().begin(), node->outputs().end(), f);
      if (found_out != node->outputs().end()) {
        return false;
      }
    }
    return true;
  }

public:
  /** @brief Constructs an empty graph with input, output, and initializer sentinel nodes. */
  Graph()
      : output_(initOutput(create(kReturn, 0))), input_(create(kParam, 0)),
        initializer_node_(create(kParam, 0)) {}

  /** @brief Returns true if a documentation string is set. */
  bool has_doc_string() const { return has_doc_string_; }
  /** @brief Returns the graph documentation string. */
  const std::string &docString() const { return doc_string_; }
  /**
   * @brief Sets the graph documentation string.
   * @param doc_string Documentation text.
   */
  void setDocString(std::string doc_string) {
    has_doc_string_ = true;
    doc_string_ = std::move(doc_string);
  }

  /**
   * @brief Adds @p initializer to the graph's initializer list.
   *
   * If the initializer has no name, a unique name is generated automatically.
   * @param initializer Tensor to register as an initializer.
   */
  void addInitializer(Tensor &initializer) {
    if (initializer.name().empty()) {
      initializer.setName(getNextUniqueName());
    }
    initializers_.push_back(initializer);
    initializer_names_.push_back(initializer.name());
  }

  /**
   * @brief Adds @p initializer to the initializer list and returns a corresponding Value.
   *
   * For IR ≥ 4, initializers are not required to appear in the graph input list.
   * This method registers the initializer and creates a @ref Value in the internal
   * initializer node list.
   *
   * @param initializer Tensor to register.
   * @return The new @ref Value representing the initializer.
   */
  Value *addInitializerAndCreateValue(Tensor &initializer) {
    addInitializer(initializer);
    auto *init_value = initializer_node_->addOutput();
    std::vector<Dimension> dim_sizes{initializer.sizes().cbegin(), initializer.sizes().cend()};
    init_value->setUniqueName(initializer.name());
    init_value->setSizes(dim_sizes);
    init_value->setElemType(initializer.elem_type());
    return init_value;
  }

  /**
   * @brief Removes the initializer with the given name from all lists.
   * @param name Name of the initializer to remove.
   */
  void eraseInitializer(const std::string &name) {
    initializers_.erase(
        std::remove_if(initializers_.begin(), initializers_.end(),
                       [&name](Tensor &initializer) { return initializer.name() == name; }),
        initializers_.end());
    initializer_names_.erase(
        std::remove(initializer_names_.begin(), initializer_names_.end(), name),
        initializer_names_.end());
    for (size_t i = 0; i < initializer_node_->outputs().size(); i++) {
      if (initializer_node_->outputs()[i]->uniqueName() == name) {
        initializer_node_->eraseOutput(i);
        break;
      }
    }
  }
  /** @brief Removes all initializers and clears the initializer name list. */
  void clearInitializers() {
    initializers_.clear();
    initializer_names_.clear();
  }
  /** @brief Returns the list of initializer tensors. */
  const std::vector<Tensor> &initializers() const { return initializers_; }
  /** @brief Returns the list of initializer names. */
  const std::vector<std::string> &initializer_names() const { return initializer_names_; }
  /**
   * @brief Returns a const iterator to the initializer with the given name, or
   *        @c initializers().end() if not found.
   * @param name Initializer name to search for.
   */
  std::vector<Tensor>::const_iterator getInitializer(const std::string &name) const {
    for (auto it = initializers_.cbegin(); it != initializers_.cend(); ++it) {
      if (name == it->name()) {
        return it;
      }
    }
    return initializers_.end();
  }
  /**
   * @brief Returns true if @p value originates from the internal initializer node
   *        (i.e. it is an IR ≥ 4 initializer not present in the input list).
   * @param value Value to test.
   */
  bool is_constant_initializer(const Value *value) const {
    return value->node() == initializer_node_;
  }
  /** @brief Returns an @c ArrayRef over the graph's formal input values. */
  ArrayRef<Value *> inputs() { return input_->outputs(); }
  /** @brief Returns a const @c ArrayRef over the graph's formal input values. */
  ArrayRef<const Value *> inputs() const {
    const auto &inputs = input_->outputs();
    return {inputs.data(), inputs.size()};
  }
  /** @brief Returns an @c ArrayRef over the graph's output values. */
  ArrayRef<Value *> outputs() { return output_->inputs(); }
  /** @brief Returns a const @c ArrayRef over the graph's output values. */
  ArrayRef<const Value *> outputs() const { return static_cast<const Node *>(output_)->inputs(); }
  /** @brief Returns a forward-iterable view over the computation nodes in topological order. */
  graph_node_list nodes() { return graph_node_list(output_, kNextDirection); }
  /** @brief Returns a const forward-iterable view over the computation nodes. */
  const_graph_node_list nodes() const { return const_graph_node_list(output_, kNextDirection); }

  /** @brief Returns a mutable reference to the operator-set version list. */
  std::vector<OpSetID> &opset_versions_mutable() { return opset_versions_; }

  /**
   * @brief Allocates and returns the next graph-unique integer identifier.
   *
   * Skips any integers whose generated name (@c _v_N) is already in use.
   */
  size_t getNextUnique() {
    std::string next_unique_name = toVarName(++next_unique_);
    while (!isNameUnique(next_unique_name)) {
      next_unique_name = toVarName(++next_unique_);
    }
    return next_unique_;
  }

  /**
   * @brief Returns a unique name string of the form @c "_v_N".
   *
   * Equivalent to @c toVarName(getNextUnique()).
   */
  std::string getNextUniqueName() { return toVarName(getNextUnique()); }

  /**
   * @brief Returns a forward iterator to the first computation node.
   *
   * Safe to call on the temporary @c graph_node_list returned by @ref nodes()
   * because the list is non-owning.
   */
  graph_node_list_iterator begin() { return nodes().begin(); }
  /** @brief Returns a const forward iterator to the first computation node. */
  const_graph_node_list_iterator begin() const { return nodes().begin(); }
  /** @brief Returns a forward sentinel iterator (one past the last node). */
  graph_node_list_iterator end() { return nodes().end(); }
  /** @brief Returns a const forward sentinel iterator. */
  const_graph_node_list_iterator end() const { return nodes().end(); }
  /** @brief Returns a reverse iterator to the last computation node. */
  graph_node_list_iterator rbegin() { return nodes().rbegin(); }
  /** @brief Returns a const reverse iterator to the last computation node. */
  const_graph_node_list_iterator rbegin() const { return nodes().rbegin(); }
  /** @brief Returns a reverse sentinel iterator. */
  graph_node_list_iterator rend() { return nodes().rend(); }
  /** @brief Returns a const reverse sentinel iterator. */
  const_graph_node_list_iterator rend() const { return nodes().rend(); }
  /**
   * @brief Returns the sentinel return node (also the circular list head/tail).
   */
  Node *return_node() { return output_; }
  /** @brief Returns the sentinel return node (const overload). */
  const Node *return_node() const { return output_; }

  /**
   * @brief Appends a new formal input @ref Value and returns it.
   */
  Value *addInput() { return input_->addOutput(); }
  /**
   * @brief Removes the formal input at index @p i.
   * @param i Zero-based input index.
   */
  void eraseInput(size_t i) { input_->eraseOutput(i); }
  /** @brief Increments the current node-stage counter. */
  void advanceStage() { new_node_stage_++; }
  /**
   * @brief Sets the current node-stage counter.
   * @param new_stage New stage value.
   */
  void setStage(size_t new_stage) { new_node_stage_ = new_stage; }
  /** @brief Returns the current node-stage counter. */
  size_t stage() const { return new_node_stage_; }
  /**
   * @brief Temporarily sets the stage to @p s and returns a @ref ResourceGuard
   *        that restores the previous value on destruction.
   * @param s Temporary stage value.
   */
  ResourceGuard setStageTemporary(size_t s) {
    auto prev_stage = new_node_stage_;
    new_node_stage_ = s;
    return ResourceGuard([prev_stage, this]() { this->new_node_stage_ = prev_stage; });
  }

  /**
   * @brief Registers @p n as a graph output and returns its output index.
   * @param n Value to register as a graph output.
   * @return Zero-based index of the newly registered output.
   */
  size_t registerOutput(Value *n) {
    output_->addInput(n);
    return outputs().size() - 1;
  }

  /**
   * @brief Creates and returns a new unattached node with @p num_outputs outputs.
   *
   * The node is added to the graph's allocation set but not yet in the
   * topological node list.  Use @ref appendNode or @ref Node::insertBefore /
   * @ref Node::insertAfter to place it.
   *
   * @param kind        Symbolic operation kind.
   * @param num_outputs Number of output values to allocate (default: 1).
   * @return The new node.
   */
  Node *create(NodeKind kind, size_t num_outputs = 1) {
    // NB: Node constructor adds node to all_nodes
    auto *n = new Node(this, kind);
    for (size_t i = 0; i < num_outputs; i++)
      n->addOutput();
    return n;
  }

  /**
   * @brief Creates a new unattached node with the given inputs and outputs.
   * @param kind        Symbolic operation kind.
   * @param inputs      Input values to wire up immediately.
   * @param num_outputs Number of output values to allocate (default: 1).
   * @return The new node.
   */
  Node *create(NodeKind kind, ArrayRef<Value *> inputs, size_t num_outputs = 1) {
    auto *n = create(kind, num_outputs);
    for (auto *i : inputs)
      n->addInput(i);
    return n;
  }

  /**
   * @brief Appends @p n at the end of the topological node list and returns it.
   * @param n Unattached node belonging to this graph.
   */
  Node *appendNode(Node *n) {
    ONNX_ASSERT(n->graph_ == this && !n->inGraphList())
    n->insertBefore(output_);
    return n;
  }

  /**
   * @brief Prepends @p n at the beginning of the topological node list and returns it.
   * @param n Unattached node belonging to this graph.
   */
  Node *prependNode(Node *n) {
    ONNX_ASSERT(n->graph_ == this && !n->inGraphList())
    n->insertAfter(output_);
    return n;
  }

  /**
   * @brief Adds @p initializer as both a graph input and an initializer entry.
   *
   * Registers the initializer in the initializer list, creates a formal input
   * @ref Value, and synchronises the initializer name, tensor name, and value name.
   *
   * @param initializer Source tensor (copied internally).
   * @param name        Name to assign to the initializer and its value.
   * @return The new input @ref Value.
   */
  Value *addInitializerAndInput(const Tensor &initializer, const std::string &name) {
    Tensor initializerCopy = initializer;
    std::vector<Dimension> dim_sizes{initializerCopy.sizes().cbegin(),
                                     initializerCopy.sizes().cend()};
    Value *new_init = addInput();
    initializerCopy.setName(name);
    new_init->setUniqueName(name);
    new_init->setSizes(dim_sizes);
    new_init->setElemType(initializerCopy.elem_type());
    addInitializer(initializerCopy);
    return new_init;
  }

  /**
   * @brief Adds @p initializer as both a graph input and an initializer entry,
   *        generating a unique name automatically.
   * @param initializer Source tensor (copied internally).
   * @return The new input @ref Value.
   */
  Value *addInitializerAndInput(const Tensor &initializer) {
    return addInitializerAndInput(initializer, getNextUniqueName());
  }

  /**
   * @brief Removes @p v from the initializer list and, if it is a graph input,
   *        from the input list as well.
   *
   * @pre @p v must have no remaining uses.
   * @param v The initializer value to erase.
   */
  void eraseInitializerAndInput(Value *v) {
    Node *node = v->node();
    const size_t offset = v->offset();
    eraseInitializer(v->uniqueName());
    if (node == input_) {
      eraseInput(offset);
    }
  }

  /** @brief Destroys all owned nodes and values. */
  ~Graph() {
    for (const Node *n : all_nodes)
      delete n;
    for (const Value *v : all_values)
      delete v;
  }

  /** @brief Returns a textual representation of the graph for debugging. */
  std::string toString() const {
    std::ostringstream oss;
    oss << *this;
    return oss.str();
  }

  /** @brief Returns true if a graph name is set. */
  bool has_name() const { return has_name_; }

  /** @brief Returns the graph name (meaningful only when @ref has_name() is true). */
  const std::string &name() const { return name_; }

  /**
   * @brief Sets the graph name.
   * @param name Name string.
   */
  void setName(std::string name) {
    has_name_ = true;
    name_ = std::move(name);
  }

  /** @brief Streams a textual representation of the graph to @p out. */
  friend std::ostream &operator<<(std::ostream &out, const Graph &g);

  /**
   * @brief Calls @p fn on this graph and recursively on every subgraph in node attributes.
   * @param fn Callable accepting a @c Graph* pointer.
   */
  void forSelfAndEachSubGraph(const std::function<void(Graph *)> &fn) {
    fn(this);

    for (const Node *node : all_nodes) {
      for (const auto &attr : node->attributeNames()) {
        if (node->kindOf(attr) == AttributeKind::g) {
          std::shared_ptr<Graph> subgraph = node->g(attr);
          subgraph->forSelfAndEachSubGraph(fn);
        } else if (node->kindOf(attr) == AttributeKind::gs) {
          for (const auto &subgraph : node->gs(attr)) {
            subgraph->forSelfAndEachSubGraph(fn);
          }
        }
      }
    }
  }

  /** @brief Const overload of @ref forSelfAndEachSubGraph. */
  void forSelfAndEachSubGraph(const std::function<void(const Graph *)> &fn) const {
    std::function<void(Graph *)> tmp_fn = [fn](Graph *graph) { fn(graph); };
    const_cast<Graph *>(this)->forSelfAndEachSubGraph(tmp_fn);
  }

  /**
   * @brief Calls @p fn on every node in this graph and all subgraphs.
   * @param fn Callable accepting a @c Node* pointer.
   */
  void forEachNode(const std::function<void(Node *)> &fn) {
    forSelfAndEachSubGraph([&fn](Graph *graph) {
      for (Node *node : graph->nodes()) {
        fn(node);
      }
    });
  }

  /** @brief Const overload of @ref forEachNode. */
  void forEachNode(const std::function<void(const Node *)> &fn) const {
    std::function<void(Node *)> tmp_fn = [fn](Node *node) { fn(node); };
    const_cast<Graph *>(this)->forEachNode(tmp_fn);
  }

private:
  // should only be called in the constructor
  Node *initOutput(Node *p) {
    p->next() = p;
    p->prev() = p;
    p->setStage(std::numeric_limits<size_t>::max());
    return p;
  }

  void freeNode(Node *n) {
    auto it = all_nodes.find(n);
    ONNX_ASSERT(it != all_nodes.end())
    delete *it;
    all_nodes.erase(it);
  }
  void freeValue(Value *v) {
    auto it = all_values.find(v);
    ONNX_ASSERT(it != all_values.end())
    delete *it;
    all_values.erase(it);
  }
};

inline Value::Value(Node *node, size_t offset)
    : node_(node), offset_(offset), unique_(node->graph_->getNextUnique()),
      stage_(node->graph_->new_node_stage_) {
  node->graph_->all_values.emplace(this);
}

inline Graph *Value::owningGraph() { return node()->owningGraph(); }

inline const Graph *Value::owningGraph() const { return node()->owningGraph(); }

// `captured` nodes in subgraph determines which value it captures
// by storing the value's unique name, so old unique names in `captured` nodes
// should also be updated.
// Initializer names are also stored in graph.initializer_names_, it should be
// updated too.
inline Value *Value::setUniqueName(const std::string &name, bool update_related_names) {
  if (has_unique_name() && update_related_names) {
    auto *graph = owningGraph();
    auto old_name = unique_name_;
    for (size_t i = 0; i < owningGraph()->initializer_names_.size(); i++) {
      auto &initializer_name = owningGraph()->initializer_names_[i];
      if (initializer_name == old_name) {
        initializer_name = name;
        owningGraph()->initializers_[i].setName(name);
      }
    }
    graph->forEachNode([this, &name, &old_name](Node *node) {
      if (node->owningGraph() == this->owningGraph()) {
        // skip non-subgraph
        return;
      }
      if (node->kind() == kCaptured) {
        Value *output = node->output();
        if (output->uniqueName() == old_name) {
          output->setUniqueName(name, false);
        }
      }
    });
  }
  unique_name_ = name;
  has_unique_name_ = true;
  return this;
}

inline void Value::replaceAllUsesWith(Value *newValue) {
  auto *graph = owningGraph();
  ONNX_ASSERT(graph == newValue->owningGraph())
  // propagate sizes and elem type
  if (this->has_sizes()) {
    newValue->setSizes(this->sizes());
  }
  if (this->elemType() != TensorProto::DataType::UNDEFINED) {
    newValue->setElemType(this->elemType());
  }
  const auto unique_name = this->uniqueName();
  // We do not want the optimization to change the graph output name
  if (std::find(graph->outputs().rbegin(), graph->outputs().rend(), this) !=
      graph->outputs().rend()) {
    newValue->setUniqueName(unique_name);
    // The "unique" semantic of unique_name should be kept or uses()
    // will return an incorrect result when the value is used in subgraph
    this->setUniqueName(graph->getNextUniqueName(), false);
  }
  newValue->uses_in_current_graph_.reserve(this->uses_in_current_graph_.size());
  for (auto u : uses_in_current_graph_) {
    u.user->inputs_[u.offset] = newValue;
    newValue->uses_in_current_graph_.push_back(u);
  }
  graph->forEachNode([this, &newValue, &unique_name](Node *node) {
    if (node->owningGraph() == this->owningGraph()) {
      // skip non-subgraph
      return;
    }
    if (node->kind() == kCaptured) {
      Value *output = node->output();
      if (output->uniqueName() == unique_name) {
        output->setUniqueName(newValue->uniqueName());
      }
    }
  });
  uses_in_current_graph_.clear();
  assert(this->uses().empty());
}

inline Node::Node(Graph *graph, NodeKind kind)
    : kind_(kind), graph_(graph), stage_(graph->new_node_stage_) {
  graph_->all_nodes.emplace(this);
}

inline void Node::eraseOutput(size_t i) {
  ONNX_ASSERT(i < outputs_.size())
  ONNX_ASSERT(outputs_[i]->uses().empty())
  Value *n = outputs_[i];
  outputs_.erase(outputs_.begin() + i);
  owningGraph()->freeValue(n);
  for (size_t j = i; j < outputs_.size(); j++) {
    outputs_[j]->offset_--;
  }
}

inline bool Node::isBefore(const Node *n) {
  if (n == nullptr || this == n) {
    // Bail out early.
    return false;
  }
  // return true if node is Param (in initializers)
  if (kind_ == kParam) {
    return true;
  }
  // return false if target node is Param (in initializers)
  if (n->kind() == kParam) {
    return false;
  }
  ONNX_ASSERT(n->inGraphList())
  for (Node *p = next(); p != *graph_->end(); p = p->next()) {
    if (p == n) {
      return true;
    }
  }
  return false;
}

inline void Node::destroy() {
  ONNX_ASSERT(inGraphList())
  while (!outputs().empty())
    eraseOutput(outputs().size() - 1);
  removeAllInputs();
  removeFromList();
  graph_->freeNode(this);
}

/************* All nodes not required to be defined before Graph **************/

inline graph_node_list_iterator Node::iterator() { return graph_node_list_iterator(this, 0); }
inline graph_node_list_iterator Node::reverseIterator() { return iterator().reverse(); }
inline const_graph_node_list_iterator Node::iterator() const {
  return const_graph_node_list_iterator(this, 0);
}
inline const_graph_node_list_iterator Node::reverseIterator() const { return iterator().reverse(); }

// Returns a list about which nodes are using this value,
// nodes in subgraph are also included.
// This method is usually used to check whether it is
// safe to delete a Value.
inline use_list Value::uses() const {
  use_list all_uses = uses_in_current_graph_;
  owningGraph()->forEachNode([this, &all_uses](const Node *node) {
    if (node->owningGraph() == this->owningGraph()) {
      // skip non-subgraph
      return;
    }
    if (node->kind() == kCaptured) {
      const Value *output = node->outputs()[0];
      if (output->uniqueName() == this->uniqueName()) {
        const auto output_uses = output->uses();
        all_uses.insert(all_uses.end(), output_uses.begin(), output_uses.end());
      }
    }
  });
  return all_uses;
}

} // namespace ONNX_NAMESPACE
