// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_verify.h"
#include <functional>
#include <iterator>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

class QuantizationParameterCatalogue;
struct RuntimeValue;

/**
 * Shares immutable sequence elements through a balanced tree.
 *
 * Copies share metadata; append and replacement copy only a logarithmic path.
 * Elements are read-only: replaces a changed element with Set(), never through
 * a mutable reference. Retained payloads follow the usual read-only contract.
 */
class RuntimeSequence {
  struct Node {
    std::shared_ptr<const Node> left, right;
    std::shared_ptr<const RuntimeValue> value;
    size_t size = 1, height = 1, depth = 0;
    bool retained = false, catalogue_independent = false;

    Node(RuntimeValue value, bool retained);
    Node(std::shared_ptr<const Node> left, std::shared_ptr<const Node> right);
  };
  using Root = std::shared_ptr<const Node>;
  Root root_;
  static Root Append(Root root, Root leaf);
  static Root Replace(const Root &root, size_t index, Root leaf);
  static Root RetainNode(const Root &root, const StructTypeCatalogue &catalogue, size_t depth);
  static Root TransformNode(const Root &root,
                            const std::function<RuntimeValue(const RuntimeValue &)> &transform);
  static Root TransformOwnedNode(const Root &root, bool shared,
                                 const std::function<void(RuntimeValue &)> &transform);
  static const RuntimeValue &At(const Root &root, size_t index);
  friend struct RuntimeValue;
  void Retain(const StructTypeCatalogue &catalogue, size_t depth);

public:
  RuntimeSequence() = default;
  explicit RuntimeSequence(std::vector<RuntimeValue> values);
  size_t size() const noexcept { return root_ ? root_->size : 0; }
  bool empty() const noexcept { return !root_; }
  void clear() noexcept { root_.reset(); }
  size_t depth() const noexcept { return root_ ? root_->depth : 0; }
  bool retained() const noexcept { return !root_ || root_->retained; }
  const RuntimeValue &at(size_t index) const;
  const RuntimeValue &operator[](size_t index) const { return at(index); }
  const RuntimeValue &front() const { return at(0); }
  const RuntimeValue &back() const { return at(size() - 1); }
  void push_back(RuntimeValue value);
  /** Replaces an element without changing any existing snapshot. */
  void Set(size_t index, RuntimeValue value);
  /** Returns a sequence with independently transformed elements. */
  RuntimeSequence
  Transform(const std::function<RuntimeValue(const RuntimeValue &)> &transform) const;
  /** Updates unpublished elements, consuming unique storage and copying shared snapshots. */
  void TransformInPlace(const std::function<void(RuntimeValue &)> &transform);

  class const_iterator {
    Root root_;
    size_t index_ = 0;

  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = RuntimeValue;
    using difference_type = std::ptrdiff_t;
    using reference = const RuntimeValue &;
    using pointer = const RuntimeValue *;
    const_iterator() = default;
    const_iterator(Root root, size_t index) : root_(std::move(root)), index_(index) {}
    reference operator*() const { return At(root_, index_); }
    pointer operator->() const { return &At(root_, index_); }
    const_iterator &operator++() {
      ++index_;
      return *this;
    }
    const_iterator operator++(int) {
      auto previous = *this;
      ++*this;
      return previous;
    }
    bool operator==(const const_iterator &other) const = default;
  };
  const_iterator begin() const { return {root_, 0}; }
  const_iterator end() const { return {root_, size()}; }

  /** Memoizes one immutable analysis policy without retaining sequence storage. */
  template <typename Result> class Memo {
    struct Entry {
      std::weak_ptr<const Node> node;
      Result result;
    };
    std::mutex mutex_;
    std::unordered_map<const Node *, Entry> entries_;
    friend class RuntimeSequence;

    template <typename Leaf, typename Join>
    Result Fold(const Root &node, const Leaf &leaf, const Join &join) {
      if (!node)
        return {};
      auto found = entries_.find(node.get());
      if (found != entries_.end() && found->second.node.lock() == node)
        return found->second.result;
      Result result = node->value
                          ? leaf(*node->value)
                          : join(Fold(node->left, leaf, join), Fold(node->right, leaf, join));
      entries_.insert_or_assign(node.get(), Entry{node, result});
      return result;
    }
  };

  /** Reduces only changed tree paths; each memo belongs to one analysis policy. */
  template <typename Result, typename Leaf, typename Join>
  Result Fold(Memo<Result> &memo, const Leaf &leaf, const Join &join) const {
    std::lock_guard lock(memo.mutex_);
    const size_t limit = std::max<size_t>(128, size() * 4);
    if (memo.entries_.size() > limit) {
      std::erase_if(memo.entries_, [](const auto &entry) { return entry.second.node.expired(); });
      // Bound memo storage even when callers retain every historical snapshot.
      if (memo.entries_.size() > limit)
        memo.entries_.clear();
    }
    return memo.Fold(root_, leaf, join);
  }
};
/**
 * Represents tensors, typed sequences, named structures and immutable encoded proto values.
 *
 * RuntimeContext stores ordinary tensor edges in its existing tensor map, not
 * here. This recursive representation supplies the structured/encoded edges
 * required by TypeProto, independently of whether an edge is persistent.
 */
struct RuntimeValue {
  static constexpr size_t kMaxDepth = 64;
  enum class Kind { kTensor, kStruct, kEncoded, kSequence };
  Kind kind = Kind::kStruct;
  Tensor tensor;
  std::unordered_map<std::string, RuntimeValue> fields;
  RuntimeSequence elements;
  std::shared_ptr<const EncodedValueProto> encoded;
  std::shared_ptr<const QuantizationParameterCatalogue> quantization_parameters;

  RuntimeValue() = default;
  explicit RuntimeValue(Tensor value) : kind(Kind::kTensor), tensor(std::move(value)) {}
  explicit RuntimeValue(std::unordered_map<std::string, RuntimeValue> value)
      : kind(Kind::kStruct), fields(std::move(value)) {}
  explicit RuntimeValue(std::vector<RuntimeValue> value)
      : kind(Kind::kSequence), elements(std::move(value)) {}
  explicit RuntimeValue(EncodedValueProto value);

  /** Returns an immutable message view with an explicit lifetime owner. */
  static RuntimeValue FromEncodedView(const EncodedValueProto &value, std::shared_ptr<void> owner);

  /** Restores a validated cache, retaining proto storage and any shared parameter catalogue. */
  static RuntimeValue
  FromPagedCache(PagedCacheProto value, const StructTypeCatalogue &catalogue = {},
                 std::shared_ptr<const QuantizationParameterCatalogue> parameters = {});

  /** Exports a cache without decoding pages; referenced types remain model-scoped. */
  PagedCacheProto ToPagedCache(const std::string &name = "",
                               const StructTypeCatalogue &catalogue = {}) const;

  /** Returns the immutable encoded message. */
  const EncodedValueProto &Encoded() const;

  /** Borrows numeric storage and materializes ordinary string fields without changing this value.
   */
  RuntimeValue BorrowView() const;

  /** Returns independent owned payloads. */
  RuntimeValue DeepCopy() const;

  /** Consumes a selected whole value and explicitly retains its backing storage. */
  RuntimeValue Retain(const StructTypeCatalogue &catalogue = {}) &&;

private:
  friend class RuntimeSequence;
  size_t Depth(size_t depth = 0) const;
  bool CatalogueIndependent(size_t depth = 0) const;
  void RetainAtDepth(size_t depth, const StructTypeCatalogue &catalogue);
  RuntimeValue CopyAtDepth(size_t depth, bool owned) const;
};

using RuntimeValueMap = std::unordered_map<std::string, RuntimeValue>;

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
