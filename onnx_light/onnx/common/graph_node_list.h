// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file graph_node_list.h
 * @brief Intrusive doubly-linked list used exclusively for Graph node lists.
 *
 * Provides `generic_graph_node_list` and `generic_graph_node_list_iterator`,
 * together with concrete type aliases for mutable and const graph node lists.
 *
 * @attention The code in this file is highly EXPERIMENTAL.
 *            The APIs will probably change.
 *
 * ### Design notes
 * The list is *intrusive*: the next/previous pointers live inside the node
 * itself (via `T::next_in_graph(size_t)`), so no external allocation is
 * needed and iterating is cache-friendly.
 *
 * Forward and reverse iteration are handled uniformly by a single iterator
 * class parameterised on a direction index (`kNextDirection` /
 * `kPrevDirection`).  A "before-first-element" sentinel node is always
 * present, which means reverse iterators physically point to the element they
 * logically denote — there is no off-by-one behaviour as with standard
 * `std::reverse_iterator`.
 *
 * ### Requirements on the element type T
 * - `T* next_in_graph(size_t direction)` — returns the neighbouring node in
 *   the given direction; used by the iterator to advance.
 * - `void destroy()` — removes `T` from the list and deallocates it.
 *
 * In practice `T` is always `Node` or `const Node`.  `destroy()` semantics
 * must be renegotiated before this template is used with any other type.
 */

#ifndef ONNX_COMMON_GRAPH_NODE_LIST_H_
#define ONNX_COMMON_GRAPH_NODE_LIST_H_

#include "onnx/common/assertions.h"

namespace ONNX_LIGHT_NAMESPACE {

/// Direction index for forward (next-pointer) traversal.
static constexpr size_t kNextDirection = 0;
/// Direction index for reverse (previous-pointer) traversal.
static constexpr size_t kPrevDirection = 1;

template <typename T> struct generic_graph_node_list;

template <typename T> struct generic_graph_node_list_iterator;

struct Node;
/// Mutable graph node list over `Node` elements.
using graph_node_list = generic_graph_node_list<Node>;
/// Read-only graph node list over `const Node` elements.
using const_graph_node_list = generic_graph_node_list<const Node>;
/// Mutable iterator over a `graph_node_list`.
using graph_node_list_iterator = generic_graph_node_list_iterator<Node>;
/// Read-only iterator over a `const_graph_node_list`.
using const_graph_node_list_iterator = generic_graph_node_list_iterator<const Node>;

/**
 * @brief Bidirectional iterator for `generic_graph_node_list`.
 *
 * The iterator stores a raw pointer to the current node and a direction index
 * so that the same class serves both forward and reverse traversal.
 * Incrementing follows the `kNextDirection` links; decrementing follows the
 * opposite direction.
 *
 * @tparam T Element type, either `Node` or `const Node`.
 */
template <typename T> struct generic_graph_node_list_iterator final {
  /// Constructs an iterator that compares equal to a default-constructed `end()`.
  generic_graph_node_list_iterator() : cur(nullptr), d(kNextDirection) {}

  /// Constructs an iterator pointing at @p cur traversing in direction @p d.
  /// @param cur Pointer to the current node (may be the sentinel head).
  /// @param d Traversal direction: `kNextDirection` (0) or `kPrevDirection` (1).
  generic_graph_node_list_iterator(T *cur, size_t d) : cur(cur), d(d) {}

  /// Dereferences the iterator.
  /// @returns Pointer to the current node.
  T *operator*() const { return cur; }

  /// Arrow dereference.
  /// @returns Pointer to the current node.
  T *operator->() const { return cur; }

  /// Pre-increment: advances to the next node in the traversal direction.
  /// @returns Reference to this iterator after the advance.
  generic_graph_node_list_iterator &operator++() {
    ONNX_ASSERT(cur)
    cur = cur->next_in_graph(d);
    return *this;
  }

  /// Post-increment: advances to the next node and returns the previous state.
  /// @returns Copy of this iterator before the advance.
  generic_graph_node_list_iterator operator++(int) {
    generic_graph_node_list_iterator old = *this;
    ++(*this);
    return old;
  }

  /// Pre-decrement: moves to the preceding node in the traversal direction.
  /// @returns Reference to this iterator after the move.
  generic_graph_node_list_iterator &operator--() {
    ONNX_ASSERT(cur)
    cur = cur->next_in_graph(reverseDir());
    return *this;
  }

  /// Post-decrement: moves to the preceding node and returns the previous state.
  /// @returns Copy of this iterator before the move.
  generic_graph_node_list_iterator operator--(int) {
    generic_graph_node_list_iterator old = *this;
    --(*this);
    return old;
  }

  /**
   * @brief Destroys the current node without invalidating this iterator.
   *
   * After the call, the iterator points to the node that preceded the
   * destroyed node in the traversal direction.  Named `destroyCurrent`
   * (rather than `destroy`) so that accidental `->` / `.` mixups do not
   * silently call the wrong function.
   */
  void destroyCurrent() {
    T *n = cur;
    cur = cur->next_in_graph(reverseDir());
    n->destroy();
  }

  /**
   * @brief Returns an iterator that traverses in the opposite direction,
   *        still pointing at the same node.
   *
   * @returns A new iterator with the direction index flipped.
   */
  generic_graph_node_list_iterator reverse() {
    return generic_graph_node_list_iterator(cur, reverseDir());
  }

private:
  /// Returns the direction index opposite to the current traversal direction.
  size_t reverseDir() { return d == kNextDirection ? kPrevDirection : kNextDirection; }

  T *cur;   ///< Pointer to the node currently pointed at.
  size_t d; ///< Traversal direction: 0 = forward (next), 1 = reverse (prev).
};

/**
 * @brief Intrusive doubly-linked list view over graph nodes.
 *
 * Does not own the nodes; it merely provides a range interface around the
 * sentinel `head` pointer and a traversal direction.  Supports both forward
 * (`begin`/`end`) and reverse (`rbegin`/`rend`) iteration.  A reversed view
 * of the same list is obtained via `reverse()`.
 *
 * @tparam T Element type, either `Node` or `const Node`.
 */
template <typename T> struct generic_graph_node_list final {
  /// Mutable iterator type.
  using iterator = generic_graph_node_list_iterator<T>;
  /// Read-only iterator type.
  using const_iterator = generic_graph_node_list_iterator<const T>;

  /// Returns a mutable iterator to the first element.
  generic_graph_node_list_iterator<T> begin() {
    return generic_graph_node_list_iterator<T>(head->next_in_graph(d), d);
  }
  /// Returns a read-only iterator to the first element.
  generic_graph_node_list_iterator<const T> begin() const {
    return generic_graph_node_list_iterator<const T>(head->next_in_graph(d), d);
  }

  /// Returns a mutable past-the-end iterator (points at the sentinel).
  generic_graph_node_list_iterator<T> end() { return generic_graph_node_list_iterator<T>(head, d); }
  /// Returns a read-only past-the-end iterator (points at the sentinel).
  generic_graph_node_list_iterator<const T> end() const {
    return generic_graph_node_list_iterator<const T>(head, d);
  }

  /// Returns a mutable iterator to the first element of the reverse sequence.
  generic_graph_node_list_iterator<T> rbegin() { return reverse().begin(); }
  /// Returns a read-only iterator to the first element of the reverse sequence.
  generic_graph_node_list_iterator<const T> rbegin() const { return reverse().begin(); }

  /// Returns a mutable past-the-end iterator for the reverse sequence.
  generic_graph_node_list_iterator<T> rend() { return reverse().end(); }
  /// Returns a read-only past-the-end iterator for the reverse sequence.
  generic_graph_node_list_iterator<const T> rend() const { return reverse().end(); }

  /**
   * @brief Returns a view of the same list traversed in the opposite direction.
   *
   * The returned object shares the same sentinel head but uses the opposite
   * direction index, so iterating it yields nodes in reverse order.
   *
   * @returns A `generic_graph_node_list` with the direction index flipped.
   */
  generic_graph_node_list reverse() {
    return generic_graph_node_list(head, d == kNextDirection ? kPrevDirection : kNextDirection);
  }
  /// @copydoc reverse()
  generic_graph_node_list reverse() const {
    return generic_graph_node_list(head, d == kNextDirection ? kPrevDirection : kNextDirection);
  }

  /**
   * @brief Constructs a list view from a sentinel head pointer and a direction.
   *
   * @param head Sentinel node that delimits the list (the "before-first" element).
   * @param d Initial traversal direction (`kNextDirection` or `kPrevDirection`).
   */
  generic_graph_node_list(T *head, size_t d) : head(head), d(d) {}

private:
  T *head;  ///< Sentinel node; its `next_in_graph` links delimit the list.
  size_t d; ///< Traversal direction: 0 = forward, 1 = reverse.
};

/**
 * @brief Equality comparison for two iterators.
 *
 * Two iterators compare equal when they point to the same node.
 *
 * @tparam T Element type.
 * @param a Left-hand iterator.
 * @param b Right-hand iterator.
 * @returns `true` if both iterators point to the same node.
 */
template <typename T>
static inline bool operator==(generic_graph_node_list_iterator<T> a,
                              generic_graph_node_list_iterator<T> b) {
  return *a == *b;
}

/**
 * @brief Inequality comparison for two iterators.
 *
 * @tparam T Element type.
 * @param a Left-hand iterator.
 * @param b Right-hand iterator.
 * @returns `true` if the iterators point to different nodes.
 */
template <typename T>
static inline bool operator!=(generic_graph_node_list_iterator<T> a,
                              generic_graph_node_list_iterator<T> b) {
  return *a != *b;
}

} // namespace ONNX_LIGHT_NAMESPACE

namespace std {

/**
 * @brief `std::iterator_traits` specialisation for `generic_graph_node_list_iterator`.
 *
 * Enables standard algorithms and range adaptors to introspect the iterator's
 * value and difference types.  The iterator category is `bidirectional_iterator_tag`
 * because `operator--` is supported but random-access arithmetic is not.
 *
 * @tparam T Element type of the underlying node list.
 */
template <typename T> struct iterator_traits<ONNX_LIGHT_NAMESPACE::generic_graph_node_list_iterator<T>> {
  using difference_type = int64_t;                      ///< Signed distance between iterators.
  using value_type = T *;                               ///< Type produced by dereferencing.
  using pointer = T **;                                 ///< Pointer-to-value type.
  using reference = T *&;                               ///< Reference-to-value type.
  using iterator_category = bidirectional_iterator_tag; ///< Supports `++` and `--`.
};

} // namespace std

#endif // ONNX_COMMON_GRAPH_NODE_LIST_H_
