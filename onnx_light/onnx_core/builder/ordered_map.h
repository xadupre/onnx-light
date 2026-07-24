// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ordered_map.h
 * @brief Minimal insertion-ordered associative container keyed by strings.
 *
 * :cpp:class:`core::builder::GraphBuilder` stores its initializers and its
 * local functions / subgraphs in an :cpp:class:`OrderedMap` so that the order
 * in which they are declared is preserved when the builder is finalised into a
 * proto (mirroring how a hand-written ONNX graph lists them). ``std::map`` is
 * not used because it reorders entries alphabetically; ``std::unordered_map``
 * loses the declaration order entirely. Lookups stay ``O(1)`` thanks to an
 * auxiliary name → index table.
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx_light_helpers.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace builder {

/**
 * Insertion-ordered map from ``std::string`` keys to ``Value``.
 *
 * Iterating over the container yields ``std::pair<const std::string, Value>``
 * references in insertion order. Existing keys keep their original position
 * when their value is replaced.
 */
template <typename Value> class OrderedMap {
public:
  using value_type = std::pair<std::string, Value>;
  using iterator = typename std::vector<value_type>::iterator;
  using const_iterator = typename std::vector<value_type>::const_iterator;

  /// Returns ``true`` when ``key`` is present.
  bool Contains(const std::string &key) const { return index_.find(key) != index_.end(); }

  /// Number of stored entries.
  std::size_t Size() const noexcept { return items_.size(); }

  /// ``true`` when the container holds no entry.
  bool Empty() const noexcept { return items_.empty(); }

  /// Inserts ``value`` under ``key`` (appending it) or replaces the value of an
  /// existing ``key`` in place. Returns a reference to the stored value.
  Value &Set(const std::string &key, Value value) {
    auto it = index_.find(key);
    if (it != index_.end()) {
      items_[it->second].second = std::move(value);
      return items_[it->second].second;
    }
    index_.emplace(key, items_.size());
    items_.emplace_back(key, std::move(value));
    return items_.back().second;
  }

  /// Returns the value stored under ``key``. Throws ``std::out_of_range`` when
  /// the key is absent.
  Value &At(const std::string &key) { return items_[IndexOf(key)].second; }
  const Value &At(const std::string &key) const { return items_[IndexOf(key)].second; }

  /// Ordered access to the stored ``key -> value`` pairs.
  const std::vector<value_type> &Items() const noexcept { return items_; }

  iterator begin() noexcept { return items_.begin(); }
  iterator end() noexcept { return items_.end(); }
  const_iterator begin() const noexcept { return items_.begin(); }
  const_iterator end() const noexcept { return items_.end(); }

private:
  std::size_t IndexOf(const std::string &key) const {
    auto it = index_.find(key);
    if (it == index_.end()) {
      throw std::out_of_range("OrderedMap: unknown key '" + key + "'.");
    }
    return it->second;
  }

  std::vector<value_type> items_;
  std::unordered_map<std::string, std::size_t> index_;
};

} // namespace builder
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
