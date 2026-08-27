// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Merges pairs of identical sibling nodes, and any identical descendant chains.
 *
 * @code
 * Before:
 *                    +----+                       +------+
 *            +-----> | Op | ---> a ------------> | Next | ---> p
 *            |       +----+                       +------+
 *            |
 *   x, c ----+
 *            |
 *            |       +----+                       +------+
 *            +-----> | Op | ---> b ------------> | Next | ---> q
 *                    +----+                       +------+
 *
 * After:
 *            +----+              +------+
 *   x, c ---> | Op | ---> a ---> | Next | ---> p
 *            +----+       |      +------+      |
 *                         |                    |
 *                         v                    v
 *                    +----------+         +----------+
 *                    | Identity | ---> b  | Identity | ---> q
 *                    +----------+         +----------+
 * @endcode
 *
 * One copy of every equivalent node is retained. Identity nodes preserve all
 * output names from the removed copy, including outputs used by later nodes.
 */
class SameChildrenPattern : public core::builder::PatternOptimization {
public:
  explicit SameChildrenPattern(int priority = 0) : PatternOptimization(priority, "SameChildren") {}

  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;

protected:
  SameChildrenPattern(int priority, std::string name)
      : PatternOptimization(priority, std::move(name)) {}
  static bool SameNode(const NodeProto &first, const NodeProto &second);
  static bool SameNodeWithAliases(
      const NodeProto &first, const NodeProto &second,
      const std::unordered_map<std::string, std::unordered_set<std::string>> &aliases);
  core::builder::MatchResult MatchWithNodes(core::builder::GraphGraph &graph,
                                            const NodeProto &candidate,
                                            const std::vector<const NodeProto *> &next_nodes) const;
};

/**
 * Merges identical nodes that consume the same graph input.
 *
 * @code
 * Before:
 *                         +---------+
 *   graph_input, c -----> | Op      | -----> a
 *                         +---------+
 *
 *                         +---------+
 *   graph_input, c -----> | Op      | -----> b
 *                         +---------+
 *
 * After:
 *                         +---------+
 *   graph_input, c -----> | Op      | -----> a
 *                         +---------+
 *                               |
 *                               v
 *                        +----------+
 *                        | Identity | -----> b
 *                        +----------+
 * @endcode
 *
 * It uses the same replacement as SameChildrenPattern, retaining one node and
 * preserving every removed sibling output with an Identity.
 */
class SameChildrenFromInputPattern final : public SameChildrenPattern {
public:
  explicit SameChildrenFromInputPattern(int priority = 0)
      : SameChildrenPattern(priority, "SameChildrenFromInput") {}

  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
};

/**
 * Merges Expand or Reshape siblings whose inferred output shapes are equal.
 *
 * @code
 * Before:
 *                    +---------+
 *   x, shape1 -----> | Reshape | -----> a
 *                    +---------+
 *
 *                    +---------+
 *   x, shape2 -----> | Reshape | -----> b
 *                    +---------+
 *
 * After:
 *                    +---------+
 *   x, shape1 -----> | Reshape | -----> a
 *                    +---------+
 *                          |
 *                          v
 *                   +----------+
 *                   | Identity | -----> b
 *                   +----------+
 * @endcode
 *
 * The inferred shapes of ``a`` and ``b`` must be equal, although the shape
 * inputs may differ. The first operation is retained and each other sibling is
 * replaced by an Identity that retains its original output name.
 */
class ShapeBasedSameChildrenPattern final : public core::builder::PatternOptimization {
public:
  explicit ShapeBasedSameChildrenPattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedSameChildren") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Replaces a shape-preserving Slice by an Identity.
 *
 * @code
 * Before:
 *   x, starts, ends, axes, steps=1
 *                |
 *                v
 *         +---------+
 *         | Slice   |
 *         +---------+
 *                |
 *                v
 *                y
 *
 * After:
 *          +----------+
 *   x ---> | Identity | ---> y
 *          +----------+
 * @endcode
 *
 * The rewrite requires equal inferred input and output shapes. When the
 * optional steps input is present, it must be a materialized all-one constant.
 */
class ShapeBasedIdentityPattern final : public core::builder::PatternOptimization {
public:
  explicit ShapeBasedIdentityPattern(int priority = 0)
      : PatternOptimization(priority, "ShapeBasedIdentity") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

/**
 * Moves a rank-preserving layout operation after a compatible following operation.
 *
 * @code
 * Before:
 *   x, perm/shape
 *         |
 *         v
 *   +-------------------+
 *   | Transpose/Reshape |
 *   +-------------------+
 *         |
 *         v
 *   +---------+
 *   | Unary   | <--- extra inputs
 *   +---------+
 *         |
 *         v
 *         y
 *
 * After:
 *          +---------+
 *   x ---> | Unary   | <--- extra inputs
 *          +---------+
 *               |
 *               v
 *        +-------------------+
 *        | Transpose/Reshape | <--- perm/shape
 *        +-------------------+
 *               |
 *               v
 *               y
 * @endcode
 *
 * The layout node is a Transpose or Reshape. The following node is unary-like,
 * or is Add, Sub, Mul, or Div with a shape-[1] second input; its extra inputs
 * and the final output name are retained.
 */
class SwapUnaryPattern final : public core::builder::PatternOptimization {
public:
  explicit SwapUnaryPattern(int priority = 0) : PatternOptimization(priority, "SwapUnary") {}

  std::set<std::string> FastOpType() const override;
  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override;
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override;
};

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
