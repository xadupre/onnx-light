// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace controlflow {

/**
 * Returns the documentation string for the If operator.
 *
 * @return Documentation string for the If operator.
 */
std::string MakeIfDoc();

/**
 * Returns the output description string for the If operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Output description string.
 */
std::string MakeIfOutputDescription(int since_version);

/**
 * Returns the documentation string for the Loop operator at the given
 * opset version. Opset 1 uses the original wording, opset 11 introduces
 * the loop-carried dependency formulation, and opset 13 adds the
 * "input/output of subgraph matching is based on order" remark.
 *
 * @param since_version Opset version for which to generate the doc.
 * @return Documentation string for the Loop operator.
 */
std::string MakeLoopDoc(int since_version);

/**
 * Returns the output description string for the Loop operator at the given
 * opset version. Opset 13 adds the "Scan outputs must be Tensors." sentence.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Output description string.
 */
std::string MakeLoopOutputDescription(int since_version);

/**
 * Returns the description string of the ``body`` GRAPH attribute of the
 * Loop operator. The text is the same across the supported opset versions
 * (1, 11, 13).
 *
 * @return Description string for the ``body`` attribute.
 */
std::string MakeLoopBodyAttributeDescription();

} // namespace controlflow
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
