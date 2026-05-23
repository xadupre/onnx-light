// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace reduction {

/**
 * Returns the documentation string for the ReduceSum operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the ReduceSum operator.
 */
std::string MakeReduceSumDoc(int since_version);

/**
 * Returns the documentation string for a generic Reduce* operator (other than
 * ReduceSum) at the given opset version. Used for ReduceMean, ReduceProd,
 * ReduceMax, ReduceMin, ReduceSumSquare, ReduceLogSum, ReduceLogSumExp,
 * ReduceL1 and ReduceL2.
 *
 * @param op_name Human-readable operator name (e.g. "mean", "product",
 *                "max", "min", "sum square", "log sum", "log sum exponent",
 *                "L1 norm", "L2 norm").
 * @param empty_value Description of the value returned when reducing over an
 *                    empty set of values.
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the Reduce* operator.
 */
std::string MakeReduceOpDoc(const std::string &op_name, const std::string &empty_value,
                            int since_version);

/**
 * Returns the documentation string for the ArgMax or ArgMin operator at the
 * given opset version.
 *
 * @param op_name Either "max" or "min".
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the ArgMax/ArgMin operator.
 */
std::string MakeArgReduceDoc(const std::string &op_name, int since_version);

} // namespace reduction
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
