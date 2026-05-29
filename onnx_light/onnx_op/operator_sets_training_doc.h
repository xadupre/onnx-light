// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace training {

/**
 * Returns the documentation string for the Gradient operator.
 *
 * @return Documentation string for the Gradient operator.
 */
std::string MakeGradientDoc();

/**
 * Returns the documentation string for the Adagrad operator.
 *
 * @return Documentation string for the Adagrad operator.
 */
std::string MakeAdagradDoc();

/**
 * Returns the documentation string for the Momentum operator.
 *
 * @return Documentation string for the Momentum operator.
 */
std::string MakeMomentumDoc();

/**
 * Returns the documentation string for the Adam operator.
 *
 * @return Documentation string for the Adam operator.
 */
std::string MakeAdamDoc();

} // namespace training
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
