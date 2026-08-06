// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "onnx_light_helpers.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_op::training {

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

} // namespace ONNX_LIGHT_NAMESPACE::onnx_op::training
