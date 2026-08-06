// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/gradient/grad_common.h"
#include "onnx_core/gradient/grad_dispatcher.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

// Import the core gradient types and helpers into the onnx_gradient namespace so
// that existing callers and gradient-implementation files can use the
// unqualified names GradFn, GradRegistry, PairStringHash,
// RegisterGradientFunction, ApplyBackward, AccumulateGrad and NewGradName
// without changing their source.
using core::gradient::AccumulateGrad;
using core::gradient::ApplyBackward;
using core::gradient::GradFn;
using core::gradient::GradRegistry;
using core::gradient::NewGradName;
using core::gradient::PairStringHash;
using core::gradient::RegisterGradientFunction;

/**
 * Returns a reference to the built-in gradient registry populated with all
 * standard ONNX operator gradient rules provided by onnx_gradient.
 *
 * The registry maps (domain, op_type) pairs to their GradFn implementations.
 * Callers who need a mutable copy should copy the returned registry and extend
 * it via RegisterGradientFunction.
 */
const GradRegistry &DefaultGradRegistry();

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
