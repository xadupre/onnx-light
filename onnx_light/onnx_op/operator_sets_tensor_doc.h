// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace tensor {

/**
 * Returns the documentation string for the Cast operator at the given opset
 * version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the Cast operator.
 */
std::string MakeCastDoc(int since_version);

/**
 * Returns the input type-constraint description for the Cast operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the Cast input.
 */
std::string MakeCastInputTypeConstraintDescription(int since_version);

/**
 * Returns the output type-constraint description for the Cast operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the Cast output.
 */
std::string MakeCastOutputTypeConstraintDescription(int since_version);

/**
 * Returns the documentation string for the CastLike operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the CastLike operator.
 */
std::string MakeCastLikeDoc(int since_version);

/**
 * Returns the input type-constraint description for the CastLike operator at
 * the given opset version (applies to both ``T1`` and ``T2``; the upstream
 * schema uses the same wording for both).
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the CastLike input.
 */
std::string MakeCastLikeInputTypeConstraintDescription(int since_version);

/**
 * Returns the output type-constraint description for the CastLike operator at
 * the given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the CastLike output.
 */
std::string MakeCastLikeOutputTypeConstraintDescription(int since_version);

/**
 * Returns the documentation string for the AffineGrid operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the AffineGrid operator.
 */
std::string MakeAffineGridDoc(int since_version);

/**
 * Returns the type-constraint description for the AffineGrid ``theta``/``grid``
 * float type parameter (``T1``) at the given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for ``T1``.
 */
std::string MakeAffineGridGridTypeConstraintDescription(int since_version);

/**
 * Returns the type-constraint description for the AffineGrid ``size`` integer
 * type parameter (``T2``) at the given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for ``T2``.
 */
std::string MakeAffineGridSizeTypeConstraintDescription(int since_version);

/**
 * Returns the documentation string for the Concat operator at the given opset
 * version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the Concat operator.
 */
std::string MakeConcatDoc(int since_version);

/**
 * Returns the type-constraint description for the Concat operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the Concat input/output.
 */
std::string MakeConcatTypeConstraintDescription(int since_version);

/**
 * Returns the documentation string for the Expand operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the Expand operator.
 */
std::string MakeExpandDoc(int since_version);

/**
 * Returns the type-constraint description for the Expand operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the Expand input/output.
 */
std::string MakeExpandTypeConstraintDescription(int since_version);

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
