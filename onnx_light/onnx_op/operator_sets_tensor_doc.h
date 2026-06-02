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
 * Returns the documentation string for the BitCast operator (opset 26).
 *
 * @return Documentation string for the BitCast operator.
 */
std::string MakeBitCastDoc();

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
 * Returns the documentation string for the GridSample operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the GridSample operator.
 */
std::string MakeGridSampleDoc(int since_version);

/**
 * Returns the type-constraint description for the GridSample ``X``/``Y``
 * type parameter (``T1``) at the given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for ``T1``.
 */
std::string MakeGridSampleInputTypeConstraintDescription(int since_version);

/**
 * Returns the type-constraint description for the GridSample ``grid`` float
 * type parameter (``T2``) at the given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for ``T2``.
 */
std::string MakeGridSampleGridTypeConstraintDescription(int since_version);

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

/**
 * Returns the documentation string for the Squeeze operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the Squeeze operator.
 */
std::string MakeSqueezeDoc(int since_version);

/**
 * Returns the type-constraint description for the Squeeze operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the Squeeze input/output.
 */
std::string MakeSqueezeTypeConstraintDescription(int since_version);

/**
 * Returns the documentation string for the Unsqueeze operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the Unsqueeze operator.
 */
std::string MakeUnsqueezeDoc(int since_version);

/**
 * Returns the type-constraint description for the Unsqueeze operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the Unsqueeze input/output.
 */
std::string MakeUnsqueezeTypeConstraintDescription(int since_version);

/**
 * Returns the documentation string for the NonZero operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the NonZero operator.
 */
std::string MakeNonZeroDoc(int since_version);

/**
 * Returns the type-constraint description for the NonZero ``T`` type
 * parameter at the given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the NonZero input.
 */
std::string MakeNonZeroTypeConstraintDescription(int since_version);

/**
 * Returns the documentation string for the Shape operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the Shape operator.
 */
std::string MakeShapeDoc(int since_version);

/**
 * Returns the type-constraint description for the Shape ``T`` type parameter
 * at the given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the Shape input.
 */
std::string MakeShapeTypeConstraintDescription(int since_version);

/**
 * Returns the documentation string for the Tile operator at the given opset
 * version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the Tile operator.
 */
std::string MakeTileDoc(int since_version);

/**
 * Returns the type-constraint description for the Tile operator at the given
 * opset version (applies to type parameter ``T``).
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the Tile input/output.
 */
std::string MakeTileTypeConstraintDescription(int since_version);

/**
 * Returns the documentation string for the Transpose operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the Transpose operator.
 */
std::string MakeTransposeDoc(int since_version);

/**
 * Returns the type-constraint description for the Transpose operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the Transpose input/output.
 */
std::string MakeTransposeTypeConstraintDescription(int since_version);

/**
 * Returns the documentation string for the DepthToSpace operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the DepthToSpace operator.
 */
std::string MakeDepthToSpaceDoc(int since_version);

/**
 * Returns the type-constraint description for the DepthToSpace ``T`` type
 * parameter at the given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the DepthToSpace input/output.
 */
std::string MakeDepthToSpaceTypeConstraintDescription(int since_version);

/**
 * Returns the documentation string for the Trilu operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the Trilu operator.
 */
std::string MakeTriluDoc(int since_version);

/**
 * Returns the type-constraint description for the Trilu operator at the given
 * opset version (applies to type parameter ``T``).
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the Trilu input/output.
 */
std::string MakeTriluTypeConstraintDescription(int since_version);

/**
 * Returns the documentation string for the Gather operator at the given
 * opset version.
 */
std::string MakeGatherDoc(int since_version);

/**
 * Returns the documentation string for the GatherElements operator at the
 * given opset version.
 */
std::string MakeGatherElementsDoc(int since_version);

/**
 * Returns the documentation string for the GatherND operator at the given
 * opset version.
 */
std::string MakeGatherNDDoc(int since_version);

/**
 * Returns the documentation string for the Compress operator at the given
 * opset version.
 */
std::string MakeCompressDoc(int since_version);

/**
 * Returns the documentation string for the Split operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the Split operator.
 */
std::string MakeSplitDoc(int since_version);

/**
 * Returns the type-constraint description for the Split ``T`` type
 * parameter at the given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the Split input/output.
 */
std::string MakeSplitTypeConstraintDescription(int since_version);

/**
 * Returns the type-constraint description for the Compress ``T`` type
 * parameter at the given opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Type-constraint description string for the Compress input/output.
 */
std::string MakeCompressTypeConstraintDescription(int since_version);

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
