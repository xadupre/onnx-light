// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file light_op_schema.h
 * @brief Backward-compatible re-export of the ``onnx_core::schema`` types.
 *
 * ``LightOpSchema`` and the surrounding operator-schema types used to live
 * directly in ``onnx_op`` but have moved to
 * ::ONNX_LIGHT_NAMESPACE::core::schema so that ``onnx_core`` remains the
 * single place where lightweight, ONNX-independent schema data structures
 * are defined. This header is kept so that the many existing
 * ``operator_sets_*.h`` files (and any other consumer) that reference these
 * types unqualified or as ``onnx_op::...`` continue to compile unchanged.
 */

#pragma once

#include "onnx_core/light_op_schema/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {

using core::schema::kOnnxDomain;

using core::schema::FormalParameter;

using core::schema::AttributeType;
using core::schema::AttributeType_Name;

using core::schema::AttributeDefault;
using core::schema::AttributeDefaultRepr;

using core::schema::AttributeParam;

using core::schema::TensorType;
using core::schema::ToTypeString;

using core::schema::TypeConstraintParam;

using core::schema::SchemaError;

using core::schema::LightOpSchema;

using core::schema::AllNumericTypes;
using core::schema::AllNumericTypesIr4;
using core::schema::AllOptionalTypes;
using core::schema::AllTensorSequenceTypes;
using core::schema::AllTensorTypes;
using core::schema::CastTypesVer13;
using core::schema::CastTypesVer19;
using core::schema::CastTypesVer1And6;
using core::schema::CastTypesVer21;
using core::schema::CastTypesVer23;
using core::schema::CastTypesVer24;
using core::schema::CastTypesVer25;
using core::schema::CastTypesVer9;
using core::schema::ConcatTypesVer1;
using core::schema::ConcatTypesVer13;
using core::schema::ConcatTypesVer4And11;
using core::schema::EqualTypesV11;
using core::schema::EqualTypesV13;
using core::schema::EqualTypesV19;
using core::schema::EqualTypesV1V7;
using core::schema::FloatTypes;
using core::schema::NumericTypesForMathReduction;
using core::schema::NumericTypesForMathReductionIr4;

using core::schema::StripDocs;

using core::schema::SchemaBuilder;

using core::schema::CollectSchemasFromBuilders;

} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
