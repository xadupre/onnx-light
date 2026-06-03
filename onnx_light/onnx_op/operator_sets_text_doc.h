// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace text {

/**
 * Returns the documentation string for the StringConcat operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the StringConcat operator.
 */
std::string MakeStringConcatDoc(int since_version);

/**
 * Returns the documentation string for the StringSplit operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the StringSplit operator.
 */
std::string MakeStringSplitDoc(int since_version);

/**
 * Returns the documentation string for the RegexFullMatch operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the RegexFullMatch operator.
 */
std::string MakeRegexFullMatchDoc(int since_version);

/**
 * Returns the documentation string for the StringNormalizer operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the StringNormalizer operator.
 */
std::string MakeStringNormalizerDoc(int since_version);

/**
 * Returns the documentation string for the TfIdfVectorizer operator at the
 * given opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the TfIdfVectorizer operator.
 */
std::string MakeTfIdfVectorizerDoc(int since_version);

} // namespace text
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
