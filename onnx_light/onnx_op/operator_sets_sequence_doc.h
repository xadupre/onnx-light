// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace sequence {

/**
 * Returns the documentation string for the SequenceEmpty operator.
 *
 * @return Documentation string for the SequenceEmpty operator.
 */
std::string MakeSequenceEmptyDoc();

/**
 * Returns the documentation string for the SequenceConstruct operator.
 *
 * @return Documentation string for the SequenceConstruct operator.
 */
std::string MakeSequenceConstructDoc();

/**
 * Returns the documentation string for the SequenceInsert operator.
 *
 * @return Documentation string for the SequenceInsert operator.
 */
std::string MakeSequenceInsertDoc();

/**
 * Returns the documentation string for the SequenceAt operator.
 *
 * @return Documentation string for the SequenceAt operator.
 */
std::string MakeSequenceAtDoc();

/**
 * Returns the documentation string for the SequenceErase operator.
 *
 * @return Documentation string for the SequenceErase operator.
 */
std::string MakeSequenceEraseDoc();

/**
 * Returns the documentation string for the SequenceLength operator.
 *
 * @return Documentation string for the SequenceLength operator.
 */
std::string MakeSequenceLengthDoc();

/**
 * Returns the documentation string for the SplitToSequence operator
 * (shared by versions 11 and 24).
 *
 * @return Documentation string for the SplitToSequence operator.
 */
std::string MakeSplitToSequenceDoc();

/**
 * Returns the documentation string for the ConcatFromSequence operator.
 *
 * @return Documentation string for the ConcatFromSequence operator.
 */
std::string MakeConcatFromSequenceDoc();

/**
 * Returns the documentation string for the SequenceMap operator.
 *
 * @return Documentation string for the SequenceMap operator.
 */
std::string MakeSequenceMapDoc();

} // namespace sequence
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
