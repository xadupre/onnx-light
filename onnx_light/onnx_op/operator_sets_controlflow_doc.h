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

/**
 * Returns the documentation string for the ``Scan`` operator at the given
 * opset version. Opset 8 uses the original wording (with ``sequence_lens``
 * input and a single ``directions`` attribute); opsets 9 and 11 drop the
 * ``sequence_lens`` input and split the directions attribute into
 * ``scan_input_directions`` / ``scan_output_directions``, additionally
 * adding the ``scan_input_axes`` / ``scan_output_axes`` attributes.
 *
 * @param since_version Opset version for which to generate the doc.
 * @return Documentation string for the ``Scan`` operator.
 */
std::string MakeScanDoc(int since_version);

/**
 * Returns the description string of the ``body`` GRAPH attribute of the
 * Scan operator. The text is the same across the supported opset versions
 * (8, 9, 11).
 *
 * @return Description string for the ``body`` attribute.
 */
std::string MakeScanBodyAttributeDescription();

/**
 * Returns the description string of the ``num_scan_inputs`` INT attribute
 * of the Scan operator. The text is the same across the supported opset
 * versions (8, 9, 11).
 *
 * @return Description string for the ``num_scan_inputs`` attribute.
 */
std::string MakeScanNumScanInputsAttributeDescription();

/**
 * Returns the description string of the opset-8 ``directions`` INTS
 * attribute of the Scan operator.
 *
 * @return Description string for the ``directions`` attribute.
 */
std::string MakeScanDirectionsAttributeDescription();

/**
 * Returns the description string of the opset-9+ ``scan_input_directions``
 * INTS attribute of the Scan operator.
 *
 * @return Description string for the ``scan_input_directions`` attribute.
 */
std::string MakeScanInputDirectionsAttributeDescription();

/**
 * Returns the description string of the opset-9+ ``scan_output_directions``
 * INTS attribute of the Scan operator.
 *
 * @return Description string for the ``scan_output_directions`` attribute.
 */
std::string MakeScanOutputDirectionsAttributeDescription();

/**
 * Returns the description string of the opset-9+ ``scan_input_axes``
 * INTS attribute of the Scan operator.
 *
 * @return Description string for the ``scan_input_axes`` attribute.
 */
std::string MakeScanInputAxesAttributeDescription();

/**
 * Returns the description string of the opset-9+ ``scan_output_axes``
 * INTS attribute of the Scan operator.
 *
 * @return Description string for the ``scan_output_axes`` attribute.
 */
std::string MakeScanOutputAxesAttributeDescription();

/**
 * Returns the description string of the ``then_branch`` GRAPH attribute of
 * the If operator. The text is the same across the supported opset versions
 * (1, 11, 13).
 *
 * @return Description string for the ``then_branch`` attribute.
 */
std::string MakeIfThenBranchAttributeDescription();

/**
 * Returns the description string of the ``else_branch`` GRAPH attribute of
 * the If operator. The text is the same across the supported opset versions
 * (1, 11, 13).
 *
 * @return Description string for the ``else_branch`` attribute.
 */
std::string MakeIfElseBranchAttributeDescription();

} // namespace controlflow
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
