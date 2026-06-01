// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace nn {

/**
 * Returns the documentation string for the AveragePool operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the documentation.
 * @return Documentation string for the AveragePool operator.
 */
std::string MakeAveragePoolDoc(int since_version);

/**
 * Returns the documentation string for the GlobalAveragePool operator at the
 * given opset version (1 or 22).
 */
std::string MakeGlobalAveragePoolDoc(int since_version);

/**
 * Returns the documentation string for the GlobalMaxPool operator at the
 * given opset version (1 or 22).
 */
std::string MakeGlobalMaxPoolDoc(int since_version);

/**
 * Returns the documentation string for the GlobalLpPool operator at the
 * given opset version (1, 2, or 22).
 */
std::string MakeGlobalLpPoolDoc(int since_version);

/**
 * Returns the documentation string for the RNN operator at the given opset
 * version (1, 7, 14, or 22).
 */
std::string MakeRNNDoc(int since_version);

/**
 * Returns the documentation string for the GRU operator at the given opset
 * version (1, 3, 7, 14, or 22).
 */
std::string MakeGRUDoc(int since_version);

/**
 * Returns the documentation string for the LSTM operator at the given opset
 * version (1, 7, 14, or 22).
 */
std::string MakeLSTMDoc(int since_version);

/**
 * Returns the documentation string for the BatchNormalization operator at the
 * given opset version (1, 6, 7, 9, 14, or 15).
 */
std::string MakeBatchNormalizationDoc(int since_version);

/**
 * Returns the documentation string for the Attention operator at the given
 * opset version (23 or 24).
 */
std::string MakeAttentionDoc(int since_version);

/**
 * Returns the documentation string for the DeformConv operator at the given
 * opset version (19 or 22). The text is identical for both opsets and
 * matches the upstream ``kDoc_DeformConv_ver19`` string.
 */
std::string MakeDeformConvDoc(int since_version);

} // namespace nn
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
