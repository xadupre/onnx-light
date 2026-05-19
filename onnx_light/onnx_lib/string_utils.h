// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file string_utils.h
 * @brief Compatibility shim for the historical ONNX include path.
 *
 * onnx-light replaces <onnx/string_utils.h> with <onnx/common/string_utils.h>.
 * Vendored headers that include "onnx/string_utils.h" are directed here.
 */

#pragma once

/// Re-exports ONNX string utilities from the canonical compatibility location.
#include "onnx_lib/common/string_utils.h"
