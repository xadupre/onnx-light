// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// onnx-light compatibility header.
//
// This file exists so that code written against the standard `onnx` C++
// package (which includes headers rooted at `onnx/`) compiles unchanged
// against onnx-light. It simply forwards to the real onnx-light header.
#pragma once

#include "onnx_lib/defs/operator_sets_training.h"
