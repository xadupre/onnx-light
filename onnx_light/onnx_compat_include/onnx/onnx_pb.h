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

#include "onnx_lib/common/onnx_pb.h"

// The standard protobuf-generated onnx headers transitively expose the
// google::protobuf types that onnxruntime references directly (RepeatedField,
// RepeatedPtrField, RepeatedFieldBackInserter, ShutdownProtobufLibrary, ...).
// onnx-light is protobuf-free, so pull in its google::protobuf compatibility
// shim here to keep onnx-light a true drop-in for consumers that include
// <onnx/onnx_pb.h> (e.g. onnxruntime).
#include "google_protobuf_compat.h"
