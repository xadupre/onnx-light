// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// onnx-light compatibility header.
//
// This file exists so that code written against the standard `onnx` C++
// package (which includes headers rooted at `onnx/`) compiles unchanged
// against onnx-light. The standard `onnx/onnx-data.pb.h` is the
// protobuf-generated translation unit that defines SequenceProto, MapProto and
// OptionalProto. onnx-light hand-writes those same message types in
// `onnx_proto/onnx.h`, which is pulled in through `onnx_lib/common/onnx_pb.h`,
// so forwarding to it makes onnx-light a true drop-in for consumers that
// include <onnx/onnx-data.pb.h> (e.g. onnxruntime's test tooling).
#pragma once

#include "onnx_lib/common/onnx_pb.h"
