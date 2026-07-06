// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// onnx-light protobuf compatibility header.
//
// Forwards a standard <google/protobuf/...> include to onnx-light's
// protobuf-free compatibility shim so that code written against the protobuf
// API (e.g. onnxruntime) compiles without linking libprotobuf.
#pragma once

#include "google_protobuf_compat.h"
