// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

#if defined(_WIN32) && defined(ONNX_THREADS_SHARED)
#if defined(lib_onnx_threads_EXPORTS)
#define ONNX_THREADS_API __declspec(dllexport)
#else
#define ONNX_THREADS_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define ONNX_THREADS_API __attribute__((visibility("default")))
#else
#define ONNX_THREADS_API
#endif
