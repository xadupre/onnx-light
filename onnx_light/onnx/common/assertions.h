// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx_pb.h"

#include <stdexcept>
#include <string>

namespace ONNX_NAMESPACE {

struct assert_error : public std::runtime_error {
public:
  explicit assert_error(const std::string &msg) : runtime_error(msg) {}
};

struct tensor_error : public assert_error {
public:
  explicit tensor_error(const std::string &msg) : assert_error(msg) {}
};

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
std::string
barf(const char *fmt, ...);

[[noreturn]] void throw_assert_error(std::string &msg);

[[noreturn]] void throw_tensor_error(std::string &msg);

} // namespace ONNX_NAMESPACE

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define _ONNX_EXPECT(x, y) (__builtin_expect((x), (y)))
#else
#define _ONNX_EXPECT(x, y) (x)
#endif

#define ONNX_ASSERT(cond)                                                                          \
  if (_ONNX_EXPECT(!(cond), 0)) {                                                                  \
    std::string error_msg = ::ONNX_NAMESPACE::barf("%s:%u: %s: Assertion `%s` failed.", __FILE__,  \
                                                   __LINE__, __func__, #cond);                     \
    throw_assert_error(error_msg);                                                                 \
  }

#define ONNX_EXPAND(x) x

#define ONNX_ASSERTM(cond, msg, ...)                                                               \
  if (_ONNX_EXPECT(!(cond), 0)) {                                                                  \
    std::string error_msg =                                                                        \
        ::ONNX_NAMESPACE::barf("%s:%u: %s: Assertion `%s` failed: " msg, __FILE__, __LINE__,       \
                               __func__, #cond, ##__VA_ARGS__);                                    \
    throw_assert_error(error_msg);                                                                 \
  }

#define TENSOR_ASSERTM(cond, msg, ...)                                                             \
  if (_ONNX_EXPECT(!(cond), 0)) {                                                                  \
    std::string error_msg =                                                                        \
        ::ONNX_NAMESPACE::barf("%s:%u: %s: Assertion `%s` failed: " msg, __FILE__, __LINE__,       \
                               __func__, #cond, ##__VA_ARGS__);                                    \
    throw_tensor_error(error_msg);                                                                 \
  }
