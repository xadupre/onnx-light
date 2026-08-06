// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/generator/include_generator_kernels.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/random.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Validates the requested output shape: dims must be non-negative.
// Returns the element count.
int64_t CheckShape(const onnx_kernels::Shape &shape, const char *op_name) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    EXT_ENFORCE_INVALID(dim >= 0, "kernel::", op_name, ": shape must not contain negative dims.");
    count *= dim;
  }
  return count;
}

// Resolves the output dtype. ``requested`` is the value of the ``dtype``
// attribute (0 means "absent"). ``default_dtype`` is used when no value
// is requested. Throws on unsupported dtype.
int32_t ResolveOutputDtype(int32_t requested, int32_t default_dtype, const char *op_name) {
  const int32_t out_dtype = (requested != 0) ? requested : default_dtype;
  switch (static_cast<DataType>(out_dtype)) {
  case DataType::FLOAT:
  case DataType::DOUBLE:
    return out_dtype;
  default:
    EXT_ENFORCE_INVALID(false, "kernel::", op_name, ": unsupported output dtype ",
                        std::to_string(out_dtype), "; only FLOAT and DOUBLE are supported.");
  }
  return out_dtype;
}

// Converts the FLOAT-typed ``seed`` attribute (passed here as int64) to
// the ``std::optional<uint64_t>`` accepted by :cpp:func:`RandUniformInto` /
// :cpp:func:`RandNormalInto`. ``kNoSeed`` (-1) means "no seed", which maps to
// ``std::nullopt``.
std::optional<uint64_t> NormalizeSeed(int64_t seed) {
  if (seed == -1) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(seed);
}

Tensor MakeUniform(const onnx_kernels::Shape &shape, double low, double high,
                   std::optional<uint64_t> seed, int32_t dtype, const char *op_name,
                   RuntimeContext *rt) {
  const int64_t count = CheckShape(shape, op_name);
  Tensor out = MakeOutputTensor(dtype, shape, static_cast<size_t>(count) * ElementSize(dtype),
                                rt ? rt->allocator() : nullptr);
  if (static_cast<DataType>(dtype) == DataType::DOUBLE) {
    RandUniformInto<double>(reinterpret_cast<double *>(out.mutable_bytes()), count, low, high,
                            seed);
  } else {
    RandUniformInto<float>(reinterpret_cast<float *>(out.mutable_bytes()), count, low, high, seed);
  }
  return out;
}

Tensor MakeNormal(const onnx_kernels::Shape &shape, double mean, double scale,
                  std::optional<uint64_t> seed, int32_t dtype, const char *op_name,
                  RuntimeContext *rt) {
  const int64_t count = CheckShape(shape, op_name);
  Tensor out = MakeOutputTensor(dtype, shape, static_cast<size_t>(count) * ElementSize(dtype),
                                rt ? rt->allocator() : nullptr);
  if (static_cast<DataType>(dtype) == DataType::DOUBLE) {
    RandNormalInto<double>(reinterpret_cast<double *>(out.mutable_bytes()), count, mean, scale,
                           seed);
  } else {
    RandNormalInto<float>(reinterpret_cast<float *>(out.mutable_bytes()), count, mean, scale, seed);
  }
  return out;
}

void WriteUniformInto(Tensor &output, const onnx_kernels::Shape &shape, double low, double high,
                      std::optional<uint64_t> seed, int32_t out_dtype, const char *op_name) {
  EXT_ENFORCE_INVALID(output.data_type == out_dtype, "kernel::", op_name,
                      " preallocated output must have the expected dtype.");
  EXT_ENFORCE_INVALID(output.shape == shape, "kernel::", op_name,
                      " preallocated output shape must match the requested shape.");
  const int64_t count = CheckShape(shape, op_name);
  if (static_cast<DataType>(out_dtype) == DataType::DOUBLE) {
    RandUniformInto<double>(reinterpret_cast<double *>(output.mutable_bytes()), count, low, high,
                            seed);
  } else {
    RandUniformInto<float>(reinterpret_cast<float *>(output.mutable_bytes()), count, low, high,
                           seed);
  }
}

void WriteNormalInto(Tensor &output, const onnx_kernels::Shape &shape, double mean, double scale,
                     std::optional<uint64_t> seed, int32_t out_dtype, const char *op_name) {
  EXT_ENFORCE_INVALID(output.data_type == out_dtype, "kernel::", op_name,
                      " preallocated output must have the expected dtype.");
  EXT_ENFORCE_INVALID(output.shape == shape, "kernel::", op_name,
                      " preallocated output shape must match the requested shape.");
  const int64_t count = CheckShape(shape, op_name);
  if (static_cast<DataType>(out_dtype) == DataType::DOUBLE) {
    RandNormalInto<double>(reinterpret_cast<double *>(output.mutable_bytes()), count, mean, scale,
                           seed);
  } else {
    RandNormalInto<float>(reinterpret_cast<float *>(output.mutable_bytes()), count, mean, scale,
                          seed);
  }
}

} // namespace

Tensor RandomNormal::operator()(const onnx_kernels::Shape &shape, double mean, double scale,
                                int64_t seed, int32_t dtype, RuntimeContext *rt) const {
  const int32_t out_dtype = ResolveOutputDtype(dtype, DataType::FLOAT, "RandomNormal");
  return MakeNormal(shape, mean, scale, NormalizeSeed(seed), out_dtype, "RandomNormal", rt);
}

void RandomNormal::operator()(const onnx_kernels::Shape &shape, double mean, double scale,
                              int64_t seed, int32_t dtype, Tensor &output) const {
  const int32_t out_dtype = ResolveOutputDtype(dtype, DataType::FLOAT, "RandomNormal");
  WriteNormalInto(output, shape, mean, scale, NormalizeSeed(seed), out_dtype, "RandomNormal");
}

Tensor RandomUniform::operator()(const onnx_kernels::Shape &shape, double low, double high,
                                 int64_t seed, int32_t dtype, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(high >= low,
                      "kernel::RandomUniform: 'high' must be greater than or equal to 'low'.");
  const int32_t out_dtype = ResolveOutputDtype(dtype, DataType::FLOAT, "RandomUniform");
  return MakeUniform(shape, low, high, NormalizeSeed(seed), out_dtype, "RandomUniform", rt);
}

void RandomUniform::operator()(const onnx_kernels::Shape &shape, double low, double high,
                               int64_t seed, int32_t dtype, Tensor &output) const {
  EXT_ENFORCE_INVALID(high >= low,
                      "kernel::RandomUniform: 'high' must be greater than or equal to 'low'.");
  const int32_t out_dtype = ResolveOutputDtype(dtype, DataType::FLOAT, "RandomUniform");
  WriteUniformInto(output, shape, low, high, NormalizeSeed(seed), out_dtype, "RandomUniform");
}

Tensor RandomNormalLike::operator()(const Tensor &input, double mean, double scale, int64_t seed,
                                    int32_t dtype, RuntimeContext *rt) const {
  const int32_t out_dtype = ResolveOutputDtype(dtype, input.data_type, "RandomNormalLike");
  return MakeNormal(input.shape, mean, scale, NormalizeSeed(seed), out_dtype, "RandomNormalLike",
                    rt);
}

void RandomNormalLike::operator()(const Tensor &input, double mean, double scale, int64_t seed,
                                  int32_t dtype, Tensor &output) const {
  const int32_t out_dtype = ResolveOutputDtype(dtype, input.data_type, "RandomNormalLike");
  WriteNormalInto(output, input.shape, mean, scale, NormalizeSeed(seed), out_dtype,
                  "RandomNormalLike");
}

Tensor RandomUniformLike::operator()(const Tensor &input, double low, double high, int64_t seed,
                                     int32_t dtype, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(high >= low,
                      "kernel::RandomUniformLike: 'high' must be greater than or equal to 'low'.");
  const int32_t out_dtype = ResolveOutputDtype(dtype, input.data_type, "RandomUniformLike");
  return MakeUniform(input.shape, low, high, NormalizeSeed(seed), out_dtype, "RandomUniformLike",
                     rt);
}

void RandomUniformLike::operator()(const Tensor &input, double low, double high, int64_t seed,
                                   int32_t dtype, Tensor &output) const {
  EXT_ENFORCE_INVALID(high >= low,
                      "kernel::RandomUniformLike: 'high' must be greater than or equal to 'low'.");
  const int32_t out_dtype = ResolveOutputDtype(dtype, input.data_type, "RandomUniformLike");
  WriteUniformInto(output, input.shape, low, high, NormalizeSeed(seed), out_dtype,
                   "RandomUniformLike");
}

void RandomNormal::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 0);
  RequireOutputCount(node, 1);
  const std::vector<int64_t> shape = GetAttributeIntsOrDefault(node, "shape", {});
  const float a = GetAttributeFloatOrDefault(node, "mean", 0.0f);
  const float b = GetAttributeFloatOrDefault(node, "scale", 1.0f);
  const int64_t seed = GetSeedAttr(node);
  const int32_t dtype = GetDtypeAttr(node);
  SetOutput(node, 0, (*this)(shape, a, b, seed, dtype, &rt), rt);
}

void RandomNormalLike::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const float a = GetAttributeFloatOrDefault(node, "mean", 0.0f);
  const float b = GetAttributeFloatOrDefault(node, "scale", 1.0f);
  const int64_t seed = GetSeedAttr(node);
  const int32_t dtype = GetDtypeAttr(node);
  const Tensor &input = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(input, a, b, seed, dtype, &rt), rt);
}

void RandomUniform::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 0);
  RequireOutputCount(node, 1);
  const std::vector<int64_t> shape = GetAttributeIntsOrDefault(node, "shape", {});
  const float a = GetAttributeFloatOrDefault(node, "low", 0.0f);
  const float b = GetAttributeFloatOrDefault(node, "high", 1.0f);
  const int64_t seed = GetSeedAttr(node);
  const int32_t dtype = GetDtypeAttr(node);
  SetOutput(node, 0, (*this)(shape, a, b, seed, dtype, &rt), rt);
}

void RandomUniformLike::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const float a = GetAttributeFloatOrDefault(node, "low", 0.0f);
  const float b = GetAttributeFloatOrDefault(node, "high", 1.0f);
  const int64_t seed = GetSeedAttr(node);
  const int32_t dtype = GetDtypeAttr(node);
  const Tensor &input = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(input, a, b, seed, dtype, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
