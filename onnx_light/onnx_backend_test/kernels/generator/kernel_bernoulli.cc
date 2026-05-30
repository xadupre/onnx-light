// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/generator/include_generator_kernels.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Default seed used when the operator's ``seed`` attribute is absent.
// Picking a fixed value keeps the kernel deterministic so that backend
// test cases can store stable expected outputs while still mirroring the
// schema's non-deterministic contract.
constexpr uint32_t kDefaultBernoulliSeed = 0u;

// Reads the input probability tensor as a vector of double. Only the
// three float dtypes accepted by the Bernoulli ``T1`` constraint are
// supported here (FLOAT, DOUBLE, FLOAT16); other dtypes throw.
std::vector<double> ReadProbabilities(const Tensor &input) {
  const int64_t n = input.element_count();
  std::vector<double> p(static_cast<std::size_t>(n));
  switch (static_cast<TensorProto::DataType>(input.data_type)) {
  case TensorProto::DataType::FLOAT: {
    const float *src = input.AsFloat();
    for (int64_t i = 0; i < n; ++i) {
      p[static_cast<std::size_t>(i)] = static_cast<double>(src[i]);
    }
    break;
  }
  case TensorProto::DataType::DOUBLE: {
    const double *src = input.AsDouble();
    for (int64_t i = 0; i < n; ++i) {
      p[static_cast<std::size_t>(i)] = src[i];
    }
    break;
  }
  case TensorProto::DataType::FLOAT16: {
    // FLOAT16 is stored as a 2-byte stride; convert via the IEEE-754
    // half-precision encoding.
    const uint16_t *src = reinterpret_cast<const uint16_t *>(input.data.data());
    for (int64_t i = 0; i < n; ++i) {
      const uint16_t h = src[i];
      const uint32_t sign = (h >> 15) & 0x1u;
      const uint32_t exp = (h >> 10) & 0x1Fu;
      const uint32_t mant = h & 0x3FFu;
      uint32_t f;
      if (exp == 0) {
        if (mant == 0) {
          f = sign << 31;
        } else {
          uint32_t m = mant;
          int32_t e = -1;
          while ((m & 0x400u) == 0) {
            m <<= 1;
            --e;
          }
          m &= 0x3FFu;
          f = (sign << 31) | (static_cast<uint32_t>(e + 127 + 1) << 23) | (m << 13);
        }
      } else if (exp == 0x1Fu) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
      } else {
        f = (sign << 31) | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
      }
      float fv;
      std::memcpy(&fv, &f, sizeof(float));
      p[static_cast<std::size_t>(i)] = static_cast<double>(fv);
    }
    break;
  }
  default:
    EXT_ENFORCE_INVALID(false, "kernel::Bernoulli: unsupported input dtype " +
                                   std::to_string(input.data_type) + ".");
  }
  return p;
}

// Number of bytes per element for the supported output dtypes.
std::size_t OutputElementSize(int32_t dtype) {
  switch (static_cast<TensorProto::DataType>(dtype)) {
  case TensorProto::DataType::FLOAT:
    return sizeof(float);
  case TensorProto::DataType::DOUBLE:
    return sizeof(double);
  case TensorProto::DataType::FLOAT16:
    return 2;
  case TensorProto::DataType::INT8:
  case TensorProto::DataType::UINT8:
  case TensorProto::DataType::BOOL:
    return 1;
  case TensorProto::DataType::INT16:
  case TensorProto::DataType::UINT16:
    return 2;
  case TensorProto::DataType::INT32:
  case TensorProto::DataType::UINT32:
    return 4;
  case TensorProto::DataType::INT64:
  case TensorProto::DataType::UINT64:
    return 8;
  default:
    EXT_ENFORCE_INVALID(false, "kernel::Bernoulli: unsupported output dtype " +
                                   std::to_string(dtype) + ".");
  }
  return 0;
}

// Stores ``sample`` (either 0 or 1) at index ``i`` of ``out`` using the
// stride for ``dtype``.
void StoreSample(int32_t dtype, std::vector<uint8_t> &out, int64_t i, int32_t sample) {
  switch (static_cast<TensorProto::DataType>(dtype)) {
  case TensorProto::DataType::FLOAT: {
    const float v = static_cast<float>(sample);
    std::memcpy(out.data() + static_cast<std::size_t>(i) * sizeof(float), &v, sizeof(float));
    break;
  }
  case TensorProto::DataType::DOUBLE: {
    const double v = static_cast<double>(sample);
    std::memcpy(out.data() + static_cast<std::size_t>(i) * sizeof(double), &v, sizeof(double));
    break;
  }
  case TensorProto::DataType::FLOAT16: {
    // 0.0 -> 0x0000, 1.0 -> 0x3C00 in IEEE-754 binary16.
    const uint16_t v = sample == 0 ? static_cast<uint16_t>(0x0000) : static_cast<uint16_t>(0x3C00);
    std::memcpy(out.data() + static_cast<std::size_t>(i) * 2, &v, sizeof(uint16_t));
    break;
  }
  case TensorProto::DataType::INT8: {
    const int8_t v = static_cast<int8_t>(sample);
    out[static_cast<std::size_t>(i)] = static_cast<uint8_t>(v);
    break;
  }
  case TensorProto::DataType::UINT8:
  case TensorProto::DataType::BOOL: {
    out[static_cast<std::size_t>(i)] = static_cast<uint8_t>(sample);
    break;
  }
  case TensorProto::DataType::INT16: {
    const int16_t v = static_cast<int16_t>(sample);
    std::memcpy(out.data() + static_cast<std::size_t>(i) * sizeof(int16_t), &v, sizeof(int16_t));
    break;
  }
  case TensorProto::DataType::UINT16: {
    const uint16_t v = static_cast<uint16_t>(sample);
    std::memcpy(out.data() + static_cast<std::size_t>(i) * sizeof(uint16_t), &v, sizeof(uint16_t));
    break;
  }
  case TensorProto::DataType::INT32: {
    const int32_t v = sample;
    std::memcpy(out.data() + static_cast<std::size_t>(i) * sizeof(int32_t), &v, sizeof(int32_t));
    break;
  }
  case TensorProto::DataType::UINT32: {
    const uint32_t v = static_cast<uint32_t>(sample);
    std::memcpy(out.data() + static_cast<std::size_t>(i) * sizeof(uint32_t), &v, sizeof(uint32_t));
    break;
  }
  case TensorProto::DataType::INT64: {
    const int64_t v = static_cast<int64_t>(sample);
    std::memcpy(out.data() + static_cast<std::size_t>(i) * sizeof(int64_t), &v, sizeof(int64_t));
    break;
  }
  case TensorProto::DataType::UINT64: {
    const uint64_t v = static_cast<uint64_t>(sample);
    std::memcpy(out.data() + static_cast<std::size_t>(i) * sizeof(uint64_t), &v, sizeof(uint64_t));
    break;
  }
  default:
    EXT_ENFORCE_INVALID(false, "kernel::Bernoulli: unsupported output dtype " +
                                   std::to_string(dtype) + ".");
  }
}

} // namespace

Tensor Bernoulli::operator()(const Tensor &input, int64_t seed, int32_t dtype) const {
  const std::vector<double> probs = ReadProbabilities(input);

  const int32_t out_dtype = (dtype == 0) ? input.data_type : dtype;
  const std::size_t es = OutputElementSize(out_dtype);
  const int64_t n = input.element_count();
  std::vector<uint8_t> out_data(static_cast<std::size_t>(n) * es, 0);

  const uint32_t engine_seed =
      (seed == kNoSeed) ? kDefaultBernoulliSeed : static_cast<uint32_t>(seed);
  std::mt19937 engine(engine_seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  for (int64_t i = 0; i < n; ++i) {
    const double p = probs[static_cast<std::size_t>(i)];
    EXT_ENFORCE_INVALID(p >= 0.0 && p <= 1.0,
                        "kernel::Bernoulli: input values must lie in [0, 1].");
    const double u = uniform(engine);
    const int32_t sample = (u < p) ? 1 : 0;
    StoreSample(out_dtype, out_data, i, sample);
  }

  return Tensor("", out_dtype, input.shape, std::move(out_data));
}

void Bernoulli::operator()(const Tensor &input, int64_t seed, int32_t dtype, Tensor &output) const {
  Tensor produced = (*this)(input, seed, dtype);
  EXT_ENFORCE_INVALID(output.data_type == produced.data_type,
                      "kernel::Bernoulli preallocated output must have the expected dtype.");
  EXT_ENFORCE_INVALID(output.shape == produced.shape,
                      "kernel::Bernoulli preallocated output shape must match the produced "
                      "tensor shape.");
  EXT_ENFORCE_INVALID(output.data.size() == produced.data.size(),
                      "kernel::Bernoulli preallocated output buffer has unexpected size in "
                      "bytes.");
  if (!produced.data.empty()) {
    std::memcpy(output.data.data(), produced.data.data(), produced.data.size());
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
