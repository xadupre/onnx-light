// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/generator/include_generator_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Default seed used when the operator's ``seed`` attribute is absent.
// Picking a fixed value keeps the kernel deterministic so that backend
// test cases can store stable expected outputs while still mirroring the
// schema's non-deterministic contract.
constexpr uint32_t kDefaultMultinomialSeed = 0u;

// Decodes a single IEEE-754 binary16 value to ``double`` (same logic as
// the Bernoulli kernel; kept local here to avoid coupling the two
// translation units).
double DecodeHalf(uint16_t h) {
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
  return static_cast<double>(std::bit_cast<float>(f));
}

// Builds the CDF for a single batch row from a typed pointer and returns the
// unnormalized probability sum. The ``cdf`` buffer is written in-place and
// must already be sized to ``class_size``. No intermediate conversion buffer
// is allocated; values are read directly from ``row``.
template <typename T> double BuildRowCdf(const T *row, int64_t class_size, double *cdf) {
  double max_logit = static_cast<double>(row[0]);
  for (int64_t c = 1; c < class_size; ++c) {
    const double v = static_cast<double>(row[c]);
    if (v > max_logit) {
      max_logit = v;
    }
  }
  double sum = 0.0;
  for (int64_t c = 0; c < class_size; ++c) {
    const double p = std::exp(static_cast<double>(row[c]) - max_logit);
    sum += p;
    cdf[static_cast<std::size_t>(c)] = sum;
  }
  return sum;
}

// Specialisation for FLOAT16 stored as raw uint16_t bytes.
template <> double BuildRowCdf<uint16_t>(const uint16_t *row, int64_t class_size, double *cdf) {
  double max_logit = DecodeHalf(row[0]);
  for (int64_t c = 1; c < class_size; ++c) {
    const double v = DecodeHalf(row[c]);
    if (v > max_logit) {
      max_logit = v;
    }
  }
  double sum = 0.0;
  for (int64_t c = 0; c < class_size; ++c) {
    const double p = std::exp(DecodeHalf(row[c]) - max_logit);
    sum += p;
    cdf[static_cast<std::size_t>(c)] = sum;
  }
  return sum;
}

// Stores ``sample`` (a class index) at the byte position pointed to by ``out``
// using the encoding for ``dtype``. The caller is responsible for advancing
// ``out`` by the element size between successive calls.
void StoreSample(int32_t dtype, uint8_t *out, int64_t sample) {
  switch (static_cast<DataType>(dtype)) {
  case DataType::INT32:
    *reinterpret_cast<int32_t *>(out) = static_cast<int32_t>(sample);
    break;
  case DataType::INT64:
    *reinterpret_cast<int64_t *>(out) = sample;
    break;
  default:
    EXT_ENFORCE_INVALID(false, "kernel::Multinomial: unsupported output dtype ",
                        std::to_string(dtype), ".");
  }
}

} // namespace

Tensor Multinomial::operator()(const Tensor &input, int64_t sample_size, int64_t seed,
                               int32_t dtype, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(input.shape.size() == 2,
                      "kernel::Multinomial: input must be a 2-D tensor of shape "
                      "[batch_size, class_size].");
  EXT_ENFORCE_INVALID(sample_size >= 0,
                      "kernel::Multinomial: sample_size must be non-negative, got ",
                      std::to_string(sample_size), ".");

  const int64_t batch_size = input.shape[0];
  const int32_t out_dtype = (dtype == 0) ? static_cast<int32_t>(DataType::INT32) : dtype;
  const std::size_t es = ElementSize(out_dtype);
  const int64_t n_out = batch_size * sample_size;
  const std::size_t out_n_bytes = static_cast<std::size_t>(n_out) * es;

  Tensor out = MakeOutputTensor(out_dtype, {batch_size, sample_size}, out_n_bytes,
                                rt ? rt->allocator() : nullptr);
  (*this)(input, sample_size, seed, dtype, out, rt);
  return out;
}

void Multinomial::operator()(const Tensor &input, int64_t sample_size, int64_t seed, int32_t dtype,
                             Tensor &output, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(input.shape.size() == 2,
                      "kernel::Multinomial: input must be a 2-D tensor of shape "
                      "[batch_size, class_size].");
  EXT_ENFORCE_INVALID(sample_size >= 0,
                      "kernel::Multinomial: sample_size must be non-negative, got ",
                      std::to_string(sample_size), ".");

  const int64_t batch_size = input.shape[0];
  const int64_t class_size = input.shape[1];
  EXT_ENFORCE_INVALID(batch_size >= 0 && class_size >= 0,
                      "kernel::Multinomial: input shape must be non-negative.");
  EXT_ENFORCE_INVALID(batch_size == 0 || class_size > 0,
                      "kernel::Multinomial: class_size must be > 0 for a non-empty batch.");

  const int32_t out_dtype = (dtype == 0) ? static_cast<int32_t>(DataType::INT32) : dtype;
  EXT_ENFORCE_INVALID(output.data_type == out_dtype,
                      "kernel::Multinomial preallocated output must have the expected dtype.");
  EXT_ENFORCE_INVALID(output.shape == (onnx_kernels::Shape{batch_size, sample_size}),
                      "kernel::Multinomial preallocated output shape must match the produced "
                      "tensor shape.");

  const std::size_t es = ElementSize(out_dtype);
  uint8_t *out_data = output.mutable_bytes();

  const uint32_t engine_seed =
      (seed == kNoSeed) ? kDefaultMultinomialSeed : static_cast<uint32_t>(seed);
  std::mt19937 engine(engine_seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  // Per-row CDF over normalized class probabilities (softmax of the
  // unnormalized log-probabilities). Reused across rows. Uses the runtime
  // allocator when available and falls back to inline storage otherwise. The
  // buffer is sized to at least one element so an empty batch (``class_size``
  // == 0) never requests a zero-byte allocation.
  detail::TemporaryTypedBuffer<double> cdf_buf(
      static_cast<std::size_t>(std::max<int64_t>(class_size, 1)), rt ? rt->allocator() : nullptr,
      "kernel::Multinomial cdf");
  double *cdf = cdf_buf.data();

  // Dispatch on input dtype once, outside the batch loop, so that typed row
  // pointers are advanced directly without any intermediate conversion buffer.
  auto run_batch = [&](auto typed_ptr) {
    for (int64_t b = 0; b < batch_size; ++b) {
      const double sum = BuildRowCdf(typed_ptr + b * class_size, class_size, cdf);
      EXT_ENFORCE_INVALID(sum > 0.0, "kernel::Multinomial: row ", std::to_string(b),
                          " produced an all-zero probability distribution.");
      // Normalize the CDF to end at exactly 1.0.
      for (int64_t c = 0; c < class_size; ++c) {
        cdf[static_cast<std::size_t>(c)] /= sum;
      }
      for (int64_t s = 0; s < sample_size; ++s) {
        const double u = uniform(engine);
        // Inverse-CDF: find the smallest index whose CDF >= u.
        const auto it = std::lower_bound(cdf, cdf + class_size, u);
        int64_t idx = static_cast<int64_t>(it - cdf);
        if (idx >= class_size) {
          idx = class_size - 1;
        }
        StoreSample(out_dtype, out_data + static_cast<std::size_t>(b * sample_size + s) * es, idx);
      }
    }
  };

  switch (static_cast<DataType>(input.data_type)) {
  case DataType::FLOAT:
    run_batch(input.AsFloat());
    break;
  case DataType::DOUBLE:
    run_batch(input.AsDouble());
    break;
  case DataType::FLOAT16:
    run_batch(reinterpret_cast<const uint16_t *>(input.bytes()));
    break;
  default:
    EXT_ENFORCE_INVALID(false, "kernel::Multinomial: unsupported input dtype ",
                        std::to_string(input.data_type), ".");
  }
}

void Multinomial::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &input = GetInput(node, 0, rt.tensors());
  const int64_t sample_size = GetAttributeIntOrDefault(node, "sample_size", 1);
  onnx_kernels::kernel::Multinomial kernel(rt.kernel_ctx());
  SetOutput(node, 0, kernel(input, sample_size, GetSeedAttr(node), GetDtypeAttr(node), &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
