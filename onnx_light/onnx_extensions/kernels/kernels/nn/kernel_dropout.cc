// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr uint32_t kDefaultDropoutSeed = 0u;

void ValidateInput(const Tensor &data, float ratio) {
  EXT_ENFORCE_INVALID(ratio >= 0.0f && ratio < 1.0f, "kernel::Dropout: ratio must be in [0, 1).");
  EXT_ENFORCE_INVALID(data.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          data.data_type == static_cast<int32_t>(DataType::DOUBLE),
                      "kernel::Dropout: only FLOAT and DOUBLE are supported.");
}

template <typename T>
void ComputeDropout(const T *src, T *dst, uint8_t *mask_data, int64_t n, float ratio,
                    bool training_mode, uint32_t seed) {
  if (!training_mode || ratio == 0.0f) {
    if (dst != src) {
      std::memcpy(dst, src, static_cast<std::size_t>(n) * sizeof(T));
    }
    if (mask_data != nullptr) {
      std::fill(mask_data, mask_data + n, static_cast<uint8_t>(1));
    }
    return;
  }

  const float scale = 1.0f / (1.0f - ratio);
  std::mt19937 engine(seed);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  for (int64_t i = 0; i < n; ++i) {
    const bool keep = uniform(engine) >= ratio;
    if (mask_data != nullptr) {
      mask_data[i] = keep ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
    }
    dst[i] = keep ? static_cast<T>(src[i] * static_cast<T>(scale)) : static_cast<T>(0);
  }
}

} // namespace

std::pair<Tensor, Tensor> Dropout::operator()(const Tensor &data, float ratio, bool training_mode,
                                              int64_t seed, RuntimeContext *rt) const {
  ValidateInput(data, ratio);

  const size_t mask_n_bytes = static_cast<std::size_t>(data.element_count());
  Tensor mask = MakeOutputTensor(static_cast<int32_t>(DataType::BOOL), data.shape, mask_n_bytes,
                                 rt ? rt->allocator() : nullptr);

  Tensor output = (*this)(data, ratio, training_mode, mask, seed, rt);
  return {std::move(output), std::move(mask)};
}

Tensor Dropout::operator()(const Tensor &data, float ratio, bool training_mode, Tensor &mask,
                           int64_t seed, RuntimeContext *rt) const {
  ValidateInput(data, ratio);
  EXT_ENFORCE_INVALID(mask.data_type == static_cast<int32_t>(DataType::BOOL),
                      "kernel::Dropout: mask must have BOOL dtype.");
  EXT_ENFORCE_INVALID(mask.shape == data.shape, "kernel::Dropout: mask shape must match input.");
  EXT_ENFORCE_INVALID(mask.size_bytes() == static_cast<std::size_t>(data.element_count()),
                      "kernel::Dropout: mask buffer must have one byte per input element.");

  const size_t output_n_bytes = data.size_bytes();
  Tensor output =
      MakeOutputTensor(data.data_type, data.shape, output_n_bytes, rt ? rt->allocator() : nullptr);
  const uint32_t engine_seed =
      (seed == kNoSeed) ? kDefaultDropoutSeed : static_cast<uint32_t>(seed);

  const int64_t n = data.element_count();
  if (data.data_type == static_cast<int32_t>(DataType::FLOAT)) {
    ComputeDropout<float>(data.AsFloat(), output.AsFloat(), mask.AsBool(), n, ratio, training_mode,
                          engine_seed);
  } else {
    ComputeDropout<double>(data.AsDouble(), output.AsDouble(), mask.AsBool(), n, ratio,
                           training_mode, engine_seed);
  }
  return output;
}

void Dropout::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputRange(node, 1, 3);
  RequireOutputRange(node, 1, 2);
  const Tensor &data = GetInput(node, 0, rt.tensors());

  // ``ratio``: from input[1] (scalar T1) when present, else from the
  // pre-opset-12 ``ratio`` attribute (FLOAT, default 0.5).
  float ratio = GetAttributeFloatOrDefault(node, "ratio", 0.5f);
  const Tensor *ratio_input = GetOptionalInput(node, 1, rt.tensors());
  if (ratio_input != nullptr) {
    EXT_ENFORCE_INVALID(!(ratio_input->element_count() != 1),
                        "RunNode: op 'Dropout' input 'ratio' must be a scalar tensor.");
    switch (ratio_input->data_type) {
    case static_cast<int32_t>(DataType::FLOAT):
      ratio = ratio_input->AsFloat()[0];
      break;
    case static_cast<int32_t>(DataType::DOUBLE):
      ratio = static_cast<float>(ratio_input->AsDouble()[0]);
      break;
    default:
      EXT_THROW_INVALID("RunNode: op 'Dropout' input 'ratio' must be FLOAT or DOUBLE.");
    }
  }

  // ``training_mode``: from input[2] (scalar BOOL) when present,
  // otherwise defaults to false (inference behaviour).
  bool training_mode = false;
  const Tensor *training_input = GetOptionalInput(node, 2, rt.tensors());
  if (training_input != nullptr) {
    EXT_ENFORCE_INVALID(!(training_input->element_count() != 1),
                        "RunNode: op 'Dropout' input 'training_mode' must be a scalar tensor.");
    EXT_ENFORCE_INVALID(!(training_input->data_type != static_cast<int32_t>(DataType::BOOL)),
                        "RunNode: op 'Dropout' input 'training_mode' must be BOOL.");
    training_mode = training_input->AsBool()[0] != 0;
  }

  const int64_t seed =
      GetAttributeIntOrDefault(node, "seed", onnx_kernels::kernel::Dropout::kNoSeed);

  onnx_kernels::kernel::Dropout k(rt.kernel_ctx());
  if (node.output_size() == 2) {
    auto out = k(data, ratio, training_mode, seed);
    SetOutput(node, 0, std::move(out.first), rt);
    SetOutput(node, 1, std::move(out.second), rt);
  } else {
    Tensor mask =
        Tensor::FromRawBytes("", static_cast<int32_t>(DataType::BOOL), data.shape,
                             RawByteBuffer(static_cast<std::size_t>(data.element_count()), 1));
    SetOutput(node, 0, k(data, ratio, training_mode, mask, seed, &rt), rt);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
