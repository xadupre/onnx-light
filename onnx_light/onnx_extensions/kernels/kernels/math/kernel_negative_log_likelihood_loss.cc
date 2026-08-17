// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Tensor NegativeLogLikelihoodLoss::operator()(const Tensor &input, const Tensor &target,
                                             const Tensor *weight, const std::string &reduction,
                                             bool has_ignore_index, int64_t ignore_index,
                                             RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(input.data_type == DataType::FLOAT,
                      "kernel::NegativeLogLikelihoodLoss only supports FLOAT input.");
  EXT_ENFORCE_INVALID(target.data_type == DataType::INT32 || target.data_type == DataType::INT64,
                      "kernel::NegativeLogLikelihoodLoss target must be INT32 or INT64.");
  EXT_ENFORCE_INVALID(input.shape.size() >= 2,
                      "kernel::NegativeLogLikelihoodLoss input rank must be >= 2.");
  EXT_ENFORCE_INVALID(target.shape.size() + 1 == input.shape.size(),
                      "kernel::NegativeLogLikelihoodLoss target rank must equal input rank - 1.");
  EXT_ENFORCE_INVALID(reduction == "none" || reduction == "sum" || reduction == "mean",
                      "kernel::NegativeLogLikelihoodLoss: reduction must be one of "
                      "'none', 'sum', 'mean'.");

  const int64_t n_batch = input.shape[0];
  const int64_t n_classes = input.shape[1];
  int64_t n_inner = 1;
  for (size_t d = 2; d < input.shape.size(); ++d) {
    n_inner *= input.shape[d];
  }
  const int64_t n_loss = n_batch * n_inner;
  EXT_ENFORCE_INVALID(target.element_count() == n_loss,
                      "kernel::NegativeLogLikelihoodLoss target element count is inconsistent "
                      "with input shape.");

  if (weight != nullptr) {
    EXT_ENFORCE_INVALID(weight->data_type == DataType::FLOAT,
                        "kernel::NegativeLogLikelihoodLoss weight must be FLOAT.");
    EXT_ENFORCE_INVALID(weight->shape.size() == 1 && weight->shape[0] == n_classes,
                        "kernel::NegativeLogLikelihoodLoss weight must have shape (C).");
  }

  const float *input_ptr = input.AsFloat();
  const float *weight_ptr = weight != nullptr ? weight->AsFloat() : nullptr;

  RawBufferAllocator *allocator = rt ? rt->execution_allocator() : nullptr;

  // Per-sample loss and applied weight (prior to reduction). Backed by the
  // runtime allocator when available, falling back to inline storage otherwise.
  detail::TemporaryTypedBuffer<float> per_sample_loss_buf(
      static_cast<size_t>(n_loss), allocator, "kernel::NegativeLogLikelihoodLoss per_sample_loss");
  detail::TemporaryTypedBuffer<float> per_sample_weight_buf(
      static_cast<size_t>(n_loss), allocator,
      "kernel::NegativeLogLikelihoodLoss per_sample_weight");
  float *per_sample_loss = per_sample_loss_buf.data();
  float *per_sample_weight = per_sample_weight_buf.data();

  for (int64_t o = 0; o < n_batch; ++o) {
    for (int64_t i = 0; i < n_inner; ++i) {
      const int64_t flat = o * n_inner + i;
      int64_t label_value;
      if (target.data_type == DataType::INT64) {
        label_value = target.AsInt64()[static_cast<size_t>(flat)];
      } else {
        label_value = static_cast<int64_t>(target.AsInt32()[static_cast<size_t>(flat)]);
      }

      if (has_ignore_index && label_value == ignore_index) {
        per_sample_loss[static_cast<size_t>(flat)] = 0.0f;
        per_sample_weight[static_cast<size_t>(flat)] = 0.0f;
        continue;
      }

      EXT_ENFORCE_INVALID(label_value >= 0 && label_value < n_classes,
                          "kernel::NegativeLogLikelihoodLoss: target is out of range [0, C).");

      const int64_t input_offset = (o * n_classes + label_value) * n_inner + i;
      const float neg_input = -input_ptr[static_cast<size_t>(input_offset)];
      const float w = weight_ptr != nullptr ? weight_ptr[static_cast<size_t>(label_value)] : 1.0f;
      per_sample_loss[static_cast<size_t>(flat)] = w * neg_input;
      per_sample_weight[static_cast<size_t>(flat)] = w;
    }
  }

  if (reduction == "none") {
    const size_t loss_n_bytes = static_cast<size_t>(n_loss) * sizeof(float);
    Tensor loss = rt ? rt->MakeOutputTensor(0, DataType::FLOAT, target.shape, loss_n_bytes)
                     : MakeOutputTensor(DataType::FLOAT, target.shape, loss_n_bytes, nullptr);
    float *out = loss.AsFloat();
    for (int64_t k = 0; k < n_loss; ++k) {
      out[static_cast<size_t>(k)] = per_sample_loss[static_cast<size_t>(k)];
    }
    return loss;
  }

  float sum_loss = 0.0f;
  float sum_weight = 0.0f;
  for (int64_t k = 0; k < n_loss; ++k) {
    sum_loss += per_sample_loss[static_cast<size_t>(k)];
    sum_weight += per_sample_weight[static_cast<size_t>(k)];
  }

  float reduced;
  if (reduction == "sum") {
    reduced = sum_loss;
  } else {
    // "mean": divide by the sum of applied weights. When ``weight`` is not
    // provided, ``per_sample_weight`` is 1.0 for every contributing sample,
    // so ``sum_weight`` equals the number of contributing samples.
    reduced = sum_weight != 0.0f ? sum_loss / sum_weight : 0.0f;
  }

  const size_t loss_n_bytes = sizeof(float);
  Tensor loss = rt ? rt->MakeOutputTensor(0, DataType::FLOAT, Shape{}, loss_n_bytes)
                   : MakeOutputTensor(DataType::FLOAT, Shape{}, loss_n_bytes, nullptr);
  loss.AsFloat()[0] = reduced;
  return loss;
}

void NegativeLogLikelihoodLoss::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputRange(node, 2, 3);
  RequireOutputCount(node, 1);
  const Tensor &input = GetInput(node, 0, rt.tensors());
  const Tensor &target = GetInput(node, 1, rt.tensors());
  const Tensor *weight = GetOptionalInput(node, 2, rt.tensors());
  const std::string reduction = GetAttributeStringOrDefault(node, "reduction", "mean");
  const bool has_ignore_index = FindAttribute(node, "ignore_index") != nullptr;
  const int64_t ignore_index = GetAttributeIntOrDefault(node, "ignore_index", 0);
  onnx_kernels::kernel::NegativeLogLikelihoodLoss k(rt.kernel_ctx());
  SetOutput(node, 0, k(input, target, weight, reduction, has_ignore_index, ignore_index, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
