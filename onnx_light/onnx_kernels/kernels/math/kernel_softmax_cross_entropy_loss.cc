// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Compute log-softmax along the class axis (axis=1) of a tensor with shape
// (N, C, D1, ..., Dk). The output buffer has the same layout as the input.
void ComputeLogSoftmaxOverClassAxis(const float *scores, float *log_prob, int64_t n_outer,
                                    int64_t n_classes, int64_t n_inner) {
  for (int64_t o = 0; o < n_outer; ++o) {
    for (int64_t i = 0; i < n_inner; ++i) {
      float max_v = -std::numeric_limits<float>::infinity();
      for (int64_t c = 0; c < n_classes; ++c) {
        const int64_t offset = (o * n_classes + c) * n_inner + i;
        max_v = std::max(max_v, scores[static_cast<size_t>(offset)]);
      }
      float sum = 0.0f;
      for (int64_t c = 0; c < n_classes; ++c) {
        const int64_t offset = (o * n_classes + c) * n_inner + i;
        sum += std::exp(scores[static_cast<size_t>(offset)] - max_v);
      }
      const float log_sum = std::log(sum);
      for (int64_t c = 0; c < n_classes; ++c) {
        const int64_t offset = (o * n_classes + c) * n_inner + i;
        log_prob[static_cast<size_t>(offset)] =
            (scores[static_cast<size_t>(offset)] - max_v) - log_sum;
      }
    }
  }
}

} // namespace

std::pair<Tensor, Tensor>
SoftmaxCrossEntropyLoss::operator()(const Tensor &scores, const Tensor &labels,
                                    const Tensor *weights, const std::string &reduction,
                                    bool has_ignore_index, int64_t ignore_index) const {
  EXT_ENFORCE_INVALID(scores.data_type == DataType::FLOAT,
                      "kernel::SoftmaxCrossEntropyLoss only supports FLOAT scores.");
  EXT_ENFORCE_INVALID(labels.data_type == DataType::INT32 || labels.data_type == DataType::INT64,
                      "kernel::SoftmaxCrossEntropyLoss labels must be INT32 or INT64.");
  EXT_ENFORCE_INVALID(scores.shape.size() >= 2,
                      "kernel::SoftmaxCrossEntropyLoss scores rank must be >= 2.");
  EXT_ENFORCE_INVALID(labels.shape.size() + 1 == scores.shape.size(),
                      "kernel::SoftmaxCrossEntropyLoss labels rank must equal scores rank - 1.");
  EXT_ENFORCE_INVALID(reduction == "none" || reduction == "sum" || reduction == "mean",
                      "kernel::SoftmaxCrossEntropyLoss: reduction must be one of "
                      "'none', 'sum', 'mean'.");

  const int64_t n_batch = scores.shape[0];
  const int64_t n_classes = scores.shape[1];
  int64_t n_inner = 1;
  for (size_t d = 2; d < scores.shape.size(); ++d) {
    n_inner *= scores.shape[d];
  }
  const int64_t n_loss = n_batch * n_inner;
  EXT_ENFORCE_INVALID(labels.element_count() == n_loss,
                      "kernel::SoftmaxCrossEntropyLoss labels element count is inconsistent "
                      "with scores shape.");

  if (weights != nullptr) {
    EXT_ENFORCE_INVALID(weights->data_type == DataType::FLOAT,
                        "kernel::SoftmaxCrossEntropyLoss weights must be FLOAT.");
    EXT_ENFORCE_INVALID(weights->shape.size() == 1 && weights->shape[0] == n_classes,
                        "kernel::SoftmaxCrossEntropyLoss weights must have shape (C).");
  }

  // Compute log-softmax along the class axis once; reused for the optional
  // ``log_prob`` output and for indexing the loss.
  Tensor log_prob(
      "", DataType::FLOAT, scores.shape,
      std::vector<uint8_t>(static_cast<size_t>(scores.element_count()) * sizeof(float)));
  ComputeLogSoftmaxOverClassAxis(scores.AsFloat(), log_prob.AsFloat(), n_batch, n_classes, n_inner);

  const float *log_prob_ptr = log_prob.AsFloat();
  const float *weights_ptr = weights != nullptr ? weights->AsFloat() : nullptr;

  // Per-sample loss and weight, prior to reduction.
  std::vector<float> per_sample_loss(static_cast<size_t>(n_loss), 0.0f);
  std::vector<float> per_sample_weight(static_cast<size_t>(n_loss), 0.0f);

  for (int64_t o = 0; o < n_batch; ++o) {
    for (int64_t i = 0; i < n_inner; ++i) {
      const int64_t flat = o * n_inner + i;
      int64_t label_value;
      if (labels.data_type == DataType::INT64) {
        label_value = labels.AsInt64()[static_cast<size_t>(flat)];
      } else {
        label_value = static_cast<int64_t>(labels.AsInt32()[static_cast<size_t>(flat)]);
      }

      if (has_ignore_index && label_value == ignore_index) {
        per_sample_loss[static_cast<size_t>(flat)] = 0.0f;
        per_sample_weight[static_cast<size_t>(flat)] = 0.0f;
        continue;
      }

      EXT_ENFORCE_INVALID(label_value >= 0 && label_value < n_classes,
                          "kernel::SoftmaxCrossEntropyLoss: label is out of range [0, C).");

      const int64_t log_prob_offset = (o * n_classes + label_value) * n_inner + i;
      const float neg_log_prob = -log_prob_ptr[static_cast<size_t>(log_prob_offset)];
      const float w = weights_ptr != nullptr ? weights_ptr[static_cast<size_t>(label_value)] : 1.0f;
      per_sample_loss[static_cast<size_t>(flat)] = w * neg_log_prob;
      per_sample_weight[static_cast<size_t>(flat)] = w;
    }
  }

  if (reduction == "none") {
    Tensor loss("", DataType::FLOAT, labels.shape,
                std::vector<uint8_t>(static_cast<size_t>(n_loss) * sizeof(float)));
    float *out = loss.AsFloat();
    for (int64_t k = 0; k < n_loss; ++k) {
      out[static_cast<size_t>(k)] = per_sample_loss[static_cast<size_t>(k)];
    }
    return std::make_pair(std::move(loss), std::move(log_prob));
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
    // "mean": divide by the sum of selected weights. When ``weights`` is not
    // provided, ``per_sample_weight`` is 1.0 for every contributing sample,
    // so ``sum_weight`` equals the number of contributing samples.
    reduced = sum_weight != 0.0f ? sum_loss / sum_weight : 0.0f;
  }

  Tensor loss("", DataType::FLOAT, std::vector<int64_t>{}, std::vector<uint8_t>(sizeof(float)));
  loss.AsFloat()[0] = reduced;
  return std::make_pair(std::move(loss), std::move(log_prob));
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
