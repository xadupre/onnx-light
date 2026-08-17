// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_extensions/kernels/kernels/traditionalml/kernel_svm_common.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Computes raw decision scores directly into the caller-provided ``out`` buffer
// (shape ``[sample_count, score_class_count]``), which is backed by the
// runtime allocator through :cpp:func:`MakeOutputTensor`.
//
// ``coefficients`` is flat row-major of shape ``[raw_class_count,
// feature_count]`` and ``intercepts`` is either empty or of length
// ``raw_class_count``. When ``binary_expand`` is true (``raw_class_count == 1``
// but two class labels are declared) the single raw score ``z`` is expanded to
// the canonical pair ``[-z, z]`` following the ONNX spec, so ``out`` has two
// columns per sample.
void ComputeLinearScoresInto(const double *x_values, int64_t sample_count, int64_t feature_count,
                             const ParamFloats &coefficients, const ParamFloats &intercepts,
                             int64_t raw_class_count, bool binary_expand, float *out) {
  EXT_ENFORCE_INVALID(raw_class_count >= 1,
                      "kernel::LinearClassifier requires at least one class.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(coefficients.size()) == raw_class_count * feature_count,
                      "kernel::LinearClassifier coefficients size must be "
                      "class_count * feature_count.");
  EXT_ENFORCE_INVALID(intercepts.empty() ||
                          static_cast<int64_t>(intercepts.size()) == raw_class_count,
                      "kernel::LinearClassifier intercepts size must be 0 or equal to "
                      "class_count.");
  const int64_t out_class_count = binary_expand ? 2 : raw_class_count;
  for (int64_t n = 0; n < sample_count; ++n) {
    const double *x_row = x_values + n * feature_count;
    float *out_row = out + n * out_class_count;
    for (int64_t c = 0; c < raw_class_count; ++c) {
      const float *w = coefficients.data() + c * feature_count;
      double value = 0.0;
      for (int64_t j = 0; j < feature_count; ++j) {
        value += x_row[j] * static_cast<double>(w[j]);
      }
      if (!intercepts.empty()) {
        value += static_cast<double>(intercepts[static_cast<size_t>(c)]);
      }
      if (binary_expand) {
        out_row[0] = -static_cast<float>(value);
        out_row[1] = static_cast<float>(value);
      } else {
        out_row[c] = static_cast<float>(value);
      }
    }
  }
}

int64_t ArgMax(const float *scores, int64_t count) {
  int64_t best = 0;
  for (int64_t i = 1; i < count; ++i) {
    if (scores[i] > scores[best]) {
      best = i;
    }
  }
  return best;
}

} // namespace

template <typename T>
std::pair<Tensor, Tensor>
LinearClassifier::operator()(const Tensor &x, const ParamFloats &coefficients,
                             const ParamFloats &intercepts,
                             const std::vector<int64_t> &class_labels,
                             const std::string &post_transform, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(post_transform == "NONE",
                      "kernel::LinearClassifier only supports post_transform == 'NONE'.");
  EXT_ENFORCE_INVALID(!class_labels.empty(),
                      "kernel::LinearClassifier requires non-empty class labels.");
  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);
  const int64_t raw_class_count =
      feature_count == 0 ? 0 : static_cast<int64_t>(coefficients.size()) / feature_count;
  RawBufferAllocator *execution_allocator = rt ? rt->execution_allocator() : ctx_.allocator;
  const Tensor x_values =
      ToDoubleRowMajorTensor<T>(x, sample_count, feature_count, execution_allocator);
  const bool binary_expand = raw_class_count == 1 && class_labels.size() == 2;
  const int64_t score_class_count = binary_expand ? 2 : raw_class_count;
  EXT_ENFORCE_INVALID(static_cast<int64_t>(class_labels.size()) == score_class_count,
                      "kernel::LinearClassifier class_labels size must match the number of "
                      "score columns.");

  const onnx_kernels::Shape score_shape = {sample_count, score_class_count};
  const size_t score_n_bytes =
      PackedByteSize(TensorElementType<float>::value, sample_count * score_class_count);
  Tensor z =
      rt ? rt->MakeOutputTensor(1, TensorElementType<float>::value, score_shape, score_n_bytes)
         : MakeOutputTensor(TensorElementType<float>::value, score_shape, score_n_bytes,
                            ctx_.allocator);
  ComputeLinearScoresInto(x_values.As<double>(), sample_count, feature_count, coefficients,
                          intercepts, raw_class_count, binary_expand, z.As<float>());

  const float *scores = z.As<float>();
  std::vector<int64_t> labels(static_cast<size_t>(sample_count));
  for (int64_t n = 0; n < sample_count; ++n) {
    const int64_t idx = ArgMax(scores + n * score_class_count, score_class_count);
    labels[static_cast<size_t>(n)] = class_labels[static_cast<size_t>(idx)];
  }
  Tensor y = rt ? rt->MakeOutputTensor(0, DataType::INT64, {sample_count},
                                       static_cast<size_t>(sample_count) * sizeof(int64_t))
                : Tensor::FromInt64("", {sample_count}, labels, ctx_.allocator);
  if (rt != nullptr) {
    std::copy(labels.begin(), labels.end(), y.AsInt64());
  }
  return std::make_pair(std::move(y), std::move(z));
}

template <typename T>
std::pair<Tensor, Tensor>
LinearClassifier::operator()(const Tensor &x, const ParamFloats &coefficients,
                             const ParamFloats &intercepts, const ParamStrings &class_labels,
                             const std::string &post_transform, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(post_transform == "NONE",
                      "kernel::LinearClassifier only supports post_transform == 'NONE'.");
  EXT_ENFORCE_INVALID(!class_labels.empty(),
                      "kernel::LinearClassifier requires non-empty class labels.");
  int64_t sample_count = 0;
  int64_t feature_count = 0;
  ValidateFeatureMatrixShape(x, sample_count, feature_count);
  const int64_t raw_class_count =
      feature_count == 0 ? 0 : static_cast<int64_t>(coefficients.size()) / feature_count;
  RawBufferAllocator *execution_allocator = rt ? rt->execution_allocator() : ctx_.allocator;
  const Tensor x_values =
      ToDoubleRowMajorTensor<T>(x, sample_count, feature_count, execution_allocator);
  const bool binary_expand = raw_class_count == 1 && class_labels.size() == 2;
  const int64_t score_class_count = binary_expand ? 2 : raw_class_count;
  EXT_ENFORCE_INVALID(static_cast<int64_t>(class_labels.size()) == score_class_count,
                      "kernel::LinearClassifier class_labels size must match the number of "
                      "score columns.");

  const onnx_kernels::Shape score_shape = {sample_count, score_class_count};
  const size_t score_n_bytes =
      PackedByteSize(TensorElementType<float>::value, sample_count * score_class_count);
  Tensor z =
      rt ? rt->MakeOutputTensor(1, TensorElementType<float>::value, score_shape, score_n_bytes)
         : MakeOutputTensor(TensorElementType<float>::value, score_shape, score_n_bytes,
                            ctx_.allocator);
  ComputeLinearScoresInto(x_values.As<double>(), sample_count, feature_count, coefficients,
                          intercepts, raw_class_count, binary_expand, z.As<float>());

  const float *scores = z.As<float>();
  std::vector<std::string> labels(static_cast<size_t>(sample_count));
  for (int64_t n = 0; n < sample_count; ++n) {
    const int64_t idx = ArgMax(scores + n * score_class_count, score_class_count);
    labels[static_cast<size_t>(n)] = class_labels[static_cast<size_t>(idx)];
  }
  Tensor y = rt ? rt->MakeOutputTensor(0, DataType::STRING, {sample_count}, 0)
                : Tensor::FromStrings("", {sample_count}, labels);
  if (rt != nullptr) {
    y.string_data = std::move(labels);
  }
  return std::make_pair(std::move(y), std::move(z));
}

#define ONNX_LIGHT_INSTANTIATE_LINEAR_CLASSIFIER(T)                                                \
  template std::pair<Tensor, Tensor> LinearClassifier::operator()<T>(                              \
      const Tensor &, const ParamFloats &, const ParamFloats &, const std::vector<int64_t> &,      \
      const std::string &, RuntimeContext *) const;                                                \
  template std::pair<Tensor, Tensor> LinearClassifier::operator()<T>(                              \
      const Tensor &, const ParamFloats &, const ParamFloats &, const ParamStrings &,              \
      const std::string &, RuntimeContext *) const

ONNX_LIGHT_INSTANTIATE_LINEAR_CLASSIFIER(float);
ONNX_LIGHT_INSTANTIATE_LINEAR_CLASSIFIER(double);
ONNX_LIGHT_INSTANTIATE_LINEAR_CLASSIFIER(int64_t);
ONNX_LIGHT_INSTANTIATE_LINEAR_CLASSIFIER(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_LINEAR_CLASSIFIER

void LinearClassifier::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 2);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const ParamFloats coefficients = GetAttributeFloatsOrDefault(node, "coefficients", {});
  const ParamFloats intercepts = GetAttributeFloatsOrDefault(node, "intercepts", {});
  const std::string post_transform = GetAttributeStringOrDefault(node, "post_transform", "NONE");
  const std::vector<int64_t> classlabels_ints =
      GetAttributeIntsOrDefault(node, "classlabels_ints", {});
  const ParamStrings classlabels_strings =
      GetAttributeStringsOrDefault(node, "classlabels_strings", {});
  const bool use_strings = !classlabels_strings.empty();
  const bool has_ints = !classlabels_ints.empty();
  EXT_ENFORCE_INVALID(use_strings != has_ints,
                      "RunNode: LinearClassifier requires exactly one of 'classlabels_ints' or "
                      "'classlabels_strings' to be set.");
  onnx_kernels::kernel::LinearClassifier cls(rt.kernel_ctx());
  std::pair<Tensor, Tensor> yz = DispatchSVMByDataType(x, "LinearClassifier", [&](auto *tag) {
    using T = std::remove_pointer_t<decltype(tag)>;
    (void)tag;
    return use_strings ? cls.template operator()<T>(x, coefficients, intercepts,
                                                    classlabels_strings, post_transform, &rt)
                       : cls.template operator()<T>(x, coefficients, intercepts, classlabels_ints,
                                                    post_transform, &rt);
  });
  SetOutput(node, 0, std::move(yz.first), rt);
  SetOutput(node, 1, std::move(yz.second), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
