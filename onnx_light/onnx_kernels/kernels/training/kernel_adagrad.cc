// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/training/include_training_kernels.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

constexpr const char *kAdagradName = "kernel::Adagrad";

// Returns the product of all dimensions in ``shape``; ``1`` for a scalar
// (empty shape). Throws on negative dimensions.
int64_t ShapeElementCount(const std::vector<int64_t> &shape, const char *label) {
  int64_t count = 1;
  for (int64_t d : shape) {
    EXT_ENFORCE_INVALID(d >= 0,
                        std::string(kAdagradName) + ": '" + label + "' has a negative dimension.");
    count *= d;
  }
  return count;
}

void CheckFloatTensor(const Tensor &t, const char *label) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT,
                      std::string(kAdagradName) + ": '" + label + "' must be a FLOAT tensor.");
}

void CheckScalar(const Tensor &t, const char *label) {
  // ONNX scalars may be encoded either with an empty shape or with a
  // single-element shape; accept both for the sake of test robustness.
  const int64_t count = ShapeElementCount(t.shape, label);
  EXT_ENFORCE_INVALID(count == 1, std::string(kAdagradName) + ": '" + label +
                                      "' must be a scalar (single-element) tensor.");
}

void CheckSameShape(const Tensor &a, const Tensor &b, const char *label_a, const char *label_b) {
  EXT_ENFORCE_INVALID(a.shape == b.shape, std::string(kAdagradName) + ": '" + label_a + "' and '" +
                                              label_b + "' must have the same shape.");
}

} // namespace

std::vector<Tensor> Adagrad::operator()(const Tensor &R, const Tensor &T,
                                        const std::vector<Tensor> &Xs,
                                        const std::vector<Tensor> &Gs,
                                        const std::vector<Tensor> &Hs, float epsilon,
                                        float decay_factor, float norm_coefficient) const {
  EXT_ENFORCE_INVALID(!Xs.empty(),
                      std::string(kAdagradName) + ": at least one optimized tensor is required.");
  EXT_ENFORCE_INVALID(Xs.size() == Gs.size() && Xs.size() == Hs.size(),
                      std::string(kAdagradName) +
                          ": 'Xs', 'Gs' and 'Hs' must have the same length.");

  std::vector<Tensor> outputs;
  outputs.reserve(Xs.size() * 2);
  // Layout: X_new_1..N, H_new_1..N.
  for (const auto &X : Xs) {
    outputs.emplace_back("", DataType::FLOAT, X.shape, std::vector<uint8_t>(X.data.size()));
  }
  for (const auto &H : Hs) {
    outputs.emplace_back("", DataType::FLOAT, H.shape, std::vector<uint8_t>(H.data.size()));
  }
  (*this)(R, T, Xs, Gs, Hs, outputs, epsilon, decay_factor, norm_coefficient);
  return outputs;
}

void Adagrad::operator()(const Tensor &R, const Tensor &T, const std::vector<Tensor> &Xs,
                         const std::vector<Tensor> &Gs, const std::vector<Tensor> &Hs,
                         std::vector<Tensor> &outputs, float epsilon, float decay_factor,
                         float norm_coefficient) const {
  EXT_ENFORCE_INVALID(!Xs.empty(),
                      std::string(kAdagradName) + ": at least one optimized tensor is required.");
  EXT_ENFORCE_INVALID(Xs.size() == Gs.size() && Xs.size() == Hs.size(),
                      std::string(kAdagradName) +
                          ": 'Xs', 'Gs' and 'Hs' must have the same length.");
  const size_t n = Xs.size();
  EXT_ENFORCE_INVALID(outputs.size() == 2 * n,
                      std::string(kAdagradName) +
                          " preallocated outputs vector must contain exactly 2 * N tensors.");

  CheckFloatTensor(R, "R");
  CheckScalar(R, "R");
  EXT_ENFORCE_INVALID(T.data_type == DataType::INT64,
                      std::string(kAdagradName) + ": 'T' must be an INT64 tensor.");
  CheckScalar(T, "T");

  const float R_val = *R.AsFloat();
  const int64_t T_val = *T.AsInt64();

  // R_adjusted: applies the decay factor based on the update count T.
  const double denom = 1.0 + static_cast<double>(T_val) * static_cast<double>(decay_factor);
  if (denom == 0.0) {
    throw std::invalid_argument(
        std::string(kAdagradName) +
        ": learning-rate decay divides by zero (1 + T * decay_factor == 0).");
  }
  const float R_adjusted = static_cast<float>(static_cast<double>(R_val) / denom);

  for (size_t i = 0; i < n; ++i) {
    const Tensor &X = Xs[i];
    const Tensor &G = Gs[i];
    const Tensor &H = Hs[i];
    Tensor &X_out = outputs[i];
    Tensor &H_out = outputs[n + i];

    CheckFloatTensor(X, "X");
    CheckFloatTensor(G, "G");
    CheckFloatTensor(H, "H");
    CheckSameShape(X, G, "X", "G");
    CheckSameShape(X, H, "X", "H");

    CheckFloatTensor(X_out, "X_new");
    CheckFloatTensor(H_out, "H_new");
    EXT_ENFORCE_INVALID(X_out.shape == X.shape,
                        std::string(kAdagradName) + " preallocated 'X_new' shape must match 'X'.");
    EXT_ENFORCE_INVALID(H_out.shape == H.shape,
                        std::string(kAdagradName) + " preallocated 'H_new' shape must match 'H'.");
    const int64_t count = ShapeElementCount(X.shape, "X");
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    EXT_ENFORCE_INVALID(X_out.data.size() == bytes && H_out.data.size() == bytes,
                        std::string(kAdagradName) +
                            " preallocated output buffers have unexpected size in bytes.");

    const float *pX = X.AsFloat();
    const float *pG = G.AsFloat();
    const float *pH = H.AsFloat();
    float *pX_out = X_out.AsFloat();
    float *pH_out = H_out.AsFloat();

    for (int64_t k = 0; k < count; ++k) {
      const double x = static_cast<double>(pX[k]);
      const double g = static_cast<double>(pG[k]);
      const double h = static_cast<double>(pH[k]);

      const double g_reg = static_cast<double>(norm_coefficient) * x + g;
      const double h_new = h + g_reg * g_reg;
      const double h_sqrt = std::sqrt(h_new) + static_cast<double>(epsilon);
      const double x_new = x - static_cast<double>(R_adjusted) * g_reg / h_sqrt;

      pX_out[k] = static_cast<float>(x_new);
      pH_out[k] = static_cast<float>(h_new);
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
