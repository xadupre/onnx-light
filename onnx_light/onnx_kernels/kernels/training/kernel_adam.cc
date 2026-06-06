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

constexpr const char *kAdamName = "kernel::Adam";

// Returns the product of all dimensions in ``shape``; ``1`` for a scalar
// (empty shape). Throws on negative dimensions.
int64_t ShapeElementCount(const std::vector<int64_t> &shape, const char *label) {
  int64_t count = 1;
  for (int64_t d : shape) {
    EXT_ENFORCE_INVALID(d >= 0,
                        std::string(kAdamName) + ": '" + label + "' has a negative dimension.");
    count *= d;
  }
  return count;
}

void CheckFloatTensor(const Tensor &t, const char *label) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT,
                      std::string(kAdamName) + ": '" + label + "' must be a FLOAT tensor.");
}

void CheckScalar(const Tensor &t, const char *label) {
  // ONNX scalars may be encoded either with an empty shape or with a
  // single-element shape; accept both for the sake of test robustness.
  const int64_t count = ShapeElementCount(t.shape, label);
  EXT_ENFORCE_INVALID(count == 1, std::string(kAdamName) + ": '" + label +
                                      "' must be a scalar (single-element) tensor.");
}

void CheckSameShape(const Tensor &a, const Tensor &b, const char *label_a, const char *label_b) {
  EXT_ENFORCE_INVALID(a.shape == b.shape, std::string(kAdamName) + ": '" + label_a + "' and '" +
                                              label_b + "' must have the same shape.");
}

} // namespace

std::vector<Tensor> Adam::operator()(const Tensor &R, const Tensor &T,
                                     const std::vector<Tensor> &Xs, const std::vector<Tensor> &Gs,
                                     const std::vector<Tensor> &Vs, const std::vector<Tensor> &Hs,
                                     float alpha, float beta, float epsilon, float norm_coefficient,
                                     float norm_coefficient_post) const {
  EXT_ENFORCE_INVALID(!Xs.empty(),
                      std::string(kAdamName) + ": at least one optimized tensor is required.");
  EXT_ENFORCE_INVALID(Xs.size() == Gs.size() && Xs.size() == Vs.size() && Xs.size() == Hs.size(),
                      std::string(kAdamName) +
                          ": 'Xs', 'Gs', 'Vs' and 'Hs' must have the same length.");

  std::vector<Tensor> outputs;
  outputs.reserve(Xs.size() * 3);
  // Layout: X_final_1..N, V_new_1..N, H_new_1..N.
  for (const auto &X : Xs) {
    outputs.emplace_back("", DataType::FLOAT, X.shape, std::vector<uint8_t>(X.data.size()));
  }
  for (const auto &V : Vs) {
    outputs.emplace_back("", DataType::FLOAT, V.shape, std::vector<uint8_t>(V.data.size()));
  }
  for (const auto &H : Hs) {
    outputs.emplace_back("", DataType::FLOAT, H.shape, std::vector<uint8_t>(H.data.size()));
  }
  (*this)(R, T, Xs, Gs, Vs, Hs, outputs, alpha, beta, epsilon, norm_coefficient,
          norm_coefficient_post);
  return outputs;
}

void Adam::operator()(const Tensor &R, const Tensor &T, const std::vector<Tensor> &Xs,
                      const std::vector<Tensor> &Gs, const std::vector<Tensor> &Vs,
                      const std::vector<Tensor> &Hs, std::vector<Tensor> &outputs, float alpha,
                      float beta, float epsilon, float norm_coefficient,
                      float norm_coefficient_post) const {
  EXT_ENFORCE_INVALID(!Xs.empty(),
                      std::string(kAdamName) + ": at least one optimized tensor is required.");
  EXT_ENFORCE_INVALID(Xs.size() == Gs.size() && Xs.size() == Vs.size() && Xs.size() == Hs.size(),
                      std::string(kAdamName) +
                          ": 'Xs', 'Gs', 'Vs' and 'Hs' must have the same length.");
  const size_t n = Xs.size();
  EXT_ENFORCE_INVALID(outputs.size() == 3 * n,
                      std::string(kAdamName) +
                          " preallocated outputs vector must contain exactly 3 * N tensors.");

  CheckFloatTensor(R, "R");
  CheckScalar(R, "R");
  EXT_ENFORCE_INVALID(T.data_type == DataType::INT64,
                      std::string(kAdamName) + ": 'T' must be an INT64 tensor.");
  CheckScalar(T, "T");

  const float R_val = *R.AsFloat();
  const int64_t T_val = *T.AsInt64();

  // R_adjusted: applies the standard Adam bias correction when T > 0.
  // For T == 0 the un-corrected learning rate is used (matches the ONNX
  // pseudo-code).
  float R_adjusted = R_val;
  if (T_val > 0) {
    const double alpha_pow = std::pow(static_cast<double>(alpha), static_cast<double>(T_val));
    const double beta_pow = std::pow(static_cast<double>(beta), static_cast<double>(T_val));
    const double denom = 1.0 - alpha_pow;
    if (denom == 0.0) {
      throw std::invalid_argument(std::string(kAdamName) +
                                  ": bias correction divides by zero (1 - alpha^T == 0); "
                                  "choose 'alpha' != 1.");
    }
    R_adjusted = static_cast<float>(static_cast<double>(R_val) * std::sqrt(1.0 - beta_pow) / denom);
  }

  for (size_t i = 0; i < n; ++i) {
    const Tensor &X = Xs[i];
    const Tensor &G = Gs[i];
    const Tensor &V = Vs[i];
    const Tensor &H = Hs[i];
    Tensor &X_out = outputs[i];
    Tensor &V_out = outputs[n + i];
    Tensor &H_out = outputs[2 * n + i];

    CheckFloatTensor(X, "X");
    CheckFloatTensor(G, "G");
    CheckFloatTensor(V, "V");
    CheckFloatTensor(H, "H");
    CheckSameShape(X, G, "X", "G");
    CheckSameShape(X, V, "X", "V");
    CheckSameShape(X, H, "X", "H");

    CheckFloatTensor(X_out, "X_new");
    CheckFloatTensor(V_out, "V_new");
    CheckFloatTensor(H_out, "H_new");
    EXT_ENFORCE_INVALID(X_out.shape == X.shape,
                        std::string(kAdamName) + " preallocated 'X_new' shape must match 'X'.");
    EXT_ENFORCE_INVALID(V_out.shape == V.shape,
                        std::string(kAdamName) + " preallocated 'V_new' shape must match 'V'.");
    EXT_ENFORCE_INVALID(H_out.shape == H.shape,
                        std::string(kAdamName) + " preallocated 'H_new' shape must match 'H'.");
    const int64_t count = ShapeElementCount(X.shape, "X");
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    EXT_ENFORCE_INVALID(
        X_out.data.size() == bytes && V_out.data.size() == bytes && H_out.data.size() == bytes,
        std::string(kAdamName) + " preallocated output buffers have unexpected size in bytes.");

    const float *pX = X.AsFloat();
    const float *pG = G.AsFloat();
    const float *pV = V.AsFloat();
    const float *pH = H.AsFloat();
    float *pX_out = X_out.AsFloat();
    float *pV_out = V_out.AsFloat();
    float *pH_out = H_out.AsFloat();

    for (int64_t k = 0; k < count; ++k) {
      const double x = static_cast<double>(pX[k]);
      const double g = static_cast<double>(pG[k]);
      const double v = static_cast<double>(pV[k]);
      const double h = static_cast<double>(pH[k]);

      const double g_reg = static_cast<double>(norm_coefficient) * x + g;
      const double v_new =
          static_cast<double>(alpha) * v + (1.0 - static_cast<double>(alpha)) * g_reg;
      const double h_new =
          static_cast<double>(beta) * h + (1.0 - static_cast<double>(beta)) * g_reg * g_reg;
      const double h_sqrt = std::sqrt(h_new) + static_cast<double>(epsilon);
      const double x_new = x - static_cast<double>(R_adjusted) * v_new / h_sqrt;
      const double x_final = (1.0 - static_cast<double>(norm_coefficient_post)) * x_new;

      pX_out[k] = static_cast<float>(x_final);
      pV_out[k] = static_cast<float>(v_new);
      pH_out[k] = static_cast<float>(h_new);
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
