// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/training/include_training_kernels.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

constexpr const char *kMomentumName = "kernel::Momentum";

int64_t ShapeElementCount(const std::vector<int64_t> &shape, const char *label) {
  int64_t count = 1;
  for (int64_t d : shape) {
    EXT_ENFORCE_INVALID(d >= 0,
                        std::string(kMomentumName) + ": '" + label + "' has a negative dimension.");
    count *= d;
  }
  return count;
}

void CheckFloatTensor(const Tensor &t, const char *label) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT,
                      std::string(kMomentumName) + ": '" + label + "' must be a FLOAT tensor.");
}

void CheckScalar(const Tensor &t, const char *label) {
  const int64_t count = ShapeElementCount(t.shape, label);
  EXT_ENFORCE_INVALID(count == 1, std::string(kMomentumName) + ": '" + label +
                                      "' must be a scalar (single-element) tensor.");
}

void CheckSameShape(const Tensor &a, const Tensor &b, const char *label_a, const char *label_b) {
  EXT_ENFORCE_INVALID(a.shape == b.shape, std::string(kMomentumName) + ": '" + label_a + "' and '" +
                                              label_b + "' must have the same shape.");
}

} // namespace

std::vector<Tensor> Momentum::operator()(const Tensor &R, const Tensor &T,
                                         const std::vector<Tensor> &Xs,
                                         const std::vector<Tensor> &Gs,
                                         const std::vector<Tensor> &Vs, float alpha, float beta,
                                         float norm_coefficient, Mode mode) const {
  EXT_ENFORCE_INVALID(!Xs.empty(),
                      std::string(kMomentumName) + ": at least one optimized tensor is required.");
  EXT_ENFORCE_INVALID(Xs.size() == Gs.size() && Xs.size() == Vs.size(),
                      std::string(kMomentumName) +
                          ": 'Xs', 'Gs' and 'Vs' must have the same length.");

  std::vector<Tensor> outputs;
  outputs.reserve(Xs.size() * 2);
  // Layout: X_new_1..N, V_new_1..N.
  for (const auto &X : Xs) {
    outputs.emplace_back("", DataType::FLOAT, X.shape, std::vector<uint8_t>(X.data.size()));
  }
  for (const auto &V : Vs) {
    outputs.emplace_back("", DataType::FLOAT, V.shape, std::vector<uint8_t>(V.data.size()));
  }
  (*this)(R, T, Xs, Gs, Vs, outputs, alpha, beta, norm_coefficient, mode);
  return outputs;
}

void Momentum::operator()(const Tensor &R, const Tensor &T, const std::vector<Tensor> &Xs,
                          const std::vector<Tensor> &Gs, const std::vector<Tensor> &Vs,
                          std::vector<Tensor> &outputs, float alpha, float beta,
                          float norm_coefficient, Mode mode) const {
  EXT_ENFORCE_INVALID(!Xs.empty(),
                      std::string(kMomentumName) + ": at least one optimized tensor is required.");
  EXT_ENFORCE_INVALID(Xs.size() == Gs.size() && Xs.size() == Vs.size(),
                      std::string(kMomentumName) +
                          ": 'Xs', 'Gs' and 'Vs' must have the same length.");
  const size_t n = Xs.size();
  EXT_ENFORCE_INVALID(outputs.size() == 2 * n,
                      std::string(kMomentumName) +
                          " preallocated outputs vector must contain exactly 2 * N tensors.");

  CheckFloatTensor(R, "R");
  CheckScalar(R, "R");
  EXT_ENFORCE_INVALID(T.data_type == DataType::INT64,
                      std::string(kMomentumName) + ": 'T' must be an INT64 tensor.");
  CheckScalar(T, "T");

  const float R_val = *R.AsFloat();
  const int64_t T_val = *T.AsInt64();

  // The first iteration (T == 0) uses a coefficient of 1 on the regularized
  // gradient; subsequent iterations use ``beta``.
  const double beta_adjusted = (T_val > 0) ? static_cast<double>(beta) : 1.0;

  for (size_t i = 0; i < n; ++i) {
    const Tensor &X = Xs[i];
    const Tensor &G = Gs[i];
    const Tensor &V = Vs[i];
    Tensor &X_out = outputs[i];
    Tensor &V_out = outputs[n + i];

    CheckFloatTensor(X, "X");
    CheckFloatTensor(G, "G");
    CheckFloatTensor(V, "V");
    CheckSameShape(X, G, "X", "G");
    CheckSameShape(X, V, "X", "V");

    CheckFloatTensor(X_out, "X_new");
    CheckFloatTensor(V_out, "V_new");
    EXT_ENFORCE_INVALID(X_out.shape == X.shape,
                        std::string(kMomentumName) + " preallocated 'X_new' shape must match 'X'.");
    EXT_ENFORCE_INVALID(V_out.shape == V.shape,
                        std::string(kMomentumName) + " preallocated 'V_new' shape must match 'V'.");
    const int64_t count = ShapeElementCount(X.shape, "X");
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    EXT_ENFORCE_INVALID(X_out.data.size() == bytes && V_out.data.size() == bytes,
                        std::string(kMomentumName) +
                            " preallocated output buffers have unexpected size in bytes.");

    const float *pX = X.AsFloat();
    const float *pG = G.AsFloat();
    const float *pV = V.AsFloat();
    float *pX_out = X_out.AsFloat();
    float *pV_out = V_out.AsFloat();

    for (int64_t k = 0; k < count; ++k) {
      const double x = static_cast<double>(pX[k]);
      const double g = static_cast<double>(pG[k]);
      const double v = static_cast<double>(pV[k]);

      const double g_reg = static_cast<double>(norm_coefficient) * x + g;
      const double v_new = static_cast<double>(alpha) * v + beta_adjusted * g_reg;
      const double x_new =
          (mode == Mode::kNesterov)
              ? x - static_cast<double>(R_val) * (g_reg + static_cast<double>(alpha) * v_new)
              : x - static_cast<double>(R_val) * v_new;

      pX_out[k] = static_cast<float>(x_new);
      pV_out[k] = static_cast<float>(v_new);
    }
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
