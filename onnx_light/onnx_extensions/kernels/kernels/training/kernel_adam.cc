// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/training/include_training_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_light_helpers.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kAdamName = "kernel::Adam";

// Returns the product of all dimensions in ``shape``; ``1`` for a scalar
// (empty shape). Throws on negative dimensions.
int64_t ShapeElementCount(const onnx_kernels::Shape &shape, const char *label) {
  int64_t count = 1;
  for (int64_t d : shape) {
    EXT_ENFORCE_INVALID(d >= 0, kAdamName, ": '", label, "' has a negative dimension.");
    count *= d;
  }
  return count;
}

void CheckFloatTensor(const Tensor &t, const char *label) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT, kAdamName, ": '", label,
                      "' must be a FLOAT tensor.");
}

void CheckScalar(const Tensor &t, const char *label) {
  // ONNX scalars may be encoded either with an empty shape or with a
  // single-element shape; accept both for the sake of test robustness.
  const int64_t count = ShapeElementCount(t.shape, label);
  EXT_ENFORCE_INVALID(count == 1, kAdamName, ": '", label,
                      "' must be a scalar (single-element) tensor.");
}

void CheckSameShape(const Tensor &a, const Tensor &b, const char *label_a, const char *label_b) {
  EXT_ENFORCE_INVALID(a.shape == b.shape, kAdamName, ": '", label_a, "' and '", label_b,
                      "' must have the same shape.");
}

} // namespace

Tensors Adam::operator()(const Tensor &R, const Tensor &T, const Tensors &Xs, const Tensors &Gs,
                         const Tensors &Vs, const Tensors &Hs, float alpha, float beta,
                         float epsilon, float norm_coefficient, float norm_coefficient_post) const {
  EXT_ENFORCE_INVALID(!Xs.empty(), kAdamName, ": at least one optimized tensor is required.");
  EXT_ENFORCE_INVALID(Xs.size() == Gs.size() && Xs.size() == Vs.size() && Xs.size() == Hs.size(),
                      kAdamName, ": 'Xs', 'Gs', 'Vs' and 'Hs' must have the same length.");

  RawBufferAllocator *allocator = ctx_.allocator;
  Tensors outputs;
  outputs.reserve(Xs.size() * 3);
  // Layout: X_final_1..N, V_new_1..N, H_new_1..N.
  for (const auto &X : Xs) {
    outputs.push_back(MakeOutputTensor(DataType::FLOAT, X.shape, X.size_bytes(), allocator));
  }
  for (const auto &V : Vs) {
    outputs.push_back(MakeOutputTensor(DataType::FLOAT, V.shape, V.size_bytes(), allocator));
  }
  for (const auto &H : Hs) {
    outputs.push_back(MakeOutputTensor(DataType::FLOAT, H.shape, H.size_bytes(), allocator));
  }
  (*this)(R, T, Xs, Gs, Vs, Hs, outputs, alpha, beta, epsilon, norm_coefficient,
          norm_coefficient_post);
  return outputs;
}

void Adam::operator()(const Tensor &R, const Tensor &T, const Tensors &Xs, const Tensors &Gs,
                      const Tensors &Vs, const Tensors &Hs, Tensors &outputs, float alpha,
                      float beta, float epsilon, float norm_coefficient,
                      float norm_coefficient_post) const {
  EXT_ENFORCE_INVALID(!Xs.empty(), kAdamName, ": at least one optimized tensor is required.");
  EXT_ENFORCE_INVALID(Xs.size() == Gs.size() && Xs.size() == Vs.size() && Xs.size() == Hs.size(),
                      kAdamName, ": 'Xs', 'Gs', 'Vs' and 'Hs' must have the same length.");
  const size_t n = Xs.size();
  EXT_ENFORCE_INVALID(outputs.size() == 3 * n, kAdamName,
                      " preallocated outputs vector must contain exactly 3 * N tensors.");

  CheckFloatTensor(R, "R");
  CheckScalar(R, "R");
  EXT_ENFORCE_INVALID(T.data_type == DataType::INT64, kAdamName, ": 'T' must be an INT64 tensor.");
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
    EXT_ENFORCE_INVALID(denom != 0.0, kAdamName,
                        ": bias correction divides by zero (1 - alpha^T == 0); "
                        "choose 'alpha' != 1.");
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
    EXT_ENFORCE_INVALID(X_out.shape == X.shape, kAdamName,
                        " preallocated 'X_new' shape must match 'X'.");
    EXT_ENFORCE_INVALID(V_out.shape == V.shape, kAdamName,
                        " preallocated 'V_new' shape must match 'V'.");
    EXT_ENFORCE_INVALID(H_out.shape == H.shape, kAdamName,
                        " preallocated 'H_new' shape must match 'H'.");
    const int64_t count = ShapeElementCount(X.shape, "X");
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    EXT_ENFORCE_INVALID(X_out.size_bytes() == bytes && V_out.size_bytes() == bytes &&
                            H_out.size_bytes() == bytes,
                        kAdamName, " preallocated output buffers have unexpected size in bytes.");

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

void Adam::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() < 6 || (node.input_size() - 2) % 4 != 0),
                      "RunNode: op 'Adam' expects 2 + 4*N inputs (got ", node.input_size(), ").");
  const int64_t n = (node.input_size() - 2) / 4;
  EXT_ENFORCE_INVALID(node.output_size() == 3 * n, "RunNode: op 'Adam' expects 3*N outputs (got ",
                      node.output_size(), " for N=", n, ").");
  const Tensor &R = GetInput(node, 0, rt.tensors());
  const Tensor &T = GetInput(node, 1, rt.tensors());
  Tensors Xs, Gs, Vs, Hs;
  Xs.reserve(n);
  Gs.reserve(n);
  Vs.reserve(n);
  Hs.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    Xs.push_back(GetInput(node, static_cast<int>(2 + i), rt.tensors()));
    Gs.push_back(GetInput(node, static_cast<int>(2 + n + i), rt.tensors()));
    Vs.push_back(GetInput(node, static_cast<int>(2 + 2 * n + i), rt.tensors()));
    Hs.push_back(GetInput(node, static_cast<int>(2 + 3 * n + i), rt.tensors()));
  }
  const float alpha = GetAttributeFloatOrDefault(node, "alpha", 0.9f);
  const float beta = GetAttributeFloatOrDefault(node, "beta", 0.999f);
  const float epsilon = GetAttributeFloatOrDefault(node, "epsilon", 1e-6f);
  const float norm_coefficient = GetAttributeFloatOrDefault(node, "norm_coefficient", 0.0f);
  const float norm_coefficient_post =
      GetAttributeFloatOrDefault(node, "norm_coefficient_post", 0.0f);
  onnx_kernels::kernel::Adam k(rt.kernel_ctx());
  Tensors outs;
  outs.reserve(static_cast<size_t>(3 * n));
  for (int64_t i = 0; i < n; ++i) {
    const Tensor &X = Xs[static_cast<size_t>(i)];
    outs.push_back(
        rt.MakeOutputTensor(static_cast<int>(i), DataType::FLOAT, X.shape, X.size_bytes()));
  }
  for (int64_t i = 0; i < n; ++i) {
    const Tensor &V = Vs[static_cast<size_t>(i)];
    outs.push_back(
        rt.MakeOutputTensor(static_cast<int>(n + i), DataType::FLOAT, V.shape, V.size_bytes()));
  }
  for (int64_t i = 0; i < n; ++i) {
    const Tensor &H = Hs[static_cast<size_t>(i)];
    outs.push_back(
        rt.MakeOutputTensor(static_cast<int>(2 * n + i), DataType::FLOAT, H.shape, H.size_bytes()));
  }
  k(R, T, Xs, Gs, Vs, Hs, outs, alpha, beta, epsilon, norm_coefficient, norm_coefficient_post);
  for (int64_t i = 0; i < 3 * n; ++i) {
    SetOutput(node, static_cast<int>(i), std::move(outs[static_cast<size_t>(i)]), rt.tensors());
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
