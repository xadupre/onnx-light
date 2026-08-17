// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/training/include_training_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include <cstddef>
#include <cstdint>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kMomentumName = "kernel::Momentum";

int64_t ShapeElementCount(const onnx_kernels::Shape &shape, const char *label) {
  int64_t count = 1;
  for (int64_t d : shape) {
    EXT_ENFORCE_INVALID(d >= 0, kMomentumName, ": '", label, "' has a negative dimension.");
    count *= d;
  }
  return count;
}

void CheckFloatTensor(const Tensor &t, const char *label) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT, kMomentumName, ": '", label,
                      "' must be a FLOAT tensor.");
}

void CheckScalar(const Tensor &t, const char *label) {
  const int64_t count = ShapeElementCount(t.shape, label);
  EXT_ENFORCE_INVALID(count == 1, kMomentumName, ": '", label,
                      "' must be a scalar (single-element) tensor.");
}

void CheckSameShape(const Tensor &a, const Tensor &b, const char *label_a, const char *label_b) {
  EXT_ENFORCE_INVALID(a.shape == b.shape, kMomentumName, ": '", label_a, "' and '", label_b,
                      "' must have the same shape.");
}

} // namespace

Tensors Momentum::operator()(const Tensor &R, const Tensor &T, const Tensors &Xs, const Tensors &Gs,
                             const Tensors &Vs, float alpha, float beta, float norm_coefficient,
                             Mode mode) const {
  EXT_ENFORCE_INVALID(!Xs.empty(), kMomentumName, ": at least one optimized tensor is required.");
  EXT_ENFORCE_INVALID(Xs.size() == Gs.size() && Xs.size() == Vs.size(), kMomentumName,
                      ": 'Xs', 'Gs' and 'Vs' must have the same length.");

  RawBufferAllocator *allocator = ctx_.allocator;
  Tensors outputs;
  outputs.reserve(Xs.size() * 2);
  // Layout: X_new_1..N, V_new_1..N.
  for (const auto &X : Xs) {
    outputs.push_back(MakeOutputTensor(DataType::FLOAT, X.shape, X.size_bytes(), allocator));
  }
  for (const auto &V : Vs) {
    outputs.push_back(MakeOutputTensor(DataType::FLOAT, V.shape, V.size_bytes(), allocator));
  }
  (*this)(R, T, Xs, Gs, Vs, outputs, alpha, beta, norm_coefficient, mode);
  return outputs;
}

void Momentum::operator()(const Tensor &R, const Tensor &T, const Tensors &Xs, const Tensors &Gs,
                          const Tensors &Vs, Tensors &outputs, float alpha, float beta,
                          float norm_coefficient, Mode mode) const {
  EXT_ENFORCE_INVALID(!Xs.empty(), kMomentumName, ": at least one optimized tensor is required.");
  EXT_ENFORCE_INVALID(Xs.size() == Gs.size() && Xs.size() == Vs.size(), kMomentumName,
                      ": 'Xs', 'Gs' and 'Vs' must have the same length.");
  const size_t n = Xs.size();
  EXT_ENFORCE_INVALID(outputs.size() == 2 * n, kMomentumName,
                      " preallocated outputs vector must contain exactly 2 * N tensors.");

  CheckFloatTensor(R, "R");
  CheckScalar(R, "R");
  EXT_ENFORCE_INVALID(T.data_type == DataType::INT64, kMomentumName,
                      ": 'T' must be an INT64 tensor.");
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
    EXT_ENFORCE_INVALID(X_out.shape == X.shape, kMomentumName,
                        " preallocated 'X_new' shape must match 'X'.");
    EXT_ENFORCE_INVALID(V_out.shape == V.shape, kMomentumName,
                        " preallocated 'V_new' shape must match 'V'.");
    const int64_t count = ShapeElementCount(X.shape, "X");
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    EXT_ENFORCE_INVALID(X_out.size_bytes() == bytes && V_out.size_bytes() == bytes, kMomentumName,
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

void Momentum::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() < 5 || (node.input_size() - 2) % 3 != 0),
                      "RunNode: op 'Momentum' expects 2 + 3*N inputs (got ", node.input_size(),
                      ").");
  const int64_t n = (node.input_size() - 2) / 3;
  EXT_ENFORCE_INVALID(node.output_size() == 2 * n,
                      "RunNode: op 'Momentum' expects 2*N outputs (got ", node.output_size(),
                      " for N=", n, ").");
  const Tensor &R = GetInput(node, 0, rt.tensors());
  const Tensor &T = GetInput(node, 1, rt.tensors());
  Tensors Xs, Gs, Vs;
  Xs.reserve(n);
  Gs.reserve(n);
  Vs.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    Xs.push_back(GetInput(node, static_cast<int>(2 + i), rt.tensors()));
    Gs.push_back(GetInput(node, static_cast<int>(2 + n + i), rt.tensors()));
    Vs.push_back(GetInput(node, static_cast<int>(2 + 2 * n + i), rt.tensors()));
  }
  const float alpha = GetAttributeFloatOrDefault(node, "alpha", 0.0f);
  const float beta = GetAttributeFloatOrDefault(node, "beta", 0.0f);
  const float norm_coefficient = GetAttributeFloatOrDefault(node, "norm_coefficient", 0.0f);
  const std::string mode_str = GetAttributeStringOrDefault(node, "mode", "standard");
  onnx_kernels::kernel::Momentum::Mode mode;
  if (mode_str == "standard") {
    mode = onnx_kernels::kernel::Momentum::Mode::kStandard;
  } else if (mode_str == "nesterov") {
    mode = onnx_kernels::kernel::Momentum::Mode::kNesterov;
  } else {
    EXT_THROW_INVALID("RunNode: Momentum 'mode' must be 'standard' or 'nesterov', got '", mode_str,
                      "'.");
  }
  onnx_kernels::kernel::Momentum k(rt.kernel_ctx());
  Tensors outs;
  outs.reserve(static_cast<size_t>(2 * n));
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
  k(R, T, Xs, Gs, Vs, outs, alpha, beta, norm_coefficient, mode);
  for (int64_t i = 0; i < 2 * n; ++i) {
    SetOutput(node, static_cast<int>(i), std::move(outs[static_cast<size_t>(i)]), rt.tensors());
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
