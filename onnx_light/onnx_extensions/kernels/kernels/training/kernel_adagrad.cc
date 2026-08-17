// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/training/include_training_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_light_helpers.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kAdagradName = "kernel::Adagrad";

// Returns the product of all dimensions in ``shape``; ``1`` for a scalar
// (empty shape). Throws on negative dimensions.
int64_t ShapeElementCount(const onnx_kernels::Shape &shape, const char *label) {
  int64_t count = 1;
  for (int64_t d : shape) {
    EXT_ENFORCE_INVALID(d >= 0, kAdagradName, ": '", label, "' has a negative dimension.");
    count *= d;
  }
  return count;
}

void CheckFloatTensor(const Tensor &t, const char *label) {
  EXT_ENFORCE_INVALID(t.data_type == DataType::FLOAT, kAdagradName, ": '", label,
                      "' must be a FLOAT tensor.");
}

void CheckScalar(const Tensor &t, const char *label) {
  // ONNX scalars may be encoded either with an empty shape or with a
  // single-element shape; accept both for the sake of test robustness.
  const int64_t count = ShapeElementCount(t.shape, label);
  EXT_ENFORCE_INVALID(count == 1, kAdagradName, ": '", label,
                      "' must be a scalar (single-element) tensor.");
}

void CheckSameShape(const Tensor &a, const Tensor &b, const char *label_a, const char *label_b) {
  EXT_ENFORCE_INVALID(a.shape == b.shape, kAdagradName, ": '", label_a, "' and '", label_b,
                      "' must have the same shape.");
}

} // namespace

Tensors Adagrad::operator()(const Tensor &R, const Tensor &T, const Tensors &Xs, const Tensors &Gs,
                            const Tensors &Hs, float epsilon, float decay_factor,
                            float norm_coefficient) const {
  EXT_ENFORCE_INVALID(!Xs.empty(), kAdagradName, ": at least one optimized tensor is required.");
  EXT_ENFORCE_INVALID(Xs.size() == Gs.size() && Xs.size() == Hs.size(), kAdagradName,
                      ": 'Xs', 'Gs' and 'Hs' must have the same length.");

  RawBufferAllocator *allocator = ctx_.allocator;
  Tensors outputs;
  outputs.reserve(Xs.size() * 2);
  // Layout: X_new_1..N, H_new_1..N.
  for (const auto &X : Xs) {
    outputs.push_back(MakeOutputTensor(DataType::FLOAT, X.shape, X.size_bytes(), allocator));
  }
  for (const auto &H : Hs) {
    outputs.push_back(MakeOutputTensor(DataType::FLOAT, H.shape, H.size_bytes(), allocator));
  }
  (*this)(R, T, Xs, Gs, Hs, outputs, epsilon, decay_factor, norm_coefficient);
  return outputs;
}

void Adagrad::operator()(const Tensor &R, const Tensor &T, const Tensors &Xs, const Tensors &Gs,
                         const Tensors &Hs, Tensors &outputs, float epsilon, float decay_factor,
                         float norm_coefficient) const {
  EXT_ENFORCE_INVALID(!Xs.empty(), kAdagradName, ": at least one optimized tensor is required.");
  EXT_ENFORCE_INVALID(Xs.size() == Gs.size() && Xs.size() == Hs.size(), kAdagradName,
                      ": 'Xs', 'Gs' and 'Hs' must have the same length.");
  const size_t n = Xs.size();
  EXT_ENFORCE_INVALID(outputs.size() == 2 * n, kAdagradName,
                      " preallocated outputs vector must contain exactly 2 * N tensors.");

  CheckFloatTensor(R, "R");
  CheckScalar(R, "R");
  EXT_ENFORCE_INVALID(T.data_type == DataType::INT64, kAdagradName,
                      ": 'T' must be an INT64 tensor.");
  CheckScalar(T, "T");

  const float R_val = *R.AsFloat();
  const int64_t T_val = *T.AsInt64();

  // R_adjusted: applies the decay factor based on the update count T.
  const double denom = 1.0 + static_cast<double>(T_val) * static_cast<double>(decay_factor);
  EXT_ENFORCE_INVALID(denom != 0.0, kAdagradName,
                      ": learning-rate decay divides by zero (1 + T * decay_factor == 0).");
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
    EXT_ENFORCE_INVALID(X_out.shape == X.shape, kAdagradName,
                        " preallocated 'X_new' shape must match 'X'.");
    EXT_ENFORCE_INVALID(H_out.shape == H.shape, kAdagradName,
                        " preallocated 'H_new' shape must match 'H'.");
    const int64_t count = ShapeElementCount(X.shape, "X");
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    EXT_ENFORCE_INVALID(X_out.size_bytes() == bytes && H_out.size_bytes() == bytes, kAdagradName,
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

void Adagrad::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() < 5 || (node.input_size() - 2) % 3 != 0),
                      "RunNode: op 'Adagrad' expects 2 + 3*N inputs (got ", node.input_size(),
                      ").");
  const int64_t n = (node.input_size() - 2) / 3;
  EXT_ENFORCE_INVALID(node.output_size() == 2 * n,
                      "RunNode: op 'Adagrad' expects 2*N outputs (got ", node.output_size(),
                      " for N=", n, ").");
  const Tensor &R = GetInput(node, 0, rt.tensors());
  const Tensor &T = GetInput(node, 1, rt.tensors());
  Tensors Xs, Gs, Hs;
  Xs.reserve(n);
  Gs.reserve(n);
  Hs.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    Xs.push_back(GetInput(node, static_cast<int>(2 + i), rt.tensors()));
    Gs.push_back(GetInput(node, static_cast<int>(2 + n + i), rt.tensors()));
    Hs.push_back(GetInput(node, static_cast<int>(2 + 2 * n + i), rt.tensors()));
  }
  const float epsilon = GetAttributeFloatOrDefault(node, "epsilon", 0.0f);
  const float decay_factor = GetAttributeFloatOrDefault(node, "decay_factor", 0.0f);
  const float norm_coefficient = GetAttributeFloatOrDefault(node, "norm_coefficient", 0.0f);
  onnx_kernels::kernel::Adagrad k(rt.kernel_ctx());
  Tensors outs;
  outs.reserve(static_cast<size_t>(2 * n));
  for (int64_t i = 0; i < n; ++i) {
    const Tensor &X = Xs[static_cast<size_t>(i)];
    outs.push_back(
        rt.MakeOutputTensor(static_cast<int>(i), DataType::FLOAT, X.shape, X.size_bytes()));
  }
  for (int64_t i = 0; i < n; ++i) {
    const Tensor &H = Hs[static_cast<size_t>(i)];
    outs.push_back(
        rt.MakeOutputTensor(static_cast<int>(n + i), DataType::FLOAT, H.shape, H.size_bytes()));
  }
  k(R, T, Xs, Gs, Hs, outs, epsilon, decay_factor, norm_coefficient);
  for (int64_t i = 0; i < 2 * n; ++i) {
    SetOutput(node, static_cast<int>(i), std::move(outs[static_cast<size_t>(i)]), rt.tensors());
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
