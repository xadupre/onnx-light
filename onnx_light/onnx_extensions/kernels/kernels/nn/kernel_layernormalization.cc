// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Returns ``axis`` normalized to the ``[0, rank]`` range. ``rank`` is a
// legal value: it means "reduce nothing", which still requires a degenerate
// loop but matches the upstream semantics.
int64_t NormalizeAxis(int64_t axis, int64_t rank) {
  if (axis < 0) {
    axis += rank;
  }
  EXT_ENFORCE_INVALID(axis >= 0 && axis <= rank,
                      "kernel::LayerNormalization: axis out of range for X's rank.");
  return axis;
}

// Validates that ``param``'s shape is unidirectionally broadcastable to the
// normalized shape ``X.shape[axis:]``.
void CheckParamBroadcast(const Shape &x_shape, int64_t axis, const Shape &param_shape,
                         const char *param_name) {
  const int64_t normalized_rank = static_cast<int64_t>(x_shape.size()) - axis;
  EXT_ENFORCE_INVALID(static_cast<int64_t>(param_shape.size()) <= normalized_rank,
                      "kernel::LayerNormalization: ", param_name,
                      " rank cannot exceed normalized rank.");
  const int64_t offset = normalized_rank - static_cast<int64_t>(param_shape.size());
  for (size_t i = 0; i < param_shape.size(); ++i) {
    const int64_t x_dim = x_shape[static_cast<size_t>(axis + offset) + i];
    const int64_t p_dim = param_shape[i];
    EXT_ENFORCE_INVALID(p_dim == x_dim || p_dim == 1, "kernel::LayerNormalization: ", param_name,
                        " shape is not broadcastable to X's normalized shape.");
  }
}

// Returns the broadcast strides of a parameter tensor of shape ``param_shape``
// over the normalized region ``x_shape[axis:]``. The returned Shape has rank
// ``rank(x_shape) - axis`` (the normalized rank, always <= rank(x_shape)); the
// entry for normalized dimension ``d`` is the stride into the parameter tensor,
// or 0 when that dimension is broadcast (``param_shape`` dimension equal to 1 or
// absent because ``param_shape`` has a smaller rank than the normalized region).
//
// Combined with a row-major coordinate over ``x_shape[axis:]``, these strides
// yield the flat index into the broadcast parameter tensor for each position.
Shape BuildBroadcastIndex(const Shape &x_shape, int64_t axis, const Shape &param_shape) {
  const int64_t normalized_rank = static_cast<int64_t>(x_shape.size()) - axis;
  const int64_t param_rank = static_cast<int64_t>(param_shape.size());
  const int64_t offset = normalized_rank - param_rank;

  Shape strides;
  strides.assign(static_cast<size_t>(normalized_rank), 0);
  if (param_rank > 0) {
    int64_t stride = 1;
    for (int64_t i = param_rank - 1; i >= 0; --i) {
      const int64_t dim = param_shape[static_cast<size_t>(i)];
      strides[static_cast<size_t>(i + offset)] = dim == 1 ? 0 : stride;
      stride *= dim;
    }
  }
  return strides;
}

// Returns the shape ``[d[0], ..., d[axis-1], 1, ..., 1]`` (rank ==
// rank(x_shape)) used for ``Mean`` and ``InvStdDev``.
Shape ReducedShape(const Shape &x_shape, int64_t axis) {
  Shape shape(x_shape);
  for (int64_t i = axis; i < static_cast<int64_t>(shape.size()); ++i) {
    shape[static_cast<size_t>(i)] = 1;
  }
  return shape;
}

} // namespace

std::tuple<Tensor, Tensor, Tensor>
LayerNormalization::operator()(const Tensor &x, const Tensor &scale, const Tensor &b, int64_t axis,
                               float epsilon, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LayerNormalization: X must be FLOAT.");
  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank >= 1, "kernel::LayerNormalization: X must have rank >= 1.");
  const int64_t normalized_axis = NormalizeAxis(axis, rank);
  const Shape reduced_shape = ReducedShape(x.shape, normalized_axis);
  int64_t reduced_elem = 1;
  for (int64_t d : reduced_shape) {
    reduced_elem *= d;
  }
  const size_t reduced_bytes = static_cast<size_t>(reduced_elem) * sizeof(float);

  const size_t y_n_bytes = x.size_bytes();
  Tensor y =
      rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::FLOAT), x.shape, y_n_bytes)
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), x.shape, y_n_bytes, nullptr);
  const bool has_mean_output =
      rt == nullptr || rt->output_slot_io_roles().empty() || rt->output_slot_io_roles().size() > 1;
  const bool has_inv_std_output =
      rt == nullptr || rt->output_slot_io_roles().empty() || rt->output_slot_io_roles().size() > 2;
  const size_t mean_n_bytes = reduced_bytes;
  Tensor mean =
      rt ? (has_mean_output ? rt->MakeOutputTensor(1, static_cast<int32_t>(DataType::FLOAT),
                                                   reduced_shape, mean_n_bytes)
                            : rt->MakeTemporaryTensor(static_cast<int32_t>(DataType::FLOAT),
                                                      reduced_shape, mean_n_bytes))
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), reduced_shape, mean_n_bytes,
                            nullptr);
  const size_t inv_std_dev_n_bytes = reduced_bytes;
  Tensor inv_std_dev =
      rt ? (has_inv_std_output ? rt->MakeOutputTensor(2, static_cast<int32_t>(DataType::FLOAT),
                                                      reduced_shape, inv_std_dev_n_bytes)
                               : rt->MakeTemporaryTensor(static_cast<int32_t>(DataType::FLOAT),
                                                         reduced_shape, inv_std_dev_n_bytes))
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), reduced_shape,
                            inv_std_dev_n_bytes, nullptr);
  (*this)(x, scale, b, y, mean, inv_std_dev, axis, epsilon);
  return {std::move(y), std::move(mean), std::move(inv_std_dev)};
}

void LayerNormalization::operator()(const Tensor &x, const Tensor &scale, const Tensor &b,
                                    Tensor &y, Tensor &mean, Tensor &inv_std_dev, int64_t axis,
                                    float epsilon) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LayerNormalization: X must be FLOAT.");
  EXT_ENFORCE_INVALID(scale.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LayerNormalization: Scale must be FLOAT.");
  EXT_ENFORCE_INVALID(y.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LayerNormalization: Y must be FLOAT.");
  EXT_ENFORCE_INVALID(mean.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LayerNormalization: Mean must be FLOAT.");
  EXT_ENFORCE_INVALID(inv_std_dev.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LayerNormalization: InvStdDev must be FLOAT.");
  EXT_ENFORCE_INVALID(y.shape == x.shape,
                      "kernel::LayerNormalization: Y must have the same shape as X.");
  EXT_ENFORCE_INVALID(y.size_bytes() == x.size_bytes(),
                      "kernel::LayerNormalization: Y buffer must have the same byte size as X.");

  const bool has_bias = !b.shape.empty() || b.size_bytes() > 0;
  if (has_bias) {
    EXT_ENFORCE_INVALID(b.data_type == static_cast<int32_t>(DataType::FLOAT),
                        "kernel::LayerNormalization: B must be FLOAT.");
  }

  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank >= 1, "kernel::LayerNormalization: X must have rank >= 1.");
  axis = NormalizeAxis(axis, rank);

  CheckParamBroadcast(x.shape, axis, scale.shape, "Scale");
  if (has_bias) {
    CheckParamBroadcast(x.shape, axis, b.shape, "B");
  }

  const Shape reduced_shape = ReducedShape(x.shape, axis);
  EXT_ENFORCE_INVALID(mean.shape == reduced_shape,
                      "kernel::LayerNormalization: Mean shape must equal X's reduced shape.");
  EXT_ENFORCE_INVALID(inv_std_dev.shape == reduced_shape,
                      "kernel::LayerNormalization: InvStdDev shape must equal X's reduced shape.");

  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) {
    outer *= x.shape[static_cast<size_t>(i)];
  }
  int64_t norm_size = 1;
  for (int64_t i = axis; i < rank; ++i) {
    norm_size *= x.shape[static_cast<size_t>(i)];
  }

  const int64_t normalized_rank = rank - axis;
  Shape norm_dims;
  norm_dims.assign(static_cast<size_t>(normalized_rank), 0);
  for (int64_t i = 0; i < normalized_rank; ++i) {
    norm_dims[static_cast<size_t>(i)] = x.shape[static_cast<size_t>(axis + i)];
  }

  const Shape scale_strides = BuildBroadcastIndex(x.shape, axis, scale.shape);
  const Shape bias_strides = has_bias ? BuildBroadcastIndex(x.shape, axis, b.shape) : Shape();

  const float *px = x.AsFloat();
  const float *ps = scale.AsFloat();
  const float *pb = has_bias ? b.AsFloat() : nullptr;
  float *py = y.AsFloat();
  float *pmean = mean.AsFloat();
  float *pinv = inv_std_dev.AsFloat();

  for (int64_t o = 0; o < outer; ++o) {
    const int64_t base = o * norm_size;
    double sum = 0.0;
    for (int64_t i = 0; i < norm_size; ++i) {
      sum += static_cast<double>(px[base + i]);
    }
    const double m = norm_size > 0 ? sum / static_cast<double>(norm_size) : 0.0;
    double sqdiff = 0.0;
    for (int64_t i = 0; i < norm_size; ++i) {
      const double d = static_cast<double>(px[base + i]) - m;
      sqdiff += d * d;
    }
    const double var = norm_size > 0 ? sqdiff / static_cast<double>(norm_size) : 0.0;
    const float inv = 1.0f / std::sqrt(static_cast<float>(var) + epsilon);
    pmean[o] = static_cast<float>(m);
    pinv[o] = inv;

    // Row-major "odometer" over ``x_shape[axis:]``, maintaining the flat index
    // into the (broadcast) Scale and B tensors incrementally via their strides.
    Shape coord;
    coord.assign(static_cast<size_t>(normalized_rank), 0);
    int64_t si = 0;
    int64_t bi = 0;
    for (int64_t i = 0; i < norm_size; ++i) {
      const float normalized = (px[base + i] - static_cast<float>(m)) * inv;
      float v = normalized * ps[si];
      if (has_bias) {
        v += pb[bi];
      }
      py[base + i] = v;

      for (int64_t d = normalized_rank - 1; d >= 0; --d) {
        ++coord[static_cast<size_t>(d)];
        si += scale_strides[static_cast<size_t>(d)];
        if (has_bias) {
          bi += bias_strides[static_cast<size_t>(d)];
        }
        if (coord[static_cast<size_t>(d)] < norm_dims[static_cast<size_t>(d)]) {
          break;
        }
        coord[static_cast<size_t>(d)] = 0;
        si -= scale_strides[static_cast<size_t>(d)] * norm_dims[static_cast<size_t>(d)];
        if (has_bias) {
          bi -= bias_strides[static_cast<size_t>(d)] * norm_dims[static_cast<size_t>(d)];
        }
      }
    }
  }
}

void LayerNormalization::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputRange(node, 2, 3);
  RequireOutputRange(node, 1, 3);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &scale = GetInput(node, 1, rt.tensors());
  const Tensor *b = GetOptionalInput(node, 2, rt.tensors());
  onnx_kernels::kernel::LayerNormalization k(rt.kernel_ctx());
  auto [y, mean, inv_std_dev] =
      k(x, scale, b != nullptr ? *b : Tensor{}, GetNormAxis(node), GetEpsilon(node), &rt);
  SetOutput(node, 0, std::move(y), rt.tensors());
  if (node.output_size() >= 2) {
    SetOutput(node, 1, std::move(mean), rt.tensors());
  }
  if (node.output_size() >= 3) {
    SetOutput(node, 2, std::move(inv_std_dev), rt.tensors());
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
