#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "onnx_core/expressions/expressions.h"
#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/symbolic_helper.h"
#include "onnx_extensions/kernels/kernels/auto_pad.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes {

// Alias to the symbolic dimension-expression library, which lives in
// ``onnx_core`` so both ``onnx_op`` and ``onnx_shapes`` can share it.
namespace expressions = ::ONNX_LIGHT_NAMESPACE::core::expressions;
namespace shapes::nn {

using onnx_kernels::kernel::AutoPad;
using onnx_kernels::kernel::AutoPadFromString;

namespace {

// Converts an ``expressions::DimType`` produced by the symbolic dimension
// helpers back into an ``SymDim``.
SymDim FromDimType(const expressions::DimType &d) {
  if (std::holds_alternative<int64_t>(d)) {
    return SymDim(std::get<int64_t>(d));
  }
  return SymDim(std::get<std::string>(d));
}

// Computes one spatial output dimension following the upstream
// ``convPoolShapeInference`` rule for Conv-like ops, honoring ``auto_pad``.
// When the input dim is symbolic the formula is evaluated symbolically so the
// result is an expression (e.g. ``H`` for a reflect-padded image that a 3×3
// VALID convolution shrinks back), never a fresh opaque dimension name.
SymDim ComputeConvSpatialDim(const SymDim &in_dim, int64_t kernel, int64_t stride,
                             int64_t pad_begin, int64_t pad_end, int64_t dilation, AutoPad auto_pad,
                             const std::string &op_name, const std::string &x_name,
                             size_t spatial_axis) {
  const std::string symbolic = op_name + "." + x_name + ":" + std::to_string(spatial_axis);
  if (stride <= 0 || kernel <= 0) {
    return SymDim(symbolic);
  }
  const int64_t eff_k = dilation * (kernel - 1) + 1;

  if (in_dim.IsInt()) {
    const int64_t iD = in_dim.AsInt();
    if (auto_pad == AutoPad::kSameUpper || auto_pad == AutoPad::kSameLower) {
      return SymDim((iD + stride - 1) / stride);
    }
    if (auto_pad == AutoPad::kValid) {
      const int64_t numer = iD - eff_k;
      if (numer < 0) {
        return SymDim(symbolic);
      }
      return SymDim(numer / stride + 1);
    }
    const int64_t numer = iD + pad_begin + pad_end - eff_k;
    if (numer < 0) {
      return SymDim(symbolic);
    }
    return SymDim(numer / stride + 1);
  }

  // Symbolic input dimension: evaluate the spatial formula with the symbolic
  // expression helpers so the output stays an expression of the input dim.
  const expressions::DimType iD = ToDimType(in_dim);
  if (auto_pad == AutoPad::kSameUpper || auto_pad == AutoPad::kSameLower) {
    return FromDimType(expressions::dim_div(
        expressions::dim_add(iD, expressions::DimType{stride - 1}), expressions::DimType{stride}));
  }
  expressions::DimType numer;
  if (auto_pad == AutoPad::kValid) {
    numer = expressions::dim_sub(iD, expressions::DimType{eff_k});
  } else {
    numer =
        expressions::dim_sub(expressions::dim_add(iD, expressions::DimType{pad_begin + pad_end}),
                             expressions::DimType{eff_k});
  }
  return FromDimType(expressions::dim_add(expressions::dim_div(numer, expressions::DimType{stride}),
                                          expressions::DimType{1}));
}

// Shared implementation for ``Conv`` and ``ConvInteger``. The output dtype
// is provided by the caller (``X.dtype`` for Conv, ``int32`` for ConvInteger).
void ComputeShapeConvLike(ShapesContext &ctx, const NodeProto &node, const char *x, const char *w,
                          const char *op_name, TensorType out_dtype) {
  const SymTensor &x_tensor = ctx.Get(x);
  const SymTensor &w_tensor = ctx.Get(w);
  const SymShape &x_shape = x_tensor.Shape();
  const SymShape &w_shape = w_tensor.Shape();

  EXT_ENFORCE_INVALID(!(x_shape.Rank() < 3), "ComputeShape", op_name, ": input '", x,
                      "' must have rank >= 3 (N, C, D1, ...).");
  EXT_ENFORCE_INVALID(w_shape.Rank() == x_shape.Rank(), "ComputeShape", op_name, ": weight '", w,
                      "' rank must match input rank.");

  const size_t n_spatial = x_shape.Rank() - 2;
  const AutoPad auto_pad =
      AutoPadFromString(GetAttributeOr<std::string>(node, "auto_pad", "NOTSET"));

  std::vector<int64_t> kernel_shape;
  GetAttributeInts(node, "kernel_shape", kernel_shape);
  if (kernel_shape.empty()) {
    kernel_shape.reserve(n_spatial);
    for (size_t i = 0; i < n_spatial; ++i) {
      const SymDim &kd = w_shape[i + 2];
      kernel_shape.push_back(kd.IsInt() ? kd.AsInt() : -1);
    }
  } else if (kernel_shape.size() != n_spatial) {
    EXT_THROW_INVALID("ComputeShape", op_name,
                      ": 'kernel_shape' size does not match input spatial rank.");
  }

  std::vector<int64_t> strides;
  GetAttributeInts(node, "strides", strides);
  if (strides.empty()) {
    strides.assign(n_spatial, 1);
  } else if (strides.size() != n_spatial) {
    EXT_THROW_INVALID("ComputeShape", op_name,
                      ": 'strides' size does not match input spatial rank.");
  }

  std::vector<int64_t> dilations;
  GetAttributeInts(node, "dilations", dilations);
  if (dilations.empty()) {
    dilations.assign(n_spatial, 1);
  } else if (dilations.size() != n_spatial) {
    EXT_THROW_INVALID("ComputeShape", op_name,
                      ": 'dilations' size does not match input spatial rank.");
  }

  std::vector<int64_t> pads;
  GetAttributeInts(node, "pads", pads);
  if (pads.empty()) {
    pads.assign(n_spatial * 2, 0);
  } else if (pads.size() != n_spatial * 2) {
    EXT_THROW_INVALID("ComputeShape", op_name, ": 'pads' size must be 2 * spatial rank.");
  }

  SymShape out_shape;
  out_shape.PushBack(x_shape[0]); // N
  out_shape.PushBack(w_shape[0]); // M
  for (size_t i = 0; i < n_spatial; ++i) {
    if (kernel_shape[i] <= 0) {
      out_shape.PushBack(SymDim(std::string(op_name) + "." + x + ":" + std::to_string(i)));
      continue;
    }
    out_shape.PushBack(ComputeConvSpatialDim(x_shape[i + 2], kernel_shape[i], strides[i], pads[i],
                                             pads[i + n_spatial], dilations[i], auto_pad, op_name,
                                             x, i));
  }

  ctx.Set(node.output(0), SymTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace

void ComputeShapeConv(ShapesContext &ctx, const NodeProto &node, const char *x, const char *w) {
  CheckNodeOpAndOutput(node, "Conv", "ComputeShapeConv");
  ComputeShapeConvLike(ctx, node, x, w, "Conv", ctx.Get(x).Dtype());
}

void ComputeShapeConvInteger(ShapesContext &ctx, const NodeProto &node, const char *x,
                             const char *w) {
  CheckNodeOpAndOutput(node, "ConvInteger", "ComputeShapeConvInteger");
  ComputeShapeConvLike(ctx, node, x, w, "ConvInteger", TensorType::kInt32);
}

} // namespace shapes::nn
} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes
