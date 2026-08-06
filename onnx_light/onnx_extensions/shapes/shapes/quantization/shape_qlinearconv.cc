#include "onnx_extensions/shapes/shapes/quantization/shape_quantization.h"

#include <cstdint>
#include <string>
#include <vector>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_extensions/kernels/kernels/auto_pad.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::quantization {

using onnx_kernels::kernel::AutoPad;
using onnx_kernels::kernel::AutoPadFromString;

namespace {

SymDim ComputeQLinearConvSpatialDim(const SymDim &in_dim, int64_t kernel, int64_t stride,
                                    int64_t pad_begin, int64_t pad_end, int64_t dilation,
                                    AutoPad auto_pad, const std::string &x_name,
                                    size_t spatial_axis) {
  const std::string symbolic =
      std::string("QLinearConv.") + x_name + ":" + std::to_string(spatial_axis);
  if (!in_dim.IsInt()) {
    return SymDim(symbolic);
  }
  const int64_t iD = in_dim.AsInt();
  const int64_t eff_k = dilation * (kernel - 1) + 1;
  if (stride <= 0 || kernel <= 0) {
    return SymDim(symbolic);
  }
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

} // namespace

void ComputeShapeQLinearConv(ShapesContext &ctx, const NodeProto &node, const char *x,
                             const char *w, const char *y_zero_point) {
  CheckNodeOpAndOutput(node, "QLinearConv", "ComputeShapeQLinearConv");

  const SymTensor &x_tensor = ctx.Get(x);
  const SymTensor &w_tensor = ctx.Get(w);
  const SymTensor &yzp_tensor = ctx.Get(y_zero_point);
  const SymShape &x_shape = x_tensor.Shape();
  const SymShape &w_shape = w_tensor.Shape();

  EXT_ENFORCE_INVALID(!(x_shape.Rank() < 3), "ComputeShapeQLinearConv: input '", x,
                      "' must have rank >= 3 (N, C, D1, ...).");
  EXT_ENFORCE_INVALID(w_shape.Rank() == x_shape.Rank(), "ComputeShapeQLinearConv: weight '", w,
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
    EXT_THROW_INVALID(
        "ComputeShapeQLinearConv: 'kernel_shape' size does not match input spatial rank.");
  }

  std::vector<int64_t> strides;
  GetAttributeInts(node, "strides", strides);
  if (strides.empty()) {
    strides.assign(n_spatial, 1);
  } else if (strides.size() != n_spatial) {
    EXT_THROW_INVALID("ComputeShapeQLinearConv: 'strides' size does not match input spatial rank.");
  }

  std::vector<int64_t> dilations;
  GetAttributeInts(node, "dilations", dilations);
  if (dilations.empty()) {
    dilations.assign(n_spatial, 1);
  } else if (dilations.size() != n_spatial) {
    EXT_THROW_INVALID(
        "ComputeShapeQLinearConv: 'dilations' size does not match input spatial rank.");
  }

  std::vector<int64_t> pads;
  GetAttributeInts(node, "pads", pads);
  if (pads.empty()) {
    pads.assign(n_spatial * 2, 0);
  } else if (pads.size() != n_spatial * 2) {
    EXT_THROW_INVALID("ComputeShapeQLinearConv: 'pads' size must be 2 * spatial rank.");
  }

  SymShape out_shape;
  out_shape.PushBack(x_shape[0]); // N
  out_shape.PushBack(w_shape[0]); // M
  for (size_t i = 0; i < n_spatial; ++i) {
    if (kernel_shape[i] <= 0) {
      out_shape.PushBack(SymDim(std::string("QLinearConv.") + x + ":" + std::to_string(i)));
      continue;
    }
    out_shape.PushBack(ComputeQLinearConvSpatialDim(x_shape[i + 2], kernel_shape[i], strides[i],
                                                    pads[i], pads[i + n_spatial], dilations[i],
                                                    auto_pad, x, i));
  }

  ctx.Set(node.output(0), SymTensor(nullptr, yzp_tensor.Dtype(), std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::quantization
