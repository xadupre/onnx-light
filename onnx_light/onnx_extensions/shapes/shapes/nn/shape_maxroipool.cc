// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <cstdint>
#include <string>
#include <vector>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {

void ComputeShapeMaxRoiPool(ShapesContext &ctx, const NodeProto &node, const char *x,
                            const char *rois) {
  CheckNodeOpAndOutput(node, "MaxRoiPool", "ComputeShapeMaxRoiPool");

  const SymTensor &input = ctx.Get(x);
  const SymShape &in_shape = input.Shape();
  EXT_ENFORCE_INVALID(in_shape.Rank() == 4, "ComputeShapeMaxRoiPool: input '", x,
                      "' must have rank 4 (N, C, H, W).");

  const SymShape &rois_shape = ctx.Get(rois).Shape();
  EXT_ENFORCE_INVALID(rois_shape.Rank() == 2, "ComputeShapeMaxRoiPool: input '", rois,
                      "' must have rank 2 (num_rois, 5).");

  std::vector<int64_t> pooled_shape;
  EXT_ENFORCE_INVALID(GetAttributeInts(node, "pooled_shape", pooled_shape),
                      "ComputeShapeMaxRoiPool: required attribute 'pooled_shape' is missing.");
  EXT_ENFORCE_INVALID(!(pooled_shape.size() != 2 || pooled_shape[0] <= 0 || pooled_shape[1] <= 0),
                      "ComputeShapeMaxRoiPool: attribute 'pooled_shape' must contain two positive "
                      "values (height, width).");

  SymShape out_shape;
  out_shape.PushBack(rois_shape[0]);
  out_shape.PushBack(in_shape[1]);
  out_shape.PushBack(SymDim(pooled_shape[0]));
  out_shape.PushBack(SymDim(pooled_shape[1]));

  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
