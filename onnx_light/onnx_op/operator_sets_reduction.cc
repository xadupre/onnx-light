// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_reduction.h"
#include "onnx_op/operator_sets_reduction_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace reduction {

std::vector<LightOpSchema> BuildReduceSumSchemas() {
  return std::vector<LightOpSchema>{
      // ReduceSum opset 13: axes moved from attribute to optional second input; added bfloat16.
      LightOpSchema("ReduceSum", kOnnxDomain, 13, MakeReduceSumDoc(13),
                    {
                        {"data", "An input tensor.", "T"},
                        {"axes",
                         "Optional input list of integers, along which to reduce. "
                         "The default is to reduce over all the dimensions of the input tensor if "
                         "noop_with_empty_axes is false, else act as an Identity op when "
                         "noop_with_empty_axes "
                         "is true. Accepted range is [-r, r-1] where r = rank(data).",
                         "tensor(int64)"},
                    },
                    {
                        {"reduced", "Reduced output tensor.", "T"},
                    },
                    {
                        {"T", NumericTypesForMathReductionIr4(),
                         "Constrain input and output types to high-precision numeric tensors."},
                    }),

      // ReduceSum opset 11: axes remain an attribute; adds support for negative axes.
      LightOpSchema("ReduceSum", kOnnxDomain, 11, MakeReduceSumDoc(11),
                    {
                        {"data", "An input tensor.", "T"},
                    },
                    {
                        {"reduced", "Reduced output tensor.", "T"},
                    },
                    {
                        {"T", NumericTypesForMathReduction(),
                         "Constrain input and output types to high-precision numeric tensors."},
                    }),

      // ReduceSum opset 1: axes are provided as an attribute.
      LightOpSchema("ReduceSum", kOnnxDomain, 1, MakeReduceSumDoc(1),
                    {
                        {"data", "An input tensor.", "T"},
                    },
                    {
                        {"reduced", "Reduced output tensor.", "T"},
                    },
                    {
                        {"T", NumericTypesForMathReduction(),
                         "Constrain input and output types to high-precision numeric tensors."},
                    }),
  };
}

std::vector<LightOpSchema> GetAllOnnxOpReductionSchemasWithHistory(bool init_doc) {
  std::vector<LightOpSchema> schemas = BuildReduceSumSchemas();
  return init_doc ? schemas : StripDocs(schemas);
}

} // namespace reduction
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
