// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include "onnx_extensions/kernels/kernels/quantization/include_quantization_kernels.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

RuntimeValue Quantize::operator()(const Tensor &x, const StructTypeProto &type,
                                  const QuantizationParameters &parameters,
                                  const StructTypeCatalogue &catalogue) const {
  return QuantizeTensor(x, type, parameters, catalogue);
}

void Quantize::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputRange(node, 1, 9);
  RequireOutputCount(node, 1);
  EXT_ENFORCE_INVALID(!node.output(0).empty(), "Quantize requires a named output.");
  const auto *attribute = FindAttribute(node, "type");
  EXT_ENFORCE_INVALID(
      attribute && attribute->type() == AttributeProto::TYPE_PROTO && attribute->has_tp() &&
          attribute->tp().has_struct_type(),
      "Quantize requires a 'type' TYPE_PROTO attribute containing StructTypeProto.");
  QuantizationParameters parameters{
      GetOptionalInput(node, 1, rt.tensors()), GetOptionalInput(node, 2, rt.tensors()),
      GetOptionalInput(node, 3, rt.tensors()), GetOptionalInput(node, 4, rt.tensors()),
      GetOptionalInput(node, 5, rt.tensors()), GetOptionalInput(node, 6, rt.tensors()),
      GetOptionalInput(node, 7, rt.tensors()), GetOptionalInput(node, 8, rt.tensors())};
  auto input = GetInput(node, 0, rt.tensors()).BorrowView();
  input.name = node.output(0);
  auto result =
      (*this)(input, attribute->tp().struct_type(), parameters, rt.struct_type_catalogue());
  rt.values().insert_or_assign(node.output(0), std::move(result));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
