// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shapes_context.h"

#include <string>
#include <utility>

#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::shapes {

// ── Opset versions ──────────────────────────────────────────────────

void ShapesContext::SetOpsetVersion(const std::string &domain, int opset_version) {
  opsets_[NormaliseDomain(domain)] = opset_version;
}

bool ShapesContext::HasOpsetVersion(const std::string &domain) const {
  return opsets_.find(NormaliseDomain(domain)) != opsets_.end();
}

int ShapesContext::OpsetVersion(const std::string &domain) const {
  auto it = opsets_.find(NormaliseDomain(domain));
  return it == opsets_.end() ? kUnknownOpsetVersion : it->second;
}

// ── Model-local functions ───────────────────────────────────────────

void ShapesContext::SetLocalFunction(std::shared_ptr<const FunctionProto> func) {
  EXT_ENFORCE_INVALID(func != nullptr, "SetLocalFunction: func must not be nullptr.");
  const std::string key = std::string(func->domain()) + ":" + std::string(func->name());
  local_functions_[key] = func.get();
  owned_local_functions_[key] = std::move(func);
}

void ShapesContext::CopyLocalFunctions(const ShapesContext &context) {
  local_functions_ = context.local_functions_;
  owned_local_functions_ = context.owned_local_functions_;
}

void ShapesContext::ClearLocalFunctions() noexcept {
  local_functions_.clear();
  owned_local_functions_.clear();
}

// ── Custom shape-inference hooks ────────────────────────────────────

void ShapesContext::SetCustomShapeInferenceFunction(const std::string &domain,
                                                    const std::string &op_type,
                                                    CustomComputeShapeFn fn) {
  EXT_ENFORCE_INVALID(!op_type.empty(),
                      "SetCustomShapeInferenceFunction: op_type must not be empty.");
  EXT_ENFORCE_INVALID(static_cast<bool>(fn),
                      "SetCustomShapeInferenceFunction: fn must not be empty.");
  custom_shape_inference_[NormaliseDomain(domain) + ":" + op_type] = std::move(fn);
}

const ShapesContext::CustomComputeShapeFn *
ShapesContext::GetCustomShapeInferenceFunction(const std::string &domain,
                                               const std::string &op_type) const {
  auto it = custom_shape_inference_.find(NormaliseDomain(domain) + ":" + op_type);
  return it == custom_shape_inference_.end() ? nullptr : &it->second;
}

bool ShapesContext::RemoveCustomShapeInferenceFunction(const std::string &domain,
                                                       const std::string &op_type) {
  return custom_shape_inference_.erase(NormaliseDomain(domain) + ":" + op_type) > 0;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::shapes
