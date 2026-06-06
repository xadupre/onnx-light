// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shape_inference.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "onnx_proto/onnx_helper.h"

#include "onnx_optim/shapes/dispatch_table.h"
#include "onnx_optim/shapes/generator/shape_generator.h"
#include "onnx_optim/shapes/preview/shape_preview.h"
#include "onnx_optim/shapes/traditionalml/shape_traditionalml.h"
#include "onnx_optim/shapes/training/shape_training.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

namespace {

// Checks the node belongs to a supported domain: the default ONNX
// domain (empty string or "ai.onnx") or the traditional ML domain
// ("ai.onnx.ml"). Throws std::invalid_argument otherwise.
// Domain-specific dispatch can be added here when other domains gain
// support.
void CheckOnnxDomain(const NodeProto &node) {
  EXT_ENFORCE_INVALID(node.domain().empty() || node.domain() == kOnnxDomain ||
                          node.domain() == traditionalml::kOnnxMlDomain ||
                          node.domain() == preview::kOnnxPreviewDomain ||
                          node.domain() == training::kOnnxPreviewTrainingDomain,
                      "ComputeShapeNode: unsupported domain '" + node.domain().as_string() +
                          "' for op '" + node.op_type().as_string() + "'.");
}

// Normalises the empty default ONNX domain to ``kOnnxDomain`` so that
// dispatch-table lookups always use a canonical key.
std::string NormaliseDispatchDomain(const NodeProto &node) {
  const std::string domain = node.domain().as_string();
  return domain.empty() ? std::string(kOnnxDomain) : domain;
}

using AnchorMap = std::unordered_map<std::string, OptimTensor>;

void AddValueInfoAsAnchor(const ValueInfoProto &vi, AnchorMap &anchors) {
  const std::string name = vi.name().as_string();
  if (name.empty()) {
    return;
  }
  OptimTensor tensor;
  if (!OptimTensorFromValueInfo(vi, tensor)) {
    return;
  }
  anchors.try_emplace(name, std::move(tensor));
}

AnchorMap CollectGraphAnchors(const GraphProto &graph) {
  AnchorMap anchors;
  // Outputs are considered more authoritative than value_info for the
  // same name (first insert wins).
  for (int i = 0; i < graph.output_size(); ++i) {
    AddValueInfoAsAnchor(graph.output(i), anchors);
  }
  for (int i = 0; i < graph.value_info_size(); ++i) {
    AddValueInfoAsAnchor(graph.value_info(i), anchors);
  }
  return anchors;
}

OptimTensor SelectTensorPreferringAnchor(const OptimTensor &inferred, const OptimTensor &anchor) {
  switch (anchor.Cmp(inferred)) {
  case OptimCmpResult::kMorePrecise:
  case OptimCmpResult::kComplementary:
  case OptimCmpResult::kConflict:
    return anchor;
  case OptimCmpResult::kLessPrecise:
    return inferred;
  default:
    // Forward compatibility in case OptimCmpResult gains new values.
    return inferred;
  }
}

void MergeAnchorsIntoContext(ShapesContext &ctx, const AnchorMap &anchors) {
  for (const auto &kv : anchors) {
    const std::string &name = kv.first;
    const OptimTensor &anchor = kv.second;
    if (!ctx.Has(name)) {
      ctx.Set(name, OptimTensor(anchor));
      continue;
    }
    const OptimTensor &current = ctx.Get(name);
    OptimTensor chosen = SelectTensorPreferringAnchor(current, anchor);
    if (chosen != current) {
      ctx.Set(name, std::move(chosen));
    }
  }
}

} // namespace

void CheckInputsAvailable(const ShapesContext &ctx, const NodeProto &node) {
  for (int i = 0; i < node.input_size(); ++i) {
    const std::string name = node.input(i).as_string();
    if (name.empty()) {
      continue;
    }
    EXT_ENFORCE_INVALID(ctx.Has(name) || ctx.HasSequence(name),
                        "CheckInputsAvailable: input '" + name + "' of op '" +
                            node.op_type().as_string() + "' is missing from ShapesContext.");
  }
}

void CheckOutputsNotAvailable(const ShapesContext &ctx, const NodeProto &node) {
  for (int i = 0; i < node.output_size(); ++i) {
    const std::string name = node.output(i).as_string();
    if (name.empty()) {
      continue;
    }
    EXT_ENFORCE_INVALID(!ctx.Has(name) && !ctx.HasSequence(name),
                        "CheckOutputsNotAvailable: output '" + name + "' of op '" +
                            node.op_type().as_string() + "' is already present in ShapesContext.");
  }
}

void ComputeShapeNode(ShapesContext &ctx, const NodeProto &node) {
  CheckOnnxDomain(node);
  CheckInputsAvailable(ctx, node);
  CheckOutputsNotAvailable(ctx, node);
  const std::string op_type = node.op_type().as_string();
  const std::string key = NormaliseDispatchDomain(node) + ":" + op_type;
  const auto &table = DispatchTable();
  auto it = table.find(key);
  EXT_ENFORCE_INVALID(it != table.end(), "ComputeShapeNode: unsupported op_type '" + op_type +
                                             "' in domain '" + NormaliseDispatchDomain(node) +
                                             "'.");
  it->second(ctx, node);
}

void ComputeShapes(ShapesContext &ctx, const utils::RepeatedProtoField<NodeProto> &nodes) {
  for (int i = 0; i < nodes.size(); ++i) {
    ComputeShapeNode(ctx, nodes[i]);
  }
}

void ComputeShapeGraph(ShapesContext &ctx, const GraphProto &graph) {
  // Seed initializers first so that they shadow any duplicate input
  // (an ONNX initializer may appear both in ``graph.initializer()``
  // and ``graph.input()``; the initializer wins).
  for (int i = 0; i < graph.initializer().size(); ++i) {
    const TensorProto &init = graph.initializer()[i];
    const std::string name = init.name().as_string();
    if (name.empty() || ctx.Has(name)) {
      continue;
    }
    OptimTensor tensor;
    if (OptimTensorFromTensorProto(init, tensor)) {
      ctx.Set(name, std::move(tensor));
    }
  }
  // Then seed graph inputs (skipping those already known via the
  // initializers or via outer-scope entries carried in ``ctx``).
  for (int i = 0; i < graph.input().size(); ++i) {
    const ValueInfoProto &vi = graph.input()[i];
    const std::string name = vi.name().as_string();
    if (name.empty() || ctx.Has(name) || ctx.HasSequence(name)) {
      continue;
    }
    OptimTensor tensor;
    if (OptimTensorFromValueInfo(vi, tensor)) {
      ctx.Set(name, std::move(tensor));
    }
  }
  ComputeShapes(ctx, graph.node());
}

void ComputeShapeModel(ShapesContext &ctx, const ModelProto &model,
                       bool prefill_with_value_info_output) {
  for (int i = 0; i < model.opset_import().size(); ++i) {
    const OperatorSetIdProto &osi = model.opset_import()[i];
    ctx.SetOpsetVersion(osi.domain().as_string(), static_cast<int>(osi.version()));
  }
  EXT_ENFORCE_INVALID(model.has_graph(),
                      "ComputeShapeModel: the ModelProto has no graph to run shape inference on.");
  AnchorMap anchors;
  if (prefill_with_value_info_output) {
    anchors = CollectGraphAnchors(model.graph());
  }
  ComputeShapeGraph(ctx, model.graph());
  if (prefill_with_value_info_output) {
    MergeAnchorsIntoContext(ctx, anchors);
  }
}

void ApplyInferredShapesToGraph(const ShapesContext &ctx, GraphProto &graph) {
  // Names that already have authoritative type/shape information in
  // the proto and must not be overwritten.
  std::unordered_set<std::string> seeded;
  for (int i = 0; i < graph.input().size(); ++i) {
    seeded.insert(graph.input()[i].name().as_string());
  }
  for (int i = 0; i < graph.initializer().size(); ++i) {
    seeded.insert(graph.initializer()[i].name().as_string());
  }
  // Update graph outputs in place.
  std::unordered_set<std::string> output_names;
  for (int i = 0; i < graph.output_size(); ++i) {
    ValueInfoProto &vi = *graph.mutable_output(i);
    const std::string name = vi.name().as_string();
    output_names.insert(name);
    if (!name.empty() && ctx.Has(name)) {
      OptimTensorToValueInfo(ctx.Get(name), vi);
    }
  }
  // Track existing value_info entries to avoid creating duplicates;
  // update them in place when the name matches.
  std::unordered_set<std::string> existing_value_info;
  for (int i = 0; i < graph.value_info_size(); ++i) {
    ValueInfoProto &vi = *graph.mutable_value_info(i);
    const std::string name = vi.name().as_string();
    existing_value_info.insert(name);
    if (!name.empty() && ctx.Has(name)) {
      OptimTensorToValueInfo(ctx.Get(name), vi);
    }
  }
  // Append a new value_info entry for every other inferred tensor.
  // Iteration order over the unordered map is not specified, so the
  // names are gathered and sorted to make the output deterministic.
  std::vector<std::string> new_names;
  new_names.reserve(ctx.Tensors().size());
  for (const auto &kv : ctx.Tensors()) {
    const std::string &name = kv.first;
    if (name.empty() || seeded.count(name) != 0 || output_names.count(name) != 0 ||
        existing_value_info.count(name) != 0) {
      continue;
    }
    new_names.push_back(name);
  }
  std::sort(new_names.begin(), new_names.end());
  for (const std::string &name : new_names) {
    const OptimTensor &tensor = ctx.Get(name);
    if (TensorTypeToDataType(tensor.Dtype()) == TensorProto::DataType::UNDEFINED) {
      continue;
    }
    ValueInfoProto *vi = graph.add_value_info();
    vi->set_name(name);
    OptimTensorToValueInfo(tensor, *vi);
  }
}

void ApplyInferredShapesToModel(const ShapesContext &ctx, ModelProto &model) {
  EXT_ENFORCE_INVALID(
      model.has_graph(),
      "ApplyInferredShapesToModel: the ModelProto has no graph to write shape inference into.");
  ApplyInferredShapesToGraph(ctx, *model.mutable_graph());
}

void InferShapesModel(ModelProto &model, bool prefill_with_value_info_output) {
  ShapesContext ctx;
  ComputeShapeModel(ctx, model, prefill_with_value_info_output);
  ApplyInferredShapesToModel(ctx, model);
}

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
