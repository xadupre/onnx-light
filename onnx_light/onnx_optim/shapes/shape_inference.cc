// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shape_inference.h"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "onnx_proto/onnx_helper.h"

#include "onnx_optim/shapes/controlflow/shape_controlflow.h"
#include "onnx_optim/shapes/generator/shape_generator.h"
#include "onnx_optim/shapes/logical/shape_logical.h"
#include "onnx_optim/shapes/math/shape_math.h"
#include "onnx_optim/shapes/nn/shape_nn.h"
#include "onnx_optim/shapes/optional/shape_optional.h"
#include "onnx_optim/shapes/preview/shape_preview.h"
#include "onnx_optim/shapes/quantization/shape_quantization.h"
#include "onnx_optim/shapes/reduction/shape_reduction.h"
#include "onnx_optim/shapes/sequence/shape_sequence.h"
#include "onnx_optim/shapes/tensor/shape_tensor.h"
#include "onnx_optim/shapes/text/shape_text.h"
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

// Verifies the node declares at least `expected` inputs.
void RequireInputs(const NodeProto &node, int expected) {
  EXT_ENFORCE_INVALID(node.input_size() >= expected,
                      "ComputeShapeNode: op '" + node.op_type().as_string() +
                          "' expects at least " + std::to_string(expected) + " input(s), got " +
                          std::to_string(node.input_size()) + ".");
}

// Signature of every per-operator ComputeShape* trampoline registered
// in kDispatchTable: it reads the node's inputs from ``ctx`` and
// inserts the resulting output descriptors back into ``ctx``.
using ComputeShapeFn = std::function<void(ShapesContext &, const NodeProto &)>;

// Returns the (normalised_domain, op_type) -> ComputeShape* dispatch
// table. Constructed on first use and shared across calls. Adding a
// new operator only requires inserting one new entry here. The
// dispatch key is the pair ``(domain, op_type)`` encoded as
// ``"<domain>:<op_type>"``; the empty default ONNX domain must be
// normalised to ``kOnnxDomain`` before lookup.
const std::unordered_map<std::string, ComputeShapeFn> &DispatchTable() {
  static const std::unordered_map<std::string, ComputeShapeFn> table = {
      {"ai.onnx:Abs",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAbs(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Acos",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAcos(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Acosh",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         math::ComputeShapeAcosh(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Add",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         math::ComputeShapeAdd(ctx, node, node.input(0).as_string().c_str(),
                               node.input(1).as_string().c_str());
       }},
      {"ai.onnx:And",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         logical::ComputeShapeAnd(ctx, node, node.input(0).as_string().c_str(),
                                  node.input(1).as_string().c_str());
       }},
      {"ai.onnx:AveragePool",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         nn::ComputeShapeAveragePool(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx:Constant",
       [](ShapesContext &ctx, const NodeProto &node) {
         generator::ComputeShapeConstant(ctx, node);
       }},
      {"ai.onnx:Concat",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         tensor::ComputeShapeConcat(ctx, node);
       }},
      {"ai.onnx:If",
       [](ShapesContext &ctx, const NodeProto &node) { controlflow::ComputeShapeIf(ctx, node); }},
      {"ai.onnx:Optional",
       [](ShapesContext &ctx, const NodeProto &node) {
         optional::ComputeShapeOptional(ctx, node);
       }},
      {"ai.onnx:QuantizeLinear",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         const std::string x_name = node.input(0).as_string();
         const std::string zp_name =
             node.input_size() >= 3 ? node.input(2).as_string() : std::string();
         quantization::ComputeShapeQuantizeLinear(ctx, node, x_name.c_str(),
                                                  zp_name.empty() ? nullptr : zp_name.c_str());
       }},
      {"ai.onnx:ReduceSum",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         const std::string data_name = node.input(0).as_string();
         const std::string axes_name =
             node.input_size() >= 2 ? node.input(1).as_string() : std::string();
         reduction::ComputeShapeReduceSum(ctx, node, data_name.c_str(),
                                          node.input_size() >= 2 ? axes_name.c_str() : nullptr);
       }},
      {"ai.onnx:Reshape",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         tensor::ComputeShapeReshape(ctx, node);
       }},
      {"ai.onnx:RoiAlign",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 3);
         nn::ComputeShapeRoiAlign(ctx, node, node.input(0).as_string().c_str(),
                                  node.input(1).as_string().c_str(),
                                  node.input(2).as_string().c_str());
       }},
      {"ai.onnx:SequenceConstruct",
       [](ShapesContext &ctx, const NodeProto &node) {
         sequence::ComputeShapeSequenceConstruct(ctx, node);
       }},
      {"ai.onnx:StringConcat",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 2);
         text::ComputeShapeStringConcat(ctx, node, node.input(0).as_string().c_str(),
                                        node.input(1).as_string().c_str());
       }},
      {"ai.onnx.ml:LabelEncoder",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 1);
         traditionalml::ComputeShapeLabelEncoder(ctx, node, node.input(0).as_string().c_str());
       }},
      {"ai.onnx.preview:FlexAttention",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 3);
         preview::ComputeShapeFlexAttention(ctx, node, node.input(0).as_string().c_str(),
                                            node.input(1).as_string().c_str(),
                                            node.input(2).as_string().c_str());
       }},
      {"ai.onnx.preview.training:Adam",
       [](ShapesContext &ctx, const NodeProto &node) {
         RequireInputs(node, 6);
         training::ComputeShapeAdam(ctx, node);
       }},
  };
  return table;
}

} // namespace

void CheckInputsAvailable(const ShapesContext &ctx, const NodeProto &node) {
  for (int i = 0; i < node.input_size(); ++i) {
    const std::string name = node.input(i).as_string();
    if (name.empty()) {
      continue;
    }
    EXT_ENFORCE_INVALID(ctx.Has(name), "CheckInputsAvailable: input '" + name + "' of op '" +
                                           node.op_type().as_string() +
                                           "' is missing from ShapesContext.");
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

namespace {

// Builds an OptimShape from a TensorShapeProto, preserving symbolic
// dimensions: ``dim_value`` becomes a concrete int dim, ``dim_param``
// becomes a symbolic dim with the same name, and an unset dim becomes
// a fresh ``"?"`` placeholder.
OptimShape ShapeFromTensorShapeProto(const TensorShapeProto &sp) {
  OptimShape shape;
  for (int i = 0; i < sp.dim().size(); ++i) {
    const TensorShapeProto::Dimension &d = sp.dim()[i];
    if (d.has_dim_value()) {
      shape.PushBack(OptimDim(static_cast<int64_t>(d.dim_value())));
    } else if (d.has_dim_param()) {
      shape.PushBack(OptimDim(std::string(d.dim_param().data(), d.dim_param().size())));
    } else {
      shape.PushBack(OptimDim(std::string("?")));
    }
  }
  return shape;
}

// Builds an OptimTensor from a ValueInfoProto wrapping a tensor type.
// Returns ``false`` when the ValueInfoProto wraps a non-tensor type
// (sequence/map/optional/sparse), in which case the caller is
// expected to skip the entry; ``OptimTensor`` does not model these.
bool OptimTensorFromValueInfo(const ValueInfoProto &vi, OptimTensor &out) {
  if (!vi.has_type() || !vi.type().has_tensor_type()) {
    return false;
  }
  const TypeProto::Tensor &tt = vi.type().tensor_type();
  const TensorType dtype = DataTypeToTensorType(tt.elem_type());
  OptimShape shape;
  if (tt.has_shape()) {
    shape = ShapeFromTensorShapeProto(tt.shape());
  }
  out = OptimTensor(nullptr, dtype, std::move(shape));
  return true;
}

// Builds an OptimTensor describing a graph initializer. Small 1-D
// integer initializers also get a ``ValueAsShape`` annotation derived
// from their content so that downstream ops (such as ``Reshape``) can
// see the actual target-shape values.
OptimTensor OptimTensorFromInitializer(const TensorProto &tp) {
  const TensorType dtype = DataTypeToTensorType(tp.data_type());
  OptimShape shape = ShapeFromTensorProtoDims(tp);
  OptimTensor tensor(nullptr, dtype, std::move(shape));
  if (IsIntegerTensorType(tensor.Dtype()) && tensor.Shape().Rank() <= 1) {
    int64_t count = 1;
    for (std::size_t i = 0; i < tensor.Shape().Rank(); ++i) {
      count *= tensor.Shape()[i].AsInt();
    }
    if (count >= 0 && count < generator::kConstantValueAsShapeMaxElements) {
      std::vector<int64_t> values;
      if (ReadIntegerValues(tp, values) && static_cast<int64_t>(values.size()) == count) {
        OptimShape value_shape;
        for (int64_t v : values) {
          value_shape.PushBack(OptimDim(v));
        }
        tensor.SetValueAsShape(std::move(value_shape));
      }
    }
  }
  return tensor;
}

} // namespace

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
    ctx.Set(name, OptimTensorFromInitializer(init));
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

void ComputeShapeModel(ShapesContext &ctx, const ModelProto &model) {
  for (int i = 0; i < model.opset_import().size(); ++i) {
    const OperatorSetIdProto &osi = model.opset_import()[i];
    ctx.SetOpsetVersion(osi.domain().as_string(), static_cast<int>(osi.version()));
  }
  EXT_ENFORCE_INVALID(model.has_graph(),
                      "ComputeShapeModel: the ModelProto has no graph to run shape inference on.");
  ComputeShapeGraph(ctx, model.graph());
}

namespace {

// Writes the (dtype, shape) of ``tensor`` into ``vi``. Concrete
// integer dimensions become ``dim_value`` entries and symbolic
// dimensions become ``dim_param`` entries. Any pre-existing
// type/shape entry on ``vi`` is overwritten so that the inferred
// information takes precedence. Returns ``false`` (and leaves ``vi``
// unchanged) when ``tensor`` has an undefined element type, since
// ``TensorProto::DataType`` does not provide a meaningful encoding
// for it.
bool WriteOptimTensorToValueInfo(const OptimTensor &tensor, ValueInfoProto &vi) {
  const TensorProto::DataType dtype = TensorTypeToDataType(tensor.Dtype());
  if (dtype == TensorProto::DataType::UNDEFINED) {
    return false;
  }
  // Reset any pre-existing type/shape information so it is replaced
  // wholesale by the inferred descriptor.
  vi.clear_type();
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(static_cast<int>(dtype));
  TensorShapeProto *sp = tt->add_shape();
  for (std::size_t i = 0; i < tensor.Shape().Rank(); ++i) {
    const OptimDim &d = tensor.Shape()[i];
    TensorShapeProto::Dimension *dim = sp->add_dim();
    if (d.IsInt()) {
      dim->set_dim_value(d.AsInt());
    } else {
      dim->set_dim_param(d.AsExpr());
    }
  }
  return true;
}

} // namespace

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
      WriteOptimTensorToValueInfo(ctx.Get(name), vi);
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
      WriteOptimTensorToValueInfo(ctx.Get(name), vi);
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
    WriteOptimTensorToValueInfo(tensor, *vi);
  }
}

void ApplyInferredShapesToModel(const ShapesContext &ctx, ModelProto &model) {
  EXT_ENFORCE_INVALID(
      model.has_graph(),
      "ApplyInferredShapesToModel: the ModelProto has no graph to write shape inference into.");
  ApplyInferredShapesToGraph(ctx, *model.mutable_graph());
}

void InferShapesModel(ModelProto &model) {
  ShapesContext ctx;
  ComputeShapeModel(ctx, model);
  ApplyInferredShapesToModel(ctx, model);
}

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
