// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/backend_test/test_case_registry.h"

#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::core::backend_test {

namespace {

// Recursively builds the TypeProto ``tp`` from ``spec``.
void BuildTypeProto(const TypeSpec &spec, TypeProto &tp) {
  switch (spec.kind) {
  case TypeSpec::Kind::kTensor: {
    TypeProto::Tensor *tt = tp.add_tensor_type();
    tt->set_elem_type(spec.elem_type);
    if (spec.has_shape) {
      TensorShapeProto *sh = tt->add_shape();
      for (int64_t d : spec.shape) {
        sh->add_dim()->set_dim_value(d);
      }
    }
    break;
  }
  case TypeSpec::Kind::kSequence: {
    TypeProto::Sequence *seq = tp.add_sequence_type();
    BuildTypeProto(spec.children.front(), *seq->add_elem_type());
    break;
  }
  case TypeSpec::Kind::kMap: {
    TypeProto::Map *mp = tp.add_map_type();
    mp->set_key_type(spec.elem_type);
    BuildTypeProto(spec.children.front(), *mp->add_value_type());
    break;
  }
  }
}

} // namespace

TypeSpec TensorTypeSpec(int32_t elem_type) {
  TypeSpec spec;
  spec.kind = TypeSpec::Kind::kTensor;
  spec.elem_type = elem_type;
  spec.has_shape = false;
  return spec;
}

TypeSpec TensorTypeSpec(int32_t elem_type, std::vector<int64_t> shape) {
  TypeSpec spec;
  spec.kind = TypeSpec::Kind::kTensor;
  spec.elem_type = elem_type;
  spec.has_shape = true;
  spec.shape = std::move(shape);
  return spec;
}

TypeSpec SequenceTypeSpec(TypeSpec elem) {
  TypeSpec spec;
  spec.kind = TypeSpec::Kind::kSequence;
  spec.children.push_back(std::move(elem));
  return spec;
}

TypeSpec MapTypeSpec(int32_t key_type, TypeSpec value) {
  TypeSpec spec;
  spec.kind = TypeSpec::Kind::kMap;
  spec.elem_type = key_type;
  spec.children.push_back(std::move(value));
  return spec;
}

void AppendValueInfo(ValueInfoProto &vi, const std::string &name, const TypeSpec &spec) {
  vi.set_name(name);
  BuildTypeProto(spec, *vi.add_type());
}

void InitModel(ModelProto &model, int64_t ir_version, const std::vector<OpsetId> &opset_imports,
               const std::string &producer_name) {
  model.set_ir_version(ir_version);
  model.set_producer_name(producer_name);
  for (const auto &osid : opset_imports) {
    OperatorSetIdProto proto;
    proto.set_domain(osid.domain);
    proto.set_version(osid.version);
    model.add_opset_import(proto);
  }
}

void AppendValueInfo(ValueInfoProto &vi, const std::string &name, int32_t elem_type,
                     const std::vector<int64_t> &shape) {
  vi.set_name(name);
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(elem_type);
  TensorShapeProto *sh = tt->add_shape();
  for (int64_t d : shape) {
    sh->add_dim()->set_dim_value(d);
  }
}

void AppendValueInfo(ValueInfoProto &vi, const std::string &name, int32_t elem_type,
                     const std::vector<DimSpec> &dims) {
  vi.set_name(name);
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(elem_type);
  TensorShapeProto *sh = tt->add_shape();
  for (const auto &d : dims) {
    auto *dim = sh->add_dim();
    if (d.value >= 0) {
      dim->set_dim_value(d.value);
    } else if (!d.param.empty()) {
      dim->set_dim_param(d.param);
    }
    // else: leave the dim unannotated (no dim_value, no dim_param).
  }
}

void AppendValueInfo(ValueInfoProto &vi, const std::string &name, TensorProto::DataType elem_type,
                     const std::vector<DimSpec> &dims) {
  vi.set_name(name);
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(elem_type);
  TensorShapeProto *sh = tt->add_shape();
  for (const auto &d : dims) {
    auto *dim = sh->add_dim();
    if (d.value >= 0) {
      dim->set_dim_value(d.value);
    } else if (!d.param.empty()) {
      dim->set_dim_param(d.param);
    }
    // else: leave the dim unannotated (no dim_value, no dim_param).
  }
}

void AppendDataSet(TestCase &tc, Tensors inputs, Tensors outputs) {
  DataSet ds;
  ds.inputs = std::move(inputs);
  ds.outputs = std::move(outputs);
  tc.data_sets().emplace_back(std::move(ds));
}

void DispatchRegisterByOpType(std::vector<TestCase> &registry, const std::string &op_type,
                              const OpRegisterMap &entries) {
  if (op_type.empty()) {
    for (const auto &entry : entries) {
      entry.second(registry);
    }
    return;
  }
  auto it = entries.find(op_type);
  if (it != entries.end()) {
    it->second(registry);
  }
}

void DispatchRegisterByOpType(std::vector<TestCase> &registry, const std::string &op_type,
                              const OpRegisterModeMap &entries, TestMode mode) {
  if (op_type.empty()) {
    for (const auto &entry : entries) {
      entry.second(registry, mode);
    }
    return;
  }
  auto it = entries.find(op_type);
  if (it != entries.end()) {
    it->second(registry, mode);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::core::backend_test