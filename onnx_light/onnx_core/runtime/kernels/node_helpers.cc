// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/node_helpers.h"

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

const Tensor &GetInput(const NodeProto &node, int index, const TensorMap &tensors) {
  const std::string &name = node.input(index);
  EXT_ENFORCE_INVALID(!(name.empty()), "RunNode: op '", node.op_type(), "' input #", index,
                      " is unset (empty name).");
  auto it = tensors.find(name);
  EXT_ENFORCE_INVALID(it != tensors.end(), "RunNode: input '", name, "' of op '", node.op_type(),
                      "' is missing from the tensor map.");
  return it->second;
}

const Tensor *GetOptionalInput(const NodeProto &node, int index, const TensorMap &tensors) {
  if (index >= node.input_size()) {
    return nullptr;
  }
  const std::string &name = node.input(index);
  if (name.empty()) {
    return nullptr;
  }
  auto it = tensors.find(name);
  EXT_ENFORCE_INVALID(it != tensors.end(), "RunNode: input '", name, "' of op '", node.op_type(),
                      "' is missing from the tensor map.");
  return &it->second;
}

void SetOutput(const NodeProto &node, int index, Tensor result, TensorMap &tensors) {
  const std::string &name = node.output(index);
  EXT_ENFORCE_INVALID(!(name.empty()), "RunNode: op '", node.op_type(), "' output #", index,
                      " is unset (empty name).");
  result.name = name;
  tensors[name] = std::move(result);
}

void SetOutput(const NodeProto &node, int index, Tensor result, RuntimeContext &rt) {
  const std::string &name = node.output(index);
  EXT_ENFORCE_INVALID(!(name.empty()), "RunNode: op '", node.op_type(), "' output #", index,
                      " is unset (empty name).");
  result.name = name;
  rt.Put(name, std::move(result), RuntimeEventKind::kIntermediate);
}

const Sequence &GetInputSequence(const NodeProto &node, int index, const RuntimeContext &rt) {
  const std::string &name = node.input(index);
  EXT_ENFORCE_INVALID(!(name.empty()), "RunNode: op '", node.op_type(), "' sequence input #", index,
                      " is unset (empty name).");
  EXT_ENFORCE_INVALID(rt.HasSequence(name), "RunNode: sequence input '", name, "' of op '",
                      node.op_type(), "' is missing from the sequence map.");
  return rt.GetSequence(name);
}

void SetOutputSequence(const NodeProto &node, int index, Sequence result, RuntimeContext &rt) {
  const std::string &name = node.output(index);
  EXT_ENFORCE_INVALID(!(name.empty()), "RunNode: op '", node.op_type(), "' sequence output #",
                      index, " is unset (empty name).");
  rt.PutSequence(name, std::move(result));
}

void RequireInputCount(const NodeProto &node, int expected) {
  EXT_ENFORCE_INVALID(!(static_cast<int>(node.input_size()) != expected), "RunNode: op '",
                      node.op_type(), "' expects ", expected, " input(s), got ", node.input_size(),
                      ".");
}

void RequireMinInputCount(const NodeProto &node, int min_expected) {
  EXT_ENFORCE_INVALID(!(static_cast<int>(node.input_size()) < min_expected), "RunNode: op '",
                      node.op_type(), "' expects at least ", min_expected, " input(s), got ",
                      node.input_size(), ".");
}

void RequireOutputCount(const NodeProto &node, int expected) {
  EXT_ENFORCE_INVALID(!(static_cast<int>(node.output_size()) != expected), "RunNode: op '",
                      node.op_type(), "' expects ", expected, " output(s), got ",
                      node.output_size(), ".");
}

const AttributeProto *FindAttribute(const NodeProto &node, const std::string &name) {
  for (size_t i = 0; i < node.attribute().size(); ++i) {
    const AttributeProto &attr = node.attribute()[i];
    if (attr.name() == name) {
      return &attr;
    }
  }
  return nullptr;
}

const GraphProto &GetRequiredGraphAttribute(const NodeProto &node, const std::string &name) {
  const AttributeProto *attr = FindAttribute(node, name);
  EXT_ENFORCE_INVALID(attr != nullptr, "RunNode: op '", node.op_type(), "' is missing '", name,
                      "' graph attribute.");
  EXT_ENFORCE_INVALID(!(attr->type() != AttributeProto::AttributeType::GRAPH),
                      "RunNode: attribute '", name, "' of op '", node.op_type(),
                      "' must be a GRAPH.");
  return attr->ref_g();
}

int64_t GetAttributeIntOrDefault(const NodeProto &node, const std::string &name, int64_t fallback) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return fallback;
  }
  EXT_ENFORCE_INVALID(!(attr->type() != AttributeProto::AttributeType::INT), "RunNode: attribute '",
                      name, "' of op '", node.op_type(), "' must be an INT.");
  return attr->i();
}

ParamInts GetAttributeIntsOrDefault(const NodeProto &node, const std::string &name,
                                    const std::vector<int64_t> &fallback) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return fallback;
  }
  EXT_ENFORCE_INVALID(!(attr->type() != AttributeProto::AttributeType::INTS),
                      "RunNode: attribute '", name, "' of op '", node.op_type(), "' must be INTS.");
  ParamInts values;
  values.reserve(attr->ints().size());
  for (size_t i = 0; i < attr->ints().size(); ++i) {
    values.push_back(attr->ints()[i]);
  }
  return values;
}

Shape GetAttributeShapeOrDefault(const NodeProto &node, const std::string &name,
                                 const Shape &fallback) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return fallback;
  }
  EXT_ENFORCE_INVALID(!(attr->type() != AttributeProto::AttributeType::INTS),
                      "RunNode: attribute '", name, "' of op '", node.op_type(), "' must be INTS.");
  Shape values;
  for (size_t i = 0; i < attr->ints().size(); ++i) {
    values.push_back(attr->ints()[i]);
  }
  return values;
}

ParamFloats GetAttributeFloatsOrDefault(const NodeProto &node, const std::string &name,
                                        const std::vector<float> &fallback) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return fallback;
  }
  EXT_ENFORCE_INVALID(!(attr->type() != AttributeProto::AttributeType::FLOATS),
                      "RunNode: attribute '", name, "' of op '", node.op_type(),
                      "' must be FLOATS.");
  ParamFloats values;
  values.reserve(attr->floats().size());
  for (size_t i = 0; i < attr->floats().size(); ++i) {
    values.push_back(attr->floats()[i]);
  }
  return values;
}

ParamStrings GetAttributeStringsOrDefault(const NodeProto &node, const std::string &name,
                                          const std::vector<std::string> &fallback) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return fallback;
  }
  EXT_ENFORCE_INVALID(!(attr->type() != AttributeProto::AttributeType::STRINGS),
                      "RunNode: attribute '", name, "' of op '", node.op_type(),
                      "' must be STRINGS.");
  ParamStrings values;
  values.reserve(attr->strings().size());
  for (size_t i = 0; i < attr->strings().size(); ++i) {
    values.push_back(attr->strings()[i]);
  }
  return values;
}

float GetAttributeFloatOrDefault(const NodeProto &node, const std::string &name, float fallback) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return fallback;
  }
  EXT_ENFORCE_INVALID(!(attr->type() != AttributeProto::AttributeType::FLOAT),
                      "RunNode: attribute '", name, "' of op '", node.op_type(),
                      "' must be a FLOAT.");
  return attr->f();
}

std::string GetAttributeStringOrDefault(const NodeProto &node, const std::string &name,
                                        const std::string &fallback) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return fallback;
  }
  EXT_ENFORCE_INVALID(!(attr->type() != AttributeProto::AttributeType::STRING),
                      "RunNode: attribute '", name, "' of op '", node.op_type(),
                      "' must be a STRING.");
  return attr->s();
}

std::string GetRequiredAttributeString(const NodeProto &node, const std::string &name) {
  const AttributeProto *attr = FindAttribute(node, name);
  EXT_ENFORCE_INVALID(attr != nullptr, "RunNode: op '", node.op_type(), "' is missing '", name,
                      "' STRING attribute.");
  EXT_ENFORCE_INVALID(!(attr->type() != AttributeProto::AttributeType::STRING),
                      "RunNode: attribute '", name, "' of op '", node.op_type(),
                      "' must be a STRING.");
  return attr->s();
}

int64_t GetRequiredAttributeInt(const NodeProto &node, const std::string &name) {
  const AttributeProto *attr = FindAttribute(node, name);
  EXT_ENFORCE_INVALID(attr != nullptr, "RunNode: op '", node.op_type(), "' is missing '", name,
                      "' INT attribute.");
  EXT_ENFORCE_INVALID(!(attr->type() != AttributeProto::AttributeType::INT), "RunNode: attribute '",
                      name, "' of op '", node.op_type(), "' must be an INT.");
  return attr->i();
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
