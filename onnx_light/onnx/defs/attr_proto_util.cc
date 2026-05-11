// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "attr_proto_util.h"

#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

#define ADD_BASIC_ATTR_IMPL(type, enumType, field)                                                 \
  AttributeProto MakeAttribute(std::string attr_name, type value) {                                \
    AttributeProto a;                                                                              \
    a.set_name(std::move(attr_name));                                                              \
    a.set_type(enumType);                                                                          \
    a.set_##field(value);                                                                          \
    return a;                                                                                      \
  }

#define ADD_ATTR_IMPL(type, enumType, field)                                                       \
  AttributeProto MakeAttribute(std::string attr_name, type value) {                                \
    AttributeProto a;                                                                              \
    a.set_name(std::move(attr_name));                                                              \
    a.set_type(enumType);                                                                          \
    a.set_##field(value);                                                                          \
    return a;                                                                                      \
  }

#define ADD_LIST_ATTR_IMPL(type, enumType, field)                                                  \
  AttributeProto MakeAttribute(std::string attr_name, std::vector<type> values) {                  \
    AttributeProto a;                                                                              \
    a.set_name(std::move(attr_name));                                                              \
    a.set_type(enumType);                                                                          \
    for (auto &&val : std::move(values)) {                                                         \
      a.add_##field() = std::move(val);                                                            \
    }                                                                                              \
    return a;                                                                                      \
  }

#define ADD_LIST_PROTO_ATTR_IMPL(type, enumType, field)                                            \
  AttributeProto MakeAttribute(std::string attr_name, std::vector<type> values) {                  \
    AttributeProto a;                                                                              \
    a.set_name(std::move(attr_name));                                                              \
    a.set_type(enumType);                                                                          \
    for (auto &&val : std::move(values)) {                                                         \
      a.add_##field(val);                                                                          \
    }                                                                                              \
    return a;                                                                                      \
  }

ADD_BASIC_ATTR_IMPL(float, AttributeProto::AttributeType::FLOAT, f)
ADD_BASIC_ATTR_IMPL(int64_t, AttributeProto::AttributeType::INT, i)
ADD_BASIC_ATTR_IMPL(int, AttributeProto::AttributeType::INT, i)
ADD_ATTR_IMPL(std::string, AttributeProto::AttributeType::STRING, s)
ADD_ATTR_IMPL(TensorProto, AttributeProto::AttributeType::TENSOR, t)
ADD_ATTR_IMPL(GraphProto, AttributeProto::AttributeType::GRAPH, g)
ADD_ATTR_IMPL(TypeProto, AttributeProto::AttributeType::TYPE_PROTO, tp)
// NOLINTNEXTLINE(performance-unnecessary-value-param)
ADD_LIST_ATTR_IMPL(float, AttributeProto::AttributeType::FLOATS, floats)
// NOLINTNEXTLINE(performance-unnecessary-value-param)
ADD_LIST_ATTR_IMPL(int64_t, AttributeProto::AttributeType::INTS, ints)
ADD_LIST_ATTR_IMPL(std::string, AttributeProto::AttributeType::STRINGS, strings)
ADD_LIST_PROTO_ATTR_IMPL(TensorProto, AttributeProto::AttributeType::TENSORS, tensors)
ADD_LIST_PROTO_ATTR_IMPL(GraphProto, AttributeProto::AttributeType::GRAPHS, graphs)
ADD_LIST_PROTO_ATTR_IMPL(TypeProto, AttributeProto::AttributeType::TYPE_PROTOS, type_protos)

#undef ADD_BASIC_ATTR_IMPL
#undef ADD_ATTR_IMPL
#undef ADD_LIST_ATTR_IMPL
#undef ADD_LIST_PROTO_ATTR_IMPL

AttributeProto MakeRefAttribute(const std::string &attr_name, AttributeProto::AttributeType type) {
  return MakeRefAttribute(attr_name, attr_name, type);
}

AttributeProto MakeRefAttribute(const std::string &attr_name, const std::string &referred_attr_name,
                                AttributeProto::AttributeType type) {
  AttributeProto a;
  a.set_name(attr_name);
  a.set_ref_attr_name(referred_attr_name);
  a.set_type(type);
  return a;
}

} // namespace ONNX_LIGHT_NAMESPACE
