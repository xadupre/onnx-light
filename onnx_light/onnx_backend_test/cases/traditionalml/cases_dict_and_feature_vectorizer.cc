// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Promote the single input ValueInfoProto from its placeholder tensor type to
// the actual map(key_type, value_type) type expected by the DictVectorizer
// schema. Mirrors ``cases_zipmap.cc``'s output-side helper.
void PromoteInputToMapType(std::vector<TestCase> &registry, int32_t key_type, int32_t value_type) {
  GraphProto &graph = registry.back().model.ref_graph();
  ValueInfoProto &in_vi = *graph.mutable_input(0);
  TypeProto &in_tp = in_vi.ref_type();
  TypeProto::Map *in_map = in_tp.mutable_map_type();
  in_map->set_key_type(key_type);
  TypeProto *map_value_type = in_map->mutable_value_type();
  map_value_type->mutable_tensor_type()->set_elem_type(value_type);
  in_tp.reset_tensor_type();
}

void AddStringsAttr(NodeProto &node, const char *name, const std::vector<std::string> &values) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::STRINGS);
  for (const std::string &v : values) {
    attr->strings().push_back(utils::String(v));
  }
}

void AddIntsAttr(NodeProto &node, const char *name, const std::vector<int64_t> &values) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::INTS);
  for (int64_t v : values) {
    attr->add_ints(v);
  }
}

} // namespace

// ---------------------------------------------------------------------------
// DictVectorizer — converts a dictionary into a 1-D output tensor whose length
// equals the vocabulary attribute length. Mirrors the upstream ONNX
// ``ai.onnx.ml::DictVectorizer`` operator (since opset 1).
// ---------------------------------------------------------------------------
void RegisterDictVectorizerCases(std::vector<TestCase> &registry) {
  const OpsetId opset("ai.onnx.ml", 1);
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::DictVectorizer dict{ctx};

  // string -> int64 dictionary with string vocabulary.
  {
    NodeProto node;
    node.set_op_type("DictVectorizer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const std::vector<std::string> vocab{"a", "c", "b", "z"};
    AddStringsAttr(node, "string_vocabulary", vocab);

    // Placeholder input tensor (its bytes are unused by the runtime; the
    // ValueInfo is rewritten below to declare a map(string, int64) input).
    Tensor x = Tensor::FromInt64("", {1}, {0});
    Tensor y = dict.operator()<std::string, int64_t>({"a", "c"}, {4, 8}, vocab);

    Expect(node, {x}, {y}, "test_cc_dict_vectorizer_string_int64", {default_opset, opset},
           "backend-test", registry);
    PromoteInputToMapType(registry, static_cast<int32_t>(DataType::STRING),
                          static_cast<int32_t>(DataType::INT64));
  }

  // int64 -> float dictionary with int64 vocabulary.
  {
    NodeProto node;
    node.set_op_type("DictVectorizer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const std::vector<int64_t> vocab{10, 20, 30};
    AddIntsAttr(node, "int64_vocabulary", vocab);

    Tensor x = Tensor::FromInt64("", {1}, {0});
    Tensor y = dict.operator()<int64_t, float>({10, 30}, {1.5f, 2.5f}, vocab);

    Expect(node, {x}, {y}, "test_cc_dict_vectorizer_int64_float", {default_opset, opset},
           "backend-test", registry);
    PromoteInputToMapType(registry, static_cast<int32_t>(DataType::INT64),
                          static_cast<int32_t>(DataType::FLOAT));
  }
}

// ---------------------------------------------------------------------------
// FeatureVectorizer — concatenates variadic input tensors along the trailing
// feature dimension. Mirrors the upstream ONNX
// ``ai.onnx.ml::FeatureVectorizer`` operator (since opset 1).
// ---------------------------------------------------------------------------
void RegisterFeatureVectorizerCases(std::vector<TestCase> &registry) {
  const OpsetId opset("ai.onnx.ml", 1);
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::FeatureVectorizer fv{ctx};

  // Two float inputs concatenated along the feature dimension.
  {
    NodeProto node;
    node.set_op_type("FeatureVectorizer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x0");
    node.add_input("x1");
    node.add_output("y");

    AddIntsAttr(node, "inputdimensions", {2, 1});

    Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor x1 = Tensor::FromFloat("", {2, 1}, {10.0f, 20.0f});
    Tensor y = fv({x0, x1}, {2, 1});

    Expect(node, {x0, x1}, {y}, "test_cc_feature_vectorizer_two_float", {default_opset, opset},
           "backend-test", registry);
  }

  // Mixed dtypes (int64 + float) cast to float.
  {
    NodeProto node;
    node.set_op_type("FeatureVectorizer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x0");
    node.add_input("x1");
    node.add_output("y");

    AddIntsAttr(node, "inputdimensions", {2, 2});

    Tensor x0 = Tensor::FromInt64("", {1, 2}, {1, 2});
    Tensor x1 = Tensor::FromFloat("", {1, 2}, {3.5f, 4.5f});
    Tensor y = fv({x0, x1}, {2, 2});

    Expect(node, {x0, x1}, {y}, "test_cc_feature_vectorizer_mixed_dtypes", {default_opset, opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
