// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/memory/simple_map.h"
#include "onnx_extensions/backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

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
//
// The map input is passed as a ``Map`` object in ``IoData::maps``; the runtime
// retrieves it by name from ``RuntimeContext::maps()``.
// ---------------------------------------------------------------------------
void RegisterDictVectorizerCases(std::vector<TestCase> &registry, TestMode /*mode*/) {
  const OpsetId opset("ai.onnx.ml", 1);
  const OpsetId default_opset = DefaultOpset(13);

  // string -> int64 dictionary with string vocabulary.
  {
    NodeProto node;
    node.set_op_type("DictVectorizer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");
    const std::vector<std::string> vocab{"a", "c", "b", "z"};
    AddStringsAttr(node, "string_vocabulary", vocab);
    Expect(registry, std::move(node), "test_cc_dict_vectorizer_string_int64",
           {default_opset, opset}, [opset, vocab]() -> IoData {
             const KernelContext dict_ctx{opset};
             const onnx_kernels::kernel::DictVectorizer dict{dict_ctx};

             const std::vector<std::string> keys{"a", "c"};
             const std::vector<int64_t> values{4, 8};
             Map x("x", Tensor::FromStrings("", {2}, keys), Tensor::FromInt64("", {2}, values));
             Tensor y = dict.template operator()<std::string, int64_t>(keys, values, vocab);
             return IoData{{}, {std::move(y)}, {std::move(x)}};
           });
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
    Expect(registry, std::move(node), "test_cc_dict_vectorizer_int64_float", {default_opset, opset},
           [opset, vocab]() -> IoData {
             const KernelContext dict_ctx{opset};
             const onnx_kernels::kernel::DictVectorizer dict{dict_ctx};

             const std::vector<int64_t> keys{10, 30};
             const std::vector<float> values{1.5f, 2.5f};
             Map x("x", Tensor::FromInt64("", {2}, keys), Tensor::FromFloat("", {2}, values));
             Tensor y = dict.template operator()<int64_t, float>(keys, values, vocab);
             return IoData{{}, {std::move(y)}, {std::move(x)}};
           });
  }
}

// ---------------------------------------------------------------------------
// FeatureVectorizer — concatenates variadic input tensors along the trailing
// feature dimension. Mirrors the upstream ONNX
// ``ai.onnx.ml::FeatureVectorizer`` operator (since opset 1).
// ---------------------------------------------------------------------------
void RegisterFeatureVectorizerCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset("ai.onnx.ml", 1);
  const OpsetId default_opset = DefaultOpset(13);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("FeatureVectorizer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x0");
    node.add_input("x1");
    node.add_output("y");

    AddIntsAttr(node, "inputdimensions", {2, 1});

    Expect(registry, std::move(node), "test_cc_feature_vectorizer_two_float_benchmark",
           {default_opset, opset}, {16384, 8192}, {24576}, [opset]() -> IoData {
             const KernelContext fv_ctx{opset};
             const onnx_kernels::kernel::FeatureVectorizer fv{fv_ctx};

             Tensor x0 = RandnTensor(DataType::FLOAT, {8192, 2}, 2711);
             Tensor x1 = RandnTensor(DataType::FLOAT, {8192, 1}, 2712);
             Tensor y = fv({x0, x1}, {2, 1});
             return IoData{{std::move(x0), std::move(x1)}, {std::move(y)}};
           });
    return;
  }

  // Two float inputs concatenated along the feature dimension.
  {
    NodeProto node;
    node.set_op_type("FeatureVectorizer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x0");
    node.add_input("x1");
    node.add_output("y");
    AddIntsAttr(node, "inputdimensions", {2, 1});
    Expect(registry, std::move(node), "test_cc_feature_vectorizer_two_float",
           {default_opset, opset}, [opset]() -> IoData {
             const KernelContext fv_ctx{opset};
             const onnx_kernels::kernel::FeatureVectorizer fv{fv_ctx};

             Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
             Tensor x1 = Tensor::FromFloat("", {2, 1}, {10.0f, 20.0f});
             Tensor y = fv({x0, x1}, {2, 1});

             return IoData{{std::move(x0), std::move(x1)}, {std::move(y)}};
           });
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
    Expect(registry, std::move(node), "test_cc_feature_vectorizer_mixed_dtypes",
           {default_opset, opset}, [opset]() -> IoData {
             const KernelContext fv_ctx{opset};
             const onnx_kernels::kernel::FeatureVectorizer fv{fv_ctx};

             Tensor x0 = Tensor::FromInt64("", {1, 2}, {1, 2});
             Tensor x1 = Tensor::FromFloat("", {1, 2}, {3.5f, 4.5f});
             Tensor y = fv({x0, x1}, {2, 2});

             return IoData{{std::move(x0), std::move(x1)}, {std::move(y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
