// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

void PromoteOutputToSequenceMapType(std::vector<TestCase> &registry, int32_t key_type) {
  GraphProto &graph = registry.back().model.ref_graph();
  ValueInfoProto &out_vi = *graph.mutable_output(0);
  TypeProto &out_tp = out_vi.ref_type();
  TypeProto::Sequence *out_seq = out_tp.mutable_sequence_type();
  TypeProto *out_elem = out_seq->mutable_elem_type();
  TypeProto::Map *out_map = out_elem->mutable_map_type();
  out_map->set_key_type(key_type);
  TypeProto *map_value_type = out_map->mutable_value_type();
  map_value_type->mutable_tensor_type()->set_elem_type(
      static_cast<int>(TensorProto::DataType::FLOAT));
  out_tp.reset_tensor_type();
}

} // namespace

void RegisterZipMapCases(std::vector<TestCase> &registry) {
  const OpsetId opset("ai.onnx.ml", 1);
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::ZipMap zipmap{ctx};

  // int64-key variant.
  {
    NodeProto node;
    node.set_op_type("ZipMap");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("z");

    const std::vector<int64_t> class_labels{10, 20, 30};
    AttributeProto *labels_attr = node.add_attribute();
    labels_attr->set_name("classlabels_int64s");
    labels_attr->set_type(AttributeProto::AttributeType::INTS);
    for (int64_t v : class_labels) {
      labels_attr->ints().push_back(v);
    }

    Tensor x = Tensor::FromFloat("", {2, 3}, {0.1f, 0.7f, 0.2f, 0.3f, 0.4f, 0.3f});
    Tensor z = zipmap(x, class_labels);

    Expect(node, {x}, {z}, "test_cc_zipmap_int64", {default_opset, opset}, "backend-test",
           registry);
    PromoteOutputToSequenceMapType(registry, static_cast<int32_t>(TensorProto::DataType::INT64));
  }

  // string-key variant.
  {
    NodeProto node;
    node.set_op_type("ZipMap");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("z");

    const std::vector<std::string> class_labels{"a", "b", "c"};
    AttributeProto *labels_attr = node.add_attribute();
    labels_attr->set_name("classlabels_strings");
    labels_attr->set_type(AttributeProto::AttributeType::STRINGS);
    for (const std::string &v : class_labels) {
      labels_attr->strings().push_back(utils::String(v));
    }

    Tensor x = Tensor::FromFloat("", {3}, {0.1f, 0.7f, 0.2f});
    Tensor z = zipmap(x, class_labels);

    Expect(node, {x}, {z}, "test_cc_zipmap_string", {default_opset, opset}, "backend-test",
           registry);
    PromoteOutputToSequenceMapType(registry, static_cast<int32_t>(TensorProto::DataType::STRING));
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
