// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_extensions/backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterZipMapCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset("ai.onnx.ml", 1);
  const OpsetId default_opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::ZipMap zipmap{ctx};

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
      labels_attr->add_ints(v);
    }
    Expect(registry, std::move(node), "test_cc_zipmap_int64", {default_opset, opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromFloat("", {2, 3}, {0.1f, 0.7f, 0.2f, 0.3f, 0.4f, 0.3f});
             Tensor z = zipmap(x, class_labels);

             return IoData{{std::move(x)}, {std::move(z)}};
           },
           "backend-test", "",
           {SequenceTypeSpec(MapTypeSpec(static_cast<int32_t>(DataType::INT64),
                                         TensorTypeSpec(static_cast<int32_t>(DataType::FLOAT))))});
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
      labels_attr->add_strings(utils::String(v));
    }
    Expect(registry, std::move(node), "test_cc_zipmap_string", {default_opset, opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromFloat("", {3}, {0.1f, 0.7f, 0.2f});
             Tensor z = zipmap(x, class_labels);

             return IoData{{std::move(x)}, {std::move(z)}};
           },
           "backend-test", "",
           {SequenceTypeSpec(MapTypeSpec(static_cast<int32_t>(DataType::STRING),
                                         TensorTypeSpec(static_cast<int32_t>(DataType::FLOAT))))});
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
