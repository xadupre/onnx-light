// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_kernels/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void RegisterTreeEnsembleClassifierCases(std::vector<TestCase> &registry) {
  // Single-tree binary classifier, single feature.
  //
  // Tree 0:
  //   node 0: feature[0] <= 0.5 => true: node 1 (class 0), false: node 2 (class 1)
  //   node 1: LEAF => class 0, weight 1.0
  //   node 2: LEAF => class 1, weight 1.0
  //
  // x = [[0.0], [1.0]]
  // Y = [0, 1]          (argmax of Z rows)
  // Z = [[1.0, 0.0],    (node 1 contributes to class 0)
  //      [0.0, 1.0]]    (node 2 contributes to class 1)

  const OpsetId opset("ai.onnx.ml", 1);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::TreeEnsembleClassifier cls{ctx};

  NodeProto node;
  node.set_op_type("TreeEnsembleClassifier");
  node.set_domain("ai.onnx.ml");
  node.add_input("x");
  node.add_output("y");
  node.add_output("z");

  auto add_ints = [&](const char *name, const std::vector<int64_t> &vals) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto::AttributeType::INTS);
    for (int64_t v : vals) {
      attr->add_ints(v);
    }
  };
  auto add_floats = [&](const char *name, const std::vector<float> &vals) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto::AttributeType::FLOATS);
    for (float v : vals) {
      attr->add_floats(v);
    }
  };
  auto add_strings = [&](const char *name, const std::vector<std::string> &vals) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto::AttributeType::STRINGS);
    for (const std::string &v : vals) {
      *attr->add_strings() = utils::String(v);
    }
  };
  auto add_string = [&](const char *name, const std::string &val) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto::AttributeType::STRING);
    attr->set_s(val);
  };

  add_ints("nodes_treeids", {0, 0, 0});
  add_ints("nodes_nodeids", {0, 1, 2});
  add_ints("nodes_featureids", {0, 0, 0});
  add_floats("nodes_values", {0.5f, 0.0f, 0.0f});
  add_strings("nodes_modes", {"BRANCH_LEQ", "LEAF", "LEAF"});
  add_ints("nodes_truenodeids", {1, 0, 0});
  add_ints("nodes_falsenodeids", {2, 0, 0});

  add_ints("class_treeids", {0, 0});
  add_ints("class_nodeids", {1, 2});
  add_ints("class_ids", {0, 1});
  add_floats("class_weights", {1.0f, 1.0f});
  add_ints("classlabels_int64s", {0, 1});
  add_string("post_transform", "NONE");

  const std::vector<int64_t> nodes_treeids{0, 0, 0};
  const std::vector<int64_t> nodes_nodeids{0, 1, 2};
  const std::vector<int64_t> nodes_featureids{0, 0, 0};
  const std::vector<float> nodes_values{0.5f, 0.0f, 0.0f};
  const std::vector<std::string> nodes_modes{"BRANCH_LEQ", "LEAF", "LEAF"};
  const std::vector<int64_t> nodes_truenodeids{1, 0, 0};
  const std::vector<int64_t> nodes_falsenodeids{2, 0, 0};
  const std::vector<int64_t> nodes_missing{};
  const std::vector<int64_t> class_treeids{0, 0};
  const std::vector<int64_t> class_nodeids{1, 2};
  const std::vector<int64_t> class_ids{0, 1};
  const std::vector<float> class_weights{1.0f, 1.0f};

  Tensor x = Tensor::FromFloat("", {2, 1}, {0.0f, 1.0f});
  auto yz = cls.operator()<float>(x, nodes_treeids, nodes_nodeids, nodes_featureids, nodes_values,
                                  nodes_modes, nodes_truenodeids, nodes_falsenodeids, nodes_missing,
                                  class_treeids, class_nodeids, class_ids, class_weights,
                                  /*classlabels_int64s=*/std::vector<int64_t>{0, 1},
                                  /*base_values=*/{}, /*post_transform=*/"NONE");

  Expect(node, {x}, {yz.first, yz.second}, "test_cc_treeensembleclassifier_int64_binary",
         {default_opset, opset}, "backend-test", registry);
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
