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

void RegisterTreeEnsembleRegressorCases(std::vector<TestCase> &registry) {
  // Two-tree ensemble, single feature, single target, aggregate=SUM.
  //
  // Tree 0:
  //   node 0: feature[0] <= 2.0 => true: node 1, false: node 2
  //   node 1: LEAF => target 0, weight 1.0
  //   node 2: LEAF => target 0, weight 3.0
  //
  // Tree 1:
  //   node 0: feature[0] <= 1.0 => true: node 1, false: node 2
  //   node 1: LEAF => target 0, weight 2.0
  //   node 2: LEAF => target 0, weight 4.0
  //
  // x = [[0.5], [3.0]]
  // y[0] = 1.0 + 2.0 = 3.0   (both trees follow true branch)
  // y[1] = 3.0 + 4.0 = 7.0   (both trees follow false branch)

  const OpsetId opset("ai.onnx.ml", 1);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::TreeEnsembleRegressor reg{ctx};

  NodeProto node;
  node.set_op_type("TreeEnsembleRegressor");
  node.set_domain("ai.onnx.ml");
  node.add_input("x");
  node.add_output("y");

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
  auto add_string_list = [&](const char *name, const std::vector<std::string> &vals) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto::AttributeType::STRINGS);
    for (const std::string &v : vals) {
      *attr->add_strings() = utils::String(v);
    }
  };
  auto add_int = [&](const char *name, int64_t val) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto::AttributeType::INT);
    attr->set_i(val);
  };
  auto add_string = [&](const char *name, const std::string &val) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto::AttributeType::STRING);
    attr->set_s(val);
  };

  add_ints("nodes_treeids", {0, 0, 0, 1, 1, 1});
  add_ints("nodes_nodeids", {0, 1, 2, 0, 1, 2});
  add_ints("nodes_featureids", {0, 0, 0, 0, 0, 0});
  add_floats("nodes_values", {2.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
  add_string_list("nodes_modes", {"BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF"});
  add_ints("nodes_truenodeids", {1, 0, 0, 1, 0, 0});
  add_ints("nodes_falsenodeids", {2, 0, 0, 2, 0, 0});

  add_ints("target_treeids", {0, 0, 1, 1});
  add_ints("target_nodeids", {1, 2, 1, 2});
  add_ints("target_ids", {0, 0, 0, 0});
  add_floats("target_weights", {1.0f, 3.0f, 2.0f, 4.0f});

  add_int("n_targets", 1);
  add_string("aggregate_function", "SUM");
  add_string("post_transform", "NONE");

  const std::vector<int64_t> nodes_treeids{0, 0, 0, 1, 1, 1};
  const std::vector<int64_t> nodes_nodeids{0, 1, 2, 0, 1, 2};
  const std::vector<int64_t> nodes_featureids{0, 0, 0, 0, 0, 0};
  const std::vector<float> nodes_values{2.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
  const std::vector<std::string> nodes_modes{"BRANCH_LEQ", "LEAF", "LEAF",
                                             "BRANCH_LEQ", "LEAF", "LEAF"};
  const std::vector<int64_t> nodes_truenodeids{1, 0, 0, 1, 0, 0};
  const std::vector<int64_t> nodes_falsenodeids{2, 0, 0, 2, 0, 0};
  const std::vector<int64_t> nodes_missing{};
  const std::vector<int64_t> target_treeids{0, 0, 1, 1};
  const std::vector<int64_t> target_nodeids{1, 2, 1, 2};
  const std::vector<int64_t> target_ids{0, 0, 0, 0};
  const std::vector<float> target_weights{1.0f, 3.0f, 2.0f, 4.0f};

  Tensor x = Tensor::FromFloat("", {2, 1}, {0.5f, 3.0f});
  Tensor y =
      reg.operator()<float>(x, nodes_treeids, nodes_nodeids, nodes_featureids, nodes_values,
                            nodes_modes, nodes_truenodeids, nodes_falsenodeids, nodes_missing,
                            target_treeids, target_nodeids, target_ids, target_weights,
                            /*n_targets=*/1, /*aggregate_function=*/"SUM",
                            /*post_transform=*/"NONE", /*base_values=*/{});

  Expect(node, {x}, {y}, "test_cc_treeensembleregressor_sum_single_target", {default_opset, opset},
         "backend-test", registry);
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
