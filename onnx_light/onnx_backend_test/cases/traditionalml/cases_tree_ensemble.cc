// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterTreeEnsembleCases(std::vector<TestCase> &registry) {
  // Single-tree ensemble (v5 encoding), single feature, single target.
  //
  // Tree 0 (root at node index 0):
  //   node 0: feature[0] <= 0.5 (mode=BRANCH_LEQ=0)
  //           true  => leaf index 0 (leaf_targetids[0]=0, leaf_weights[0]=1.0)
  //           false => leaf index 1 (leaf_targetids[1]=0, leaf_weights[1]=2.0)
  //
  // x = [[0.0], [1.0]]
  // y[0,0] = 1.0   (follows true  => leaf 0, weight 1.0)
  // y[1,0] = 2.0   (follows false => leaf 1, weight 2.0)

  const OpsetId opset("ai.onnx.ml", 5);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::TreeEnsemble te{ctx};

  NodeProto node;
  node.set_op_type("TreeEnsemble");
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
  auto add_int = [&](const char *name, int64_t val) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto::AttributeType::INT);
    attr->set_i(val);
  };
  auto add_float_tensor = [&](const char *name, const std::vector<float> &vals) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *tp = attr->add_t();
    tp->set_data_type(TensorProto::FLOAT);
    tp->add_dims(static_cast<uint64_t>(vals.size()));
    std::vector<uint8_t> raw(vals.size() * sizeof(float));
    std::memcpy(raw.data(), vals.data(), raw.size());
    tp->set_raw_data(utils::ByteSpan(raw));
  };

  add_ints("nodes_featureids", {0});
  add_ints("nodes_truenodeids", {0});
  add_ints("nodes_falsenodeids", {1});
  add_ints("nodes_trueleafs", {1});
  add_ints("nodes_falseleafs", {1});

  add_float_tensor("nodes_splits", {0.5f});

  {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("nodes_modes");
    attr->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *tp = attr->add_t();
    tp->set_data_type(TensorProto::UINT8);
    tp->add_dims(1);
    std::vector<uint8_t> raw{0x00};
    tp->set_raw_data(utils::ByteSpan(raw));
  }

  add_ints("leaf_targetids", {0, 0});
  add_float_tensor("leaf_weights", {1.0f, 2.0f});

  add_ints("tree_roots", {0});
  add_int("n_targets", 1);
  add_int("aggregate_function", 1);
  add_int("post_transform", 0);

  const std::vector<int64_t> tree_roots_v{0};
  const std::vector<int64_t> nodes_featureids_v{0};
  const std::vector<float> nodes_splits_v{0.5f};
  const std::vector<uint8_t> nodes_modes_v{0};
  const std::vector<int64_t> nodes_truenodeids_v{0};
  const std::vector<int64_t> nodes_falsenodeids_v{1};
  const std::vector<int64_t> nodes_trueleafs_v{1};
  const std::vector<int64_t> nodes_falseleafs_v{1};
  const std::vector<int64_t> nodes_missing_v{};
  const std::vector<int64_t> leaf_targetids_v{0, 0};
  const std::vector<float> leaf_weights_v{1.0f, 2.0f};

  Tensor x = Tensor::FromFloat("", {2, 1}, {0.0f, 1.0f});
  Tensor y =
      te.operator()<float>(x, tree_roots_v, nodes_featureids_v, nodes_splits_v, nodes_modes_v,
                           nodes_truenodeids_v, nodes_falsenodeids_v, nodes_trueleafs_v,
                           nodes_falseleafs_v, nodes_missing_v, leaf_targetids_v, leaf_weights_v,
                           /*n_targets=*/1, /*aggregate_function=*/1,
                           /*post_transform=*/0);

  Expect(node, {x}, {y}, "test_cc_treeensemble_single_tree_float", {default_opset, opset},
         "backend-test", registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
