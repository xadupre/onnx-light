// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

/// Helpers for populating attributes on a ``TreeEnsemble`` ``NodeProto``.
void AddInts(NodeProto &node, const char *name, const std::vector<int64_t> &vals) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::INTS);
  for (int64_t v : vals) {
    attr->add_ints(v);
  }
}

void AddInt(NodeProto &node, const char *name, int64_t val) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(val);
}

template <typename T>
void AddTypedTensor(NodeProto &node, const char *name, TensorProto::DataType dtype,
                    const std::vector<T> &vals) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto *tp = attr->add_t();
  tp->set_data_type(dtype);
  tp->add_dims(static_cast<uint64_t>(vals.size()));
  std::vector<uint8_t> raw(vals.size() * sizeof(T));
  std::memcpy(raw.data(), vals.data(), raw.size());
  tp->set_raw_data(utils::ByteSpan(raw));
}

void AddUint8Tensor(NodeProto &node, const char *name, const std::vector<uint8_t> &vals) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto *tp = attr->add_t();
  tp->set_data_type(TensorProto::UINT8);
  tp->add_dims(static_cast<uint64_t>(vals.size()));
  tp->set_raw_data(utils::ByteSpan(vals));
}

} // namespace

void RegisterTreeEnsembleCases(std::vector<TestCase> &registry) {
  const OpsetId opset("ai.onnx.ml", 5);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::TreeEnsemble tree_ens{ctx};

  // ---------------------------------------------------------------------------
  // Single-tree ensemble (v5 encoding), single feature, single target, float.
  //
  // Tree 0 (root at node index 0):
  //   node 0: feature[0] <= 0.5 (mode=BRANCH_LEQ=0)
  //           true  => leaf index 0 (leaf_targetids[0]=0, leaf_weights[0]=1.0)
  //           false => leaf index 1 (leaf_targetids[1]=0, leaf_weights[1]=2.0)
  //
  // x = [[0.0], [1.0]] => y = [[1.0], [2.0]].
  // ---------------------------------------------------------------------------
  {
    NodeProto node;
    node.set_op_type("TreeEnsemble");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    AddInts(node, "nodes_featureids", {0});
    AddInts(node, "nodes_truenodeids", {0});
    AddInts(node, "nodes_falsenodeids", {1});
    AddInts(node, "nodes_trueleafs", {1});
    AddInts(node, "nodes_falseleafs", {1});
    AddTypedTensor<float>(node, "nodes_splits", TensorProto::FLOAT, {0.5f});
    AddUint8Tensor(node, "nodes_modes", {0x00});
    AddInts(node, "leaf_targetids", {0, 0});
    AddTypedTensor<float>(node, "leaf_weights", TensorProto::FLOAT, {1.0f, 2.0f});
    AddInts(node, "tree_roots", {0});
    AddInt(node, "n_targets", 1);
    AddInt(node, "aggregate_function", 1);
    AddInt(node, "post_transform", 0);

    Tensor x = Tensor::FromFloat("", {2, 1}, {0.0f, 1.0f});
    Tensor y =
        tree_ens.operator()<float>(x, /*tree_roots=*/{0}, /*nodes_featureids=*/{0},
                                   /*nodes_splits=*/{0.5f}, /*nodes_modes=*/{0},
                                   /*nodes_truenodeids=*/{0}, /*nodes_falsenodeids=*/{1},
                                   /*nodes_trueleafs=*/{1}, /*nodes_falseleafs=*/{1},
                                   /*nodes_missing=*/{}, /*leaf_targetids=*/{0, 0},
                                   /*leaf_weights=*/{1.0f, 2.0f}, /*membership_values=*/{},
                                   /*n_targets=*/1, /*aggregate_function=*/1, /*post_transform=*/0);

    Expect(node, {x}, {y}, "test_cc_treeensemble_single_tree_float", {default_opset, opset},
           "backend-test", registry);
  }

  // ---------------------------------------------------------------------------
  // Mirrors upstream ONNX node test
  // ``test_ai_onnx_ml_tree_ensemble_single_tree`` (see
  // ``onnx/backend/test/case/node/ai_onnx_ml/tree_ensemble.py``): single tree
  // with two interior nodes, two-feature ``double`` input, ``n_targets=2``.
  // ---------------------------------------------------------------------------
  {
    NodeProto node;
    node.set_op_type("TreeEnsemble");
    node.set_domain("ai.onnx.ml");
    node.add_input("X");
    node.add_output("Y");

    AddInt(node, "n_targets", 2);
    AddInt(node, "aggregate_function", 1);
    AddInt(node, "post_transform", 0);
    AddInts(node, "tree_roots", {0});
    AddUint8Tensor(node, "nodes_modes", {0x00, 0x00, 0x00});
    AddInts(node, "nodes_featureids", {0, 0, 0});
    AddTypedTensor<double>(node, "nodes_splits", TensorProto::DOUBLE, {3.14, 1.2, 4.2});
    AddInts(node, "nodes_truenodeids", {1, 0, 1});
    AddInts(node, "nodes_trueleafs", {0, 1, 1});
    AddInts(node, "nodes_falsenodeids", {2, 2, 3});
    AddInts(node, "nodes_falseleafs", {0, 1, 1});
    AddInts(node, "leaf_targetids", {0, 1, 0, 1});
    AddTypedTensor<double>(node, "leaf_weights", TensorProto::DOUBLE, {5.23, 12.12, -12.23, 7.21});

    Tensor x = Tensor::FromDouble("", {3, 2}, {1.2, 3.4, -0.12, 1.66, 4.14, 1.77});
    Tensor y = tree_ens.operator()<double>(
        x, /*tree_roots=*/{0}, /*nodes_featureids=*/{0, 0, 0},
        /*nodes_splits=*/{3.14, 1.2, 4.2}, /*nodes_modes=*/{0, 0, 0},
        /*nodes_truenodeids=*/{1, 0, 1}, /*nodes_falsenodeids=*/{2, 2, 3},
        /*nodes_trueleafs=*/{0, 1, 1}, /*nodes_falseleafs=*/{0, 1, 1},
        /*nodes_missing=*/{}, /*leaf_targetids=*/{0, 1, 0, 1},
        /*leaf_weights=*/{5.23, 12.12, -12.23, 7.21}, /*membership_values=*/{},
        /*n_targets=*/2, /*aggregate_function=*/1, /*post_transform=*/0);

    Expect(node, {x}, {y}, "test_cc_treeensemble_single_tree_double", {default_opset, opset},
           "backend-test", registry);
  }

  // ---------------------------------------------------------------------------
  // Mirrors upstream ONNX node test
  // ``test_ai_onnx_ml_tree_ensemble_set_membership`` (see
  // ``onnx/backend/test/case/node/ai_onnx_ml/tree_ensemble.py``): single tree
  // with two ``BRANCH_MEMBER`` (mode 6) interior nodes whose set members are
  // encoded in ``membership_values`` and delimited by ``NaN`` sentinels.
  // ---------------------------------------------------------------------------
  {
    NodeProto node;
    node.set_op_type("TreeEnsemble");
    node.set_domain("ai.onnx.ml");
    node.add_input("X");
    node.add_output("Y");

    const float kNaN = std::numeric_limits<float>::quiet_NaN();

    AddInt(node, "n_targets", 4);
    AddInt(node, "aggregate_function", 1);
    AddInt(node, "post_transform", 0);
    AddInts(node, "tree_roots", {0});
    AddUint8Tensor(node, "nodes_modes", {0x00, 0x06, 0x06});
    AddInts(node, "nodes_featureids", {0, 0, 0});
    AddTypedTensor<float>(node, "nodes_splits", TensorProto::FLOAT, {11.0f, 232344.0f, kNaN});
    AddInts(node, "nodes_truenodeids", {1, 0, 1});
    AddInts(node, "nodes_trueleafs", {0, 1, 1});
    AddInts(node, "nodes_falsenodeids", {2, 2, 3});
    AddInts(node, "nodes_falseleafs", {1, 0, 1});
    AddInts(node, "leaf_targetids", {0, 1, 2, 3});
    AddTypedTensor<float>(node, "leaf_weights", TensorProto::FLOAT, {1.0f, 10.0f, 1000.0f, 100.0f});
    AddTypedTensor<float>(node, "membership_values", TensorProto::FLOAT,
                          {1.2f, 3.7f, 8.0f, 9.0f, kNaN, 12.0f, 7.0f, kNaN});

    Tensor x = Tensor::FromFloat("", {6, 1}, {1.2f, 3.4f, -0.12f, kNaN, 12.0f, 7.0f});
    Tensor y = tree_ens.operator()<float>(
        x, /*tree_roots=*/{0}, /*nodes_featureids=*/{0, 0, 0},
        /*nodes_splits=*/{11.0f, 232344.0f, kNaN}, /*nodes_modes=*/{0, 6, 6},
        /*nodes_truenodeids=*/{1, 0, 1}, /*nodes_falsenodeids=*/{2, 2, 3},
        /*nodes_trueleafs=*/{0, 1, 1}, /*nodes_falseleafs=*/{1, 0, 1},
        /*nodes_missing=*/{}, /*leaf_targetids=*/{0, 1, 2, 3},
        /*leaf_weights=*/{1.0f, 10.0f, 1000.0f, 100.0f},
        /*membership_values=*/{1.2f, 3.7f, 8.0f, 9.0f, kNaN, 12.0f, 7.0f, kNaN},
        /*n_targets=*/4, /*aggregate_function=*/1, /*post_transform=*/0);

    Expect(node, {x}, {y}, "test_cc_treeensemble_set_membership", {default_opset, opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
