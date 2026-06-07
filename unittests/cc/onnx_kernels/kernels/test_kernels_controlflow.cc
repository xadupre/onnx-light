// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/controlflow/include_controlflow_kernels.h"
#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/run_nodes.h"
#include "onnx_kernels/runtime_context.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::If;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::Loop;
using onnx_kernels::kernel::Scan;

namespace Test {

TEST(KernelClass, IfClassSelectsThenBranchWhenCondTrue) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  Tensor cond("", onnx_kernels::DataType::BOOL, {}, {1});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor out = if_kernel(cond, then_v, else_v);
  ASSERT_EQ(out.element_count(), 2);
  EXPECT_EQ(out.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  EXPECT_FLOAT_EQ(out.AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(out.AsFloat()[1], 2.0f);
}

TEST(KernelClass, IfClassSelectsElseBranchWhenCondFalse) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  Tensor cond("", onnx_kernels::DataType::BOOL, {}, {0});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor out = if_kernel(cond, then_v, else_v);
  ASSERT_EQ(out.element_count(), 2);
  EXPECT_FLOAT_EQ(out.AsFloat()[0], 3.0f);
  EXPECT_FLOAT_EQ(out.AsFloat()[1], 4.0f);
}

TEST(KernelClass, IfClassRejectsInvalidInputs) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  Tensor cond_bool("", onnx_kernels::DataType::BOOL, {}, {1});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

  // Non-bool cond is rejected.
  Tensor cond_float = Tensor::FromFloat("", {}, {1.0f});
  EXPECT_THROW((void)if_kernel(cond_float, then_v, else_v), std::invalid_argument);

  // Multi-element cond is rejected.
  Tensor cond_vec("", onnx_kernels::DataType::BOOL, {2}, {1, 0});
  EXPECT_THROW((void)if_kernel(cond_vec, then_v, else_v), std::invalid_argument);

  // Mismatched branch types are rejected.
  Tensor else_int = Tensor::From<int32_t>("", {2}, {3, 4});
  EXPECT_THROW((void)if_kernel(cond_bool, then_v, else_int), std::invalid_argument);

  // Mismatched branch shapes are rejected.
  Tensor else_short = Tensor::FromFloat("", {1}, {3.0f});
  EXPECT_THROW((void)if_kernel(cond_bool, then_v, else_short), std::invalid_argument);
}

TEST(KernelClass, IfInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

  // cond = true → then-branch.
  {
    Tensor cond("", onnx_kernels::DataType::BOOL, {}, {1});
    Tensor out("", onnx_kernels::DataType::FLOAT, {2}, std::vector<uint8_t>(2 * sizeof(float)));
    if_kernel(cond, then_v, else_v, out);
    EXPECT_FLOAT_EQ(out.AsFloat()[0], 1.0f);
    EXPECT_FLOAT_EQ(out.AsFloat()[1], 2.0f);
  }

  // cond = false → else-branch.
  {
    Tensor cond("", onnx_kernels::DataType::BOOL, {}, {0});
    Tensor out("", onnx_kernels::DataType::FLOAT, {2}, std::vector<uint8_t>(2 * sizeof(float)));
    if_kernel(cond, then_v, else_v, out);
    EXPECT_FLOAT_EQ(out.AsFloat()[0], 3.0f);
    EXPECT_FLOAT_EQ(out.AsFloat()[1], 4.0f);
  }
}

TEST(KernelClass, IfInPlaceRejectsBadOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  Tensor cond("", onnx_kernels::DataType::BOOL, {}, {1});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

  // Wrong dtype.
  Tensor bad_dtype("", onnx_kernels::DataType::INT32, {2},
                   std::vector<uint8_t>(2 * sizeof(int32_t)));
  EXPECT_THROW(if_kernel(cond, then_v, else_v, bad_dtype), std::invalid_argument);

  // Wrong shape.
  Tensor bad_shape("", onnx_kernels::DataType::FLOAT, {3}, std::vector<uint8_t>(3 * sizeof(float)));
  EXPECT_THROW(if_kernel(cond, then_v, else_v, bad_shape), std::invalid_argument);

  // Wrong buffer byte count.
  Tensor bad_bytes("", onnx_kernels::DataType::FLOAT, {2}, std::vector<uint8_t>(1 * sizeof(float)));
  EXPECT_THROW(if_kernel(cond, then_v, else_v, bad_bytes), std::invalid_argument);
}

namespace {
// Builds a subgraph containing a single FLOAT initializer and an ``Add``
// node that doubles it; declares the doubled tensor as the only output.
// This avoids depending on the ``Constant`` op which is not registered in
// :cpp:func:`KernelDispatchTable`.
void BuildDoubleInitBranchGraph(GraphProto &g, const std::string &graph_name,
                                const std::string &init_name, const std::string &output_name,
                                float value) {
  g.set_name(graph_name);
  TensorProto *init = g.add_initializer();
  init->set_name(init_name);
  init->set_data_type(onnx_kernels::DataType::FLOAT);
  init->add_dims(1);
  init->add_float_data(value);
  NodeProto *node = g.add_node();
  node->set_op_type("Add");
  node->add_input(init_name);
  node->add_input(init_name);
  node->add_output(output_name);
  ValueInfoProto *vi = g.add_output();
  vi->set_name(output_name);
  TypeProto::Tensor *tt = vi->mutable_type()->mutable_tensor_type();
  tt->set_elem_type(onnx_kernels::DataType::FLOAT);
  tt->mutable_shape()->add_dim()->set_dim_value(1);
}
} // namespace

TEST(KernelClass, IfBranchOverloadExecutesThenSubgraph) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  onnx_kernels::RuntimeContext rt(ctx);

  GraphProto then_graph;
  GraphProto else_graph;
  BuildDoubleInitBranchGraph(then_graph, "then_g", "t", "out", 5.0f);
  BuildDoubleInitBranchGraph(else_graph, "else_g", "e", "out", -1.0f);

  Tensor cond_true("", onnx_kernels::DataType::BOOL, {}, {1});
  std::vector<Tensor> outs = if_kernel(cond_true, then_graph, else_graph, rt);
  ASSERT_EQ(outs.size(), 1u);
  ASSERT_EQ(outs[0].element_count(), 1);
  EXPECT_FLOAT_EQ(outs[0].AsFloat()[0], 10.0f);
}

TEST(KernelClass, IfBranchOverloadExecutesElseSubgraph) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  onnx_kernels::RuntimeContext rt(ctx);

  GraphProto then_graph;
  GraphProto else_graph;
  BuildDoubleInitBranchGraph(then_graph, "then_g", "t", "out", 5.0f);
  BuildDoubleInitBranchGraph(else_graph, "else_g", "e", "out", -1.0f);

  Tensor cond_false("", onnx_kernels::DataType::BOOL, {}, {0});
  std::vector<Tensor> outs = if_kernel(cond_false, then_graph, else_graph, rt);
  ASSERT_EQ(outs.size(), 1u);
  EXPECT_FLOAT_EQ(outs[0].AsFloat()[0], -2.0f);
}

TEST(KernelClass, IfBranchOverloadInheritsOuterScope) {
  // Validates that the kernel inherits the caller's tensor map so a branch
  // subgraph can reference an outer-scope value by name (here ``x``,
  // operated on by ``Neg`` in the then-branch).
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  onnx_kernels::RuntimeContext rt(ctx);
  Tensor x = Tensor::FromFloat("x", {3}, {1.0f, -2.0f, 3.0f});
  rt.Set("x", x);

  GraphProto then_graph;
  then_graph.set_name("neg_x");
  {
    NodeProto *n = then_graph.add_node();
    n->set_op_type("Neg");
    n->add_input("x");
    n->add_output("y");
  }
  ValueInfoProto *vi = then_graph.add_output();
  vi->set_name("y");
  TypeProto::Tensor *tt = vi->mutable_type()->mutable_tensor_type();
  tt->set_elem_type(onnx_kernels::DataType::FLOAT);
  tt->mutable_shape()->add_dim()->set_dim_value(3);

  GraphProto else_graph;
  else_graph.set_name("abs_x");
  {
    NodeProto *n = else_graph.add_node();
    n->set_op_type("Abs");
    n->add_input("x");
    n->add_output("y");
  }
  ValueInfoProto *vi2 = else_graph.add_output();
  vi2->set_name("y");
  TypeProto::Tensor *tt2 = vi2->mutable_type()->mutable_tensor_type();
  tt2->set_elem_type(onnx_kernels::DataType::FLOAT);
  tt2->mutable_shape()->add_dim()->set_dim_value(3);

  Tensor cond_true("", onnx_kernels::DataType::BOOL, {}, {1});
  std::vector<Tensor> outs = if_kernel(cond_true, then_graph, else_graph, rt);
  ASSERT_EQ(outs.size(), 1u);
  ASSERT_EQ(outs[0].element_count(), 3);
  EXPECT_FLOAT_EQ(outs[0].AsFloat()[0], -1.0f);
  EXPECT_FLOAT_EQ(outs[0].AsFloat()[1], 2.0f);
  EXPECT_FLOAT_EQ(outs[0].AsFloat()[2], -3.0f);
  // The caller's tensor map is not polluted by the subgraph's intermediate
  // names ("y" in this case is only present in the child context).
  EXPECT_FALSE(rt.Has("y"));
}

TEST(KernelClass, IfBranchOverloadReturnsAllOutputsInOrder) {
  // Validates that the kernel returns *every* declared subgraph output, in
  // the order declared by ``branch.output()``.
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  onnx_kernels::RuntimeContext rt(ctx);

  GraphProto then_graph;
  then_graph.set_name("two_outputs");
  // Initializer "a" of shape [2] = [3, 4]; initializer "b" of shape [1] = [5].
  {
    TensorProto *a = then_graph.add_initializer();
    a->set_name("a");
    a->set_data_type(onnx_kernels::DataType::FLOAT);
    a->add_dims(2);
    a->add_float_data(3.0f);
    a->add_float_data(4.0f);
    TensorProto *b = then_graph.add_initializer();
    b->set_name("b");
    b->set_data_type(onnx_kernels::DataType::FLOAT);
    b->add_dims(1);
    b->add_float_data(5.0f);
  }
  // Two ``Neg`` nodes producing ``out_a`` and ``out_b``.
  for (const auto &p :
       std::vector<std::pair<std::string, std::string>>{{"a", "out_a"}, {"b", "out_b"}}) {
    NodeProto *n = then_graph.add_node();
    n->set_op_type("Neg");
    n->add_input(p.first);
    n->add_output(p.second);
  }
  for (const auto &p : std::vector<std::pair<std::string, int64_t>>{{"out_a", 2}, {"out_b", 1}}) {
    ValueInfoProto *vi = then_graph.add_output();
    vi->set_name(p.first);
    TypeProto::Tensor *tt = vi->mutable_type()->mutable_tensor_type();
    tt->set_elem_type(onnx_kernels::DataType::FLOAT);
    tt->mutable_shape()->add_dim()->set_dim_value(static_cast<uint64_t>(p.second));
  }

  // Else-branch must declare the same number of outputs (validated by the
  // kernel) but their values are not reached when cond is true.
  GraphProto else_graph = then_graph;

  Tensor cond_true("", onnx_kernels::DataType::BOOL, {}, {1});
  std::vector<Tensor> outs = if_kernel(cond_true, then_graph, else_graph, rt);
  ASSERT_EQ(outs.size(), 2u);
  ASSERT_EQ(outs[0].element_count(), 2);
  EXPECT_FLOAT_EQ(outs[0].AsFloat()[0], -3.0f);
  EXPECT_FLOAT_EQ(outs[0].AsFloat()[1], -4.0f);
  ASSERT_EQ(outs[1].element_count(), 1);
  EXPECT_FLOAT_EQ(outs[1].AsFloat()[0], -5.0f);
}

TEST(KernelClass, IfBranchOverloadRejectsMismatchedOutputCounts) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  onnx_kernels::RuntimeContext rt(ctx);
  GraphProto then_graph;
  GraphProto else_graph;
  BuildDoubleInitBranchGraph(then_graph, "then_g", "t", "y", 1.0f);
  // else_graph has no outputs declared, so mismatched output count.
  else_graph.set_name("empty");

  Tensor cond_true("", onnx_kernels::DataType::BOOL, {}, {1});
  EXPECT_THROW((void)if_kernel(cond_true, then_graph, else_graph, rt), std::invalid_argument);
}

TEST(KernelClass, IfBranchOverloadRejectsNonBoolCond) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  onnx_kernels::RuntimeContext rt(ctx);
  GraphProto then_graph;
  GraphProto else_graph;
  BuildDoubleInitBranchGraph(then_graph, "then_g", "t", "y", 1.0f);
  BuildDoubleInitBranchGraph(else_graph, "else_g", "e", "y", 2.0f);

  Tensor cond_float = Tensor::FromFloat("", {}, {1.0f});
  EXPECT_THROW((void)if_kernel(cond_float, then_graph, else_graph, rt), std::invalid_argument);

  Tensor cond_vec("", onnx_kernels::DataType::BOOL, {2}, {1, 0});
  EXPECT_THROW((void)if_kernel(cond_vec, then_graph, else_graph, rt), std::invalid_argument);
}

} // namespace Test
namespace Test {

namespace {
// Builds an INT64 scalar tensor carrying ``v``.
Tensor Int64Scalar(int64_t v) {
  std::vector<uint8_t> bytes(sizeof(int64_t));
  std::memcpy(bytes.data(), &v, sizeof(int64_t));
  return Tensor("", onnx_kernels::DataType::INT64, {}, std::move(bytes));
}
} // namespace

TEST(KernelClass, LoopStacksScanOutputsAcrossIterations) {
  const KernelContext ctx{DefaultOpset(13)};
  Loop loop_kernel{ctx};
  // M = 3, no cond, no carried deps, K = 1 scan output of shape [2] per iter.
  Tensor M = Int64Scalar(3);
  Tensor cond_undef;
  Tensor s0 = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor s1 = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor s2 = Tensor::FromFloat("", {2}, {5.0f, 6.0f});
  std::vector<Tensor> out =
      loop_kernel(M, cond_undef, /*v_initial=*/{}, /*final_state=*/{}, {{s0, s1, s2}});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  ASSERT_EQ(out[0].shape.size(), 2u);
  EXPECT_EQ(out[0].shape[0], 3);
  EXPECT_EQ(out[0].shape[1], 2);
  ASSERT_EQ(out[0].element_count(), 6);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[4], 5.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[5], 6.0f);
}

TEST(KernelClass, LoopReturnsInitialStateWhenTripCountIsZero) {
  const KernelContext ctx{DefaultOpset(13)};
  Loop loop_kernel{ctx};
  Tensor M = Int64Scalar(0);
  Tensor cond_undef;
  Tensor v0 = Tensor::FromFloat("", {2}, {7.0f, 8.0f});
  Tensor v0_final = Tensor::FromFloat("", {2}, {0.0f, 0.0f});
  std::vector<Tensor> out = loop_kernel(M, cond_undef, {v0}, {v0_final}, {});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 7.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[1], 8.0f);
}

TEST(KernelClass, LoopHonorsCondFalseEvenWhenMIsLarge) {
  const KernelContext ctx{DefaultOpset(13)};
  Loop loop_kernel{ctx};
  Tensor M = Int64Scalar(5);
  Tensor cond_false("", onnx_kernels::DataType::BOOL, {}, {0});
  Tensor scan = Tensor::FromFloat("", {1}, {1.0f});
  std::vector<Tensor> out = loop_kernel(M, cond_false, /*v_initial=*/{}, /*final_state=*/{},
                                        {{scan, scan, scan, scan, scan}});
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].shape.size(), 2u);
  EXPECT_EQ(out[0].shape[0], 0);
  EXPECT_EQ(out[0].shape[1], 1);
  EXPECT_TRUE(out[0].data.empty());
}

TEST(KernelClass, LoopUsesPerIterRowLengthWhenMIsAbsent) {
  // No M and no cond → trip count is the per-iteration row length (2).
  const KernelContext ctx{DefaultOpset(13)};
  Loop loop_kernel{ctx};
  Tensor M_undef;
  Tensor cond_undef;
  Tensor s0 = Tensor::FromFloat("", {1}, {1.0f});
  Tensor s1 = Tensor::FromFloat("", {1}, {2.0f});
  std::vector<Tensor> out =
      loop_kernel(M_undef, cond_undef, /*v_initial=*/{}, /*final_state=*/{}, {{s0, s1}});
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].shape.size(), 2u);
  EXPECT_EQ(out[0].shape[0], 2);
  EXPECT_EQ(out[0].shape[1], 1);
}

TEST(KernelClass, LoopRejectsMismatchedFinalStateAndVInitial) {
  const KernelContext ctx{DefaultOpset(13)};
  Loop loop_kernel{ctx};
  Tensor M = Int64Scalar(1);
  Tensor cond_undef;
  Tensor v0_float = Tensor::FromFloat("", {2}, {0.0f, 0.0f});
  // Mismatched dtype: float vs int32.
  Tensor v0_final_int = Tensor::From<int32_t>("", {2}, {0, 0});
  EXPECT_THROW((void)loop_kernel(M, cond_undef, {v0_float}, {v0_final_int}, {}),
               std::invalid_argument);
}

TEST(KernelClass, LoopRejectsScanRowsOfDifferentLengths) {
  const KernelContext ctx{DefaultOpset(13)};
  Loop loop_kernel{ctx};
  Tensor M = Int64Scalar(2);
  Tensor cond_undef;
  Tensor s = Tensor::FromFloat("", {1}, {1.0f});
  EXPECT_THROW(
      (void)loop_kernel(M, cond_undef, /*v_initial=*/{}, /*final_state=*/{}, {{s, s}, {s}}),
      std::invalid_argument);
}

TEST(KernelClass, ScanStacksPerIterAlongLeadingAxisByDefault) {
  const KernelContext ctx{DefaultOpset(11)};
  Scan scan_kernel{ctx};
  // T = 3, no state vars, K = 1 scan output of shape [2] per iter.
  Tensor s0 = Tensor::FromFloat("", {2}, {0.0f, 1.0f});
  Tensor s1 = Tensor::FromFloat("", {2}, {2.0f, 3.0f});
  Tensor s2 = Tensor::FromFloat("", {2}, {4.0f, 5.0f});
  std::vector<Tensor> out =
      scan_kernel(3, /*initial_state=*/{}, /*final_state=*/{}, {{s0, s1, s2}});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  ASSERT_EQ(out[0].shape, (std::vector<int64_t>{3, 2}));
  ASSERT_EQ(out[0].element_count(), 6);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 0.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[1], 1.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[2], 2.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[3], 3.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[4], 4.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[5], 5.0f);
}

TEST(KernelClass, ScanReturnsInitialStateWhenTripCountIsZero) {
  const KernelContext ctx{DefaultOpset(11)};
  Scan scan_kernel{ctx};
  Tensor initial = Tensor::FromFloat("", {2}, {7.0f, 8.0f});
  Tensor final_ignored = Tensor::FromFloat("", {2}, {9.0f, 10.0f});
  Tensor s = Tensor::FromFloat("", {2}, {0.0f, 0.0f});
  std::vector<Tensor> out = scan_kernel(0, {initial}, {final_ignored}, {{s}});
  ASSERT_EQ(out.size(), 2u);
  // State output equals the initial value when T = 0.
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 7.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[1], 8.0f);
  // Scan output has shape [0, 2] (empty along the new leading axis).
  EXPECT_EQ(out[1].shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(out[1].element_count(), 0);
}

TEST(KernelClass, ScanReversesPerIterWhenDirectionPrepend) {
  const KernelContext ctx{DefaultOpset(11)};
  Scan scan_kernel{ctx};
  Tensor s0 = Tensor::FromFloat("", {1}, {10.0f});
  Tensor s1 = Tensor::FromFloat("", {1}, {20.0f});
  Tensor s2 = Tensor::FromFloat("", {1}, {30.0f});
  // direction = 1 → prepend = reverse before stacking.
  std::vector<Tensor> out = scan_kernel(3, {}, {}, {{s0, s1, s2}},
                                        /*scan_output_axes=*/{},
                                        /*scan_output_directions=*/{1});
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].shape, (std::vector<int64_t>{3, 1}));
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 30.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[1], 20.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[2], 10.0f);
}

TEST(KernelClass, ScanStacksAlongNonLeadingAxisWhenRequested) {
  const KernelContext ctx{DefaultOpset(11)};
  Scan scan_kernel{ctx};
  // Per-iter element shape [3] → stacking along axis=1 yields shape [3, T].
  Tensor s0 = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor s1 = Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f});
  std::vector<Tensor> out = scan_kernel(2, {}, {}, {{s0, s1}}, /*scan_output_axes=*/{1});
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].shape, (std::vector<int64_t>{3, 2}));
  // Memory layout (row-major, axis 1 = trip):
  //   [s0[0], s1[0], s0[1], s1[1], s0[2], s1[2]] = [1, 4, 2, 5, 3, 6].
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[1], 4.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[2], 2.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[3], 5.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[4], 3.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[5], 6.0f);
}

TEST(KernelClass, ScanRejectsMismatchedInitialAndFinalState) {
  const KernelContext ctx{DefaultOpset(11)};
  Scan scan_kernel{ctx};
  Tensor initial = Tensor::FromFloat("", {1}, {1.0f});
  // Different number of state tensors than initial_state.
  EXPECT_THROW((void)scan_kernel(1, {initial}, {}, {}), std::invalid_argument);
}

TEST(KernelClass, ScanRejectsNegativeTripCount) {
  const KernelContext ctx{DefaultOpset(11)};
  Scan scan_kernel{ctx};
  EXPECT_THROW((void)scan_kernel(-1, {}, {}, {}), std::invalid_argument);
}

} // namespace Test
