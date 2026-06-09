// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/run_nodes.h"
#include "onnx_kernels/runtime_context.h"
#include "onnx_kernels/simple_tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTestCases;
using onnx_backend_test::DataSet;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::TestCase;
using onnx_kernels::RunModel;
using onnx_kernels::RuntimeContext;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::KernelContext;

namespace Test {

namespace {

constexpr int64_t kFallbackDefaultOpsetVersion = 18;

int64_t GetDefaultOpsetVersion(const ModelProto &model) {
  for (const auto &opset : model.ref_opset_import()) {
    if (opset.ref_domain().empty()) {
      return opset.version();
    }
  }
  return kFallbackDefaultOpsetVersion;
}

void ExpectTensorBitEqual(const Tensor &actual, const Tensor &expected) {
  EXPECT_EQ(actual.data_type, expected.data_type);
  EXPECT_EQ(actual.shape, expected.shape);
  EXPECT_EQ(actual.string_data, expected.string_data);
  ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
  EXPECT_EQ(std::vector<uint8_t>(actual.bytes(), actual.bytes() + actual.size_bytes()),
            std::vector<uint8_t>(expected.bytes(), expected.bytes() + expected.size_bytes()));
}

// Runs ``RunModel`` on every backend test case whose top-level graph contains
// a single node whose ``op_type`` matches ``op_type``, and verifies that the
// computed outputs match the expected outputs of every ``DataSet``. The
// per-case dataset tensors carry the graph input/output names (see
// :cpp:func:`Expect`), so we can wire them by name into the
// :cpp:class:`RuntimeContext` directly. The expected outputs in the
// ``cases_*`` registry are themselves produced by the very same kernels that
// :cpp:func:`RunModel` will dispatch to, so a bit-exact comparison is
// appropriate.
void RunBackendCasesFor(const std::string &op_type) {
  const std::vector<TestCase> cases = CollectTestCases(op_type);
  ASSERT_FALSE(cases.empty()) << "No backend test cases found for op_type=" << op_type;

  size_t executed = 0;
  for (const TestCase &tc : cases) {
    const auto &graph = tc.model.ref_graph();
    if (graph.ref_node().size() != 1u) {
      continue;
    }
    const NodeProto &node = graph.ref_node()[0];
    if (node.ref_op_type().as_string() != op_type) {
      continue;
    }
    SCOPED_TRACE(tc.name);

    for (size_t ds_idx = 0; ds_idx < tc.data_sets.size(); ++ds_idx) {
      const DataSet &ds = tc.data_sets[ds_idx];
      RuntimeContext rt(KernelContext(DefaultOpset(GetDefaultOpsetVersion(tc.model))));
      for (const Tensor &t : ds.inputs) {
        rt.Set(t.name, t);
      }

      ASSERT_NO_THROW(RunModel(tc.model, rt)) << "RunModel threw for case " << tc.name;

      for (const Tensor &expected : ds.outputs) {
        ASSERT_TRUE(rt.Has(expected.name))
            << "Missing output '" << expected.name << "' for case " << tc.name;
        ExpectTensorBitEqual(rt.Get(expected.name), expected);
      }
    }
    ++executed;
  }
  EXPECT_GT(executed, 0u) << "No single-node test cases exercised for op_type=" << op_type;
}

} // namespace

// One TEST per kernel currently registered in ``KernelDispatchTable``
// (see ``onnx_kernels/run_nodes.cc``). When a new kernel is registered there
// the corresponding TEST below should be added so its backend cases are
// exercised through the full :cpp:func:`RunModel` path (and therefore catch
// regressions in graph wiring, kernel dispatch and the kernel itself in one
// shot).

TEST(BackendRunModel, Abs) { RunBackendCasesFor("Abs"); }
TEST(BackendRunModel, Neg) { RunBackendCasesFor("Neg"); }
TEST(BackendRunModel, Add) { RunBackendCasesFor("Add"); }
TEST(BackendRunModel, Sub) { RunBackendCasesFor("Sub"); }
TEST(BackendRunModel, Mul) { RunBackendCasesFor("Mul"); }
TEST(BackendRunModel, Div) { RunBackendCasesFor("Div"); }

// Additional unary, no-attribute math kernels registered in
// :cpp:func:`KernelDispatchTable` alongside the original baseline.
TEST(BackendRunModel, Acos) { RunBackendCasesFor("Acos"); }
TEST(BackendRunModel, Acosh) { RunBackendCasesFor("Acosh"); }
TEST(BackendRunModel, Asin) { RunBackendCasesFor("Asin"); }
TEST(BackendRunModel, Asinh) { RunBackendCasesFor("Asinh"); }
TEST(BackendRunModel, Atan) { RunBackendCasesFor("Atan"); }
TEST(BackendRunModel, Atanh) { RunBackendCasesFor("Atanh"); }
TEST(BackendRunModel, Ceil) { RunBackendCasesFor("Ceil"); }
TEST(BackendRunModel, Cos) { RunBackendCasesFor("Cos"); }
TEST(BackendRunModel, Cosh) { RunBackendCasesFor("Cosh"); }
TEST(BackendRunModel, Det) { RunBackendCasesFor("Det"); }
TEST(BackendRunModel, Erf) { RunBackendCasesFor("Erf"); }
TEST(BackendRunModel, Exp) { RunBackendCasesFor("Exp"); }
TEST(BackendRunModel, Floor) { RunBackendCasesFor("Floor"); }
TEST(BackendRunModel, HardSwish) { RunBackendCasesFor("HardSwish"); }
TEST(BackendRunModel, Log) { RunBackendCasesFor("Log"); }
TEST(BackendRunModel, Mish) { RunBackendCasesFor("Mish"); }
TEST(BackendRunModel, Reciprocal) { RunBackendCasesFor("Reciprocal"); }
TEST(BackendRunModel, Relu) { RunBackendCasesFor("Relu"); }
TEST(BackendRunModel, Round) { RunBackendCasesFor("Round"); }
TEST(BackendRunModel, Sigmoid) { RunBackendCasesFor("Sigmoid"); }
TEST(BackendRunModel, Sign) { RunBackendCasesFor("Sign"); }
TEST(BackendRunModel, Sin) { RunBackendCasesFor("Sin"); }
TEST(BackendRunModel, Sinh) { RunBackendCasesFor("Sinh"); }
TEST(BackendRunModel, Softplus) { RunBackendCasesFor("Softplus"); }
TEST(BackendRunModel, Softsign) { RunBackendCasesFor("Softsign"); }
TEST(BackendRunModel, Sqrt) { RunBackendCasesFor("Sqrt"); }
TEST(BackendRunModel, Tan) { RunBackendCasesFor("Tan"); }
TEST(BackendRunModel, Tanh) { RunBackendCasesFor("Tanh"); }

// Additional binary, no-attribute math kernels.
TEST(BackendRunModel, MatMul) { RunBackendCasesFor("MatMul"); }
TEST(BackendRunModel, PRelu) { RunBackendCasesFor("PRelu"); }
TEST(BackendRunModel, Pow) { RunBackendCasesFor("Pow"); }

// Variadic element-wise reducers.
TEST(BackendRunModel, Sum) { RunBackendCasesFor("Sum"); }
TEST(BackendRunModel, Max) { RunBackendCasesFor("Max"); }
TEST(BackendRunModel, Min) { RunBackendCasesFor("Min"); }
TEST(BackendRunModel, Mean) { RunBackendCasesFor("Mean"); }

// Reduction kernels.
TEST(BackendRunModel, ArgMax) { RunBackendCasesFor("ArgMax"); }
TEST(BackendRunModel, ArgMin) { RunBackendCasesFor("ArgMin"); }
TEST(BackendRunModel, ReduceL1) { RunBackendCasesFor("ReduceL1"); }
TEST(BackendRunModel, ReduceL2) { RunBackendCasesFor("ReduceL2"); }
TEST(BackendRunModel, ReduceLogSum) { RunBackendCasesFor("ReduceLogSum"); }
TEST(BackendRunModel, ReduceLogSumExp) { RunBackendCasesFor("ReduceLogSumExp"); }
TEST(BackendRunModel, ReduceMax) { RunBackendCasesFor("ReduceMax"); }
TEST(BackendRunModel, ReduceMean) { RunBackendCasesFor("ReduceMean"); }
TEST(BackendRunModel, ReduceMin) { RunBackendCasesFor("ReduceMin"); }
TEST(BackendRunModel, ReduceProd) { RunBackendCasesFor("ReduceProd"); }
TEST(BackendRunModel, ReduceSum) { RunBackendCasesFor("ReduceSum"); }
TEST(BackendRunModel, ReduceSumSquare) { RunBackendCasesFor("ReduceSumSquare"); }

// Attribute-driven unary math kernels.
TEST(BackendRunModel, Celu) { RunBackendCasesFor("Celu"); }
TEST(BackendRunModel, Elu) { RunBackendCasesFor("Elu"); }
TEST(BackendRunModel, LeakyRelu) { RunBackendCasesFor("LeakyRelu"); }
TEST(BackendRunModel, Swish) { RunBackendCasesFor("Swish"); }
TEST(BackendRunModel, ThresholdedRelu) { RunBackendCasesFor("ThresholdedRelu"); }
TEST(BackendRunModel, Hardmax) { RunBackendCasesFor("Hardmax"); }
TEST(BackendRunModel, LogSoftmax) { RunBackendCasesFor("LogSoftmax"); }
TEST(BackendRunModel, Softmax) { RunBackendCasesFor("Softmax"); }
TEST(BackendRunModel, HardSigmoid) { RunBackendCasesFor("HardSigmoid"); }
TEST(BackendRunModel, Selu) { RunBackendCasesFor("Selu"); }
TEST(BackendRunModel, Shrink) { RunBackendCasesFor("Shrink"); }
TEST(BackendRunModel, Gelu) { RunBackendCasesFor("Gelu"); }
TEST(BackendRunModel, Mod) { RunBackendCasesFor("Mod"); }
TEST(BackendRunModel, Clip) { RunBackendCasesFor("Clip"); }

} // namespace Test
