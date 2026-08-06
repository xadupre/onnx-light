// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/run_nodes.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::CollectTestCases;
using core::backend_test::DataSet;
using core::backend_test::DefaultOpset;
using core::backend_test::TestCase;
using core::runtime::DataType;
using core::runtime::ExecutionPlan;
using core::runtime::Map;
using core::runtime::RegisterModelFunctions;
using core::runtime::RuntimeContext;
using core::runtime::RuntimeSession;
using core::runtime::Tensor;
using core::runtime::TensorFromProto;
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

// Registers `model`'s local functions in `rt`, seeds `model.graph()`'s
// initializers into `rt`, and runs the graph by building its ExecutionPlan
// and driving it through a fresh RuntimeSession. This is what the removed
// `RunModel` used to do internally; every call site now builds the
// ExecutionPlan/RuntimeSession itself.
void RunModelViaSession(const ModelProto &model, RuntimeContext &rt) {
  RegisterModelFunctions(model, rt);
  const GraphProto &graph = model.ref_graph();
  const auto &inits = graph.initializer();
  for (size_t i = 0; i < inits.size(); ++i) {
    const TensorProto &tp = inits[i];
    const std::string init_name = tp.name();
    if (!rt.Has(init_name)) {
      rt.Set(init_name, TensorFromProto(tp), core::runtime::RuntimeEventKind::kInitializer);
    }
  }
  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan);
  session.Run(rt);
}

// Runs the model (via `RunModelViaSession`) on every backend test case whose top-level graph
// contains a single node whose ``op_type`` matches ``op_type``, and verifies that the computed
// outputs match the expected outputs of every ``DataSet``. The per-case dataset tensors carry the
// graph input/output names (see :cpp:func:`Expect`), so we can wire them by name into the
// :cpp:class:`RuntimeContext` directly. The expected outputs in the
// ``cases_*`` registry are themselves produced by the very same kernels that
// :cpp:func:`RunModelViaSession` will dispatch to, so a bit-exact comparison is
// appropriate.
//
// Evaluates the optional ``accept_test_case`` predicate once per
// ``TestCase`` (before any ``DataSet`` is examined); returning ``false``
// skips the entire case. Evaluates the optional ``accept_data_set``
// predicate per ``DataSet`` within an accepted case.
void RunBackendCasesFor(const std::string &op_type,
                        const std::function<bool(const TestCase &)> &accept_test_case,
                        const std::function<bool(const DataSet &)> &accept_data_set) {
  const std::vector<TestCase> cases = CollectTestCases(op_type);
  ASSERT_FALSE(cases.empty()) << "No backend test cases found for op_type=" << op_type;

  size_t executed = 0;
  for (const TestCase &tc : cases) {
    const auto &graph = tc.model().ref_graph();
    if (graph.ref_node().size() != 1u) {
      continue;
    }
    const NodeProto &node = graph.ref_node()[0];
    if (node.ref_op_type() != op_type) {
      continue;
    }
    if (!accept_test_case(tc)) {
      continue;
    }
    SCOPED_TRACE(tc.name);

    bool ran_case = false;
    for (size_t ds_idx = 0; ds_idx < tc.data_sets().size(); ++ds_idx) {
      const DataSet &ds = tc.data_sets()[ds_idx];
      if (!accept_data_set(ds)) {
        continue;
      }
      RuntimeContext rt(KernelContext(DefaultOpset(GetDefaultOpsetVersion(tc.model()))));
      for (const Tensor &t : ds.inputs) {
        rt.Set(t.name, t);
      }
      for (const Map &m : ds.maps) {
        rt.PutMap(m.name, m);
      }

      ASSERT_NO_THROW(RunModelViaSession(tc.model(), rt))
          << "Running the model threw for case " << tc.name;

      for (const Tensor &expected : ds.outputs) {
        ASSERT_TRUE(rt.Has(expected.name))
            << "Missing output '" << expected.name << "' for case " << tc.name;
        ExpectTensorBitEqual(rt.Get(expected.name), expected);
      }
      ran_case = true;
    }
    if (ran_case) {
      ++executed;
    }
  }
  EXPECT_GT(executed, 0u) << "No single-node test cases exercised for op_type=" << op_type;
}

void RunBackendCasesFor(
    const std::string &op_type, const std::function<bool(const DataSet &)> &accept_data_set =
                                    [](const DataSet &) { return true; }) {
  RunBackendCasesFor(op_type, [](const TestCase &) { return true; }, accept_data_set);
}

} // namespace

// One TEST per kernel currently registered in ``KernelDispatchTable``
// (see ``onnx_kernels/run_nodes.cc``). When a new kernel is registered there
// the corresponding TEST below should be added so its backend cases are
// exercised through the full model-run path (and therefore catch
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
TEST(BackendRunModel, ReverseSequence) { RunBackendCasesFor("ReverseSequence"); }
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
TEST(BackendRunModel, MatMulInteger) { RunBackendCasesFor("MatMulInteger"); }
TEST(BackendRunModel, PRelu) { RunBackendCasesFor("PRelu"); }
TEST(BackendRunModel, Pow) { RunBackendCasesFor("Pow"); }

// Comparison / logical kernels.
TEST(BackendRunModel, Equal) { RunBackendCasesFor("Equal"); }
TEST(BackendRunModel, Greater) { RunBackendCasesFor("Greater"); }
TEST(BackendRunModel, GreaterOrEqual) { RunBackendCasesFor("GreaterOrEqual"); }
TEST(BackendRunModel, Less) { RunBackendCasesFor("Less"); }
TEST(BackendRunModel, LessOrEqual) { RunBackendCasesFor("LessOrEqual"); }
TEST(BackendRunModel, Where) { RunBackendCasesFor("Where"); }

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
TEST(BackendRunModel, SwiGLU) { RunBackendCasesFor("SwiGLU"); }
TEST(BackendRunModel, ThresholdedRelu) { RunBackendCasesFor("ThresholdedRelu"); }
TEST(BackendRunModel, TopK) { RunBackendCasesFor("TopK"); }
TEST(BackendRunModel, Hardmax) { RunBackendCasesFor("Hardmax"); }
TEST(BackendRunModel, LogSoftmax) { RunBackendCasesFor("LogSoftmax"); }
TEST(BackendRunModel, Flatten) { RunBackendCasesFor("Flatten"); }
TEST(BackendRunModel, Softmax) { RunBackendCasesFor("Softmax"); }
TEST(BackendRunModel, SoftmaxCrossEntropyLoss) { RunBackendCasesFor("SoftmaxCrossEntropyLoss"); }
TEST(BackendRunModel, NegativeLogLikelihoodLoss) {
  RunBackendCasesFor("NegativeLogLikelihoodLoss");
}
TEST(BackendRunModel, HardSigmoid) { RunBackendCasesFor("HardSigmoid"); }
TEST(BackendRunModel, Selu) { RunBackendCasesFor("Selu"); }
TEST(BackendRunModel, Shrink) { RunBackendCasesFor("Shrink"); }
TEST(BackendRunModel, GatherElements) { RunBackendCasesFor("GatherElements"); }
TEST(BackendRunModel, ScatterElements) { RunBackendCasesFor("ScatterElements"); }
TEST(BackendRunModel, ScatterND) { RunBackendCasesFor("ScatterND"); }
TEST(BackendRunModel, Gelu) { RunBackendCasesFor("Gelu"); }
TEST(BackendRunModel, Mod) { RunBackendCasesFor("Mod"); }
TEST(BackendRunModel, Clip) { RunBackendCasesFor("Clip"); }
TEST(BackendRunModel, Compress) { RunBackendCasesFor("Compress"); }
TEST(BackendRunModel, Unique) { RunBackendCasesFor("Unique"); }
TEST(BackendRunModel, NonZero) { RunBackendCasesFor("NonZero"); }
TEST(BackendRunModel, Concat) { RunBackendCasesFor("Concat"); }
TEST(BackendRunModel, CumSum) { RunBackendCasesFor("CumSum"); }
TEST(BackendRunModel, CumProd) { RunBackendCasesFor("CumProd"); }
TEST(BackendRunModel, DFT) { RunBackendCasesFor("DFT"); }
TEST(BackendRunModel, STFT) { RunBackendCasesFor("STFT"); }
TEST(BackendRunModel, MelWeightMatrix) { RunBackendCasesFor("MelWeightMatrix"); }
TEST(BackendRunModel, Attention) {
  RunBackendCasesFor("Attention", [](const DataSet &ds) {
    return ds.inputs.size() >= 3 && ds.inputs[0].data_type == DataType::FLOAT &&
           ds.inputs[1].data_type == DataType::FLOAT && ds.inputs[2].data_type == DataType::FLOAT;
  });
}
TEST(BackendRunModel, Cast) { RunBackendCasesFor("Cast"); }
TEST(BackendRunModel, CastLike) { RunBackendCasesFor("CastLike"); }
TEST(BackendRunModel, CenterCropPad) { RunBackendCasesFor("CenterCropPad"); }
TEST(BackendRunModel, Pad) { RunBackendCasesFor("Pad"); }
TEST(BackendRunModel, Slice) { RunBackendCasesFor("Slice"); }
TEST(BackendRunModel, BitCast) { RunBackendCasesFor("BitCast"); }
TEST(BackendRunModel, CausalConvWithState) { RunBackendCasesFor("CausalConvWithState"); }
TEST(BackendRunModel, Conv) { RunBackendCasesFor("Conv"); }
TEST(BackendRunModel, ConvInteger) { RunBackendCasesFor("ConvInteger"); }
TEST(BackendRunModel, ConvTranspose) { RunBackendCasesFor("ConvTranspose"); }
TEST(BackendRunModel, Transpose) { RunBackendCasesFor("Transpose"); }
TEST(BackendRunModel, Trilu) { RunBackendCasesFor("Trilu"); }
TEST(BackendRunModel, TensorScatter) { RunBackendCasesFor("TensorScatter"); }
TEST(BackendRunModel, Col2Im) { RunBackendCasesFor("Col2Im"); }
TEST(BackendRunModel, DeformConv) { RunBackendCasesFor("DeformConv"); }
TEST(BackendRunModel, DepthToSpace) { RunBackendCasesFor("DepthToSpace"); }
TEST(BackendRunModel, SpaceToDepth) { RunBackendCasesFor("SpaceToDepth"); }
TEST(BackendRunModel, Upsample) { RunBackendCasesFor("Upsample"); }
TEST(BackendRunModel, BatchNormalization) { RunBackendCasesFor("BatchNormalization"); }
TEST(BackendRunModel, GroupNormalization) { RunBackendCasesFor("GroupNormalization"); }
TEST(BackendRunModel, GridSample) { RunBackendCasesFor("GridSample"); }
TEST(BackendRunModel, Resize) { RunBackendCasesFor("Resize"); }
TEST(BackendRunModel, RoiAlign) { RunBackendCasesFor("RoiAlign"); }
TEST(BackendRunModel, MaxRoiPool) { RunBackendCasesFor("MaxRoiPool"); }
TEST(BackendRunModel, InstanceNormalization) { RunBackendCasesFor("InstanceNormalization"); }
TEST(BackendRunModel, LayerNormalization) { RunBackendCasesFor("LayerNormalization"); }
TEST(BackendRunModel, RMSNormalization) { RunBackendCasesFor("RMSNormalization"); }
TEST(BackendRunModel, RNN) { RunBackendCasesFor("RNN"); }
TEST(BackendRunModel, GRU) { RunBackendCasesFor("GRU"); }
TEST(BackendRunModel, LSTM) { RunBackendCasesFor("LSTM"); }
TEST(BackendRunModel, RotaryEmbedding) { RunBackendCasesFor("RotaryEmbedding"); }
TEST(BackendRunModel, MeanVarianceNormalization) {
  RunBackendCasesFor("MeanVarianceNormalization");
}
TEST(BackendRunModel, Dropout) { RunBackendCasesFor("Dropout"); }
TEST(BackendRunModel, AveragePool) { RunBackendCasesFor("AveragePool"); }
TEST(BackendRunModel, GlobalAveragePool) { RunBackendCasesFor("GlobalAveragePool"); }
TEST(BackendRunModel, GlobalMaxPool) { RunBackendCasesFor("GlobalMaxPool"); }
TEST(BackendRunModel, GlobalLpPool) { RunBackendCasesFor("GlobalLpPool"); }
TEST(BackendRunModel, LpPool) { RunBackendCasesFor("LpPool"); }
TEST(BackendRunModel, LpNormalization) { RunBackendCasesFor("LpNormalization"); }
TEST(BackendRunModel, LRN) { RunBackendCasesFor("LRN"); }
TEST(BackendRunModel, MaxPool) {
  RunBackendCasesFor("MaxPool", [](const DataSet &ds) {
    return !ds.inputs.empty() && ds.inputs[0].data_type == DataType::FLOAT;
  });
}
TEST(BackendRunModel, MaxUnpool) { RunBackendCasesFor("MaxUnpool"); }

// ai.onnx.preview.training optimizer kernels.
TEST(BackendRunModel, Adagrad) { RunBackendCasesFor("Adagrad"); }
TEST(BackendRunModel, Adam) { RunBackendCasesFor("Adam"); }
TEST(BackendRunModel, Momentum) { RunBackendCasesFor("Momentum"); }

// Random / sampling kernels (ai.onnx).
TEST(BackendRunModel, Bernoulli) { RunBackendCasesFor("Bernoulli"); }
TEST(BackendRunModel, DelayedInitializer) { RunBackendCasesFor("DelayedInitializer"); }
TEST(BackendRunModel, RandomNormal) { RunBackendCasesFor("RandomNormal"); }
TEST(BackendRunModel, RandomNormalLike) { RunBackendCasesFor("RandomNormalLike"); }
TEST(BackendRunModel, RandomUniform) { RunBackendCasesFor("RandomUniform"); }
TEST(BackendRunModel, RandomUniformLike) { RunBackendCasesFor("RandomUniformLike"); }
TEST(BackendRunModel, Multinomial) { RunBackendCasesFor("Multinomial"); }

// Window-generation kernels (ai.onnx, opset 17).
TEST(BackendRunModel, BlackmanWindow) { RunBackendCasesFor("BlackmanWindow"); }
TEST(BackendRunModel, HannWindow) { RunBackendCasesFor("HannWindow"); }
TEST(BackendRunModel, HammingWindow) { RunBackendCasesFor("HammingWindow"); }

// ai.onnx.ml kernels.
TEST(BackendRunModel, CastMap) { RunBackendCasesFor("CastMap"); }
TEST(BackendRunModel, DictVectorizer) { RunBackendCasesFor("DictVectorizer"); }
TEST(BackendRunModel, SVMRegressor) { RunBackendCasesFor("SVMRegressor"); }
TEST(BackendRunModel, SVMClassifier) { RunBackendCasesFor("SVMClassifier"); }
TEST(BackendRunModel, LinearRegressor) { RunBackendCasesFor("LinearRegressor"); }
TEST(BackendRunModel, LinearClassifier) { RunBackendCasesFor("LinearClassifier"); }
TEST(BackendRunModel, TreeEnsembleRegressor) { RunBackendCasesFor("TreeEnsembleRegressor"); }
TEST(BackendRunModel, TreeEnsembleClassifier) { RunBackendCasesFor("TreeEnsembleClassifier"); }
TEST(BackendRunModel, TreeEnsemble) { RunBackendCasesFor("TreeEnsemble"); }
TEST(BackendRunModel, ArrayFeatureExtractor) { RunBackendCasesFor("ArrayFeatureExtractor"); }
TEST(BackendRunModel, Binarizer) { RunBackendCasesFor("Binarizer"); }
TEST(BackendRunModel, Normalizer) { RunBackendCasesFor("Normalizer"); }
TEST(BackendRunModel, CategoryMapper) { RunBackendCasesFor("CategoryMapper"); }
TEST(BackendRunModel, FeatureVectorizer) { RunBackendCasesFor("FeatureVectorizer"); }
TEST(BackendRunModel, Imputer) { RunBackendCasesFor("Imputer"); }
TEST(BackendRunModel, LabelEncoder) { RunBackendCasesFor("LabelEncoder"); }
TEST(BackendRunModel, OneHotEncoder) { RunBackendCasesFor("OneHotEncoder"); }
TEST(BackendRunModel, Scaler) { RunBackendCasesFor("Scaler"); }

// Text kernels.
TEST(BackendRunModel, StringConcat) { RunBackendCasesFor("StringConcat"); }
TEST(BackendRunModel, StringNormalizer) { RunBackendCasesFor("StringNormalizer"); }
TEST(BackendRunModel, StringSplit) { RunBackendCasesFor("StringSplit"); }
TEST(BackendRunModel, TfIdfVectorizer) { RunBackendCasesFor("TfIdfVectorizer"); }

// Control-flow kernels. ``Scan`` body-aware execution is owned by
// :cpp:class:`kernel::Scan`; :cpp:func:`RunScanNode` additionally handles
// the opset-8 batched form (leading ``sequence_lens`` placeholder + outer
// batch dim on every state / scan input/output) by running the Scan-9
// kernel once per batch element and stacking the per-batch outputs.
//
// ``test_cc_scan_zero_trip_count`` is included: when trip_count==0 the body
// is run once with zero-filled dummy slices so the body-aware overload can
// recover the scan-output element type/shape and emit the expected FLOAT
// ``[0, 2]`` output (instead of a degenerate UNDEFINED ``[0]`` tensor).
TEST(BackendRunModel, Scan) { RunBackendCasesFor("Scan"); }

// Quantization kernels.
// The reference QuantizeLinear/DequantizeLinear kernels support per-tensor and
// per-axis quantization with FLOAT scales, covering integer (UINT8/INT8/UINT16/
// INT16), float8, and sub-byte (INT4/UINT4/INT2/UINT2/FLOAT4E2M1) output types.
// Skip blocked / FLOAT16-scale cases which are not yet implemented.
TEST(BackendRunModel, QuantizeLinear) {
  RunBackendCasesFor("QuantizeLinear", [](const DataSet &ds) {
    if (ds.inputs.size() < 2) {
      return false;
    }
    if (ds.inputs[1].data_type != static_cast<int32_t>(DataType::FLOAT)) {
      return false;
    }
    if (ds.outputs.empty()) {
      return false;
    }
    const int32_t y_dtype = ds.outputs[0].data_type;
    return y_dtype == static_cast<int32_t>(DataType::UINT8) ||
           y_dtype == static_cast<int32_t>(DataType::INT8) ||
           y_dtype == static_cast<int32_t>(DataType::UINT16) ||
           y_dtype == static_cast<int32_t>(DataType::INT16) ||
           y_dtype == static_cast<int32_t>(DataType::FLOAT8E4M3FN) ||
           y_dtype == static_cast<int32_t>(DataType::FLOAT8E4M3FNUZ) ||
           y_dtype == static_cast<int32_t>(DataType::FLOAT8E5M2) ||
           y_dtype == static_cast<int32_t>(DataType::FLOAT8E5M2FNUZ) ||
           y_dtype == static_cast<int32_t>(DataType::INT4) ||
           y_dtype == static_cast<int32_t>(DataType::UINT4) ||
           y_dtype == static_cast<int32_t>(DataType::INT2) ||
           y_dtype == static_cast<int32_t>(DataType::UINT2) ||
           y_dtype == static_cast<int32_t>(DataType::FLOAT4E2M1);
  });
}
TEST(BackendRunModel, DequantizeLinear) {
  RunBackendCasesFor("DequantizeLinear", [](const DataSet &ds) {
    if (ds.inputs.size() < 2) {
      return false;
    }
    const int32_t scale_dtype = ds.inputs[1].data_type;
    const int64_t scale_count = ds.inputs[1].element_count();
    const bool is_scalar_scale = scale_count == 1;
    // Per-axis: 1-D x_scale with multiple elements; FLOAT only (no FLOAT16
    // per-axis). Blocked: N-D FLOAT x_scale that divides x along the
    // quantization axis. Both are handled by the reference kernel.
    const bool is_per_axis_scale = scale_count > 1 && ds.inputs[1].shape.size() == 1 &&
                                   scale_dtype == static_cast<int32_t>(DataType::FLOAT);
    const bool is_blocked_scale = scale_count > 1 && ds.inputs[1].shape.size() > 1 &&
                                  scale_dtype == static_cast<int32_t>(DataType::FLOAT);
    if (!is_scalar_scale && !is_per_axis_scale && !is_blocked_scale) {
      return false;
    }
    if (scale_dtype != static_cast<int32_t>(DataType::FLOAT) &&
        scale_dtype != static_cast<int32_t>(DataType::FLOAT16)) {
      return false;
    }
    if (ds.outputs.empty() || ds.outputs[0].data_type != scale_dtype) {
      return false;
    }
    const int32_t x_dtype = ds.inputs[0].data_type;
    return x_dtype == static_cast<int32_t>(DataType::UINT8) ||
           x_dtype == static_cast<int32_t>(DataType::INT8) ||
           x_dtype == static_cast<int32_t>(DataType::UINT16) ||
           x_dtype == static_cast<int32_t>(DataType::INT16) ||
           x_dtype == static_cast<int32_t>(DataType::INT32) ||
           x_dtype == static_cast<int32_t>(DataType::FLOAT8E4M3FN) ||
           x_dtype == static_cast<int32_t>(DataType::FLOAT8E4M3FNUZ) ||
           x_dtype == static_cast<int32_t>(DataType::FLOAT8E5M2) ||
           x_dtype == static_cast<int32_t>(DataType::FLOAT8E5M2FNUZ) ||
           x_dtype == static_cast<int32_t>(DataType::INT4) ||
           x_dtype == static_cast<int32_t>(DataType::UINT4) ||
           x_dtype == static_cast<int32_t>(DataType::INT2) ||
           x_dtype == static_cast<int32_t>(DataType::UINT2) ||
           x_dtype == static_cast<int32_t>(DataType::FLOAT4E2M1);
  });
}
TEST(BackendRunModel, DynamicQuantizeLinear) { RunBackendCasesFor("DynamicQuantizeLinear"); }

// The reference QLinearMatMul kernel supports per-tensor quantization
// (scalar scales/zero-points) with FLOAT or FLOAT16 scales.
TEST(BackendRunModel, QLinearMatMul) {
  RunBackendCasesFor("QLinearMatMul", [](const DataSet &ds) {
    if (ds.inputs.size() < 8 || ds.inputs[1].element_count() != 1) {
      return false;
    }
    auto is_float_scale = [](int32_t dt) {
      return dt == static_cast<int32_t>(DataType::FLOAT) ||
             dt == static_cast<int32_t>(DataType::FLOAT16);
    };
    return is_float_scale(ds.inputs[1].data_type) && is_float_scale(ds.inputs[4].data_type) &&
           is_float_scale(ds.inputs[6].data_type);
  });
}

// The reference QLinearConv kernel supports per-tensor (or per-output-channel
// ``w``-side) quantization with INT8/UINT8 ``x``/``w``/``y`` and FLOAT scales.
TEST(BackendRunModel, QLinearConv) {
  RunBackendCasesFor("QLinearConv", [](const DataSet &ds) {
    if (ds.inputs.size() < 8 || ds.inputs[1].element_count() != 1 ||
        ds.inputs[6].element_count() != 1) {
      return false;
    }
    return ds.inputs[1].data_type == static_cast<int32_t>(DataType::FLOAT) &&
           ds.inputs[4].data_type == static_cast<int32_t>(DataType::FLOAT) &&
           ds.inputs[6].data_type == static_cast<int32_t>(DataType::FLOAT);
  });
}

// LinearAttention (opset 27) and FlexAttention (ai.onnx.preview) kernels.
TEST(BackendRunModel, LinearAttention) {
  RunBackendCasesFor("LinearAttention", [](const DataSet &ds) {
    if (ds.inputs.size() < 3) {
      return false;
    }
    const int32_t dtype = ds.inputs[0].data_type;
    const bool supported = dtype == static_cast<int32_t>(DataType::FLOAT) ||
                           dtype == static_cast<int32_t>(DataType::FLOAT16);
    return supported && ds.inputs[1].data_type == dtype && ds.inputs[2].data_type == dtype;
  });
}
TEST(BackendRunModel, FlexAttention) {
  // The dispatch-table kernel handles the base FlexAttention path (Q, K, V ->
  // Y) including the optional ``score_mod`` and ``prob_mod`` modifier
  // subgraphs (executed via SubgraphSession). Some score_mod cases use ops
  // not yet wired through the dispatch table; skip those by case name.
  RunBackendCasesFor(
      "FlexAttention",
      [](const TestCase &tc) {
        // ``test_cc_flexattention_soft_cap`` uses Constant/Div/Tanh
        // chains in its score_mod subgraph; some of those paths are not
        // yet covered.
        if (tc.name == "test_cc_flexattention_soft_cap") {
          return false;
        }
        // On 32-bit builds (e.g. Windows x86), the ``score_mod`` case
        // diverges by a single ULP: its expected output was produced by a
        // ``double``-precision reference applying the modifier as a C++
        // lambda, whereas the kernel applies it as an ONNX subgraph in
        // ``float``. The two float rounding paths only agree bit-for-bit on
        // 64-bit targets, so the byte-exact comparison is skipped here on
        // 32-bit (it still runs and matches on every 64-bit target).
        if constexpr (sizeof(void *) == 4) {
          if (tc.name == "test_cc_flexattention_score_mod") {
            return false;
          }
        }
        return true;
      },
      [](const DataSet &ds) {
        // Exercise both the FLOAT cases and the DOUBLE case
        // (``test_cc_flexattention_double``); Q/K/V always share one dtype.
        if (ds.inputs.size() < 3) {
          return false;
        }
        const bool float_inputs = ds.inputs[0].data_type == DataType::FLOAT &&
                                  ds.inputs[1].data_type == DataType::FLOAT &&
                                  ds.inputs[2].data_type == DataType::FLOAT;
        const bool double_inputs = ds.inputs[0].data_type == DataType::DOUBLE &&
                                   ds.inputs[1].data_type == DataType::DOUBLE &&
                                   ds.inputs[2].data_type == DataType::DOUBLE;
        return float_inputs || double_inputs;
      });
}

// SequenceMap is a sequence-typed control-flow op: its outputs are
// :cpp:struct:`Sequence` values held in ``RuntimeContext::sequences``,
// not :cpp:struct:`Tensor` values in ``RuntimeContext::tensors``, and
// every backend test case wraps the SequenceMap node in a 2-node
// graph (``SequenceConstruct`` + ``SequenceMap``). The single-node
// ``RunBackendCasesFor`` helper therefore does not apply; instead the
// per-case expected stacked tensor (registered in
// ``cases_sequence_map.cc``) is compared against the actual output
// sequence by re-stacking the latter through
// :cpp:class:`kernel::SequenceConstruct`.
TEST(BackendRunModel, SequenceMap) {
  const std::vector<TestCase> cases = CollectTestCases("SequenceMap");
  ASSERT_FALSE(cases.empty()) << "No backend test cases found for SequenceMap";

  std::size_t executed = 0;
  for (const TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);
    const onnx_kernels::kernel::KernelContext kctx(
        DefaultOpset(GetDefaultOpsetVersion(tc.model())));

    for (const DataSet &ds : tc.data_sets()) {
      RuntimeContext rt(kctx);
      for (const Tensor &t : ds.inputs) {
        rt.Set(t.name, t);
      }
      ASSERT_NO_THROW(RunModelViaSession(tc.model(), rt))
          << "Running the model threw for case " << tc.name;

      // Each expected output is the stacked-tensor materialisation of
      // the corresponding output sequence (see
      // ``cases_sequence_map.cc``); re-stack the actual sequence and
      // compare bit-exactly.
      for (const Tensor &expected : ds.outputs) {
        ASSERT_TRUE(rt.HasSequence(expected.name))
            << "Missing output sequence '" << expected.name << "' for case " << tc.name;
        const auto &out_seq = rt.GetSequence(expected.name);
        const std::vector<Tensor> values(out_seq.values.begin(), out_seq.values.end());
        Tensor actual = onnx_kernels::kernel::SequenceConstruct(kctx)(values);
        actual.name = expected.name;
        ExpectTensorBitEqual(actual, expected);
      }
      ++executed;
    }
  }
  EXPECT_GT(executed, 0u) << "No SequenceMap test cases exercised.";
}

// SplitToSequence is a sequence-typed op: its output is a
// :cpp:struct:`Sequence` value held in ``RuntimeContext::sequences``
// rather than a :cpp:struct:`Tensor` in ``RuntimeContext::tensors``, so the
// single-node ``RunBackendCasesFor`` helper (which looks the output up by
// name in the tensor store) does not apply. Each backend case (see
// ``cases_split_to_sequence.cc``) registers its expected output as the
// stacked-tensor materialisation of the output sequence, so the actual
// output sequence is re-stacked through
// :cpp:class:`kernel::SequenceConstruct` and compared bit-exactly. Running
// through :cpp:func:`RunModelViaSession` drives the kernel with a real
// :cpp:class:`RuntimeContext`, exercising the runtime-allocator path for the
// kernel's split-size buffer.
TEST(BackendRunModel, SplitToSequence) {
  const std::vector<TestCase> cases = CollectTestCases("SplitToSequence");
  ASSERT_FALSE(cases.empty()) << "No backend test cases found for SplitToSequence";

  std::size_t executed = 0;
  for (const TestCase &tc : cases) {
    const auto &graph = tc.model().ref_graph();
    if (graph.ref_node().size() != 1u || graph.ref_node()[0].ref_op_type() != "SplitToSequence") {
      continue;
    }
    SCOPED_TRACE(tc.name);
    const onnx_kernels::kernel::KernelContext kctx(
        DefaultOpset(GetDefaultOpsetVersion(tc.model())));

    for (const DataSet &ds : tc.data_sets()) {
      RuntimeContext rt(kctx);
      for (const Tensor &t : ds.inputs) {
        rt.Set(t.name, t);
      }
      ASSERT_NO_THROW(RunModelViaSession(tc.model(), rt))
          << "Running the model threw for case " << tc.name;

      // Each expected output is the stacked-tensor materialisation of the
      // corresponding output sequence (see ``cases_split_to_sequence.cc``);
      // re-stack the actual sequence and compare bit-exactly.
      for (const Tensor &expected : ds.outputs) {
        ASSERT_TRUE(rt.HasSequence(expected.name))
            << "Missing output sequence '" << expected.name << "' for case " << tc.name;
        const auto &out_seq = rt.GetSequence(expected.name);
        const std::vector<Tensor> values(out_seq.values.begin(), out_seq.values.end());
        Tensor actual = onnx_kernels::kernel::SequenceConstruct(kctx)(values);
        actual.name = expected.name;
        ExpectTensorBitEqual(actual, expected);
      }
      ++executed;
    }
  }
  EXPECT_GT(executed, 0u) << "No SplitToSequence test cases exercised.";
}

} // namespace Test
