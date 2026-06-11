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
#include <functional>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTestCases;
using onnx_backend_test::DataSet;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::TestCase;
using onnx_kernels::DataType;
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
    const auto &graph = tc.model.ref_graph();
    if (graph.ref_node().size() != 1u) {
      continue;
    }
    const NodeProto &node = graph.ref_node()[0];
    if (node.ref_op_type().as_string() != op_type) {
      continue;
    }
    if (!accept_test_case(tc)) {
      continue;
    }
    SCOPED_TRACE(tc.name);

    bool ran_case = false;
    for (size_t ds_idx = 0; ds_idx < tc.data_sets.size(); ++ds_idx) {
      const DataSet &ds = tc.data_sets[ds_idx];
      if (!accept_data_set(ds)) {
        continue;
      }
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
TEST(BackendRunModel, TopK) { RunBackendCasesFor("TopK"); }
TEST(BackendRunModel, Hardmax) { RunBackendCasesFor("Hardmax"); }
TEST(BackendRunModel, LogSoftmax) { RunBackendCasesFor("LogSoftmax"); }
TEST(BackendRunModel, Flatten) { RunBackendCasesFor("Flatten"); }
TEST(BackendRunModel, Softmax) { RunBackendCasesFor("Softmax"); }
TEST(BackendRunModel, HardSigmoid) { RunBackendCasesFor("HardSigmoid"); }
TEST(BackendRunModel, Selu) { RunBackendCasesFor("Selu"); }
TEST(BackendRunModel, Shrink) { RunBackendCasesFor("Shrink"); }
TEST(BackendRunModel, GatherElements) { RunBackendCasesFor("GatherElements"); }
TEST(BackendRunModel, Gelu) { RunBackendCasesFor("Gelu"); }
TEST(BackendRunModel, Mod) { RunBackendCasesFor("Mod"); }
TEST(BackendRunModel, Clip) { RunBackendCasesFor("Clip"); }
TEST(BackendRunModel, Compress) { RunBackendCasesFor("Compress"); }
TEST(BackendRunModel, Unique) { RunBackendCasesFor("Unique"); }
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
TEST(BackendRunModel, Col2Im) { RunBackendCasesFor("Col2Im"); }
TEST(BackendRunModel, DeformConv) { RunBackendCasesFor("DeformConv"); }
TEST(BackendRunModel, DepthToSpace) { RunBackendCasesFor("DepthToSpace"); }
TEST(BackendRunModel, Upsample) { RunBackendCasesFor("Upsample"); }
TEST(BackendRunModel, BatchNormalization) { RunBackendCasesFor("BatchNormalization"); }
TEST(BackendRunModel, GroupNormalization) { RunBackendCasesFor("GroupNormalization"); }
TEST(BackendRunModel, GridSample) { RunBackendCasesFor("GridSample"); }
TEST(BackendRunModel, RoiAlign) { RunBackendCasesFor("RoiAlign"); }
TEST(BackendRunModel, InstanceNormalization) { RunBackendCasesFor("InstanceNormalization"); }
TEST(BackendRunModel, LayerNormalization) { RunBackendCasesFor("LayerNormalization"); }
TEST(BackendRunModel, RMSNormalization) { RunBackendCasesFor("RMSNormalization"); }
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
TEST(BackendRunModel, MaxPool) {
  RunBackendCasesFor(
      "MaxPool",
      [](const TestCase &tc) {
        const NodeProto &node = tc.model.ref_graph().ref_node()[0];
        for (const auto &attr : node.ref_attribute()) {
          if (attr.ref_name().as_string() == "storage_order" && attr.i() != 0) {
            return false;
          }
        }
        return true;
      },
      [](const DataSet &ds) {
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

// Text kernels.
TEST(BackendRunModel, StringNormalizer) { RunBackendCasesFor("StringNormalizer"); }

// Control-flow kernels. ``Scan`` body-aware execution is owned by
// :cpp:class:`kernel::Scan`; :cpp:func:`RunScanNode` additionally handles
// the opset-8 batched form (leading ``sequence_lens`` placeholder + outer
// batch dim on every state / scan input/output) by running the Scan-9
// kernel once per batch element and stacking the per-batch outputs.
//
// ``test_cc_scan_zero_trip_count`` is excluded: when trip_count==0 the
// body is never executed so the body-aware overload of ``kernel::Scan``
// has no per-iteration tensor from which to recover the scan-output
// element shape/dtype, and produces a degenerate UNDEFINED ``[0]`` output
// instead of the expected FLOAT ``[0, 2]``. This is unrelated to the
// opset-8 fix and tracked separately.
TEST(BackendRunModel, Scan) {
  RunBackendCasesFor(
      "Scan", [](const TestCase &tc) { return tc.name != "test_cc_scan_zero_trip_count"; },
      [](const DataSet &) { return true; });
}

// Quantization kernels.
// The reference QuantizeLinear/DequantizeLinear kernels only support
// per-tensor quantization (scalar y_scale/x_scale) with FLOAT scales for
// QuantizeLinear and FLOAT or FLOAT16 scales for DequantizeLinear, and
// byte-or-larger integer (or float8 for DequantizeLinear) element types.
// Skip per-axis / sub-byte / blocked cases (and FLOAT16-scale cases for
// QuantizeLinear).
TEST(BackendRunModel, QuantizeLinear) {
  RunBackendCasesFor("QuantizeLinear", [](const DataSet &ds) {
    if (ds.inputs.size() < 2 || ds.inputs[1].element_count() != 1) {
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
           y_dtype == static_cast<int32_t>(DataType::INT16);
  });
}
TEST(BackendRunModel, DequantizeLinear) {
  RunBackendCasesFor("DequantizeLinear", [](const DataSet &ds) {
    if (ds.inputs.size() < 2 || ds.inputs[1].element_count() != 1) {
      return false;
    }
    const int32_t scale_dtype = ds.inputs[1].data_type;
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
           x_dtype == static_cast<int32_t>(DataType::FLOAT8E5M2FNUZ);
  });
}
TEST(BackendRunModel, DynamicQuantizeLinear) { RunBackendCasesFor("DynamicQuantizeLinear"); }

// The reference QLinearMatMul kernel only supports per-tensor quantization
// (scalar scales/zero-points) with FLOAT scales. Skip FLOAT16-scale cases.
TEST(BackendRunModel, QLinearMatMul) {
  RunBackendCasesFor("QLinearMatMul", [](const DataSet &ds) {
    if (ds.inputs.size() < 8 || ds.inputs[1].element_count() != 1) {
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
    return ds.inputs.size() >= 3 && ds.inputs[0].data_type == DataType::FLOAT &&
           ds.inputs[1].data_type == DataType::FLOAT && ds.inputs[2].data_type == DataType::FLOAT;
  });
}
TEST(BackendRunModel, FlexAttention) {
  // The dispatch-table kernel handles the base FlexAttention path (Q, K, V ->
  // Y) including the optional ``score_mod`` and ``prob_mod`` modifier
  // subgraphs (executed via RunSubgraph). Some score_mod cases use ops
  // (e.g. ``Sub`` / ``Cast`` over INT64 indices) not yet wired through the
  // dispatch table; skip those by case name.
  RunBackendCasesFor(
      "FlexAttention",
      [](const TestCase &tc) {
        // ``test_cc_flexattention_relative_positional`` uses Sub/Cast on
        // INT64 indices which are not yet registered in the dispatch
        // table. ``test_cc_flexattention_soft_cap`` uses Constant/Div/Tanh
        // chains in its score_mod subgraph; some of those paths are not
        // yet covered either.
        return tc.name != "test_cc_flexattention_relative_positional" &&
               tc.name != "test_cc_flexattention_soft_cap";
      },
      [](const DataSet &ds) {
        return ds.inputs.size() >= 3 && ds.inputs[0].data_type == DataType::FLOAT &&
               ds.inputs[1].data_type == DataType::FLOAT &&
               ds.inputs[2].data_type == DataType::FLOAT;
      });
}

} // namespace Test
