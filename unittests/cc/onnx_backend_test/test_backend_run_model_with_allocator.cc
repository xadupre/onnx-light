// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Exercises every registered backend kernel with a ``SimpleRawBufferAllocator``
// attached to the ``RuntimeContext``.  After ``RunModel`` returns each
// non-STRING, non-empty output tensor must satisfy ``has_allocation() == true``
// and ``allocation_owner() != nullptr``, proving that the allocator is
// threaded through the full kernel dispatch / output-commit path.
//
// The test layout mirrors ``test_backend_run_model.cc`` one-to-one so that
// adding a new kernel requires parallel updates in both files.

#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/kernels/sequence/include_sequence_kernels.h"
#include "onnx_kernels/raw_buffer_allocator.h"
#include "onnx_kernels/run_nodes.h"
#include "onnx_kernels/runtime_context.h"
#include "onnx_kernels/simple_tensor.h"

#include <gtest/gtest.h>

#include <cstddef>
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
using onnx_kernels::Map;
using onnx_kernels::RunModel;
using onnx_kernels::RuntimeContext;
using onnx_kernels::SimpleRawBufferAllocator;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::KernelContext;

namespace Test {

namespace {

constexpr int64_t kFallbackDefaultOpsetVersion = 18;

// Generous upper bound on the number of live allocations in the parent
// RuntimeContext for any single-node test case (at most N_inputs +
// N_outputs, which never exceeds ~15 across all registered backend cases;
// subgraph child contexts do not share the parent allocator).
constexpr std::size_t kAllocatorCapacity = 64;

int64_t GetDefaultOpsetVersion(const ModelProto &model) {
  for (const auto &opset : model.ref_opset_import()) {
    if (opset.ref_domain().empty()) {
      return opset.version();
    }
  }
  return kFallbackDefaultOpsetVersion;
}

// Runs every single-node backend test case for ``op_type`` with a
// ``SimpleRawBufferAllocator`` set on the ``RuntimeContext``, and asserts
// that every non-STRING, non-empty output tensor stored in the context after
// ``RunModel`` is backed by that allocator.
//
// STRING tensors store their payload in ``string_data``; the allocator does
// not own a raw byte buffer for them and ``has_allocation()`` is intentionally
// false in that case.  Zero-byte tensors (e.g. empty scan/loop outputs) are
// similarly exempt.
void RunBackendCasesForWithAllocator(const std::string &op_type,
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

      SimpleRawBufferAllocator alloc(kAllocatorCapacity);
      RuntimeContext rt(KernelContext(DefaultOpset(GetDefaultOpsetVersion(tc.model))));
      rt.set_allocator(&alloc);

      for (const Tensor &t : ds.inputs) {
        rt.Set(t.name, t);
      }
      for (const Map &m : ds.maps) {
        rt.PutMap(m.name, m);
      }

      ASSERT_NO_THROW(RunModel(tc.model, rt)) << "RunModel threw for case " << tc.name;

      for (const Tensor &expected : ds.outputs) {
        ASSERT_TRUE(rt.Has(expected.name))
            << "Missing output '" << expected.name << "' for case " << tc.name;
        const Tensor &actual = rt.Get(expected.name);
        // STRING tensors and zero-size tensors are exempt (EnsureAllocatorBacked
        // skips them deliberately).
        if (static_cast<DataType>(actual.data_type) == DataType::STRING ||
            actual.size_bytes() == 0) {
          continue;
        }
        EXPECT_TRUE(actual.has_allocation())
            << "Output '" << expected.name << "' is not allocator-backed in case " << tc.name;
        EXPECT_EQ(actual.allocation_owner(), &alloc)
            << "Output '" << expected.name << "' has unexpected allocator owner in case "
            << tc.name;
      }
      ran_case = true;
    }
    if (ran_case) {
      ++executed;
    }
  }
  EXPECT_GT(executed, 0u) << "No single-node test cases exercised for op_type=" << op_type;
}

void RunBackendCasesForWithAllocator(
    const std::string &op_type, const std::function<bool(const DataSet &)> &accept_data_set =
                                    [](const DataSet &) { return true; }) {
  RunBackendCasesForWithAllocator(op_type, [](const TestCase &) { return true; }, accept_data_set);
}

} // namespace

// ---------------------------------------------------------------------------
// One TEST per registered kernel (mirrors BackendRunModel in
// test_backend_run_model.cc).
// ---------------------------------------------------------------------------

TEST(BackendRunModelWithAllocator, Abs) { RunBackendCasesForWithAllocator("Abs"); }
TEST(BackendRunModelWithAllocator, Neg) { RunBackendCasesForWithAllocator("Neg"); }
TEST(BackendRunModelWithAllocator, Add) { RunBackendCasesForWithAllocator("Add"); }
TEST(BackendRunModelWithAllocator, Sub) { RunBackendCasesForWithAllocator("Sub"); }
TEST(BackendRunModelWithAllocator, Mul) { RunBackendCasesForWithAllocator("Mul"); }
TEST(BackendRunModelWithAllocator, Div) { RunBackendCasesForWithAllocator("Div"); }

TEST(BackendRunModelWithAllocator, Acos) { RunBackendCasesForWithAllocator("Acos"); }
TEST(BackendRunModelWithAllocator, Acosh) { RunBackendCasesForWithAllocator("Acosh"); }
TEST(BackendRunModelWithAllocator, Asin) { RunBackendCasesForWithAllocator("Asin"); }
TEST(BackendRunModelWithAllocator, Asinh) { RunBackendCasesForWithAllocator("Asinh"); }
TEST(BackendRunModelWithAllocator, Atan) { RunBackendCasesForWithAllocator("Atan"); }
TEST(BackendRunModelWithAllocator, Atanh) { RunBackendCasesForWithAllocator("Atanh"); }
TEST(BackendRunModelWithAllocator, Ceil) { RunBackendCasesForWithAllocator("Ceil"); }
TEST(BackendRunModelWithAllocator, Cos) { RunBackendCasesForWithAllocator("Cos"); }
TEST(BackendRunModelWithAllocator, Cosh) { RunBackendCasesForWithAllocator("Cosh"); }
TEST(BackendRunModelWithAllocator, Det) { RunBackendCasesForWithAllocator("Det"); }
TEST(BackendRunModelWithAllocator, Erf) { RunBackendCasesForWithAllocator("Erf"); }
TEST(BackendRunModelWithAllocator, Exp) { RunBackendCasesForWithAllocator("Exp"); }
TEST(BackendRunModelWithAllocator, Floor) { RunBackendCasesForWithAllocator("Floor"); }
TEST(BackendRunModelWithAllocator, HardSwish) { RunBackendCasesForWithAllocator("HardSwish"); }
TEST(BackendRunModelWithAllocator, Log) { RunBackendCasesForWithAllocator("Log"); }
TEST(BackendRunModelWithAllocator, Mish) { RunBackendCasesForWithAllocator("Mish"); }
TEST(BackendRunModelWithAllocator, Reciprocal) { RunBackendCasesForWithAllocator("Reciprocal"); }
TEST(BackendRunModelWithAllocator, Relu) { RunBackendCasesForWithAllocator("Relu"); }
TEST(BackendRunModelWithAllocator, ReverseSequence) {
  RunBackendCasesForWithAllocator("ReverseSequence");
}
TEST(BackendRunModelWithAllocator, Round) { RunBackendCasesForWithAllocator("Round"); }
TEST(BackendRunModelWithAllocator, Sigmoid) { RunBackendCasesForWithAllocator("Sigmoid"); }
TEST(BackendRunModelWithAllocator, Sign) { RunBackendCasesForWithAllocator("Sign"); }
TEST(BackendRunModelWithAllocator, Sin) { RunBackendCasesForWithAllocator("Sin"); }
TEST(BackendRunModelWithAllocator, Sinh) { RunBackendCasesForWithAllocator("Sinh"); }
TEST(BackendRunModelWithAllocator, Softplus) { RunBackendCasesForWithAllocator("Softplus"); }
TEST(BackendRunModelWithAllocator, Softsign) { RunBackendCasesForWithAllocator("Softsign"); }
TEST(BackendRunModelWithAllocator, Sqrt) { RunBackendCasesForWithAllocator("Sqrt"); }
TEST(BackendRunModelWithAllocator, Tan) { RunBackendCasesForWithAllocator("Tan"); }
TEST(BackendRunModelWithAllocator, Tanh) { RunBackendCasesForWithAllocator("Tanh"); }

TEST(BackendRunModelWithAllocator, MatMul) { RunBackendCasesForWithAllocator("MatMul"); }
TEST(BackendRunModelWithAllocator, MatMulInteger) {
  RunBackendCasesForWithAllocator("MatMulInteger");
}
TEST(BackendRunModelWithAllocator, PRelu) { RunBackendCasesForWithAllocator("PRelu"); }
TEST(BackendRunModelWithAllocator, Pow) { RunBackendCasesForWithAllocator("Pow"); }

TEST(BackendRunModelWithAllocator, Equal) { RunBackendCasesForWithAllocator("Equal"); }
TEST(BackendRunModelWithAllocator, Greater) { RunBackendCasesForWithAllocator("Greater"); }
TEST(BackendRunModelWithAllocator, GreaterOrEqual) {
  RunBackendCasesForWithAllocator("GreaterOrEqual");
}
TEST(BackendRunModelWithAllocator, Less) { RunBackendCasesForWithAllocator("Less"); }
TEST(BackendRunModelWithAllocator, LessOrEqual) { RunBackendCasesForWithAllocator("LessOrEqual"); }
TEST(BackendRunModelWithAllocator, Where) { RunBackendCasesForWithAllocator("Where"); }

TEST(BackendRunModelWithAllocator, Sum) { RunBackendCasesForWithAllocator("Sum"); }
TEST(BackendRunModelWithAllocator, Max) { RunBackendCasesForWithAllocator("Max"); }
TEST(BackendRunModelWithAllocator, Min) { RunBackendCasesForWithAllocator("Min"); }
TEST(BackendRunModelWithAllocator, Mean) { RunBackendCasesForWithAllocator("Mean"); }

TEST(BackendRunModelWithAllocator, ArgMax) { RunBackendCasesForWithAllocator("ArgMax"); }
TEST(BackendRunModelWithAllocator, ArgMin) { RunBackendCasesForWithAllocator("ArgMin"); }
TEST(BackendRunModelWithAllocator, ReduceL1) { RunBackendCasesForWithAllocator("ReduceL1"); }
TEST(BackendRunModelWithAllocator, ReduceL2) { RunBackendCasesForWithAllocator("ReduceL2"); }
TEST(BackendRunModelWithAllocator, ReduceLogSum) {
  RunBackendCasesForWithAllocator("ReduceLogSum");
}
TEST(BackendRunModelWithAllocator, ReduceLogSumExp) {
  RunBackendCasesForWithAllocator("ReduceLogSumExp");
}
TEST(BackendRunModelWithAllocator, ReduceMax) { RunBackendCasesForWithAllocator("ReduceMax"); }
TEST(BackendRunModelWithAllocator, ReduceMean) { RunBackendCasesForWithAllocator("ReduceMean"); }
TEST(BackendRunModelWithAllocator, ReduceMin) { RunBackendCasesForWithAllocator("ReduceMin"); }
TEST(BackendRunModelWithAllocator, ReduceProd) { RunBackendCasesForWithAllocator("ReduceProd"); }
TEST(BackendRunModelWithAllocator, ReduceSum) { RunBackendCasesForWithAllocator("ReduceSum"); }
TEST(BackendRunModelWithAllocator, ReduceSumSquare) {
  RunBackendCasesForWithAllocator("ReduceSumSquare");
}

TEST(BackendRunModelWithAllocator, Celu) { RunBackendCasesForWithAllocator("Celu"); }
TEST(BackendRunModelWithAllocator, Elu) { RunBackendCasesForWithAllocator("Elu"); }
TEST(BackendRunModelWithAllocator, LeakyRelu) { RunBackendCasesForWithAllocator("LeakyRelu"); }
TEST(BackendRunModelWithAllocator, Swish) { RunBackendCasesForWithAllocator("Swish"); }
TEST(BackendRunModelWithAllocator, ThresholdedRelu) {
  RunBackendCasesForWithAllocator("ThresholdedRelu");
}
TEST(BackendRunModelWithAllocator, TopK) { RunBackendCasesForWithAllocator("TopK"); }
TEST(BackendRunModelWithAllocator, Hardmax) { RunBackendCasesForWithAllocator("Hardmax"); }
TEST(BackendRunModelWithAllocator, LogSoftmax) { RunBackendCasesForWithAllocator("LogSoftmax"); }
TEST(BackendRunModelWithAllocator, Flatten) { RunBackendCasesForWithAllocator("Flatten"); }
TEST(BackendRunModelWithAllocator, Softmax) { RunBackendCasesForWithAllocator("Softmax"); }
TEST(BackendRunModelWithAllocator, HardSigmoid) { RunBackendCasesForWithAllocator("HardSigmoid"); }
TEST(BackendRunModelWithAllocator, Selu) { RunBackendCasesForWithAllocator("Selu"); }
TEST(BackendRunModelWithAllocator, Shrink) { RunBackendCasesForWithAllocator("Shrink"); }
TEST(BackendRunModelWithAllocator, GatherElements) {
  RunBackendCasesForWithAllocator("GatherElements");
}
TEST(BackendRunModelWithAllocator, ScatterElements) {
  RunBackendCasesForWithAllocator("ScatterElements");
}
TEST(BackendRunModelWithAllocator, ScatterND) { RunBackendCasesForWithAllocator("ScatterND"); }
TEST(BackendRunModelWithAllocator, Gelu) { RunBackendCasesForWithAllocator("Gelu"); }
TEST(BackendRunModelWithAllocator, Mod) { RunBackendCasesForWithAllocator("Mod"); }
TEST(BackendRunModelWithAllocator, Clip) { RunBackendCasesForWithAllocator("Clip"); }
TEST(BackendRunModelWithAllocator, Compress) { RunBackendCasesForWithAllocator("Compress"); }
TEST(BackendRunModelWithAllocator, Unique) { RunBackendCasesForWithAllocator("Unique"); }
TEST(BackendRunModelWithAllocator, NonZero) { RunBackendCasesForWithAllocator("NonZero"); }
TEST(BackendRunModelWithAllocator, Concat) { RunBackendCasesForWithAllocator("Concat"); }
TEST(BackendRunModelWithAllocator, CumSum) { RunBackendCasesForWithAllocator("CumSum"); }
TEST(BackendRunModelWithAllocator, CumProd) { RunBackendCasesForWithAllocator("CumProd"); }
TEST(BackendRunModelWithAllocator, DFT) { RunBackendCasesForWithAllocator("DFT"); }
TEST(BackendRunModelWithAllocator, STFT) { RunBackendCasesForWithAllocator("STFT"); }
TEST(BackendRunModelWithAllocator, MelWeightMatrix) {
  RunBackendCasesForWithAllocator("MelWeightMatrix");
}
TEST(BackendRunModelWithAllocator, Attention) {
  RunBackendCasesForWithAllocator("Attention", [](const DataSet &ds) {
    return ds.inputs.size() >= 3 && ds.inputs[0].data_type == DataType::FLOAT &&
           ds.inputs[1].data_type == DataType::FLOAT && ds.inputs[2].data_type == DataType::FLOAT;
  });
}
TEST(BackendRunModelWithAllocator, Cast) { RunBackendCasesForWithAllocator("Cast"); }
TEST(BackendRunModelWithAllocator, CastLike) { RunBackendCasesForWithAllocator("CastLike"); }
TEST(BackendRunModelWithAllocator, CenterCropPad) {
  RunBackendCasesForWithAllocator("CenterCropPad");
}
TEST(BackendRunModelWithAllocator, Pad) { RunBackendCasesForWithAllocator("Pad"); }
TEST(BackendRunModelWithAllocator, Slice) { RunBackendCasesForWithAllocator("Slice"); }
TEST(BackendRunModelWithAllocator, BitCast) { RunBackendCasesForWithAllocator("BitCast"); }
TEST(BackendRunModelWithAllocator, CausalConvWithState) {
  RunBackendCasesForWithAllocator("CausalConvWithState");
}
TEST(BackendRunModelWithAllocator, Conv) { RunBackendCasesForWithAllocator("Conv"); }
TEST(BackendRunModelWithAllocator, ConvInteger) { RunBackendCasesForWithAllocator("ConvInteger"); }
TEST(BackendRunModelWithAllocator, ConvTranspose) {
  RunBackendCasesForWithAllocator("ConvTranspose");
}
TEST(BackendRunModelWithAllocator, Transpose) { RunBackendCasesForWithAllocator("Transpose"); }
TEST(BackendRunModelWithAllocator, Trilu) { RunBackendCasesForWithAllocator("Trilu"); }
TEST(BackendRunModelWithAllocator, TensorScatter) {
  RunBackendCasesForWithAllocator("TensorScatter");
}
TEST(BackendRunModelWithAllocator, Col2Im) { RunBackendCasesForWithAllocator("Col2Im"); }
TEST(BackendRunModelWithAllocator, DeformConv) { RunBackendCasesForWithAllocator("DeformConv"); }
TEST(BackendRunModelWithAllocator, DepthToSpace) {
  RunBackendCasesForWithAllocator("DepthToSpace");
}
TEST(BackendRunModelWithAllocator, SpaceToDepth) {
  RunBackendCasesForWithAllocator("SpaceToDepth");
}
TEST(BackendRunModelWithAllocator, Upsample) { RunBackendCasesForWithAllocator("Upsample"); }
TEST(BackendRunModelWithAllocator, BatchNormalization) {
  RunBackendCasesForWithAllocator("BatchNormalization");
}
TEST(BackendRunModelWithAllocator, GroupNormalization) {
  RunBackendCasesForWithAllocator("GroupNormalization");
}
TEST(BackendRunModelWithAllocator, GridSample) { RunBackendCasesForWithAllocator("GridSample"); }
TEST(BackendRunModelWithAllocator, RoiAlign) { RunBackendCasesForWithAllocator("RoiAlign"); }
TEST(BackendRunModelWithAllocator, MaxRoiPool) { RunBackendCasesForWithAllocator("MaxRoiPool"); }
TEST(BackendRunModelWithAllocator, InstanceNormalization) {
  RunBackendCasesForWithAllocator("InstanceNormalization");
}
TEST(BackendRunModelWithAllocator, LayerNormalization) {
  RunBackendCasesForWithAllocator("LayerNormalization");
}
TEST(BackendRunModelWithAllocator, RMSNormalization) {
  RunBackendCasesForWithAllocator("RMSNormalization");
}
TEST(BackendRunModelWithAllocator, RNN) { RunBackendCasesForWithAllocator("RNN"); }
TEST(BackendRunModelWithAllocator, RotaryEmbedding) {
  RunBackendCasesForWithAllocator("RotaryEmbedding");
}
TEST(BackendRunModelWithAllocator, MeanVarianceNormalization) {
  RunBackendCasesForWithAllocator("MeanVarianceNormalization");
}
TEST(BackendRunModelWithAllocator, Dropout) { RunBackendCasesForWithAllocator("Dropout"); }
TEST(BackendRunModelWithAllocator, AveragePool) { RunBackendCasesForWithAllocator("AveragePool"); }
TEST(BackendRunModelWithAllocator, GlobalAveragePool) {
  RunBackendCasesForWithAllocator("GlobalAveragePool");
}
TEST(BackendRunModelWithAllocator, GlobalMaxPool) {
  RunBackendCasesForWithAllocator("GlobalMaxPool");
}
TEST(BackendRunModelWithAllocator, GlobalLpPool) {
  RunBackendCasesForWithAllocator("GlobalLpPool");
}
TEST(BackendRunModelWithAllocator, LpPool) { RunBackendCasesForWithAllocator("LpPool"); }
TEST(BackendRunModelWithAllocator, LpNormalization) {
  RunBackendCasesForWithAllocator("LpNormalization");
}
TEST(BackendRunModelWithAllocator, LRN) { RunBackendCasesForWithAllocator("LRN"); }
TEST(BackendRunModelWithAllocator, MaxPool) {
  RunBackendCasesForWithAllocator("MaxPool", [](const DataSet &ds) {
    return !ds.inputs.empty() && ds.inputs[0].data_type == DataType::FLOAT;
  });
}
TEST(BackendRunModelWithAllocator, MaxUnpool) { RunBackendCasesForWithAllocator("MaxUnpool"); }

TEST(BackendRunModelWithAllocator, Adagrad) { RunBackendCasesForWithAllocator("Adagrad"); }
TEST(BackendRunModelWithAllocator, Adam) { RunBackendCasesForWithAllocator("Adam"); }
TEST(BackendRunModelWithAllocator, Momentum) { RunBackendCasesForWithAllocator("Momentum"); }

TEST(BackendRunModelWithAllocator, Bernoulli) { RunBackendCasesForWithAllocator("Bernoulli"); }
TEST(BackendRunModelWithAllocator, DelayedInitializer) {
  RunBackendCasesForWithAllocator("DelayedInitializer");
}
TEST(BackendRunModelWithAllocator, RandomNormal) {
  RunBackendCasesForWithAllocator("RandomNormal");
}
TEST(BackendRunModelWithAllocator, RandomNormalLike) {
  RunBackendCasesForWithAllocator("RandomNormalLike");
}
TEST(BackendRunModelWithAllocator, RandomUniform) {
  RunBackendCasesForWithAllocator("RandomUniform");
}
TEST(BackendRunModelWithAllocator, RandomUniformLike) {
  RunBackendCasesForWithAllocator("RandomUniformLike");
}
TEST(BackendRunModelWithAllocator, Multinomial) { RunBackendCasesForWithAllocator("Multinomial"); }

TEST(BackendRunModelWithAllocator, BlackmanWindow) {
  RunBackendCasesForWithAllocator("BlackmanWindow");
}
TEST(BackendRunModelWithAllocator, HannWindow) { RunBackendCasesForWithAllocator("HannWindow"); }
TEST(BackendRunModelWithAllocator, HammingWindow) {
  RunBackendCasesForWithAllocator("HammingWindow");
}

TEST(BackendRunModelWithAllocator, CastMap) { RunBackendCasesForWithAllocator("CastMap"); }
TEST(BackendRunModelWithAllocator, DictVectorizer) {
  RunBackendCasesForWithAllocator("DictVectorizer");
}
TEST(BackendRunModelWithAllocator, SVMRegressor) {
  RunBackendCasesForWithAllocator("SVMRegressor");
}
TEST(BackendRunModelWithAllocator, SVMClassifier) {
  RunBackendCasesForWithAllocator("SVMClassifier");
}
TEST(BackendRunModelWithAllocator, LinearRegressor) {
  RunBackendCasesForWithAllocator("LinearRegressor");
}
TEST(BackendRunModelWithAllocator, LinearClassifier) {
  RunBackendCasesForWithAllocator("LinearClassifier");
}
TEST(BackendRunModelWithAllocator, TreeEnsembleRegressor) {
  RunBackendCasesForWithAllocator("TreeEnsembleRegressor");
}
TEST(BackendRunModelWithAllocator, TreeEnsembleClassifier) {
  RunBackendCasesForWithAllocator("TreeEnsembleClassifier");
}
TEST(BackendRunModelWithAllocator, TreeEnsemble) {
  RunBackendCasesForWithAllocator("TreeEnsemble");
}
TEST(BackendRunModelWithAllocator, ArrayFeatureExtractor) {
  RunBackendCasesForWithAllocator("ArrayFeatureExtractor");
}
TEST(BackendRunModelWithAllocator, Binarizer) { RunBackendCasesForWithAllocator("Binarizer"); }
TEST(BackendRunModelWithAllocator, Normalizer) { RunBackendCasesForWithAllocator("Normalizer"); }
TEST(BackendRunModelWithAllocator, CategoryMapper) {
  RunBackendCasesForWithAllocator("CategoryMapper");
}
TEST(BackendRunModelWithAllocator, FeatureVectorizer) {
  RunBackendCasesForWithAllocator("FeatureVectorizer");
}
TEST(BackendRunModelWithAllocator, Imputer) { RunBackendCasesForWithAllocator("Imputer"); }
TEST(BackendRunModelWithAllocator, LabelEncoder) {
  RunBackendCasesForWithAllocator("LabelEncoder");
}
TEST(BackendRunModelWithAllocator, OneHotEncoder) {
  RunBackendCasesForWithAllocator("OneHotEncoder");
}
TEST(BackendRunModelWithAllocator, Scaler) { RunBackendCasesForWithAllocator("Scaler"); }

TEST(BackendRunModelWithAllocator, StringConcat) {
  RunBackendCasesForWithAllocator("StringConcat");
}
TEST(BackendRunModelWithAllocator, StringNormalizer) {
  RunBackendCasesForWithAllocator("StringNormalizer");
}
TEST(BackendRunModelWithAllocator, StringSplit) { RunBackendCasesForWithAllocator("StringSplit"); }
TEST(BackendRunModelWithAllocator, TfIdfVectorizer) {
  RunBackendCasesForWithAllocator("TfIdfVectorizer");
}

TEST(BackendRunModelWithAllocator, Scan) { RunBackendCasesForWithAllocator("Scan"); }

TEST(BackendRunModelWithAllocator, QuantizeLinear) {
  RunBackendCasesForWithAllocator("QuantizeLinear", [](const DataSet &ds) {
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
TEST(BackendRunModelWithAllocator, DequantizeLinear) {
  RunBackendCasesForWithAllocator("DequantizeLinear", [](const DataSet &ds) {
    if (ds.inputs.size() < 2) {
      return false;
    }
    const int32_t scale_dtype = ds.inputs[1].data_type;
    const int64_t scale_count = ds.inputs[1].element_count();
    const bool is_scalar_scale = scale_count == 1;
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
TEST(BackendRunModelWithAllocator, DynamicQuantizeLinear) {
  RunBackendCasesForWithAllocator("DynamicQuantizeLinear");
}
TEST(BackendRunModelWithAllocator, QLinearMatMul) {
  RunBackendCasesForWithAllocator("QLinearMatMul", [](const DataSet &ds) {
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
TEST(BackendRunModelWithAllocator, QLinearConv) {
  RunBackendCasesForWithAllocator("QLinearConv", [](const DataSet &ds) {
    if (ds.inputs.size() < 8 || ds.inputs[1].element_count() != 1 ||
        ds.inputs[6].element_count() != 1) {
      return false;
    }
    return ds.inputs[1].data_type == static_cast<int32_t>(DataType::FLOAT) &&
           ds.inputs[4].data_type == static_cast<int32_t>(DataType::FLOAT) &&
           ds.inputs[6].data_type == static_cast<int32_t>(DataType::FLOAT);
  });
}
TEST(BackendRunModelWithAllocator, LinearAttention) {
  RunBackendCasesForWithAllocator("LinearAttention", [](const DataSet &ds) {
    if (ds.inputs.size() < 3) {
      return false;
    }
    const int32_t dtype = ds.inputs[0].data_type;
    const bool supported = dtype == static_cast<int32_t>(DataType::FLOAT) ||
                           dtype == static_cast<int32_t>(DataType::FLOAT16);
    return supported && ds.inputs[1].data_type == dtype && ds.inputs[2].data_type == dtype;
  });
}
TEST(BackendRunModelWithAllocator, FlexAttention) {
  RunBackendCasesForWithAllocator(
      "FlexAttention",
      [](const TestCase &tc) { return tc.name != "test_cc_flexattention_soft_cap"; },
      [](const DataSet &ds) {
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

// SequenceMap outputs are Sequence values (not Tensor values), so the
// allocator-backed tensor check does not apply.  This test verifies that
// attaching an allocator to the RuntimeContext does not break SequenceMap
// execution.
TEST(BackendRunModelWithAllocator, SequenceMap) {
  const std::vector<TestCase> cases = CollectTestCases("SequenceMap");
  ASSERT_FALSE(cases.empty()) << "No backend test cases found for SequenceMap";

  std::size_t executed = 0;
  for (const TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);
    const KernelContext kctx(DefaultOpset(GetDefaultOpsetVersion(tc.model)));

    for (const DataSet &ds : tc.data_sets) {
      SimpleRawBufferAllocator alloc(kAllocatorCapacity);
      RuntimeContext rt(kctx);
      rt.set_allocator(&alloc);
      for (const Tensor &t : ds.inputs) {
        rt.Set(t.name, t);
      }
      ASSERT_NO_THROW(RunModel(tc.model, rt)) << "RunModel threw for case " << tc.name;
      ++executed;
    }
  }
  EXPECT_GT(executed, 0u) << "No SequenceMap test cases exercised.";
}

} // namespace Test
