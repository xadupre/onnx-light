// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Exercises every registered backend kernel with a ``SimpleRawBufferAllocator``
// attached to the ``RuntimeContext``.  After ``RunModel`` returns each
// non-STRING, non-empty output tensor must satisfy ``has_allocation() == true``
// and ``allocation_owner() != nullptr``, proving that the allocator is
// threaded through the full kernel dispatch / output-commit path.
//
// A single TEST (``AllKernels``) iterates over every registered op type; the
// ``SCOPED_TRACE`` at each iteration reports the currently-executing op type
// and test-case name in failure messages.  Add new op types to the loop (or
// to the special-predicate block below) when a kernel is registered in
// ``KernelDispatchTable``.

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
// A single TEST exercises every registered kernel.  SCOPED_TRACE at each
// iteration reports the op type and test-case name in failure messages.
// ---------------------------------------------------------------------------

TEST(BackendRunModelWithAllocator, AllKernels) {
  // Kernels that run with the default accept predicates.
  for (const char *op_type : {"Abs",
                              "Neg",
                              "Add",
                              "Sub",
                              "Mul",
                              "Div",
                              "Acos",
                              "Acosh",
                              "Asin",
                              "Asinh",
                              "Atan",
                              "Atanh",
                              "Ceil",
                              "Cos",
                              "Cosh",
                              "Det",
                              "Erf",
                              "Exp",
                              "Floor",
                              "HardSwish",
                              "Log",
                              "Mish",
                              "Reciprocal",
                              "Relu",
                              "ReverseSequence",
                              "Round",
                              "Sigmoid",
                              "Sign",
                              "Sin",
                              "Sinh",
                              "Softplus",
                              "Softsign",
                              "Sqrt",
                              "Tan",
                              "Tanh",
                              "MatMul",
                              "MatMulInteger",
                              "PRelu",
                              "Pow",
                              "Equal",
                              "Greater",
                              "GreaterOrEqual",
                              "Less",
                              "LessOrEqual",
                              "Where",
                              "Sum",
                              "Max",
                              "Min",
                              "Mean",
                              "ArgMax",
                              "ArgMin",
                              "ReduceL1",
                              "ReduceL2",
                              "ReduceLogSum",
                              "ReduceLogSumExp",
                              "ReduceMax",
                              "ReduceMean",
                              "ReduceMin",
                              "ReduceProd",
                              "ReduceSum",
                              "ReduceSumSquare",
                              "Celu",
                              "Elu",
                              "LeakyRelu",
                              "Swish",
                              "ThresholdedRelu",
                              "TopK",
                              "Hardmax",
                              "LogSoftmax",
                              "Flatten",
                              "Softmax",
                              "HardSigmoid",
                              "Selu",
                              "Shrink",
                              "GatherElements",
                              "ScatterElements",
                              "ScatterND",
                              "Gelu",
                              "Mod",
                              "Clip",
                              "Compress",
                              "Unique",
                              "NonZero",
                              "Concat",
                              "CumSum",
                              "CumProd",
                              "DFT",
                              "STFT",
                              "MelWeightMatrix",
                              "Cast",
                              "CastLike",
                              "CenterCropPad",
                              "Pad",
                              "Slice",
                              "BitCast",
                              "CausalConvWithState",
                              "Conv",
                              "ConvInteger",
                              "ConvTranspose",
                              "Transpose",
                              "Trilu",
                              "TensorScatter",
                              "Col2Im",
                              "DeformConv",
                              "DepthToSpace",
                              "SpaceToDepth",
                              "Upsample",
                              "BatchNormalization",
                              "GroupNormalization",
                              "GridSample",
                              "RoiAlign",
                              "MaxRoiPool",
                              "InstanceNormalization",
                              "LayerNormalization",
                              "RMSNormalization",
                              "RNN",
                              "RotaryEmbedding",
                              "MeanVarianceNormalization",
                              "Dropout",
                              "AveragePool",
                              "GlobalAveragePool",
                              "GlobalMaxPool",
                              "GlobalLpPool",
                              "LpPool",
                              "LpNormalization",
                              "LRN",
                              "MaxUnpool",
                              "Adagrad",
                              "Adam",
                              "Momentum",
                              "Bernoulli",
                              "DelayedInitializer",
                              "RandomNormal",
                              "RandomNormalLike",
                              "RandomUniform",
                              "RandomUniformLike",
                              "Multinomial",
                              "BlackmanWindow",
                              "HannWindow",
                              "HammingWindow",
                              "CastMap",
                              "DictVectorizer",
                              "SVMRegressor",
                              "SVMClassifier",
                              "LinearRegressor",
                              "LinearClassifier",
                              "TreeEnsembleRegressor",
                              "TreeEnsembleClassifier",
                              "TreeEnsemble",
                              "ArrayFeatureExtractor",
                              "Binarizer",
                              "Normalizer",
                              "CategoryMapper",
                              "FeatureVectorizer",
                              "Imputer",
                              "LabelEncoder",
                              "OneHotEncoder",
                              "Scaler",
                              "StringConcat",
                              "StringNormalizer",
                              "StringSplit",
                              "TfIdfVectorizer",
                              "Scan",
                              "DynamicQuantizeLinear"}) {
    SCOPED_TRACE(op_type);
    RunBackendCasesForWithAllocator(op_type);
  }

  // Kernels that require a custom accept_data_set predicate.
  {
    SCOPED_TRACE("Attention");
    RunBackendCasesForWithAllocator("Attention", [](const DataSet &ds) {
      return ds.inputs.size() >= 3 && ds.inputs[0].data_type == DataType::FLOAT &&
             ds.inputs[1].data_type == DataType::FLOAT && ds.inputs[2].data_type == DataType::FLOAT;
    });
  }
  {
    SCOPED_TRACE("MaxPool");
    RunBackendCasesForWithAllocator("MaxPool", [](const DataSet &ds) {
      return !ds.inputs.empty() && ds.inputs[0].data_type == DataType::FLOAT;
    });
  }
  {
    SCOPED_TRACE("QuantizeLinear");
    RunBackendCasesForWithAllocator("QuantizeLinear", [](const DataSet &ds) {
      if (ds.inputs.size() < 2 || ds.inputs[1].data_type != static_cast<int32_t>(DataType::FLOAT) ||
          ds.outputs.empty()) {
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
  {
    SCOPED_TRACE("DequantizeLinear");
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
  {
    SCOPED_TRACE("QLinearMatMul");
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
  {
    SCOPED_TRACE("QLinearConv");
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
  {
    SCOPED_TRACE("LinearAttention");
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
  {
    SCOPED_TRACE("FlexAttention");
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
  // allocator-backed tensor check does not apply.  Verify that attaching an
  // allocator to the RuntimeContext does not break SequenceMap execution.
  {
    SCOPED_TRACE("SequenceMap");
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
}

} // namespace Test
