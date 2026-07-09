// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Exercises every registered backend kernel with a ``SimpleRawBufferAllocator``
// attached to the ``RuntimeContext``.  After ``RunModel`` returns each
// non-STRING, non-empty output tensor must satisfy ``has_allocation() == true``
// and ``allocation_owner() != nullptr``, proving that the allocator is
// threaded through the full kernel dispatch / output-commit path.
//
// A single TEST (``AllKernels``) collects all backend test cases at once via
// ``CollectTestCases()`` and iterates over them; ``SCOPED_TRACE(tc.name)``
// identifies the failing case in failure messages.

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

// Returns false for DataSets that use dtype combinations not supported by the
// given op.  These mirror the filters in test_backend_run_model.cc.
bool AcceptDataSet(const std::string &op_type, const DataSet &ds) {
  if (op_type == "Attention") {
    return ds.inputs.size() >= 3 && ds.inputs[0].data_type == DataType::FLOAT &&
           ds.inputs[1].data_type == DataType::FLOAT && ds.inputs[2].data_type == DataType::FLOAT;
  }
  if (op_type == "MaxPool") {
    return !ds.inputs.empty() && ds.inputs[0].data_type == DataType::FLOAT;
  }
  if (op_type == "QuantizeLinear") {
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
  }
  if (op_type == "DequantizeLinear") {
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
  }
  if (op_type == "QLinearMatMul") {
    if (ds.inputs.size() < 8 || ds.inputs[1].element_count() != 1) {
      return false;
    }
    auto is_float_scale = [](int32_t dt) {
      return dt == static_cast<int32_t>(DataType::FLOAT) ||
             dt == static_cast<int32_t>(DataType::FLOAT16);
    };
    return is_float_scale(ds.inputs[1].data_type) && is_float_scale(ds.inputs[4].data_type) &&
           is_float_scale(ds.inputs[6].data_type);
  }
  if (op_type == "QLinearConv") {
    if (ds.inputs.size() < 8 || ds.inputs[1].element_count() != 1 ||
        ds.inputs[6].element_count() != 1) {
      return false;
    }
    return ds.inputs[1].data_type == static_cast<int32_t>(DataType::FLOAT) &&
           ds.inputs[4].data_type == static_cast<int32_t>(DataType::FLOAT) &&
           ds.inputs[6].data_type == static_cast<int32_t>(DataType::FLOAT);
  }
  if (op_type == "LinearAttention") {
    if (ds.inputs.size() < 3) {
      return false;
    }
    const int32_t dtype = ds.inputs[0].data_type;
    const bool supported = dtype == static_cast<int32_t>(DataType::FLOAT) ||
                           dtype == static_cast<int32_t>(DataType::FLOAT16);
    return supported && ds.inputs[1].data_type == dtype && ds.inputs[2].data_type == dtype;
  }
  if (op_type == "FlexAttention") {
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
  }
  return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Single TEST that collects all registered backend test cases at once and
// exercises each one with a SimpleRawBufferAllocator.  SCOPED_TRACE reports
// the failing case name in failure messages.
// ---------------------------------------------------------------------------

TEST(BackendRunModelWithAllocator, AllKernels) {
  const std::vector<TestCase> cases = CollectTestCases();
  ASSERT_FALSE(cases.empty());

  for (const TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);

    const auto &graph = tc.model.ref_graph();
    if (graph.ref_node().size() != 1u) {
      continue;
    }
    const std::string op_type = graph.ref_node()[0].ref_op_type().as_string();

    // One FlexAttention case uses a soft-cap variant not yet supported.
    if (tc.name == "test_cc_flexattention_soft_cap") {
      continue;
    }

    for (const DataSet &ds : tc.data_sets) {
      if (!AcceptDataSet(op_type, ds)) {
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

      ASSERT_NO_THROW(RunModel(tc.model, rt));

      // SequenceMap outputs are Sequence values, not Tensors; skip the
      // allocator-backed check for them.
      if (op_type == "SequenceMap") {
        continue;
      }

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
    }
  }
}

} // namespace Test
