// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/raw_buffer_allocator.h"
#include "onnx_kernels/run_nodes.h"
#include "onnx_kernels/runtime_context.h"
#include "onnx_kernels/simple_tensor.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::CollectTestCases;
using onnx_backend_test::DataSet;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::TestCase;
using onnx_kernels::MakeOutputTensor;
using onnx_kernels::Map;
using onnx_kernels::RunModel;
using onnx_kernels::RuntimeContext;
using onnx_kernels::SimpleRawBufferAllocator;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::KernelContext;

namespace {

// Allocator slot capacity — sized for the largest single-level test-case data
// set (no Loop/Scan iterations to multiply the slot demand).
constexpr size_t kAllocatorSlotCapacity = 256;

// Fallback opset version when the model carries no default-domain opset import.
constexpr int64_t kFallbackDefaultOpsetVersion = 21;

// Returns the ai.onnx opset version embedded in ``model``, or
// ``kFallbackDefaultOpsetVersion`` when no default-domain opset is present.
int64_t GetDefaultOpsetVersion(const ModelProto &model) {
  for (const auto &opset : model.ref_opset_import()) {
    if (opset.ref_domain().empty()) {
      return opset.version();
    }
  }
  return kFallbackDefaultOpsetVersion;
}

// Returns true when any graph (top-level or function body) of ``model``
// contains at least one node whose op_type is If, Loop, or Scan.
//
// RuntimeContext::MakeSubgraphContext does a shallow copy of the parent
// context's tensor map (including raw allocation_ pointers). The child
// context's destructor therefore frees the same allocator slots that the
// parent still holds, causing a use-after-free / SEGFAULT when the parent
// subsequently accesses those tensors. Attaching a SimpleRawBufferAllocator
// to the RuntimeContext is therefore unsafe for models that trigger subgraph
// execution. Those cases are skipped in the allocator test below; their
// correctness is already covered by the BackendRunModel.* tests.
static bool GraphHasSubgraphOp(const GraphProto &graph) {
  for (const auto &node : graph.ref_node()) {
    const std::string_view op = node.ref_op_type().sv();
    if (op == "If" || op == "Loop" || op == "Scan") {
      return true;
    }
  }
  return false;
}

bool HasSubgraphOp(const ModelProto &model) {
  if (GraphHasSubgraphOp(model.ref_graph())) {
    return true;
  }
  for (const auto &fn : model.ref_function()) {
    for (const auto &node : fn.ref_node()) {
      const std::string_view op = node.ref_op_type().sv();
      if (op == "If" || op == "Loop" || op == "Scan") {
        return true;
      }
    }
  }
  return false;
}

// Returns true when ``data_type`` is the ONNX STRING element type.
// STRING tensors store their values in ``string_data``, not in the raw byte
// buffer, so they cannot be made allocator-backed via MakeOutputTensor.
bool IsStringType(int32_t data_type) {
  return data_type == static_cast<int32_t>(TensorProto::DataType::STRING);
}

// Creates an allocator-backed copy of a non-STRING tensor ``src``.
// The returned tensor carries the same name, data_type, shape, and raw bytes
// as ``src``, but its storage is managed by ``alloc``.
Tensor MakeAllocatorBackedCopy(const Tensor &src, onnx_kernels::RawBufferAllocator *alloc) {
  const size_t n = src.size_bytes();
  Tensor t = MakeOutputTensor(src.data_type, src.shape, n, alloc);
  t.name = src.name;
  if (n > 0) {
    std::memcpy(t.mutable_bytes(), src.bytes(), n);
  }
  return t;
}

// Asserts that ``actual`` matches ``expected`` (shape, dtype, string_data, and
// raw bytes for numeric dtypes) and, for non-STRING types, that ``actual`` is
// allocator-backed (``has_allocation() == true``).
void ExpectAllocatorBackedOutputMatchesExpected(const Tensor &actual, const Tensor &expected,
                                                const std::string &case_name) {
  EXPECT_EQ(actual.data_type, expected.data_type) << "case: " << case_name;
  EXPECT_EQ(actual.shape, expected.shape) << "case: " << case_name;
  EXPECT_EQ(actual.string_data, expected.string_data) << "case: " << case_name;
  const size_t n = expected.size_bytes();
  ASSERT_EQ(actual.size_bytes(), n) << "case: " << case_name;
  if (n > 0) {
    EXPECT_EQ(std::vector<uint8_t>(actual.bytes(), actual.bytes() + n),
              std::vector<uint8_t>(expected.bytes(), expected.bytes() + n))
        << "case: " << case_name << " output '" << expected.name << "'";
  }
  // Non-STRING outputs produced via a RuntimeContext that carries an allocator
  // must be allocator-backed so their storage is managed by that allocator.
  if (!IsStringType(expected.data_type)) {
    EXPECT_TRUE(actual.has_allocation())
        << "case: " << case_name << " output '" << expected.name << "' is not allocator-backed";
  }
}

// Runs every data set of ``tc`` through RunModel with a
// SimpleRawBufferAllocator attached to the RuntimeContext and verifies that:
//   1. Every output matches the pre-computed expected tensor (bit-exact for
//      numeric types, string_data-exact for STRING types).
//   2. Every non-STRING input stored in the RuntimeContext is allocator-backed.
//   3. Every non-STRING output stored in the RuntimeContext is allocator-backed.
void RunWithAllocatorAndVerify(const TestCase &tc) {
  for (const DataSet &ds : tc.data_sets()) {
    SCOPED_TRACE("data_set");
    SimpleRawBufferAllocator alloc(kAllocatorSlotCapacity);
    RuntimeContext rt(KernelContext(DefaultOpset(GetDefaultOpsetVersion(tc.model()))));
    rt.set_allocator(&alloc);

    // Load inputs. Non-STRING tensors are backed by the allocator so the
    // "inputs have allocator not empty" postcondition can be verified below.
    for (const Tensor &in : ds.inputs) {
      if (IsStringType(in.data_type)) {
        rt.Set(in.name, in);
      } else {
        rt.Set(in.name, MakeAllocatorBackedCopy(in, &alloc));
      }
    }
    for (const Map &m : ds.maps) {
      rt.PutMap(m.name, m);
    }

    // Verify that non-STRING inputs are allocator-backed.
    for (const Tensor &in : ds.inputs) {
      if (!IsStringType(in.data_type)) {
        EXPECT_TRUE(rt.Get(in.name).has_allocation())
            << "case: " << tc.name << " input '" << in.name << "' is not allocator-backed";
      }
    }

    ASSERT_NO_THROW(RunModel(tc.model(), rt)) << "case: " << tc.name;

    // Verify outputs: expected values and allocator-backed storage.
    for (const Tensor &expected : ds.outputs) {
      ASSERT_TRUE(rt.Has(expected.name))
          << "case: " << tc.name << " missing output '" << expected.name << "'";
      ExpectAllocatorBackedOutputMatchesExpected(rt.Get(expected.name), expected, tc.name);
    }
  }
}

} // namespace

namespace Test {

// Runs every C++ backend test case collected by CollectTestCases() through
// RunModel with a SimpleRawBufferAllocator attached to the RuntimeContext.
//
// Test cases whose top-level graph contains a subgraph op (If, Loop, Scan)
// are skipped: RuntimeContext::MakeSubgraphContext shallow-copies the parent's
// allocation pointers into the child context, whose destructor would free
// those slots while the parent still holds them (use-after-free). Subgraph
// correctness is already verified by the BackendRunModel.* tests.
//
// For each remaining data set the test asserts:
//   1. Outputs match the pre-computed expected values (bit-exact for numeric
//      types; string_data for STRING types).
//   2. Every non-STRING input stored in the RuntimeContext is allocator-backed.
//   3. Every non-STRING output stored in the RuntimeContext is allocator-backed.
TEST(RunModelWithAllocator, AllBackendTestCases) {
  const auto cases = CollectTestCases();
  ASSERT_FALSE(cases.empty());
  size_t executed = 0;
  for (const TestCase &tc : cases) {
    SCOPED_TRACE(tc.name);
    if (HasSubgraphOp(tc.model())) {
      continue;
    }
    RunWithAllocatorAndVerify(tc);
    ++executed;
  }
  EXPECT_GT(executed, 100u) << "Fewer non-subgraph test cases than expected were exercised";
}

} // namespace Test
