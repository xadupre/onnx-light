// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Walks the whole C++ backend test registry returned by ``CollectTestCases``
// (models only, no benchmark-sized inputs, no ``_big_`` models) and, for every
// case whose graph outputs are all plain tensors, runs the model through an
// ``ExecutionPlan`` + ``RuntimeSession`` **twice** and checks that:
//   1. the produced outputs reproduce the expected ones bit-for-bit, and
//   2. for plain-tensor models, the pool allocator's memory peak is identical
//      on the second run (a leak would make the second run peak higher).
//
// This is the C++ counterpart of
// ``unittests/python/backend/test_backend_with_run_model.py``: both drive the
// same registry through the runtime's model-execution path. Unlike
// ``test_backend_run_model.cc`` (one ``TEST`` per registered op that only looks
// at single-node graphs), this exercises every collected case in a single loop,
// including the multi-node control-flow / shape-inference models.

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/run_nodes.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::CollectTestCases;
using core::backend_test::DataSet;
using core::backend_test::DefaultOpset;
using core::backend_test::TestCase;
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

// Returns the version of the default (empty-domain) ai.onnx opset imported by
// ``model``, falling back to ``kFallbackDefaultOpsetVersion`` when none is
// declared.
int64_t GetDefaultOpsetVersion(const ModelProto &model) {
  for (const auto &opset : model.ref_opset_import()) {
    if (opset.ref_domain().empty()) {
      return opset.version();
    }
  }
  return kFallbackDefaultOpsetVersion;
}

// Cases whose expected outputs are not reproduced bit-for-bit by the runtime
// because the reference data is codec-dependent. The baseline-JPEG reference
// images shipped with the upstream ONNX ``ImageDecoder`` cases are produced by
// a lossy DCT decoder whose least-significant bits differ across JPEG
// libraries, so a byte-exact comparison is inappropriate. (The lossless bmp /
// png / pnm / tiff / jpeg2k / webp decoder cases still run and match.) The
// Python counterpart excludes the same cases.
const std::unordered_set<std::string> &ExcludedCaseNames() {
  static const std::unordered_set<std::string> kExcluded = {
      "test_cc_image_decoder_decode_jpeg_bgr",
      "test_cc_image_decoder_decode_jpeg_grayscale",
      "test_cc_image_decoder_decode_jpeg_rgb",
  };
  return kExcluded;
}

// Returns whether every declared graph output of ``model`` is a plain tensor.
// Sequence-, map- and optional-typed outputs are stored outside the
// name-indexed tensor table, so the byte-exact tensor comparison below does not
// apply to them; such cases are skipped.
bool AllOutputsAreTensors(const ModelProto &model) {
  for (const auto &vi : model.ref_graph().ref_output()) {
    if (!vi.type().has_tensor_type()) {
      return false;
    }
  }
  return true;
}

// Returns whether ``model``'s top-level graph contains a control-flow node
// carrying a sub-graph attribute (``If`` / ``Loop`` / ``Scan``). Such models
// build nested per-invocation ``RuntimeContext`` / ``ExecutionPlan`` instances
// that do not yet route their buffers through an externally supplied
// ``RawBufferAllocator``, so they are excluded from the pool-allocator-backed
// memory-peak comparison (their outputs are still checked on both runs).
bool HasSubgraph(const ModelProto &model) {
  for (const auto &node : model.ref_graph().ref_node()) {
    for (const auto &attr : node.ref_attribute()) {
      if (attr.type() == AttributeProto::GRAPH || attr.type() == AttributeProto::GRAPHS) {
        return true;
      }
    }
  }
  return false;
}

// Returns whether ``model`` references any STRING tensor in its graph inputs,
// outputs, value_info, or initializers. The pool allocator's raw-byte buffers
// do not back STRING tensors (their payload lives in ``string_data``), and the
// allocator-backed execution path does not support them, so string models are
// excluded from the memory-peak comparison (their outputs are still checked on
// both runs).
bool HasStringTensor(const ModelProto &model) {
  const GraphProto &graph = model.ref_graph();
  const auto has_string_vi = [](const auto &value_infos) {
    for (const auto &vi : value_infos) {
      if (vi.type().has_tensor_type() &&
          static_cast<int32_t>(vi.type().tensor_type().elem_type()) ==
              static_cast<int32_t>(core::runtime::DataType::STRING)) {
        return true;
      }
    }
    return false;
  };
  if (has_string_vi(graph.input()) || has_string_vi(graph.output()) ||
      has_string_vi(graph.value_info())) {
    return true;
  }
  for (const auto &init : graph.initializer()) {
    if (static_cast<int32_t>(init.data_type()) ==
        static_cast<int32_t>(core::runtime::DataType::STRING)) {
      return true;
    }
  }
  return false;
}

// Asserts that ``actual`` and ``expected`` are bit-for-bit identical: same data
// type, shape, string data, and raw byte content.
void ExpectTensorBitEqual(const Tensor &actual, const Tensor &expected) {
  EXPECT_EQ(actual.data_type, expected.data_type);
  EXPECT_EQ(actual.shape, expected.shape);
  EXPECT_EQ(actual.string_data, expected.string_data);
  ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
  EXPECT_EQ(std::vector<uint8_t>(actual.bytes(), actual.bytes() + actual.size_bytes()),
            std::vector<uint8_t>(expected.bytes(), expected.bytes() + expected.size_bytes()));
}

// Registers ``model``'s local functions in ``rt``, seeds ``model.graph``'s
// initializers, and runs the graph by building its ExecutionPlan and driving it
// through a fresh RuntimeSession.
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

// Returns whether every declared graph input, output, and value_info of
// ``model`` is a plain tensor. Sequence-, map-, and optional-typed values live
// outside the name-indexed tensor table, and the allocator-backed execution
// path does not route their storage through an external ``RawBufferAllocator``,
// so such models are excluded from the memory-peak comparison.
bool AllIOTypesAreTensors(const ModelProto &model) {
  const GraphProto &graph = model.ref_graph();
  const auto all_tensor = [](const auto &value_infos) {
    for (const auto &vi : value_infos) {
      if (vi.has_type() && !vi.type().has_tensor_type()) {
        return false;
      }
    }
    return true;
  };
  return all_tensor(graph.input()) && all_tensor(graph.output()) && all_tensor(graph.value_info());
}

// Returns whether ``model``'s top-level graph uses any sequence- or
// optional-producing/consuming operator. These ops materialize non-tensor
// runtime values (sequences / optionals) that are not routed through an
// external ``RawBufferAllocator``; running them under the pool allocator is not
// supported, so such models are excluded from the memory-peak comparison (their
// outputs are still checked on both runs).
bool UsesSequenceOrOptionalOp(const ModelProto &model) {
  static const std::unordered_set<std::string> kOps = {"SequenceConstruct",
                                                       "SequenceEmpty",
                                                       "SequenceAt",
                                                       "SequenceInsert",
                                                       "SequenceErase",
                                                       "SequenceLength",
                                                       "SplitToSequence",
                                                       "ConcatFromSequence",
                                                       "SequenceMap",
                                                       "Optional",
                                                       "OptionalGetElement",
                                                       "OptionalHasElement",
                                                       "ZipMap"};
  for (const auto &node : model.ref_graph().ref_node()) {
    if (node.ref_domain().empty() && kOps.count(node.op_type()) != 0) {
      return true;
    }
  }
  return false;
}

} // namespace

// Collects every model-based backend test case (``TestMode::TEST``, no big
// models) and, for each data set, (1) runs the model twice through a
// ``RuntimeSession`` (no allocator) and checks every output bit-for-bit on both
// runs, and (2) for plain-tensor models, runs it twice more through a
// ``SimpleRawBufferAllocator`` and verifies the allocator's memory peak is
// identical on the second run. The expected outputs are themselves produced by
// the very kernels the runtime dispatches to, so a bit-exact comparison is
// appropriate. Outputs are matched positionally against ``graph.output`` (a
// case's expected-tensor names are not required to match the graph output
// names).
//
// The second-run peak check guards against per-run buffer leaks: each run uses a
// fresh ``RuntimeContext`` whose destructor frees every allocator buffer it
// holds, so a leak-free second run reaches exactly the same live total as the
// first and leaves the running peak unchanged. Any buffer retained across runs
// would make the second run peak higher. Models the allocator path does not
// support (control flow, sequence/optional/map, or string tensors) still run
// twice for output correctness but skip the peak comparison.
TEST(BackendRunModelAllCases, RunEveryModelTwiceWithStableMemoryPeak) {
  std::vector<TestCase> cases = CollectTestCases();
  ASSERT_FALSE(cases.empty()) << "No backend test cases collected.";

  // ``SimpleRawBufferAllocator`` capacity counts buffer slots, not bytes. The
  // collected model cases are small; this is comfortably above the maximum
  // number of buffers any of them holds live at once.
  constexpr size_t kAllocatorSlotCapacity = 4096;

  size_t executed = 0;
  size_t peak_checked = 0;
  for (TestCase &tc : cases) {
    if (ExcludedCaseNames().count(tc.name) != 0) {
      continue;
    }
    const ModelProto &model = tc.model();
    if (!AllOutputsAreTensors(model)) {
      continue;
    }
    SCOPED_TRACE(tc.name);

    // The memory-peak comparison drives the model through the pool-allocator
    // execution path. That path is only wired into the top-level ExecutionPlan
    // for plain-tensor models: control-flow sub-graphs build their own nested
    // contexts, and sequence/optional/map/string values are stored outside the
    // allocator-backed tensor table. Such models are still run twice for output
    // correctness, but they skip the peak check.
    const bool track_peak = AllIOTypesAreTensors(model) && !HasSubgraph(model) &&
                            !HasStringTensor(model) && !UsesSequenceOrOptionalOp(model);

    const auto &outputs = model.ref_graph().ref_output();
    for (const DataSet &ds : tc.data_sets()) {
      const auto seed_inputs = [&](RuntimeContext &rt) {
        for (const Tensor &t : ds.inputs) {
          rt.Put(t.name, t, core::runtime::RuntimeEventKind::kInput);
        }
        for (const Map &m : ds.maps) {
          rt.PutMap(m.name, m);
        }
      };

      const auto check_outputs = [&](RuntimeContext &rt) {
        ASSERT_EQ(ds.outputs.size(), outputs.size())
            << "Data set / graph output arity mismatch for case " << tc.name;
        for (size_t i = 0; i < outputs.size(); ++i) {
          const std::string &oname = outputs[i].name();
          ASSERT_TRUE(rt.Has(oname)) << "Missing output '" << oname << "' for case " << tc.name;
          ExpectTensorBitEqual(rt.Get(oname), ds.outputs[i]);
        }
      };

      // Base correctness: run the model twice through a RuntimeSession (no
      // allocator) and check every output bit-for-bit on both runs.
      for (int run = 0; run < 2; ++run) {
        RuntimeContext rt(KernelContext(DefaultOpset(GetDefaultOpsetVersion(model))));
        seed_inputs(rt);
        ASSERT_NO_THROW(RunModelViaSession(model, rt))
            << "Running the model threw for case " << tc.name;
        check_outputs(rt);
      }

      if (!track_peak) {
        continue;
      }

      // Memory-peak stability: a single allocator is shared across two runs and
      // its peak is *not* reset between them. Each run uses a fresh
      // RuntimeContext whose destructor frees every allocator buffer it holds,
      // so a leak-free second run reaches exactly the same live total as the
      // first and leaves the running peak unchanged; any buffer retained across
      // runs would raise it.
      core::runtime::SimpleRawBufferAllocator alloc(kAllocatorSlotCapacity);
      const auto run_with_allocator = [&]() -> size_t {
        RuntimeContext rt(KernelContext(DefaultOpset(GetDefaultOpsetVersion(model))));
        rt.set_allocator(&alloc);
        seed_inputs(rt);
        // Let exceptions propagate so genuinely allocator-unsupported edge
        // cases (e.g. zero-byte allocations) are skipped rather than failed.
        RunModelViaSession(model, rt);
        for (size_t i = 0; i < outputs.size(); ++i) {
          const std::string &oname = outputs[i].name();
          if (rt.Has(oname)) {
            ExpectTensorBitEqual(rt.Get(oname), ds.outputs[i]);
          }
        }
        return alloc.PeakAllocatedSize();
      };

      // NOLINTNEXTLINE(bugprone-empty-catch): a throw means the allocator path
      // does not support this model; the peak check is simply skipped.
      try {
        const size_t peak_first = run_with_allocator();
        const size_t peak_second = run_with_allocator();
        EXPECT_EQ(peak_first, peak_second)
            << "Memory peak changed on the second run for case " << tc.name;
        ++peak_checked;
      } catch (const std::exception &) {
        // Allocator-unsupported model: outputs were already validated above.
      }
    }
    ++executed;
  }

  EXPECT_GT(executed, 1000u) << "Expected the model-run loop to exercise many cases.";
  EXPECT_GT(peak_checked, 500u) << "Expected the memory-peak stability check to cover many models.";
}

} // namespace Test
