// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Walks the whole C++ backend test registry returned by ``CollectTestCases``
// (models only, no benchmark-sized inputs, no ``_big_`` models) and, for every
// case whose graph outputs are all plain tensors, builds a single
// ``RuntimeContext`` / ``ExecutionPlan`` / ``RuntimeSession`` and runs that same
// session **twice**, checking that:
//   1. the produced outputs reproduce the expected ones bit-for-bit on both
//      runs, and
//   2. for plain-tensor models, the pool allocator's memory peak is identical
//      after the second run (a leak would make the second run peak higher).
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
// because the reference data is codec-dependent. Two families are excluded:
//
//   * The baseline-JPEG reference images shipped with the upstream ONNX
//     ``ImageDecoder`` cases are produced by a lossy DCT decoder whose
//     least-significant bits differ across JPEG libraries, so a byte-exact
//     comparison is inappropriate.
//   * The JPEG 2000 and WebP cases decode through optional runtime codecs
//     (libopenjp2 / libwebp). Those libraries are present on the Linux CI
//     runners but not on the macOS / Windows ones, where ``ImageDecoder``
//     falls back to an empty output; even where a codec is available its
//     version-dependent output is only guaranteed to match within a small
//     tolerance, not bit-for-bit. (The lossless bmp / png / pnm / tiff decoder
//     cases use the bundled decoders and still run and match everywhere.)
//
// The Python counterpart excludes the same cases.
const std::unordered_set<std::string> &ExcludedCaseNames() {
  static const std::unordered_set<std::string> kExcluded = {
      "test_cc_image_decoder_decode_jpeg_bgr", "test_cc_image_decoder_decode_jpeg_grayscale",
      "test_cc_image_decoder_decode_jpeg_rgb", "test_cc_image_decoder_decode_jpeg2k_rgb",
      "test_cc_image_decoder_decode_webp_rgb",
  };
  return kExcluded;
}

// Cases whose plain-tensor outputs are genuinely empty (zero elements): the
// pool allocator cannot back a zero-byte buffer (it hands back a null data
// pointer, which the kernels reject), so these models are excluded from the
// allocator-backed memory-peak comparison. They still run twice, without an
// allocator, for output correctness.
const std::unordered_set<std::string> &AllocatorUnsupportedCaseNames() {
  static const std::unordered_set<std::string> kUnsupported = {
      "test_cc_reducelogsumexp_empty_set_non_reduced_axis_zero",
  };
  return kUnsupported;
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
// models) and, for each data set, builds a single ``RuntimeContext``,
// ``ExecutionPlan`` and ``RuntimeSession`` and runs that same session twice.
// Every output is checked bit-for-bit on both runs (the expected outputs are
// themselves produced by the very kernels the runtime dispatches to, so a
// bit-exact comparison is appropriate). Outputs are matched positionally
// against ``graph.output`` (a case's expected-tensor names are not required to
// match the graph output names).
//
// For plain-tensor models the context is backed by a shared
// ``SimpleRawBufferAllocator`` and the allocator's memory peak must be
// identical after the second run: replaying the plan on the same context must
// not accumulate buffers, so a leak-free second run leaves the running peak
// unchanged, whereas any buffer retained across runs would raise it. Models the
// allocator path does not support (control flow, sequence/optional/map, or
// string tensors) still run twice for output correctness but skip the peak
// comparison.
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
                            !HasStringTensor(model) && !UsesSequenceOrOptionalOp(model) &&
                            AllocatorUnsupportedCaseNames().count(tc.name) == 0;

    const GraphProto &graph = model.ref_graph();
    const auto &outputs = graph.ref_output();
    for (const DataSet &ds : tc.data_sets()) {
      ASSERT_EQ(ds.outputs.size(), outputs.size())
          << "Data set / graph output arity mismatch for case " << tc.name;

      // One allocator, one RuntimeContext, one ExecutionPlan and one
      // RuntimeSession per data set; the same session is run twice below.
      // Intermediates are released as scheduled so each run allocates and frees
      // the same buffers.
      core::runtime::SimpleRawBufferAllocator alloc(kAllocatorSlotCapacity);
      RuntimeContext rt(
          KernelContext(DefaultOpset(GetDefaultOpsetVersion(model))),
          core::runtime::RuntimeContextOptions{.allocator = track_peak ? &alloc : nullptr});
      rt.set_release_intermediates(true);
      RegisterModelFunctions(model, rt);

      const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
      RuntimeSession session(plan);

      // Seed inputs, maps and initializers once. These are the tensors that
      // must survive between the two runs; every other tensor a run produces is
      // cleared afterwards so the second run replays from the identical state.
      std::unordered_set<std::string> seeded_names;
      for (const Tensor &t : ds.inputs) {
        rt.Put(t.name, t, core::runtime::RuntimeEventKind::kInput);
        seeded_names.insert(t.name);
      }
      for (const Map &m : ds.maps) {
        rt.PutMap(m.name, m);
      }
      for (const TensorProto &tp : graph.initializer()) {
        if (!rt.Has(tp.name())) {
          rt.Set(tp.name(), TensorFromProto(tp), core::runtime::RuntimeEventKind::kInitializer);
        }
        seeded_names.insert(tp.name());
      }

      size_t memory_peaks[2] = {0, 0};
      for (int run = 0; run < 2; ++run) {
        ASSERT_NO_THROW(session.Run(rt)) << "Run " << (run + 1) << " threw for case " << tc.name;
        for (size_t i = 0; i < outputs.size(); ++i) {
          const std::string &oname = outputs[i].name();
          ASSERT_TRUE(rt.Has(oname)) << "Missing output '" << oname << "' for case " << tc.name;
          ExpectTensorBitEqual(rt.Get(oname), ds.outputs[i]);
        }
        memory_peaks[run] = alloc.PeakAllocatedSize();

        // Restore the seeded baseline: drop every tensor the run produced —
        // graph outputs, released-late intermediates, and dead node outputs
        // (e.g. an unused ``TopK`` ``indices``). Holding these live while the
        // next run rebuilds them would inflate the second run's peak, so the
        // second run of the same session starts from the identical state and
        // reaches the identical peak when there is no leak.
        std::vector<std::string> produced_names;
        for (const auto &named_tensor : rt.tensors()) {
          if (seeded_names.count(named_tensor.first) == 0) {
            produced_names.push_back(named_tensor.first);
          }
        }
        for (const std::string &name : produced_names) {
          rt.Remove(name);
        }
      }

      if (track_peak) {
        EXPECT_EQ(memory_peaks[0], memory_peaks[1])
            << "Memory peak changed on the second run for case " << tc.name;
        ++peak_checked;
      }
    }
    ++executed;
  }

  EXPECT_GT(executed, 1000u) << "Expected the model-run loop to exercise many cases.";
  EXPECT_GT(peak_checked, 500u) << "Expected the memory-peak stability check to cover many models.";
}

} // namespace Test
