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
#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>
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
// because the reference data is codec-dependent:
//
//   * The baseline-JPEG reference images shipped with the upstream ONNX
//     ``ImageDecoder`` cases are produced by a lossy DCT decoder whose
//     least-significant bits differ across JPEG libraries, so a byte-exact
//     comparison is inappropriate. (The lossless bmp / png / pnm decoder
//     cases use the bundled decoders and still run and match everywhere.)
//
// The Python counterpart excludes the same cases.
const std::unordered_set<std::string> &ExcludedCaseNames() {
  static const std::unordered_set<std::string> kExcluded = {
      "test_cc_image_decoder_decode_jpeg_bgr",
      "test_cc_image_decoder_decode_jpeg_grayscale",
      "test_cc_image_decoder_decode_jpeg_rgb",
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

// Cases whose byte-exact comparison is only reliable on 64-bit targets. The
// FlexAttention ``score_mod`` / ``soft_cap`` cases apply their modifier as a
// C++ lambda in the ``double``-precision reference that produced the expected
// outputs, but as an ONNX sub-graph (Add / Div / Tanh / ...) evaluated by the
// ``float`` kernel at run time. Those two float rounding paths agree
// bit-for-bit on 64-bit builds but diverge by a single ULP on 32-bit builds
// (e.g. Windows x86). The math is correct on every platform; only the
// byte-exact comparison is inappropriate on 32-bit, so these cases are skipped
// there (and still run and match on every 64-bit target).
const std::unordered_set<std::string> &BitInexactOn32BitCaseNames() {
  static const std::unordered_set<std::string> kNames = [] {
    std::unordered_set<std::string> names;
    if constexpr (sizeof(void *) == 4) {
      names.insert("test_cc_flexattention_score_mod");
      names.insert("test_cc_flexattention_soft_cap");
    }
    return names;
  }();
  return kNames;
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

void AddOpsetImport(ModelProto &model, const std::string &domain, int64_t version) {
  OperatorSetIdProto *opset = model.add_opset_import();
  if (!domain.empty()) {
    opset->set_domain(domain);
  }
  opset->set_version(version);
}

void AddFunctionNode(FunctionProto &function, const std::string &op_type, const std::string &domain,
                     const std::vector<std::string> &inputs,
                     const std::vector<std::string> &outputs) {
  NodeProto *node = function.add_node();
  node->set_op_type(op_type);
  if (!domain.empty()) {
    node->set_domain(domain);
  }
  for (const std::string &input : inputs) {
    node->add_input(input);
  }
  for (const std::string &output : outputs) {
    node->add_output(output);
  }
}

std::pair<ModelProto, std::vector<DataSet>> BuildThreeLevelNestedLocalFunctionCase() {
  ModelProto model;
  model.set_ir_version(10);
  model.set_producer_name("backend-test");
  AddOpsetImport(model, "", 18);
  AddOpsetImport(model, "custom", 1);

  FunctionProto *inner = model.add_functions();
  inner->set_name("Inner");
  inner->set_domain("custom");
  inner->add_input("x");
  inner->add_output("y");
  AddFunctionNode(*inner, "Add", "", {"x", "x"}, {"y"});

  FunctionProto *middle = model.add_functions();
  middle->set_name("Middle");
  middle->set_domain("custom");
  middle->add_input("x");
  middle->add_output("y");
  AddFunctionNode(*middle, "Inner", "custom", {"x"}, {"t"});
  AddFunctionNode(*middle, "Inner", "custom", {"t"}, {"y"});

  FunctionProto *outer = model.add_functions();
  outer->set_name("Outer");
  outer->set_domain("custom");
  outer->add_input("x");
  outer->add_output("y");
  AddFunctionNode(*outer, "Inner", "custom", {"x"}, {"t"});
  AddFunctionNode(*outer, "Middle", "custom", {"t"}, {"y"});

  GraphProto &graph = model.ref_graph();
  graph.set_name("test_cc_local_function_three_level_nested_calls");
  NodeProto *call = graph.add_node();
  call->set_op_type("Outer");
  call->set_domain("custom");
  call->add_input("x");
  call->add_output("y");

  Tensor x = Tensor::FromFloat("x", {3}, {1.0f, 2.5f, -3.0f});
  Tensor y = Tensor::FromFloat("y", {3}, {8.0f, 20.0f, -24.0f});
  FillValueInfo(x, *graph.add_input());
  FillValueInfo(y, *graph.add_output());

  DataSet ds;
  ds.inputs = {x};
  ds.outputs = {y};
  std::vector<DataSet> data_sets;
  data_sets.push_back(std::move(ds));
  return {std::move(model), std::move(data_sets)};
}

std::pair<ModelProto, std::vector<DataSet>> BuildLinkedAttributeLocalFunctionCase() {
  ModelProto model;
  model.set_ir_version(10);
  model.set_producer_name("backend-test");
  AddOpsetImport(model, "", 18);
  AddOpsetImport(model, "custom", 1);

  FunctionProto *function = model.add_functions();
  function->set_name("Pick");
  function->set_domain("custom");
  function->add_input("cond");
  function->add_output("out");
  function->add_attribute("then_branch");
  function->add_attribute("else_branch");
  {
    NodeProto *if_node = function->add_node();
    if_node->set_op_type("If");
    if_node->add_input("cond");
    if_node->add_output("out");
    AttributeProto *then_ref = if_node->add_attribute();
    then_ref->set_name("then_branch");
    then_ref->set_ref_attr_name("then_branch");
    then_ref->set_type(AttributeProto::AttributeType::GRAPH);
    AttributeProto *else_ref = if_node->add_attribute();
    else_ref->set_name("else_branch");
    else_ref->set_ref_attr_name("else_branch");
    else_ref->set_type(AttributeProto::AttributeType::GRAPH);
  }

  GraphProto &graph = model.ref_graph();
  graph.set_name("test_cc_local_function_linked_attribute");
  NodeProto *call = graph.add_node();
  call->set_op_type("Pick");
  call->set_domain("custom");
  call->add_input("cond");
  call->add_output("out");
  auto fill_branch = [](GraphProto &branch, const std::string &branch_name,
                        const std::string &init_name, float value) {
    branch.set_name(branch_name);
    TensorProto *init = branch.add_initializer();
    init->set_name(init_name);
    init->set_data_type(TensorProto::DataType::FLOAT);
    init->add_float_data(value);
    NodeProto *add = branch.add_node();
    add->set_op_type("Add");
    add->add_input(init_name);
    add->add_input(init_name);
    add->add_output("z");
    branch.add_output()->set_name("z");
  };
  AttributeProto *then_attr = call->add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(AttributeProto::AttributeType::GRAPH);
  fill_branch(*then_attr->mutable_g(), "then_g", "t", 10.0f);
  AttributeProto *else_attr = call->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(AttributeProto::AttributeType::GRAPH);
  fill_branch(*else_attr->mutable_g(), "else_g", "e", 1.0f);

  Tensor cond_true = Tensor::FromBool("cond", {}, {1});
  Tensor cond_false = Tensor::FromBool("cond", {}, {0});
  Tensor out_true = Tensor::FromFloat("out", {}, {20.0f});
  Tensor out_false = Tensor::FromFloat("out", {}, {2.0f});
  FillValueInfo(cond_true, *graph.add_input());
  FillValueInfo(out_true, *graph.add_output());

  DataSet ds_true;
  ds_true.inputs = {cond_true};
  ds_true.outputs = {out_true};
  DataSet ds_false;
  ds_false.inputs = {cond_false};
  ds_false.outputs = {out_false};
  std::vector<DataSet> data_sets;
  data_sets.push_back(std::move(ds_true));
  data_sets.push_back(std::move(ds_false));
  return {std::move(model), std::move(data_sets)};
}

void ExpectModelOutputsMatchDataSet(const ModelProto &model, const DataSet &ds) {
  RuntimeContext rt(KernelContext(DefaultOpset(GetDefaultOpsetVersion(model))));
  const GraphProto &graph = model.ref_graph();
  RegisterModelFunctions(model, rt);
  for (const Tensor &t : ds.inputs) {
    rt.Put(t.name, t, core::runtime::RuntimeEventKind::kInput);
  }
  for (const Map &m : ds.maps) {
    rt.PutMap(m.name, m);
  }
  for (const TensorProto &tp : graph.initializer()) {
    if (!rt.Has(tp.name())) {
      rt.Set(tp.name(), TensorFromProto(tp), core::runtime::RuntimeEventKind::kInitializer);
    }
  }
  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan);
  ASSERT_NO_THROW(session.Run(rt));
  ASSERT_EQ(ds.outputs.size(), graph.output().size());
  for (size_t i = 0; i < graph.output().size(); ++i) {
    const std::string output_name = graph.output()[i].name();
    ASSERT_TRUE(rt.Has(output_name)) << "Missing output '" << output_name << "'.";
    ExpectTensorBitEqual(rt.Get(output_name), ds.outputs[i]);
  }
}

} // namespace

TEST(BackendRunModelAllCases, GraphBuilderRoundTripKeepsLocalFunctionsRunnable) {
  auto case_data = BuildThreeLevelNestedLocalFunctionCase();
  const ModelProto &original = case_data.first;
  core::builder::GraphBuilder rebuilt(original);
  ASSERT_TRUE(rebuilt.HasLocalFunction("Inner"));
  ASSERT_TRUE(rebuilt.HasLocalFunction("Middle"));
  ASSERT_TRUE(rebuilt.HasLocalFunction("Outer"));
  const ModelProto round_tripped = rebuilt.ToModel(original.ir_version());
  ASSERT_EQ(round_tripped.functions().size(), original.functions().size());
  for (const DataSet &ds : case_data.second) {
    ExpectModelOutputsMatchDataSet(original, ds);
    ExpectModelOutputsMatchDataSet(round_tripped, ds);
  }
}

TEST(BackendRunModelAllCases, GraphBuilderRoundTripMaterializesGraphRefAttributes) {
  auto case_data = BuildLinkedAttributeLocalFunctionCase();
  const ModelProto &original = case_data.first;
  core::builder::GraphBuilder rebuilt(original);
  ASSERT_EQ(rebuilt.Subgraphs().size(), 2u);
  ASSERT_EQ(rebuilt.Nodes().size(), 1u);

  const ModelProto round_tripped = rebuilt.ToModel(original.ir_version());
  ASSERT_EQ(round_tripped.graph().node().size(), 1);
  const NodeProto &round_tripped_call = round_tripped.graph().node(0);
  bool has_then_graph = false;
  bool has_else_graph = false;
  for (const auto &attribute : round_tripped_call.ref_attribute()) {
    if (attribute.name() == "then_branch") {
      has_then_graph = true;
      EXPECT_EQ(attribute.type(), AttributeProto::AttributeType::GRAPH);
    }
    if (attribute.name() == "else_branch") {
      has_else_graph = true;
      EXPECT_EQ(attribute.type(), AttributeProto::AttributeType::GRAPH);
    }
  }
  EXPECT_TRUE(has_then_graph);
  EXPECT_TRUE(has_else_graph);

  for (const DataSet &ds : case_data.second) {
    ExpectModelOutputsMatchDataSet(original, ds);
    ExpectModelOutputsMatchDataSet(round_tripped, ds);
  }
}

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
    if (ExcludedCaseNames().count(tc.name) != 0 ||
        BitInexactOn32BitCaseNames().count(tc.name) != 0) {
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
      // RuntimeSession per data set; the same serial session is run twice below.
      // Intermediates are released as scheduled so each run allocates and frees
      // the same buffers. Serial execution keeps the measured peak independent
      // of worker scheduling.
      core::runtime::SimpleRawBufferAllocator alloc(kAllocatorSlotCapacity);
      RuntimeContext rt(
          KernelContext(DefaultOpset(GetDefaultOpsetVersion(model))),
          core::runtime::RuntimeContextOptions{.allocator = track_peak ? &alloc : nullptr});
      rt.set_release_intermediates(true);
      RegisterModelFunctions(model, rt);

      const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
      RuntimeSession session(plan, core::runtime::RuntimeSessionOptions{
                                       .parameters = core::runtime::RuntimeParameters(1),
                                   });

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
