// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Walks the whole C++ backend test registry returned by ``CollectTestCases``
// (models only, no benchmark-sized inputs, no ``_big_`` models) and, for every
// case whose graph outputs are all plain tensors, runs the model through an
// ``ExecutionPlan`` + ``RuntimeSession`` and checks that the produced outputs
// reproduce the expected ones bit-for-bit.
//
// This is the C++ counterpart of
// ``unittests/python/backend/test_backend_with_run_model.py``: both drive the
// same registry through the runtime's model-execution path. Unlike
// ``test_backend_run_model.cc`` (one ``TEST`` per registered op that only looks
// at single-node graphs), this exercises every collected case in a single loop,
// including the multi-node control-flow / shape-inference models.

#include "onnx_core/backend_test/expect.h"
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

} // namespace

// Collects every model-based backend test case (``TestMode::TEST``, no big
// models) and runs each one through a ``RuntimeSession``, verifying every data
// set's expected outputs are reproduced bit-for-bit. The expected outputs are
// themselves produced by the very kernels the runtime dispatches to, so a
// bit-exact comparison is appropriate. Outputs are matched positionally against
// ``graph.output`` (a case's expected-tensor names are not required to match
// the graph output names).
TEST(BackendRunModelAllCases, RunEveryModel) {
  std::vector<TestCase> cases = CollectTestCases();
  ASSERT_FALSE(cases.empty()) << "No backend test cases collected.";

  size_t executed = 0;
  for (TestCase &tc : cases) {
    if (ExcludedCaseNames().count(tc.name) != 0) {
      continue;
    }
    const ModelProto &model = tc.model();
    if (!AllOutputsAreTensors(model)) {
      continue;
    }
    SCOPED_TRACE(tc.name);

    const auto &outputs = model.ref_graph().ref_output();
    for (const DataSet &ds : tc.data_sets()) {
      RuntimeContext rt(KernelContext(DefaultOpset(GetDefaultOpsetVersion(model))));
      for (const Tensor &t : ds.inputs) {
        rt.Set(t.name, t);
      }
      for (const Map &m : ds.maps) {
        rt.PutMap(m.name, m);
      }

      ASSERT_NO_THROW(RunModelViaSession(model, rt))
          << "Running the model threw for case " << tc.name;

      ASSERT_EQ(ds.outputs.size(), outputs.size())
          << "Data set / graph output arity mismatch for case " << tc.name;
      for (size_t i = 0; i < outputs.size(); ++i) {
        const std::string &oname = outputs[i].name();
        ASSERT_TRUE(rt.Has(oname)) << "Missing output '" << oname << "' for case " << tc.name;
        ExpectTensorBitEqual(rt.Get(oname), ds.outputs[i]);
      }
    }
    ++executed;
  }

  EXPECT_GT(executed, 1000u) << "Expected the model-run loop to exercise many cases.";
}

} // namespace Test
