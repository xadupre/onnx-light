// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/sequence/include_sequence_cases.h"
#include "onnx_backend_test/kernels/sequence/include_sequence_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// IR version used by the manually-built models below.
constexpr int64_t kDefaultIrVersion = 13;

// Builds and registers one SequenceEmpty test case.
//
// SequenceEmpty has no inputs and produces a sequence value. The
// runtime harness cannot directly compare sequence outputs, so the
// model also chains a ``SequenceLength`` node on the output of
// ``SequenceEmpty`` and exposes the resulting scalar INT64 length
// (always 0) as the graph output. The ``dtype`` attribute (when
// ``has_dtype`` is true) is forwarded verbatim and used to populate
// the empty sequence's element dtype.
void RegisterSequenceEmptyCase(const std::string &name, bool has_dtype, int64_t dtype,
                               const OpsetId &opset, std::vector<TestCase> &registry) {
  const kernel::KernelContext ctx{opset};

  // Compute the expected output: SequenceLength of an empty sequence
  // is always 0.
  const Sequence empty_seq = has_dtype ? kernel::SequenceEmpty(ctx)(static_cast<int32_t>(dtype))
                                       : kernel::SequenceEmpty(ctx)();
  Tensor expected = kernel::SequenceLength(ctx)(empty_seq);
  expected.name = "length";

  TestCase tc;
  tc.name = name;
  tc.model_name = name;
  tc.kind = "node";
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  model.set_ir_version(kDefaultIrVersion);
  model.set_producer_name("backend-test");
  OperatorSetIdProto proto;
  proto.set_domain(opset.domain);
  proto.set_version(opset.version);
  model.add_opset_import(proto);

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  // Node 1: SequenceEmpty [dtype] → empty_seq.
  NodeProto *empty_node = graph->add_node();
  empty_node->set_op_type("SequenceEmpty");
  empty_node->add_output("empty_seq");
  if (has_dtype) {
    AddAttribute<int64_t>(*empty_node, "dtype", dtype);
  }

  // Node 2: SequenceLength empty_seq → length.
  NodeProto *len_node = graph->add_node();
  len_node->set_op_type("SequenceLength");
  len_node->add_input("empty_seq");
  len_node->add_output("length");

  // No graph inputs. Graph output is the scalar INT64 length.
  FillValueInfo(expected, *graph->add_output());

  DataSet ds;
  ds.outputs.push_back(expected);
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));
}

} // namespace

// ---------------------------------------------------------------------------
// SequenceEmpty — constructs an empty tensor sequence (since opset 11 in the
// ai.onnx domain).
//
// Two cases are registered:
//   * ``test_cc_sequence_empty_default``: dtype attribute omitted (defaults
//     to FLOAT).
//   * ``test_cc_sequence_empty_int64``:   dtype attribute set to INT64.
// In both cases the graph chains ``SequenceLength`` after ``SequenceEmpty``
// and the test harness compares the resulting scalar INT64 length (always
// ``0``) to the expected value computed by the reference kernels.
// ---------------------------------------------------------------------------
void RegisterSequenceEmptyCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(11);

  RegisterSequenceEmptyCase("test_cc_sequence_empty_default", /*has_dtype=*/false,
                            /*dtype=*/0, opset, registry);
  RegisterSequenceEmptyCase("test_cc_sequence_empty_int64", /*has_dtype=*/true,
                            /*dtype=*/static_cast<int64_t>(TensorProto::INT64), opset, registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
