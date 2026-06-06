// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/text/include_text_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/text/include_text_kernels.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// StringConcat — z[i] = x[i] + y[i] element-wise with NumPy-style
// broadcasting (since opset 20 in the ai.onnx domain). Inputs and outputs
// are ``tensor(string)``; the reference kernel supports equal-shape and
// scalar broadcasting.
// ---------------------------------------------------------------------------
void RegisterStringConcatCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(20);
  const kernel::KernelContext ctx{opset};
  const kernel::StringConcat string_concat{ctx};

  // Equal-shape variant: element-wise concatenation of two 1-D string
  // tensors.
  {
    NodeProto node;
    node.set_op_type("StringConcat");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromStrings("", {3}, {"abc", "", "hello "});
    Tensor y = Tensor::FromStrings("", {3}, {"def", "xyz", "world"});
    Tensor z = string_concat(x, y);

    Expect(node, {x, y}, {z}, "test_cc_string_concat", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant: z[i] = x[i] + y (scalar).
  {
    NodeProto node;
    node.set_op_type("StringConcat");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromStrings("", {2, 2}, {"a", "b", "c", "d"});
    Tensor y = Tensor::FromStrings("", {}, {"!"});
    Tensor z = string_concat(x, y);

    Expect(node, {x, y}, {z}, "test_cc_string_concat_bcast", {opset}, "backend-test", registry);
  }

  // Zero-dimensional (scalar) variant: both inputs are 0-D string tensors.
  // Mirrors upstream onnx ``test_string_concat_zero_dimensional``.
  {
    NodeProto node;
    node.set_op_type("StringConcat");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromStrings("", {}, {"cat"});
    Tensor y = Tensor::FromStrings("", {}, {"s"});
    Tensor z = string_concat(x, y);

    Expect(node, {x, y}, {z}, "test_cc_string_concat_zero_dimensional", {opset}, "backend-test",
           registry);
  }

  // UTF-8 variant: element-wise concatenation of multi-byte UTF-8 strings.
  // Mirrors upstream onnx ``test_string_concat_utf8``.
  {
    NodeProto node;
    node.set_op_type("StringConcat");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromStrings("", {2}, {"\xe7\x9a\x84", "\xe4\xb8\xad"});
    Tensor y = Tensor::FromStrings("", {2}, {"\xe7\x9a\x84", "\xe4\xb8\xad"});
    Tensor z = string_concat(x, y);

    Expect(node, {x, y}, {z}, "test_cc_string_concat_utf8", {opset}, "backend-test", registry);
  }

  // Chained variant with 3 inputs and 1 output: w = (x + y) + z, built as
  // two StringConcat nodes in a single graph. Exercises shape inference and
  // execution across multiple StringConcat invocations.
  {
    TestCase tc("test_cc_string_concat_chained_3in_1out", "test_cc_string_concat_chained_3in_1out");
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.model;
    model.set_ir_version(13);
    model.set_producer_name("backend-test");
    OperatorSetIdProto *osid = model.add_opset_import();
    osid->set_domain(opset.domain);
    osid->set_version(opset.version);

    GraphProto *graph = model.add_graph();
    graph->set_name(tc.name);

    // First node: tmp = StringConcat(x, y)
    NodeProto *node1 = graph->add_node();
    node1->set_op_type("StringConcat");
    node1->add_input("x");
    node1->add_input("y");
    node1->add_output("tmp");

    // Second node: w = StringConcat(tmp, z)
    NodeProto *node2 = graph->add_node();
    node2->set_op_type("StringConcat");
    node2->add_input("tmp");
    node2->add_input("z");
    node2->add_output("w");

    Tensor x = Tensor::FromStrings("x", {3}, {"abc", "def", "ghi"});
    Tensor y = Tensor::FromStrings("y", {3}, {"-", "/", "."});
    Tensor z = Tensor::FromStrings("z", {3}, {"123", "456", "789"});
    Tensor tmp = string_concat(x, y);
    Tensor w = string_concat(tmp, z);
    w.name = "w";

    FillValueInfo(x, *graph->add_input());
    FillValueInfo(y, *graph->add_input());
    FillValueInfo(z, *graph->add_input());
    FillValueInfo(w, *graph->add_output());

    DataSet ds;
    ds.inputs = {x, y, z};
    ds.outputs = {w};
    tc.data_sets.emplace_back(std::move(ds));

    registry.emplace_back(std::move(tc));
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
