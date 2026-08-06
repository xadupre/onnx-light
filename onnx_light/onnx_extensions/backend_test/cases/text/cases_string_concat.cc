// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/text/include_text_cases.h"
#include "onnx_extensions/kernels/kernels/text/include_text_kernels.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// StringConcat — z[i] = x[i] + y[i] element-wise with NumPy-style
// broadcasting (since opset 20 in the ai.onnx domain). Inputs and outputs
// are ``tensor(string)``; the reference kernel supports equal-shape and
// scalar broadcasting.
// ---------------------------------------------------------------------------
void RegisterStringConcatCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(20);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::StringConcat string_concat{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("StringConcat");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    constexpr int64_t count = 262144;
    Expect(registry, std::move(node), "test_cc_string_concat_benchmark", {opset}, {count, count},
           {count}, [string_concat, count]() -> IoData {
             std::vector<std::string> x_values(static_cast<size_t>(count));
             std::vector<std::string> y_values(static_cast<size_t>(count));
             for (size_t i = 0; i < x_values.size(); ++i) {
               x_values[i] = (i % 2 == 0) ? "abc" : "hello ";
               y_values[i] = (i % 2 == 0) ? "def" : "world";
             }
             Tensor x = Tensor::FromStrings("", {count}, x_values);
             Tensor y = Tensor::FromStrings("", {count}, y_values);
             Tensor z = string_concat(x, y);
             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
    return;
  }

  // Equal-shape variant: element-wise concatenation of two 1-D string
  // tensors.
  {
    NodeProto node;
    node.set_op_type("StringConcat");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");
    Expect(registry, std::move(node), "test_cc_string_concat", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromStrings("", {3}, {"abc", "", "hello "});
      Tensor y = Tensor::FromStrings("", {3}, {"def", "xyz", "world"});
      Tensor z = string_concat(x, y);

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Scalar broadcast variant: z[i] = x[i] + y (scalar).
  {
    NodeProto node;
    node.set_op_type("StringConcat");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");
    Expect(registry, std::move(node), "test_cc_string_concat_bcast", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromStrings("", {2, 2}, {"a", "b", "c", "d"});
      Tensor y = Tensor::FromStrings("", {}, {"!"});
      Tensor z = string_concat(x, y);

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Zero-dimensional (scalar) variant: both inputs are 0-D string tensors.
  // Mirrors upstream onnx ``test_string_concat_zero_dimensional``.
  {
    NodeProto node;
    node.set_op_type("StringConcat");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");
    Expect(registry, std::move(node), "test_cc_string_concat_zero_dimensional", {opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromStrings("", {}, {"cat"});
             Tensor y = Tensor::FromStrings("", {}, {"s"});
             Tensor z = string_concat(x, y);

             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
  }

  // UTF-8 variant: element-wise concatenation of multi-byte UTF-8 strings.
  // Mirrors upstream onnx ``test_string_concat_utf8``.
  {
    NodeProto node;
    node.set_op_type("StringConcat");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");
    Expect(registry, std::move(node), "test_cc_string_concat_utf8", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromStrings("", {2}, {"\xe7\x9a\x84", "\xe4\xb8\xad"});
      Tensor y = Tensor::FromStrings("", {2}, {"\xe7\x9a\x84", "\xe4\xb8\xad"});
      Tensor z = string_concat(x, y);

      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // Broadcasting variant: mirrors upstream onnx
  // ``test_string_concat_broadcasting`` where ``y`` is a length-1 1-D tensor
  // broadcast across ``x`` (e.g. cats/dogs/snakes).
  {
    NodeProto node;
    node.set_op_type("StringConcat");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");
    Expect(registry, std::move(node), "test_cc_string_concat_broadcasting", {opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromStrings("", {3}, {"cat", "dog", "snake"});
             Tensor y = Tensor::FromStrings("", {1}, {"s"});
             Tensor z = string_concat(x, y);

             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
  }

  // Empty-string variant: mirrors upstream onnx
  // ``test_string_concat_empty_string`` where concatenation with empty
  // strings yields the non-empty operand unchanged.
  {
    NodeProto node;
    node.set_op_type("StringConcat");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");
    Expect(registry, std::move(node), "test_cc_string_concat_empty_string", {opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromStrings("", {2}, {"abc", ""});
             Tensor y = Tensor::FromStrings("", {2}, {"", "abc"});
             Tensor z = string_concat(x, y);

             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
  }

  // Chained variant with 3 inputs and 1 output: w = (x + y) + z, built as
  // two StringConcat nodes in a single graph. Exercises shape inference and
  // execution across multiple StringConcat invocations.
  {
    TestCase tc("test_cc_string_concat_chained_3in_1out", "test_cc_string_concat_chained_3in_1out");
    tc.rtol = 1e-3;
    tc.atol = 1e-7;

    ModelProto &model = tc.emplace_model();
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
    tmp.name = "tmp";
    Tensor w = string_concat(tmp, z);
    w.name = "w";

    FillValueInfo(x, *graph->add_input());
    FillValueInfo(y, *graph->add_input());
    FillValueInfo(z, *graph->add_input());
    FillValueInfo(w, *graph->add_output());
    FillValueInfo(tmp, *graph->add_value_info());

    DataSet ds;
    ds.inputs = {x, y, z};
    ds.outputs = {w};
    tc.data_sets().emplace_back(std::move(ds));

    registry.emplace_back(std::move(tc));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
